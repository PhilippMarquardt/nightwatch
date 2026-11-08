"""TriSAR-3D (TU Darmstadt, CC-BY-4.0): LiDAR (Ouster 128ch) + RGB + thermal, 3D person boxes with pose tags.

Everything is rendered into ONE reference frame: the (undistorted) thermal camera.
    rgb      warped with the infinite-depth homography K_t R_tr K_r^-1. The two cameras sit
             ~10 cm apart, so the residual parallax is a few px at >3 m -- far below one
             32-px visual token.
    lidar    projected with a z-buffer into the thermal image -> 3 channels
             (inverse range, intensity, hit mask), max-dilated to close scan-line gaps.
    boxes    3D boxes (lidar frame, axis-aligned mins/maxs) -> 8 corners -> tight 2D box
             in the thermal frame; per person we keep pose tag + 3D range.
So the RGB+T(+L) stream lands on the exact token grid the LLVIP pipeline already uses.

Layout (release): <root>/dataset.json + <root>/rosbag2_*/{links,samples,calibrations,annotations,split}.json
                  + samples/{lidar/*.pcd, rgb/*.jpg, thermal/*.jpg}
Annotations are keyed by the *lidar* sample token.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

TW, TH = 640, 512          # thermal native resolution = reference frame (5:4, same as LLVIP)
POSES = ["lying", "sitting", "walking", "standing"]   # priority order when several tags are set

# Site-held-out split (ours). The official split.json is frame-level *within* each sequence, so test
# frames are near-duplicates of training frames. We hold out whole sequences instead:
#   test = every forest sequence (a site never seen in training, recorded at night)
#          + two lab sequences with lying persons (one day/outdoor, one night/indoor)
#   val  = one night (outdoor lab) and one lying/sitting sequence; train = the remaining 15 sequences.
SITE_SPLIT = {
    "rosbag2_2025_11_17-17_55_10": "test", "rosbag2_2025_11_17-17_56_29": "test",
    "rosbag2_2025_11_17-17_59_40": "test", "rosbag2_2025_11_17-18_03_11": "test",
    "rosbag2_2025_09_18-16_28_32": "test", "rosbag2_2025_11_14-18_08_17": "test",
    "rosbag2_2025_11_14-17_15_15": "val", "rosbag2_2025_09_18-16_19_41": "val",
}


def site_split(scene: str) -> str:
    return next((v for k, v in SITE_SPLIT.items() if scene.startswith(k)), "train")


def is_night(scene: str) -> bool:
    """November recordings started after 17:00 local time -> after sunset in Darmstadt."""
    return "_2025_11_" in scene


# ---------------------------------------------------------------- geometry
def quat_to_R(q):
    """xyzw quaternion -> rotation matrix (scipy convention, as used by the dataset tools)."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_pose(cal):
    """Camera pose in the lidar frame (lidar calibration is identity): x_lidar = R x_cam + t."""
    return quat_to_R(cal["rotation"]), np.asarray(cal["translation"], dtype=np.float64)


def intrinsics(cal):
    K = np.asarray(cal.get("camera_intrinisc") or cal.get("camera_intrinsic"), dtype=np.float64).reshape(3, 3)
    d = cal.get("distortion_coefficients")
    return K, (np.asarray(d, dtype=np.float64) if d is not None else np.zeros(5))


def scaled_K(K, nominal_wh, actual_wh):
    K = K.copy()
    sx, sy = actual_wh[0] / nominal_wh[0], actual_wh[1] / nominal_wh[1]
    K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy
    return K


def lidar_to_pixels(xyz, R, t, K):
    """lidar points -> (u, v, depth). depth<=0 means behind the camera."""
    pc = (xyz - t) @ R          # R^T (x - t), row-vector form
    z = pc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = pc[:, 0] / zs * K[0, 0] + K[0, 2]
    v = pc[:, 1] / zs * K[1, 1] + K[1, 2]
    return u, v, z


def box_corners(mins, maxs):
    return np.array([[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])])


# ---------------------------------------------------------------- io
def read_pcd(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            k, _, v = line.partition(" ")
            header[k] = v
            if k == "DATA":
                break
        fields = header["FIELDS"].split()
        n = int(header["POINTS"])
        if header["DATA"].lower() == "ascii":
            import pandas as pd
            arr = pd.read_csv(io.BytesIO(f.read()), sep=" ", header=None, dtype=np.float32, engine="c").to_numpy()
        else:
            sizes = [int(s) for s in header["SIZE"].split()]
            types = header["TYPE"].split()
            dt = np.dtype([(fn, {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}[(ty, sz)])
                           for fn, ty, sz in zip(fields, types, sizes)])
            s = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            arr = np.stack([s[fn].astype(np.float32) for fn in fields], 1)
    xyz = arr[:, :3].astype(np.float64)
    inten = arr[:, fields.index("intensity")] if "intensity" in fields else np.zeros(len(arr), np.float32)
    ok = np.isfinite(xyz).all(1)
    return xyz[ok], inten[ok]


def _find_root(root: Path) -> Path:
    for p in [root, *root.glob("*"), *root.glob("*/*")]:
        if (p / "dataset.json").is_file():
            return p
    raise FileNotFoundError(f"dataset.json not found under {root}")


def _split_lookup(scene_dir: Path):
    """split.json format is not documented; accept {split: [tokens]} or [{token, split}] or {token: split}."""
    p = scene_dir / "split.json"
    if not p.is_file():
        return {}
    d = json.load(open(p))
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list):
                for tok in v:
                    out[tok if isinstance(tok, str) else json.dumps(tok)] = k
            elif isinstance(v, str):
                out[k] = v
    elif isinstance(d, list):
        for row in d:
            if isinstance(row, dict) and "split" in row:
                out[row.get("token") or row.get("lidar_token") or row.get("sample_token")] = row["split"]
    return out


def build_index(root: str, out: str):
    root = _find_root(Path(root))
    meta = {m["subset_folder"]: m for m in json.load(open(root / "dataset.json"))}
    recs, unknown_split = [], 0
    for scene, m in meta.items():
        sd = root / scene
        if not (sd / "links.json").is_file():
            continue
        links = json.load(open(sd / "links.json"))
        samples = {s["token"]: s for s in json.load(open(sd / "samples.json"))}
        cals = {c["token"]: c["calibration"] for c in json.load(open(sd / "calibrations.json"))}
        anns = {}
        for a in json.load(open(sd / "annotations.json")):
            anns.setdefault(a["sample_token"], []).append(a)
        split = _split_lookup(sd)
        for lk in links:
            lid, th, rgb = samples.get(lk["lidar_token"]), samples.get(lk["thermal_token"]), samples.get(lk["rgb_token"])
            if not (lid and th and rgb):
                continue
            official = split.get(lk["lidar_token"]) or split.get(lk["token"])
            if official is None:
                unknown_split += 1
            sp = site_split(scene)
            persons = []
            for a in anns.get(lk["lidar_token"], []):
                tags = a.get("tags", {})
                pose = next((p for p in POSES if tags.get(p)), "standing")
                c = np.asarray(a["translation"], dtype=float)
                persons.append({"mins": a["mins"], "maxs": a["maxs"], "pose": pose, "occluded": bool(tags.get("occluded")),
                                "range_m": float(np.linalg.norm(c[:2])), "track_id": a.get("track_id"),
                                "num_points": a.get("num_points", 0)})
            recs.append({
                "id": lk["lidar_token"], "scene": scene, "location": m.get("location", scene),
                "outside": bool(m.get("isOutside", False)), "weather": m.get("weather", []), "action": m.get("action", []),
                "split": sp, "official_split": official, "night": is_night(scene), "sync_ms": 1000 * float(lk.get("max_time_diff", 0.0)),
                "lidar": str(sd / "samples" / lid["filename"].lstrip("/")),
                "rgb": str(sd / "samples" / rgb["filename"].lstrip("/")),
                "thermal": str(sd / "samples" / th["filename"].lstrip("/")),
                "cal_rgb": cals[rgb["calibrated_sensor_token"]], "cal_th": cals[th["calibrated_sensor_token"]],
                "wh_rgb": [rgb["width"], rgb["height"]], "wh_th": [th["width"], th["height"]],
                "persons": persons, "t_ns": lid["timestamp"],
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(recs, open(out, "w"))
    return recs, unknown_split


# ---------------------------------------------------------------- rendering into the thermal frame
def render(rec, max_range=60.0, dilate=3):
    """-> rgb_w (TH,TW,3) u8, thermal (TH,TW) u8, lidar (TH,TW,3) u8, boxes [[x1,y1,x2,y2]] (thermal px), keep mask."""
    th = cv2.imread(rec["thermal"], cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(cv2.imread(rec["rgb"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    Kt, dt = intrinsics(rec["cal_th"]); Kr, dr = intrinsics(rec["cal_rgb"])
    Kt = scaled_K(Kt, rec["wh_th"], (th.shape[1], th.shape[0]))
    Kr = scaled_K(Kr, rec["wh_rgb"], (rgb.shape[1], rgb.shape[0]))
    # reference frame = undistorted thermal, resized to TWxTH (pinhole after undistortion)
    th_u = cv2.undistort(th, Kt, dt)
    sx, sy = TW / th.shape[1], TH / th.shape[0]
    Kref = Kt.copy(); Kref[0] *= sx; Kref[1] *= sy
    th_u = cv2.resize(th_u, (TW, TH), interpolation=cv2.INTER_LINEAR)
    Rt, tt = cam_pose(rec["cal_th"]); Rr, tr = cam_pose(rec["cal_rgb"])
    # RGB: undistort, then infinite homography into the thermal frame
    rgb_u = cv2.undistort(rgb, Kr, dr)
    Htr = Kref @ (Rt.T @ Rr) @ np.linalg.inv(Kr)
    rgb_w = cv2.warpPerspective(rgb_u, Htr, (TW, TH), flags=cv2.INTER_LINEAR, borderValue=0)
    # LiDAR -> z-buffered maps
    xyz, inten = read_pcd(rec["lidar"])
    u, v, z = lidar_to_pixels(xyz, Rt, tt, Kref)
    rng = np.linalg.norm(xyz, axis=1)
    ok = (z > 0.3) & (u >= 0) & (u < TW) & (v >= 0) & (v < TH) & (rng < max_range)
    ui, vi, r, it = u[ok].astype(np.int32), v[ok].astype(np.int32), rng[ok], inten[ok]
    order = np.argsort(-r)                      # far first, near overwrites
    inv = np.zeros((TH, TW), np.float32); ins = np.zeros((TH, TW), np.float32); hit = np.zeros((TH, TW), np.uint8)
    inv[vi[order], ui[order]] = 1.0 / np.maximum(r[order], 0.5)
    ins[vi[order], ui[order]] = it[order]
    hit[vi[order], ui[order]] = 255
    if dilate > 1:
        k = np.ones((dilate, dilate), np.uint8)
        inv = cv2.dilate(inv, k); ins = cv2.dilate(ins, k); hit = cv2.dilate(hit, k)
    imax = np.percentile(it, 99) if len(it) else 1.0
    lidar = np.stack([np.clip(inv * 2 * 255, 0, 255),            # 1/r scaled: 0.5 m -> 255, 4 m -> 64, 40 m -> 6
                      np.clip(ins / max(imax, 1e-6) * 255, 0, 255), hit], -1).astype(np.uint8)
    # boxes
    boxes, keep = [], []
    for p in rec["persons"]:
        cu, cv_, cz = lidar_to_pixels(box_corners(p["mins"], p["maxs"]), Rt, tt, Kref)
        if (cz <= 0.1).any():
            keep.append(False); boxes.append([0, 0, 0, 0]); continue
        x1, y1, x2, y2 = cu.min(), cv_.min(), cu.max(), cv_.max()
        a0 = (x2 - x1) * (y2 - y1)
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(TW, x2), min(TH, y2)
        a1 = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        keep.append(a0 > 0 and a1 / a0 >= 0.5)
        boxes.append([float(cx1), float(cy1), float(cx2), float(cy2)])
    return rgb_w, th_u, lidar, boxes, keep


def _render_to_disk(args):
    rec, out_dir = args
    try:
        rgb_w, th, lid, boxes, keep = render(rec)
    except Exception as e:  # corrupt frame: skip, but say so
        return rec["id"], None, repr(e)[:200]
    base = os.path.join(out_dir, rec["id"])
    cv2.imwrite(base + "_rgb.jpg", cv2.cvtColor(rgb_w, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(base + "_t.png", th)
    cv2.imwrite(base + "_l.png", cv2.cvtColor(lid, cv2.COLOR_RGB2BGR))
    persons = [{**{k: v for k, v in p.items() if k not in ("mins", "maxs")}, "box": b}
               for p, b, k in zip(rec["persons"], boxes, keep) if k]
    return rec["id"], persons, None


def preprocess(index_path, out_dir, workers=16, stride=1):
    """Render every `stride`-th frame into aligned PNG/JPG triplets + a compact frame index."""
    recs = json.load(open(index_path))
    recs = [r for i, r in enumerate(sorted(recs, key=lambda r: (r["scene"], r["t_ns"]))) if i % stride == 0]
    os.makedirs(out_dir, exist_ok=True)
    frames, errors = [], 0
    with ProcessPoolExecutor(workers) as ex:
        for rec, (iid, persons, err) in zip(recs, ex.map(_render_to_disk, [(r, out_dir) for r in recs], chunksize=8)):
            if err:
                errors += 1
                continue
            frames.append({"id": iid, "scene": rec["scene"], "location": rec["location"], "outside": rec["outside"],
                           "night": rec["night"], "weather": rec["weather"], "split": rec["split"],
                           "official_split": rec["official_split"], "sync_ms": rec["sync_ms"], "persons": persons})
    json.dump(frames, open(os.path.join(out_dir, "frames.json"), "w"))
    print(f"rendered {len(frames)} frames, {errors} errors -> {out_dir}")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default="data/trisar_index.json")
    ap.add_argument("--out", default="data/trisar_fused")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    recs, unk = build_index(a.root, a.index)
    from collections import Counter
    print(len(recs), "frames;", "split", Counter(r["split"] for r in recs), "unknown split", unk)
    print("locations", Counter(r["location"] for r in recs))
    print("poses", Counter(p["pose"] for r in recs for p in r["persons"]))
    preprocess(a.index, a.out, a.workers, a.stride)
