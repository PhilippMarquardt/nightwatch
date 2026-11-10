"""Native point-cloud tokens for TriSAR LiDAR: frozen Utonia (Pointcept, ICML 2026) features pooled onto the VLM's
16x20 merged-token grid of the 640x512 thermal reference frame (docs/PLAN_POINTCLOUD.md).

The full 360° Ouster cloud goes through Utonia (outdoor recipe: scale 0.2 -> 5 cm grid, no CenterShift, intensity as
grey "colour", zero normals). Point features (two up-cast levels concatenated, as in Utonia's outdoor demo) are then
projected into the thermal frame with the same calibration as nightwatch.data.trisar.render and mean-pooled per 32-px
cell. Output per frame: feat (320, D) float16 and count (320,) int16 (points per cell; 0 = no return).

python -m nightwatch.slow.pc_encode --smoke                # V3: one frame, timing, dims, memory
python -m nightwatch.slow.pc_encode --out data/trisar_pc   # all frames in data/trisar_fused/frames.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/Utonia")
from nightwatch.data.trisar import TH, TW, cam_pose, intrinsics, lidar_to_pixels, read_pcd, scaled_K

PACK = os.environ.get("NW_PC_PACK")
PACK_META = json.load(open(os.path.join(PACK, "meta.json"))) if PACK else None
CELL = 32
GH, GW = TH // CELL, TW // CELL          # 16 x 20 = 320 tokens, row-major like the merged ViT grid


def load_model():
    import utonia
    try:
        import flash_attn  # noqa: F401
        cfg = None
    except ImportError:
        cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    m = utonia.load("utonia", repo_id="Pointcept/Utonia", custom_config=cfg).cuda().eval()
    return m, utonia.transform.default(0.2, apply_z_positive=False)


def encode(model, transform, xyz, inten):
    """-> per-original-point features (N, D) on CPU float32."""
    i = inten.astype(np.float32)
    hi = np.percentile(i, 99) if len(i) else 1.0
    g = np.clip(i / max(hi, 1e-6) * 255.0, 0, 255)
    point = {"coord": xyz.astype(np.float32), "color": np.repeat(g[:, None], 3, 1).astype(np.float32),
             "normal": np.zeros_like(xyz, dtype=np.float32)}
    point = transform(point)
    with torch.inference_mode():
        for k in point.keys():
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda(non_blocking=True)
        point = model(point)
        for _ in range(2):
            parent, inv = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent, inv = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = point.feat[inv]
            point = parent
        return point.feat[point.inverse].float().cpu()


def project(rec, xyz, max_range=60.0):
    """Same thermal reference frame as nightwatch.data.trisar.render -> (cell index per point, valid mask)."""
    tw, th_ = rec["th_size"] if "th_size" in rec else Image.open(rec["thermal"]).size
    Kt, _ = intrinsics(rec["cal_th"])
    Kt = scaled_K(Kt, rec["wh_th"], (tw, th_))
    Kref = Kt.copy(); Kref[0] *= TW / tw; Kref[1] *= TH / th_
    Rt, tt = cam_pose(rec["cal_th"])
    u, v, z = lidar_to_pixels(xyz, Rt, tt, Kref)
    ok = (z > 0.3) & (u >= 0) & (u < TW) & (v >= 0) & (v < TH) & (np.linalg.norm(xyz, axis=1) < max_range)
    cell = (v.astype(np.int64) // CELL) * GW + (u.astype(np.int64) // CELL)
    return cell, ok


def pool(feat, cell, ok, xyz):
    """-> feat (320, D) mean-pooled, count (320,), geo (320, 3) = [min range, mean range, mean height] in metres
    (LiDAR frame; 0 where the cell has no return). geo gives the VLM metric range explicitly (tri_pc3)."""
    D = feat.shape[1]
    out = torch.zeros(GH * GW, D); cnt = torch.zeros(GH * GW)
    c = torch.from_numpy(cell[ok]); f = feat[torch.from_numpy(ok)]
    out.index_add_(0, c, f); cnt.index_add_(0, c, torch.ones(len(c)))
    out = out / cnt.clamp(min=1).unsqueeze(1)
    p = torch.from_numpy(xyz[ok].astype(np.float32))
    r = p.norm(dim=1)
    rmin = torch.full((GH * GW,), 1e9).scatter_reduce(0, c, r, reduce="amin")
    rmean = torch.zeros(GH * GW).index_add_(0, c, r) / cnt.clamp(min=1)
    zmean = torch.zeros(GH * GW).index_add_(0, c, p[:, 2]) / cnt.clamp(min=1)
    geo = torch.stack([torch.where(cnt > 0, rmin, torch.zeros(())), rmean, zmean], 1)
    return out.half().numpy(), cnt.short().numpy(), geo.half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/trisar_index.json")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--out", default="data/trisar_pc")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    recs = {r["id"]: r for r in json.load(open(a.index))}
    ids = [f["id"] for f in json.load(open(os.path.join(a.fused_dir, "frames.json")))]
    ids = ids[a.shard::a.nshard]
    model, transform = load_model()
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        ids = ids[:3]
    t0 = time.time()
    for k, fid in enumerate(ids):
        dst = os.path.join(a.out, f"{fid}.npz")
        if os.path.exists(dst) and not a.smoke:
            try:                                   # a killed run can leave a truncated file
                z = np.load(dst)
                if z["feat"].shape == (GH * GW, 1224) and z["count"].shape == (GH * GW,) and z["geo"].shape == (GH * GW, 3):
                    continue
            except Exception:
                pass
        rec = recs[fid]
        if PACK:      # quantised clouds uploaded from another machine (scripts/pc_pack.py): 5 mm xyz, uint16 intensity
            zz = np.load(os.path.join(PACK, f"{fid}.npz"))
            xyz, inten = zz["xyz"].astype(np.float64) * PACK_META["quant_m"], zz["inten"].astype(np.float32) / 65535.0
            rec = {**rec, "th_size": PACK_META["thermal_size"][fid]}
        else:
            xyz, inten = read_pcd(rec["lidar"])
        ts = time.time()
        feat = encode(model, transform, xyz, inten)
        te = time.time() - ts
        cell, ok = project(rec, xyz)
        g, cnt, geo = pool(feat, cell, ok, xyz)
        np.savez_compressed(dst, feat=g, count=cnt, geo=geo)
        if a.smoke:
            print(f"{fid}: points {len(xyz)}, in thermal view {ok.sum()}, feat dim {feat.shape[1]}, "
                  f"cells with points {int((cnt > 0).sum())}/320, encode {te:.2f}s, "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        elif k % 200 == 0:
            el = time.time() - t0
            print(f"{k}/{len(ids)} {el:.0f}s, ETA {el / max(k, 1) * (len(ids) - k) / 60:.1f} min", flush=True)
    print("PC_ENCODE_DONE", len(ids))


if __name__ == "__main__":
    main()
