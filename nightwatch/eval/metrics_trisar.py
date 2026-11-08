"""Score TriSAR-3D slow-head (report/lying/nearest/count) and fast-head predictions.

Detection is matched in the thermal reference frame (640x512) at IoU 0.5. On matched persons we score
3-class posture (upright/sitting/lying) and metric distance. Slices: indoor/outdoor, location, and
per-person range bins / occlusion / lying.

python -m nightwatch.eval.metrics_trisar --pred runs/t_fusion_l/pred_test.jsonl --out runs/t_fusion_l/pred_test_metrics.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.slow.data_trisar import POSE3, load_frames

RANGE_BINS = [(0, 10, "0-10m"), (10, 20, "10-20m"), (20, 40, "20-40m"), (40, 60, "40-60m"), (60, 1e9, "60m+")]
OBS_MIN_POINTS, OBS_MAX_RANGE = 5, 60.0   # a person is "observable" if it has >=5 LiDAR points and lies inside the render range


def observable(gp):
    return gp.get("num_points", 0) >= OBS_MIN_POINTS and gp["range_m"] < OBS_MAX_RANGE


def range_bin(r):
    return next(n for lo, hi, n in RANGE_BINS if lo <= r < hi)


def score_report(preds, frames, ids, score_thr=None):
    tp = fp = fn = 0
    pose_ok = pose_n = 0
    lying_hit = lying_n = 0
    d_err, d_rel = [], []
    per_bin = defaultdict(lambda: [0, 0])
    per_bin_derr = defaultdict(list)
    occ = [0, 0]
    obs = [0, 0]
    joint = 0
    n_gt = 0
    for i in ids:
        g = frames[i]["persons"]
        p = preds.get(i, {}).get("persons", [])
        if score_thr is not None and p and "score" in p[0]:
            p = sorted([q for q in p if q["score"] >= score_thr], key=lambda q: -q["score"])
        gb = [q["box"] for q in g]
        pb = [q["box"] for q in p]
        m = T.iou_matrix(pb, gb)
        used = np.zeros(len(g), bool)
        match = {}
        for k in range(len(p)):
            if not len(g):
                break
            cand = np.where(~used, m[k], -1)
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                match[j] = k
        tp += len(match); fp += len(p) - len(match); fn += len(g) - len(match)
        n_gt += len(g)
        for j, gp in enumerate(g):
            b = range_bin(gp["range_m"])
            per_bin[b][1] += 1
            if gp.get("occluded"):
                occ[1] += 1
            if observable(gp):
                obs[1] += 1
            if gp["pose"] == "lying":
                lying_n += 1
            if j not in match:
                continue
            per_bin[b][0] += 1
            occ[0] += int(bool(gp.get("occluded")))
            obs[0] += int(observable(gp))
            q = p[match[j]]
            gt_pose = POSE3[gp["pose"]]
            if q.get("pose") is not None:
                pose_n += 1
                pose_ok += int(q["pose"] == gt_pose)
            if (q.get("pose") == gt_pose and q.get("distance") is not None
                    and abs(q["distance"] - gp["range_m"]) <= max(1.0, 0.1 * gp["range_m"])):
                joint += 1                     # box + posture + distance (<= max(1 m, 10 %)) all right
            if gp["pose"] == "lying" and q.get("pose") == "lying":
                lying_hit += 1
            if q.get("distance") is not None:
                e = abs(q["distance"] - gp["range_m"])
                d_err.append(e); d_rel.append(e / max(gp["range_m"], 0.5)); per_bin_derr[b].append(e)
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    out = {"n_images": len(ids), "precision": P, "recall": R, "f1": 2 * P * R / max(1e-9, P + R),
           "pose_acc": pose_ok / pose_n if pose_n else float("nan"),
           "lying_recall": lying_hit / lying_n if lying_n else float("nan"),
           "dist_mae_m": float(np.mean(d_err)) if d_err else float("nan"),
           "dist_rel_err": float(np.mean(d_rel)) if d_rel else float("nan"),
           "occluded_recall": occ[0] / occ[1] if occ[1] else float("nan"),
           "observable_recall": obs[0] / obs[1] if obs[1] else float("nan"),
           "joint_report_recall": joint / n_gt if n_gt else float("nan"),
           "observable_frac": obs[1] / max(1, sum(len(frames[i]["persons"]) for i in ids))}
    for b, (h, n) in per_bin.items():
        out[f"recall_{b}"] = h / max(1, n)
        if per_bin_derr[b]:
            out[f"dist_mae_{b}"] = float(np.mean(per_bin_derr[b]))
    return out


def slices(frames):
    sl = {"all": list(frames)}
    sl["outdoor"] = [i for i, f in frames.items() if f["outside"]]
    sl["indoor"] = [i for i, f in frames.items() if not f["outside"]]
    sl["night"] = [i for i, f in frames.items() if f.get("night")]
    sl["day"] = [i for i, f in frames.items() if not f.get("night")]
    sl["has_lying"] = [i for i, f in frames.items() if any(p["pose"] == "lying" for p in f["persons"])]
    for loc in sorted({f["location"] for f in frames.values()}):
        sl[f"loc_{loc}"] = [i for i, f in frames.items() if f["location"] == loc]
    return {k: v for k, v in sl.items() if v}


def score_file(pred_path, fused_dir, split="test", score_thr=None, ids_from=None):
    frames = {f["id"]: f for f in load_frames(fused_dir, {split})}
    if ids_from:     # same-subset scoring: restrict to the frames another run was evaluated on
        keep = {json.loads(l)["id"] for l in open(ids_from)}
        frames = {k: v for k, v in frames.items() if k in keep}
    rows = defaultdict(dict)
    for line in open(pred_path):
        d = json.loads(line)
        if d["id"] in frames:
            rows[(d["task"], d["cond"])][d["id"]] = d
    res = {}
    for (task, cond), preds in sorted(rows.items()):
        key = f"{task}/{cond}"
        res[key] = {}
        for sname, ids in slices(frames).items():
            ids = [i for i in ids if i in preds]
            if not ids:
                continue
            if task in ("report", "detect"):
                res[key][sname] = score_report(preds, frames, ids, score_thr)
            elif task == "lying":
                v = [T.parse_yesno(preds[i]["text"]) for i in ids]
                g = [any(p["pose"] == "lying" for p in frames[i]["persons"]) for i in ids]
                res[key][sname] = {"n_images": len(ids), "accuracy": float(np.mean([a == b for a, b in zip(v, g)]))}
            elif task == "nearest":
                errs, fa, n_empty = [], 0, 0
                for i in ids:
                    g = [p["range_m"] for p in frames[i]["persons"]]
                    t = preds[i]["text"].strip().lower()
                    try:
                        v = float(t.split()[0].rstrip("m"))
                    except (ValueError, IndexError):
                        v = None
                    if g and v is not None:
                        errs.append(abs(v - min(g)))
                    elif g:
                        errs.append(min(g))           # said "none" with someone there: full miss
                    if not g:
                        n_empty += 1
                        fa += int(v is not None)       # a distance for nobody = hallucination
                res[key][sname] = {"n_images": len(ids), "nearest_mae_m": float(np.mean(errs)) if errs else float("nan"),
                                   "nearest_false_alarm_rate": fa / n_empty if n_empty else float("nan")}
            elif task == "count":
                v = [T.parse_int(preds[i]["text"]) for i in ids]
                g = [len(frames[i]["persons"]) for i in ids]
                res[key][sname] = {"n_images": len(ids), "count_exact": float(np.mean([a == b for a, b in zip(v, g)]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--ids-from", default=None, help="only score frames present in this prediction file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = score_file(a.pred, a.fused_dir, a.split, a.score_thr, a.ids_from)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, {m: round(x, 3) for m, x in v.get("all", {}).items() if isinstance(x, float)})


if __name__ == "__main__":
    main()
