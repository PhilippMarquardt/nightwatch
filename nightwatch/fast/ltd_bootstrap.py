"""Track B statistics: paired clip-level bootstrap on the LTDv2 test months (clips are the independent units).

For each seed and contrast, resample test clips with replacement (same clips for both arms) and recompute pooled F1 at
each arm's own (LLVIP-val) threshold. Contrasts: B3-B1 (pre-registered primary), B1-B0p (pre-registered secondary),
B2-B1 (post hoc), plus B2-B0p and B4-B2 for context.

python -m nightwatch.fast.ltd_bootstrap   (on the pod: needs data/cache_ltd/test_gt.json)
"""
from __future__ import annotations

import json
import os

import numpy as np

from nightwatch.data import tasks as T

ARMS = ["B0p", "B1_self", "B2_vlm", "B3_consensus", "B4_gt"]
CONTRASTS = [("B3_consensus", "B1_self", "primary (pre-registered)"), ("B1_self", "B0p", "secondary (pre-registered)"),
             ("B2_vlm", "B1_self", "post hoc"), ("B2_vlm", "B0p", "context"), ("B4_gt", "B2_vlm", "context")]


def per_clip(run, gt, ids, iou=0.5):
    thr = json.load(open(f"{run}/train_summary.json"))["score_thr"]
    P = {d["id"]: d for d in map(json.loads, open(f"{run}/pred_ltd_test.jsonl"))}
    clips = {}
    for i in ids:
        p = P[i]
        b = [x for x, s in sorted(zip(p["boxes"], p["scores"]), key=lambda t: -t[1]) if s >= thr]
        tp, fp, fn, _ = T.greedy_match(b, gt[i]["boxes"], iou)
        c = clips.setdefault(gt[i]["clip"], np.zeros(3))
        c += (tp, fp, fn)
    return clips


def f1(c):
    return 2 * c[..., 0] / np.maximum(1e-9, 2 * c[..., 0] + c[..., 1] + c[..., 2])


def main(reps=5000):
    gt = json.load(open("data/cache_ltd/test_gt.json"))
    ids = json.load(open("data/cache_ltd/test_ids.json"))
    out = {"n_frames": len(ids), "n_clips": len({gt[i]["clip"] for i in ids}), "per_seed": {}, "contrasts": {}}
    C = {}
    for s in (0, 1, 2):
        for a in ARMS:
            r = f"runs/ltd_{a}_s{s}"
            if os.path.exists(f"{r}/pred_ltd_test.jsonl"):
                C[(a, s)] = per_clip(r, gt, ids)
    clips = sorted({gt[i]["clip"] for i in ids})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(clips), size=(reps, len(clips)))
    for (a, s), c in C.items():
        M = np.stack([c[k] for k in clips])
        out["per_seed"][f"{a}_s{s}"] = float(f1(M.sum(0)))
    for hi, lo, kind in CONTRASTS:
        rows = []
        for s in (0, 1, 2):
            if (hi, s) not in C or (lo, s) not in C:
                continue
            A = np.stack([C[(hi, s)][k] for k in clips]); B = np.stack([C[(lo, s)][k] for k in clips])
            d = f1(A[idx].sum(1)) - f1(B[idx].sum(1))
            rows.append({"seed": s, "delta": float(f1(A.sum(0)) - f1(B.sum(0))),
                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         "p_le_0": float((d <= 0).mean())})
        out["contrasts"][f"{hi} - {lo}"] = {"kind": kind, "seeds": rows}
        print(f"{hi:13s} - {lo:13s} [{kind}]: " + "; ".join(f"s{r['seed']} {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" for r in rows))
    os.makedirs("runs/ltd", exist_ok=True)
    json.dump(out, open("runs/ltd/bootstrap.json", "w"), indent=1)
    print("clips", out["n_clips"], "frames", out["n_frames"])


if __name__ == "__main__":
    main()
