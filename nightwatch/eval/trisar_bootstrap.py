"""Paired bootstrap for TriSAR slow heads: is model A better than model B on the same test frames?

The primary contrast of the follow-up plan is three images (`tri`) vs camera + thermal (`rgbt_i`), seed-paired.
TriSAR's test set has only 6 sequences (4 forest, 2 lab), so two resampling units are reported:
- sequence: resample whole sequences (honest, but only 6 units, so the interval is coarse)
- block: resample blocks of 50 consecutive frames within the test set (keeps short-range correlation, more units)

Counts use the same greedy IoU-0.5 matching as nightwatch.eval.metrics_trisar (report task, normal condition).

python -m nightwatch.eval.trisar_bootstrap --pairs t_sft_tri:t_sft_rgbt_i t_sft_tri_s1:t_sft_rgbt_i_s1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.slow.data_trisar import load_frames

BLOCK = 50


def counts(run, frames, slice_fn=None):
    P = {}
    for line in open(f"runs/{run}/pred_test.jsonl"):
        d = json.loads(line)
        if d.get("task") == "report" and d.get("cond") == "normal":
            P[d["id"]] = d
    out = {}
    for i, f in frames.items():
        g = [q["box"] for q in f["persons"]]
        p = [q["box"] for q in P.get(i, {}).get("persons", [])]
        m = T.iou_matrix(p, g)
        used = np.zeros(len(g), bool)
        tp = 0
        for k in range(len(p)):
            if not len(g):
                break
            cand = np.where(~used, m[k], -1)
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                tp += 1
        out[i] = (tp, len(p) - tp, len(g) - tp)
    return out


def f1(c):
    return 2 * c[..., 0] / np.maximum(1e-9, 2 * c[..., 0] + c[..., 1] + c[..., 2])


def units(frames, kind):
    ids = sorted(frames)
    if kind == "sequence":
        keys = sorted({frames[i]["scene"] for i in ids})
        return [[i for i in ids if frames[i]["scene"] == k] for k in keys]
    out = []
    for sc in sorted({frames[i]["scene"] for i in ids}):
        s = [i for i in ids if frames[i]["scene"] == sc]
        out += [s[k:k + BLOCK] for k in range(0, len(s), BLOCK)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B run-name pairs (A - B)")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--out", default="runs/trisar_bootstrap.json")
    a = ap.parse_args()
    all_frames = {f["id"]: f for f in load_frames(a.fused_dir, {"test"})}
    res = {}
    for pair in a.pairs:
        A, B = pair.split(":")
        if not (os.path.exists(f"runs/{A}/pred_test.jsonl") and os.path.exists(f"runs/{B}/pred_test.jsonl")):
            print("skip", pair); continue
        cA, cB = counts(A, all_frames), counts(B, all_frames)
        res[pair] = {}
        for sl, frames in [("all", all_frames),
                           ("forest", {k: v for k, v in all_frames.items() if v["location"] == "forest"}),
                           ("lab", {k: v for k, v in all_frames.items() if v["location"] == "lab"})]:
            for kind in ("sequence", "block"):
                U = units(frames, kind)
                UA = np.array([np.sum([cA[i] for i in u], 0) for u in U], float)
                UB = np.array([np.sum([cB[i] for i in u], 0) for u in U], float)
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(U), size=(a.reps, len(U)))
                d = f1(UA[idx].sum(1)) - f1(UB[idx].sum(1))
                delta = float(f1(UA.sum(0)) - f1(UB.sum(0)))
                r = {"delta": delta, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "p_le_0": float((d <= 0).mean()), "n_units": len(U)}
                res[pair][f"{sl}/{kind}"] = r
                print(f"{A} - {B}  {sl:6s} {kind:8s} ({len(U):3d} units): {delta:+.3f} "
                      f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  P(<=0) {r['p_le_0']:.3f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
