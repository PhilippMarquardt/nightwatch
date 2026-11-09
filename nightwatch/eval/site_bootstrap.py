"""Per-site results with a site-level block bootstrap (LLVIP test: 7 held-out scenes).

The unit of generalisation is the site, not the frame. Frames from one camera are strongly correlated, so a
frame-level confidence interval is far too narrow. Here: per-scene F1 for every run on the SAME frames, and a
bootstrap that resamples whole scenes (with replacement) to get intervals for each run and for differences.

python -m nightwatch.eval.site_bootstrap --runs zs_rgb sft_rgb sft_fusion fast_gt_labeled fast_distill \
    --pairs sft_fusion:sft_rgb fast_distill:fast_gt_labeled
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.data.llvip import load_index

RUNS = os.environ.get("NW_RUNS", "runs")


def load_preds(run):
    thr = None
    s = f"{RUNS}/{run}/train_summary.json"
    if os.path.exists(s):
        thr = json.load(open(s)).get("score_thr")
    out = {}
    for d in map(json.loads, open(f"{RUNS}/{run}/pred_test.jsonl")):
        if d["task"] != "detect" or d["cond"] != "normal":
            continue
        b, sc = d["boxes"], d.get("scores")
        if sc is not None and thr is not None:
            order = sorted([k for k, v in enumerate(sc) if v >= thr], key=lambda k: -sc[k])
            b = [b[k] for k in order]
        out[d["id"]] = b
    return out


def counts(preds, gts, ids, scene_of):
    c = {}
    for i in ids:
        tp, fp, fn, _ = T.greedy_match(preds.get(i, []), gts[i])
        s = c.setdefault(scene_of[i], np.zeros(3))
        s += (tp, fp, fn)
    return c


def f1(tp, fp, fn):
    return 2 * tp / max(1e-9, 2 * tp + fp + fn)


NAMES = {"zs_rgb": "VLM zero-shot, RGB", "zs_dual": "VLM zero-shot, RGB + thermal image", "sft_rgb": "VLM fine-tuned, RGB",
         "sft_thermal": "VLM fine-tuned, thermal", "sft_dual": "VLM fine-tuned, 2 images", "sft_fusion": "VLM fine-tuned, fused",
         "fast_gt_labeled": "fast, 7 labelled sites", "fast_distill": "fast, + VLM pseudo-labels",
         "fast_selfdistill": "fast, + own pseudo-labels", "fast_distill_rgbteacher": "fast, + RGB-VLM pseudo-labels",
         "fast_gt_labeled_long": "fast, labelled only, matched compute", "fast_oracle": "fast, all sites labelled (oracle)"}


def fig_sites(res):
    """One row per run: per-site F1 (dots), pooled F1 (bar) with the site-bootstrap 95 % interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK, INK2, save
    runs = list(res["runs"])
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(runs) + 1.4))
    for k, r in enumerate(runs[::-1]):
        v = res["runs"][r]
        ax.plot(v["ci95"], [k, k], color=INK2, lw=6, alpha=.35, solid_capstyle="round")
        ax.scatter(list(v["per_scene"].values()), [k] * len(v["per_scene"]), s=16, color=INK2, alpha=.8, zorder=3)
        ax.scatter([v["f1"]], [k], s=70, color="#3987e5", zorder=4, marker="D")
        ax.text(1.005, k, f'{v["f1"]:.3f}', va="center", fontsize=9, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(runs)), [NAMES.get(r, r) for r in runs[::-1]])
    ax.set_xlabel("person F1 @ IoU 0.5 (dots: individual held-out sites; band: 95 % site-bootstrap interval)")
    ax.set_title(f'LLVIP: {res["n_frames"]} test frames from {len(res["scenes"])} unseen sites, same frames for every model')
    fig.tight_layout()
    return save(fig, "fig_llvip_sites")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="candidate:baseline")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--out", default=f"{RUNS}/site_bootstrap.json")
    ap.add_argument("--fig", action="store_true", help="also write report/fig_llvip_sites.png")
    a = ap.parse_args()
    runs = [r for r in a.runs if os.path.exists(f"{RUNS}/{r}/pred_test.jsonl")]
    recs = load_index(a.index, ("test",))
    gts = {r.id: r.boxes for r in recs}
    scene_of = {r.id: r.scene for r in recs}
    P = {r: load_preds(r) for r in runs}
    ids = sorted(set.intersection(*[set(p) for p in P.values()]) & set(gts))   # same frames for every run
    scenes = sorted({scene_of[i] for i in ids})
    C = {r: counts(P[r], gts, ids, scene_of) for r in runs}
    M = {r: np.stack([C[r][s] for s in scenes]) for r in runs}                  # scenes x (tp, fp, fn)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(scenes), size=(a.reps, len(scenes)))
    res = {"n_frames": len(ids), "scenes": scenes,
           "frames_per_scene": {s: sum(scene_of[i] == s for i in ids) for s in scenes}, "runs": {}, "pairs": {}}
    boot = {}
    for r in runs:
        tot = M[r].sum(0)
        b = M[r][idx].sum(1)                                                     # reps x 3
        boot[r] = 2 * b[:, 0] / np.maximum(1e-9, 2 * b[:, 0] + b[:, 1] + b[:, 2])
        res["runs"][r] = {"f1": f1(*tot), "ci95": [float(np.percentile(boot[r], 2.5)), float(np.percentile(boot[r], 97.5))],
                          "per_scene": {s: f1(*M[r][k]) for k, s in enumerate(scenes)}}
    for p in a.pairs:
        c, b = p.split(":")
        if c in boot and b in boot:
            d = boot[c] - boot[b]
            ps = [res["runs"][c]["per_scene"][s] - res["runs"][b]["per_scene"][s] for s in scenes]
            res["pairs"][p] = {"delta": res["runs"][c]["f1"] - res["runs"][b]["f1"],
                               "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                               "p_le_0": float((d <= 0).mean()), "scenes_better": int(sum(x > 0 for x in ps)),
                               "n_scenes": len(scenes)}
    json.dump(res, open(a.out, "w"), indent=1)
    if a.fig:
        fig_sites(res)
    print(f"{len(ids)} frames, {len(scenes)} scenes")
    for r, v in res["runs"].items():
        print(f"{r:24s} F1 {v['f1']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  " +
              " ".join(f"{s}:{x:.2f}" for s, x in v["per_scene"].items()))
    for p, v in res["pairs"].items():
        print(f"{p:40s} delta {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  better on {v['scenes_better']}/{v['n_scenes']} sites")


if __name__ == "__main__":
    main()
