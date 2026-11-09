"""TriSAR fast heads: F1 vs score threshold, per family (plain vs thermal-normalised), mean and range over seeds.

Shows whether a difference between families is an operating-point artefact (curves cross) or holds at every threshold.

python -m nightwatch.report.thr_sweep
"""
from __future__ import annotations

import json
import os

import numpy as np

from nightwatch.eval.metrics_trisar import score_file

RUNS = os.environ.get("NW_RUNS", "runs")
FUSED = os.environ.get("NW_TRISAR_FUSED", "data/trisar_fused")
FAMILIES = {"plain": ["tf_rgbtl", "tf_rgbtl_s1", "tf_rgbtl_s2"],
            "thermal-normalised": ["tf_rgbtl_tnorm_s0", "tf_rgbtl_tnorm_s1", "tf_rgbtl_tnorm_s2"]}
THRS = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
SLICES = [("all", "all held-out"), ("loc_forest", "unseen forest"), ("loc_lab", "held-out lab")]


def main():
    out = {}
    for fam, runs in FAMILIES.items():
        for r in runs:
            p = f"{RUNS}/{r}/pred_test.jsonl"
            if not os.path.exists(p):
                continue
            rows = {}
            for t in THRS:
                m = score_file(p, FUSED, score_thr=t)["report/normal"]
                rows[t] = {s: m[s]["f1"] for s, _ in SLICES}
            out[r] = {"family": fam, "val_thr": json.load(open(f"{RUNS}/{r}/train_summary.json"))["score_thr"], "f1": rows}
            print(r, {t: round(v["loc_forest"], 3) for t, v in rows.items()})
    json.dump(out, open(f"{RUNS}/thr_sweep.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK2, save
    col = {"plain": "#9aa0a6", "thermal-normalised": "#d95926"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (s, title) in zip(axs, SLICES):
        for fam in FAMILIES:
            rs = [r for r in out if out[r]["family"] == fam]
            if not rs:
                continue
            Y = np.array([[out[r]["f1"][t][s] for t in THRS] for r in rs])
            ax.fill_between(THRS, Y.min(0), Y.max(0), color=col[fam], alpha=.22, lw=0)
            ax.plot(THRS, Y.mean(0), color=col[fam], lw=2, label=f"{fam} (mean of {len(rs)} seeds, band = range)")
            for r in rs:                                    # each seed's val-tuned operating point
                t = min(THRS, key=lambda x: abs(x - out[r]["val_thr"]))
                ax.scatter([t], [out[r]["f1"][t][s]], color=col[fam], s=18, zorder=3, edgecolor="none")
        ax.set_title(title); ax.set_xlabel("fast-head score threshold"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("person F1 @ IoU 0.5"); axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("TriSAR fast head: F1 across thresholds (dots: each seed's val-tuned threshold)", color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    print(save(fig, "fig_trisar_thr_sweep"))


if __name__ == "__main__":
    main()
