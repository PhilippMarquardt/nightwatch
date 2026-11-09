"""LLVIP fast students: F1 vs score threshold on the full test set (diagnostic, not a result).

Shows whether a student's F1 deficit is ranking (curve lower everywhere) or operating point (curve as high, but its
val-tuned threshold sits off the peak). Dots mark each run's val-tuned threshold, the only one used for reported numbers.

python -m nightwatch.report.llvip_thr_sweep
"""
from __future__ import annotations

import json
import os

import numpy as np

from nightwatch.eval.metrics import score_file

RUNS = os.environ.get("NW_RUNS", "runs")
STUDENTS = [("fast_gt_labeled", "labelled-only", "#9aa0a6"), ("fast_selfdistill", "self-training", "#199e70"),
            ("fast_distill", "VLM labels", "#d95926"), ("fast_distill_consensus", "VLM labels, consensus", "#9085e9"),
            ("fast_oracle", "human labels (oracle)", "#3987e5")]
THRS = [round(x, 2) for x in np.arange(0.15, 0.81, 0.05)]


def main():
    out = {}
    for r, lab, c in STUDENTS:
        p = f"{RUNS}/{r}/pred_test.jsonl"
        if not os.path.exists(p):
            continue
        f1 = [score_file(p, "data/index.json", score_thr=t)["detect/normal"]["all"]["f1"] for t in THRS]
        out[r] = {"f1": dict(zip(THRS, f1)), "val_thr": json.load(open(f"{RUNS}/{r}/train_summary.json"))["score_thr"]}
        print(f"{r:24s} max F1 {max(f1):.3f} at {THRS[int(np.argmax(f1))]}  val-thr {out[r]['val_thr']:.2f}")
    json.dump(out, open(f"{RUNS}/llvip_thr_sweep.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import save
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for r, lab, c in STUDENTS:
        if r not in out:
            continue
        ys = [out[r]["f1"][t] for t in THRS]
        ax.plot(THRS, ys, color=c, lw=2, label=lab)
        t = min(THRS, key=lambda x: abs(x - out[r]["val_thr"]))
        ax.scatter([t], [out[r]["f1"][t]], color=c, s=40, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("fast-head score threshold"); ax.set_ylabel("person F1 @ IoU 0.5 (3,463 test frames)")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="x", visible=False)
    ax.set_title("LLVIP students across thresholds (dots: val-tuned threshold used for every reported number)", loc="left",
                 fontsize=10)
    fig.tight_layout()
    print(save(fig, "fig_llvip_thr_sweep"))


if __name__ == "__main__":
    main()
