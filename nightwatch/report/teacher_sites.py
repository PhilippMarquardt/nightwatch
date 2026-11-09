"""LLVIP: how good are the slow head's pseudo-labels, site by site, measured against the withheld GT of the
'unlabeled' sites (08-18), next to its F1 on the test sites (19-26). Diagnostic figure; nothing is tuned on it.

python -m nightwatch.report.teacher_sites
"""
from __future__ import annotations

import json
import os

import numpy as np

RUNS = os.environ.get("NW_RUNS", "runs")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK2, save
    unl = json.load(open(f"{RUNS}/sft_fusion/pred_unlabeled_metrics.json"))["detect/normal"]
    tst = json.load(open(f"{RUNS}/sft_fusion/pred_test_metrics.json"))["detect/normal"]
    cq = f"{RUNS}/pseudo_labels_unlabeled_consensus_quality.json"
    cons = json.load(open(cq))["consensus"] if os.path.exists(cq) else {}
    su = sorted(k for k in unl if k.startswith("site_"))
    st = sorted(k for k in tst if k.startswith("site_"))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    x = np.arange(len(su) + len(st) + 1)
    xs_u, xs_t = x[:len(su)], x[len(su) + 1:]
    ax.bar(xs_u - 0.2, [unl[s]["precision"] for s in su], 0.38, color="#d95926", label="teacher labels: precision")
    ax.bar(xs_u + 0.2, [unl[s]["recall"] for s in su], 0.38, color="#9aa0a6", label="teacher labels: recall")
    if cons:
        ax.scatter(xs_u - 0.2, [cons[s]["precision"] for s in su], marker="_", s=260, color="#f0efec", zorder=4,
                   label="after consensus filter: precision")
    ax.bar(xs_t, [tst[s]["f1"] for s in st], 0.6, color="#3987e5", label="teacher F1 on test sites")
    ax.axvline(len(su), color=INK2, lw=0.8, ls="--")
    ax.text(len(su) / 2 - 0.5, 1.04, "unlabeled sites: teacher writes the student's labels (scored vs withheld GT)",
            ha="center", fontsize=9, color=INK2)
    ax.text(len(su) + 1 + len(st) / 2 - 0.5, 1.04, "test sites", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(list(xs_u) + list(xs_t), [s.split("_")[1] for s in su + st])
    ax.set_ylim(0, 1.1); ax.set_ylabel("score"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    ax.set_title("LLVIP: fused-VLM teacher, label quality per site", loc="left")
    fig.tight_layout()
    print(save(fig, "fig_llvip_teacher_sites"))


if __name__ == "__main__":
    main()
