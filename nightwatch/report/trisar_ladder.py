"""TriSAR: how the input route decides whether the 9B uses a sensor. One bar group per slow head:
held-out lab F1, unseen-forest F1 (with forest precision as a marker), and lab F1 with the camera dead.
Seeds are averaged where several exist (the number of seeds is printed in the label).

python -m nightwatch.report.trisar_ladder
"""
from __future__ import annotations

import json
import os

import numpy as np

RUNS = os.environ.get("NW_RUNS", "runs")
LADDER = [  # (label, [runs...])
    ("camera only", ["t_sft_rgb"]),
    ("camera + thermal + LiDAR\nside branch", ["t_sft_fusion_l_cd0", "t_sft_fusion_l_cd0_s1"]),
    ("thermal\nas image", ["t_sft_thermal_i"]),
    ("camera + thermal\nas images", ["t_sft_rgbt_i", "t_sft_rgbt_i_s1", "t_sft_rgbt_i_s2"]),
    ("camera + thermal + LiDAR\nas images", ["t_sft_tri", "t_sft_tri_s1", "t_sft_tri_s2"]),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    for lab, runs in LADDER:
        vals = []
        for r in runs:
            m = jl(f"{RUNS}/{r}/pred_test_metrics.json")
            if not m:
                continue
            n = m["report/normal"]
            rb = jl(f"{RUNS}/{r}/pred_test_robust_metrics.json") or {}
            cam_off = rb.get("report/rgb_off", {}).get("loc_lab", {}).get("f1")
            vals.append((n["loc_lab"]["f1"], n["loc_forest"]["f1"], n["loc_forest"]["precision"], n["all"]["f1"], cam_off))
        if vals:
            v = np.array([[x if x is not None else np.nan for x in t] for t in vals], float)
            rows.append((f"{lab}\n({len(vals)} seed{'s' if len(vals) > 1 else ''})", np.nanmean(v, 0), v))
            print(lab.replace("\n", " "), len(vals), np.round(np.nanmean(v, 0), 3))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK, INK2, SURF, save
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(rows)); w = 0.26
    cols = [("held-out lab", 0, "#9aa0a6"), ("unseen forest, night", 1, "#d95926"), ("lab, camera dead", 4, "#3987e5")]
    for k, (name, idx, c) in enumerate(cols):
        means = [r[1][idx] for r in rows]
        ax.bar(x + (k - 1) * w, np.nan_to_num(means), w * 0.92, color=c, label=name, edgecolor=SURF, linewidth=1.5)
        for xi, (r, mval) in enumerate(zip(rows, means)):
            if np.isnan(mval):
                ax.text(xi + (k - 1) * w, 0.02, "n/a", ha="center", fontsize=7, color=INK2, rotation=90)
                continue
            ax.text(xi + (k - 1) * w, mval + 0.015, f"{mval:.2f}", ha="center", fontsize=8, color=INK)
            if len(r[2]) > 1:                                   # seed range as a thin line
                lo, hi = np.nanmin(r[2][:, idx]), np.nanmax(r[2][:, idx])
                ax.plot([xi + (k - 1) * w] * 2, [lo, hi], color=INK, lw=1)
    prec = [r[1][2] if r[1][1] >= 0.01 else np.nan for r in rows]          # precision is meaningless at ~0 F1
    ax.scatter(x, prec, marker="_", s=300, color="#f0efec", zorder=4, label="forest precision")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5); ax.set_ylim(0, 1); ax.set_ylabel("person F1 @ IoU 0.5")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="upper left", ncol=4)
    ax.set_title("TriSAR-3D: the input route decides whether the 9B uses a sensor (held-out test, 2,751 frames)",
                 loc="left", fontsize=10.5)
    fig.text(0.01, 0.005, "camera-dead bar: lab subset of the fixed 600-frame robustness set, averaged over the seeds whose robustness eval "
             "has finished; thin lines = seed range; thermal-only has no camera input (n/a)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    print(save(fig, "fig_trisar_sensor_ladder"))


if __name__ == "__main__":
    main()
