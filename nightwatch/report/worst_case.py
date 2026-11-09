"""LLVIP: average F1 vs worst-single-sensor-failure F1 for every head (the reason to fuse, in one picture).

Worst case = min over the sensor-failure conditions the head was evaluated under (camera dead, thermal dead). A head that
does not use a sensor is unaffected by losing it, so only the conditions present in its metrics count. Slow heads:
normal F1 on 1,500 frames, failures on the fixed 800-frame subset; fast heads: full test set for both.

python -m nightwatch.report.worst_case
"""
from __future__ import annotations

import json
import os

RUNS = os.environ.get("NW_RUNS", "runs")
HEADS = [("zs_rgb", "VLM zero-shot, RGB", "#8a8983"), ("zs_dual", "VLM zero-shot, 2 images", "#6b6a65"),
         ("sft_rgb", "VLM, RGB only", "#199e70"), ("sft_thermal", "VLM, thermal only", "#e66767"),
         ("sft_dual", "VLM, 2 images", "#c98500"), ("sft_fusion", "VLM, thermal fused", "#3987e5"),
         ("fast_gt_labeled", "fast head, labelled", "#9085e9"), ("fast_selfdistill", "fast head, self-training", "#9085e9"),
         ("fast_oracle", "fast head, oracle", "#9085e9")]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    pts = []
    for r, lab, c in HEADS:
        m = {**jl(f"{RUNS}/{r}/pred_test_robust_metrics.json"), **jl(f"{RUNS}/{r}/pred_test_metrics.json")}
        if "detect/normal" not in m:
            continue
        fails = [m[k]["all"]["f1"] for k in ("detect/rgb_off", "detect/thermal_off") if k in m]
        if not fails:
            continue
        pts.append((r, lab, c, m["detect/normal"]["all"]["f1"], min(fails)))
        print(f"{lab:28s} mean {pts[-1][3]:.3f}  worst failure {pts[-1][4]:.3f}")
    json.dump([{"run": p[0], "f1": p[3], "worst_failure_f1": p[4]} for p in pts], open(f"{RUNS}/worst_case.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK, INK2, save
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r, lab, c, x, y in pts:
        ax.scatter(x, y, s=120 if r.startswith("sft") else 60, color=c, zorder=3, edgecolor="white", linewidth=0.8,
                   marker="o" if not r.startswith("fast") else "s")
        off = {"zs_rgb": (-40, 14), "sft_rgb": (-30, 26), "sft_thermal": (-30, 14), "fast_gt_labeled": (-70, -16),
               "fast_oracle": (8, -14), "fast_selfdistill": (8, 4)}.get(r, (7, -3))
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6) if r in ("zs_rgb", "sft_rgb", "sft_thermal") else None)
    ax.set_xlabel("person F1, all sensors working"); ax.set_ylabel("person F1, worst single sensor failure")
    ax.set_xlim(0.66, 0.96); ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title("LLVIP, 7 unseen sites: average vs worst case (circles: 9B VLM heads, squares: 12M fast heads)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    print(save(fig, "fig_llvip_worst_case"))


if __name__ == "__main__":
    main()
