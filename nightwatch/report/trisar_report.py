"""TriSAR-3D section of the report: the sensor ladder RGB -> +thermal -> +LiDAR.

python -m nightwatch.report.trisar_report     (also called from make_report)
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from nightwatch.report.make_report import COL, GRAY, GRID, INK, INK2, OUT, RUNS, SURF, VLM, hbar, jload, save

FUSED = os.environ.get("NW_TRISAR_FUSED", "data/trisar_fused")
TCOL = {"t_sft_fusion_l_cd50": "#9085e9", "t_zoom": "#c98500", "t_zs_rgb": GRAY, "t_sft_rgb": "#199e70", "t_sft_fusion": "#d95926", "t_sft_fusion_l": "#3987e5",
        "tf_rgb": "#199e70", "tf_rgbt": "#d95926", "tf_tl": "#9085e9", "tf_rgbtl": "#3987e5"}
TNAME = {"t_sft_fusion_l_cd50": "slow head, RGB+T+LiDAR, camera-dropout 0.5", "t_zoom": "zoom cascade (fast proposes, slow inspects)",
         "t_zs_rgb": f"{VLM} zero-shot, RGB", "t_sft_rgb": "slow head, RGB", "t_sft_fusion": "slow head, RGB + thermal",
         "t_sft_fusion_l": "slow head, RGB + thermal + LiDAR",
         "tf_rgb": "fast head, RGB", "tf_rgbt": "fast head, RGB + thermal", "tf_tl": "fast head, thermal + LiDAR",
         "tf_rgbtl": "fast head, RGB + thermal + LiDAR"}
SLOW = ["t_zs_rgb", "t_sft_rgb", "t_sft_fusion", "t_sft_fusion_l", "t_sft_fusion_l_cd50", "t_zoom"]
FAST = ["tf_rgb", "tf_rgbt", "tf_tl", "tf_rgbtl"]


def m(run, kind="pred_test_metrics"):
    return jload(f"{RUNS}/{run}/{kind}.json") or {}


def g(d, key, sl, metric):
    try:
        v = d[key][sl][metric]
        return None if v != v else v
    except Exception:
        return None


def fig_ladder():
    metrics = [("f1", "person F1"), ("pose_acc", "posture accuracy"), ("lying_recall", "lying-person recall"),
               ("dist_mae_m", "distance error (m) - lower is better")]
    runs = [r for r in SLOW + FAST if m(r)]
    if not runs:
        return None
    fig, axs = plt.subplots(1, 4, figsize=(15, 0.42 * len(runs) + 1.6), sharey=True)
    for ax, (k, title) in zip(axs, metrics):
        vals = [g(m(r), "report/normal", "all", k) for r in runs]
        y = np.arange(len(runs))[::-1]
        ax.barh(y, [v or 0 for v in vals], color=[TCOL[r] for r in runs], height=0.62, edgecolor=SURF, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text((v or 0) * 1.02 + 0.005, yi, "-" if v is None else f"{v:.2f}", va="center", color=INK, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max([v or 0 for v in vals] + [1e-3]) * 1.3)
    axs[0].set_yticks(np.arange(len(runs))[::-1], [TNAME[r] for r in runs])
    fig.suptitle("TriSAR-3D held-out sites (unseen forest at night + held-out lab scenes): sensor ladder (the RGB+T middle rung was replaced by the camera-dropout run)",
                 color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, "fig_trisar_ladder")


def fig_distance():
    bins = ["0-10m", "10-20m", "20-40m", "40-60m", "60m+"]
    runs = [r for r in ["t_sft_rgb", "t_sft_fusion", "t_sft_fusion_l", "tf_rgbtl"] if m(r)]
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        vals = [g(m(r), "report/normal", "all", f"dist_mae_{b}") or 0 for b in bins]
        ax.bar(np.arange(len(bins)) + (k - (len(runs) - 1) / 2) * w, vals, width=w * 0.92, color=TCOL[r], label=TNAME[r],
               edgecolor=SURF, linewidth=2)
    ax.set_xticks(range(len(bins)), bins); ax.set_ylabel("mean |distance error| (m)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9)
    ax.set_title("Distance error by range (matched persons only; see joint recall for coverage)")
    return save(fig, "fig_trisar_distance")


def fig_robust():
    conds = [("normal", "all sensors"), ("rgb_off", "camera dead"), ("thermal_off", "thermal dead"), ("lidar_off", "LiDAR dead")]
    runs = [r for r in ["t_sft_fusion", "t_sft_fusion_l", "tf_rgbtl"] if m(r)]
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        d = {**m(r, "pred_test_robust_metrics"), **m(r)}
        vals = [g(d, f"report/{c}", "all", "f1") for c, _ in conds]
        if r == "t_sft_fusion":            # no LiDAR in this model: losing LiDAR cannot hurt it
            vals[3] = vals[0]
        ax.bar(np.arange(4) + (k - (len(runs) - 1) / 2) * w, [v or 0 for v in vals], width=w * 0.92, color=TCOL[r],
               label=TNAME[r], edgecolor=SURF, linewidth=2)
    ax.set_xticks(range(4), [c[1] for c in conds]); ax.set_ylim(0, 1); ax.set_ylabel("person F1")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Sensor failure on held-out sites")
    return save(fig, "fig_trisar_robust")


def fig_router():
    """Fast + slow combined per frame from model outputs only (nightwatch.eval.router_trisar)."""
    res = jload(f"{RUNS}/t_router/results.json")
    if not res:
        return None
    rows = [("fast", "fast head alone", "#199e70"),
            ("slow__t_sft_fusion_l_cd50", "slow head (camera-dropout) alone", "#9085e9"),
            ("union__t_sft_fusion_l", "slow (no dropout) + fast", "#9aa0a6"),
            ("union__t_sft_fusion_l_cd50", "slow (camera-dropout) + fast", "#3987e5")]
    rows = [r for r in rows if r[0] in res]
    sls = [("all", "all held-out"), ("night", "night"), ("loc_forest", "unseen forest"), ("loc_lab", "held-out lab")]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    w = 0.8 / len(rows)
    for k, (key, lab, c) in enumerate(rows):
        vals = [g(res, key, s, "f1") or 0 for s, _ in sls]
        x = np.arange(len(sls)) + (k - (len(rows) - 1) / 2) * w
        ax.bar(x, vals, width=w * 0.92, color=c, label=lab, edgecolor=SURF, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(sls)), [s[1] for s in sls]); ax.set_ylim(0, 1); ax.set_ylabel("person F1 (IoU 0.5)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_title("Combining heads per frame: slow-head persons + fast detections that do not overlap them")
    return save(fig, "fig_trisar_router")


def fig_examples(n=4):
    pred = {}
    p = f"{RUNS}/t_sft_fusion_l/pred_test.jsonl"
    if not os.path.exists(p) or not os.path.exists(f"{FUSED}/frames.json"):
        return None
    for line in open(p):
        d = json.loads(line)
        if d["task"] == "report" and d["cond"] == "normal":
            pred[d["id"]] = d
    frames = {f["id"]: f for f in json.load(open(f"{FUSED}/frames.json")) if f["split"] == "test"}
    # pick: forest at night, lying, crowded, far
    want = [lambda f: f["location"] == "forest" and len(f["persons"]) >= 2,
            lambda f: any(q["pose"] == "lying" for q in f["persons"]) and f.get("night"),
            lambda f: any(q["pose"] == "lying" for q in f["persons"]) and not f.get("night"),
            lambda f: any(q["range_m"] > 18 for q in f["persons"])]
    picks = []
    for cond in want:
        c = [i for i, f in frames.items() if i in pred and cond(f) and i not in picks]
        if c:
            picks.append(c[len(c) // 2])
    if not picks:
        return None
    rows = []
    for i in picks[:n]:
        th = Image.open(f"{FUSED}/{i}_t.png").convert("RGB")
        li = np.asarray(Image.open(f"{FUSED}/{i}_l.png").convert("RGB"))
        rgb = Image.open(f"{FUSED}/{i}_rgb.jpg").convert("RGB")
        lid = np.asarray(th).copy()
        mask = li[..., 2] > 0
        inv = li[..., 0].astype(float)
        col = np.stack([inv, 255 - inv, np.full_like(inv, 60)], -1)
        lid[mask] = (0.35 * lid[mask] + 0.65 * col[mask]).astype(np.uint8)
        out = Image.fromarray(lid)
        d = ImageDraw.Draw(out)
        for q in pred[i].get("persons", []):
            b = q["box"]
            d.rectangle(b, outline="#3987e5", width=3)
            lab = f'{q.get("pose") or "?"} {q["distance"]:.1f} m' if q.get("distance") is not None else (q.get("pose") or "")
            d.rectangle([b[0], max(0, b[1] - 16), b[0] + 7 * len(lab) + 6, b[1]], fill="#3987e5")
            d.text((b[0] + 3, max(0, b[1] - 15)), lab, fill="white")
        gt = Image.fromarray(np.asarray(th).copy())
        dg = ImageDraw.Draw(gt)
        for q in frames[i]["persons"]:
            b = q["box"]
            dg.rectangle(b, outline="#f0efec", width=2)
            lab = f'{q["pose"]} {q["range_m"]:.1f} m'
            dg.text((b[0] + 3, max(0, b[1] - 13)), lab, fill="#f0efec")
        rows.append((rgb, gt, out, frames[i]))
    fig, axs = plt.subplots(len(rows), 3, figsize=(15, 4.1 * len(rows)))
    axs = np.atleast_2d(axs)
    heads = ["visible camera (warped into thermal frame)", "thermal + ground truth", "LiDAR overlay + slow-head report"]
    for a, (rgb, gt, out, f) in enumerate(rows):
        for b, im in enumerate([rgb, gt, out]):
            axs[a, b].imshow(im); axs[a, b].set_xticks([]); axs[a, b].set_yticks([]); axs[a, b].grid(False)
            for s in axs[a, b].spines.values():
                s.set_visible(False)
            if a == 0:
                axs[a, b].set_title(heads[b], fontsize=12, loc="center")
        axs[a, 0].set_ylabel(f'{f["location"]}, {"night" if f.get("night") else "day"}', color=INK2)
    fig.tight_layout()
    return save(fig, "fig_trisar_examples")


def table():
    lines = ["| model | F1 | posture acc | lying recall | distance MAE (m) | F1 night | F1 forest | lying? acc | nearest MAE (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    n = 0
    for r in SLOW + FAST:
        d = m(r)
        if not d:
            continue
        f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
        qa = {**m(r), **m(r, 'pred_test_qa_metrics')}      # SFT models write QA to a separate file
        lines.append(f"| {TNAME[r]} | {f(g(d, 'report/normal', 'all', 'f1'))} | {f(g(d, 'report/normal', 'all', 'pose_acc'))} | "
                     f"{f(g(d, 'report/normal', 'all', 'lying_recall'))} | {f(g(d, 'report/normal', 'all', 'dist_mae_m'), 2)} | "
                     f"{f(g(d, 'report/normal', 'night', 'f1'))} | {f(g(d, 'report/normal', 'loc_forest', 'f1'))} | "
                     f"{f(g(qa, 'lying/normal', 'all', 'accuracy'))} | {f(g(qa, 'nearest/normal', 'all', 'nearest_mae_m'), 2)} |")
        n += 1
    return "\n".join(lines) if n else ""


def build():
    figs = {"trisar_ladder": fig_ladder(), "trisar_distance": fig_distance(), "trisar_robust": fig_robust(),
            "trisar_router": fig_router()}
    try:
        figs["trisar_examples"] = fig_examples()
    except Exception as e:
        print("trisar examples skipped:", e)
    return {k: v for k, v in figs.items() if v}, table()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f, t = build()
    print(f)
    print(t)
