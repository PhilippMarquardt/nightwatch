"""Build figures + a blog-ready markdown post + a self-contained HTML report from runs/.

Every section is optional: missing runs are skipped, so this can be re-run at any point.
python -m nightwatch.report.make_report
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from nightwatch.data import tasks as T
from nightwatch.data.llvip import load_index

OUT = os.environ.get("NW_REPORT", "report")
RUNS = os.environ.get("NW_RUNS", "runs")
VLM = os.environ.get("NW_MODEL", "Qwen/Qwen3.5-9B").split("/")[-1]
# dark surface + validated dark categorical steps (fixed per entity, never by rank)
SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3936"
GRAY = "#8a8983"
COL = {
    "sft_fusion": "#3987e5", "grpo_fusion": "#9085e9", "sft_dual": "#d95926", "sft_rgb": "#199e70",
    "zs_rgb": GRAY, "zs_dual": "#6b6a65",
    "fast_distill": "#c98500", "fast_oracle": "#d55181", "fast_gt_labeled": "#9085e9",
    "fast_oracle_rgb": "#199e70", "fast_oracle_t": "#e66767",
}
NAME = {
    "zs_rgb": f"{VLM} zero-shot (RGB)", "zs_dual": f"{VLM} zero-shot (RGB + T image)",
    "sft_rgb": "LoRA, RGB only", "sft_dual": "LoRA, thermal as 2nd image", "sft_fusion": "LoRA + native thermal fusion (ours)", "grpo_fusion": "fusion + GRPO (F1 reward)",
    "fast_gt_labeled": "Fast head, 7 labeled sites", "fast_distill": "Fast head, distilled from slow head",
    "fast_oracle": "Fast head, all sites hand-labeled", "fast_oracle_rgb": "Fast head, RGB only (oracle labels)",
    "fast_oracle_t": "Fast head, thermal only (oracle labels)",
}
SLOW = ["zs_rgb", "zs_dual", "sft_rgb", "sft_dual", "sft_fusion", "grpo_fusion"]
FAST = ["fast_gt_labeled", "fast_distill", "fast_oracle", "fast_oracle_rgb", "fast_oracle_t"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def metrics(run, kind="pred_test_metrics"):
    return jload(f"{RUNS}/{run}/{kind}.json")


def get(m, key, sl, metric):
    try:
        return m[key][sl][metric]
    except Exception:
        # a single-sensor model is unaffected by losing the *other* sensor: report its normal score
        if key == "detect/thermal_off" and m and "detect/rgb_off" not in m:
            return get(m, "detect/normal", sl, metric)
        if key == "detect/rgb_off" and m and "detect/thermal_off" not in m and "detect/normal" in m and m.get("_modal") == "t":
            return get(m, "detect/normal", sl, metric)
        return None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def hbar(ax, labels, vals, colors, fmt="{:.3f}", xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, fmt.format(v), va="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, min(1.0, max(vals) * 1.15 + 0.02))
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ figures
def fig_headline():
    rows = []
    for r in SLOW + ["fast_distill", "fast_oracle"]:
        m = metrics(r)
        f = get(m, "detect/normal", "all", "f1") if m else None
        if f is not None:
            rows.append((r, f))
    if not rows:
        return None, {}
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.3))
    hbar(ax, [NAME[r] for r, _ in rows], [f for _, f in rows], [COL[r] for r, _ in rows], xlabel="person detection F1 @ IoU 0.5")
    ax.set_title("Held-out sites (LLVIP test scenes, never seen in training)")
    return save(fig, "fig_headline"), dict(rows)


def fig_slices():
    cols = [("detect/normal", "all", "f1", "all frames"), ("detect/normal", "darkest_25pct", "f1", "darkest 25%"),
            ("detect/normal", "crowded_5plus", "f1", "crowded (5+)"), ("detect/normal", "all", "recall_far", "far persons\n(recall)"),
            ("detect/rgb_off", "all", "f1", "visible cam\ndead"), ("detect/thermal_off", "all", "f1", "thermal\ndead")]
    runs, mat = [], []
    for r in SLOW + FAST:
        m, mr = metrics(r) or {}, metrics(r, "pred_test_robust_metrics") or {}
        both = {**mr, **m}
        row = [get(both, k, s, me) for k, s, me, _ in cols]
        if any(v is not None for v in row):
            runs.append(r); mat.append([np.nan if v is None else v for v in row])
    if not runs:
        return None
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(runs) + 1.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#256abf", "#5598e7", "#9ec5f4"])
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="#0b0b0b" if (not np.isnan(v) and v > 0.7) else INK, fontsize=10)
    ax.set_xticks(range(len(cols)), [c[3] for c in cols]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(runs)), [NAME[r] for r in runs])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Every model on identical slices (F1 unless noted)", pad=38)
    return save(fig, "fig_slices")


def fig_robust():
    conds = [("normal", "both sensors"), ("rgb_off", "visible camera dead"), ("thermal_off", "thermal dead")]
    runs = [r for r in ["sft_rgb", "sft_dual", "sft_fusion", "fast_distill"] if metrics(r, "pred_test_robust_metrics") or metrics(r)]
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        m = {**(metrics(r, "pred_test_robust_metrics") or {}), **(metrics(r) or {})}
        vals = [get(m, f"detect/{c}", "all", "f1") or 0 for c, _ in conds]
        xs = np.arange(len(conds)) + (k - (len(runs) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=COL[r], label=NAME[r], edgecolor=SURF, linewidth=2)
    ax.set_xticks(range(len(conds)), [c[1] for c in conds]); ax.set_ylim(0, 1); ax.set_ylabel("F1 @ IoU 0.5")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    ax.set_title("Sensor failure: what happens when one stream goes dark")
    return save(fig, "fig_robust")


def fig_escalation():
    e = jload(f"{RUNS}/escalation.json")
    if not e:
        return None, None
    fig, ax = plt.subplots(figsize=(8, 4.4))
    r = np.array(e["rates"]) * 100
    sty = {"uncertainty/merge": ("#3987e5", "-", "uncertainty router (ours)"), "random/merge": (GRAY, "--", "random escalation"),
           "oracle/merge": ("#c98500", ":", "oracle router (upper bound)")}
    for k, (c, ls, lab) in sty.items():
        ax.plot(r, e["curves"][k], color=c, ls=ls, lw=2, label=lab)
    ax.axhline(e["slow_only_f1"], color=INK2, lw=1, ls="-.")
    ax.text(r[-1], e["slow_only_f1"] + 0.004, "slow head on every frame", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_xlabel("% of frames escalated to the slow head"); ax.set_ylabel("F1 @ IoU 0.5")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("Fast head sees every frame; hard frames escalate")
    return save(fig, "fig_escalation"), e


def fig_distill():
    rows = []
    for r in ["fast_gt_labeled", "fast_distill", "fast_oracle"]:
        m, s = metrics(r), jload(f"{RUNS}/{r}/train_summary.json")
        if m and s:
            rows.append((r, get(m, "detect/normal", "all", "ap50"), get(m, "detect/normal", "all", "f1")))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 2.9))
    hbar(ax, [NAME[r] for r, _, _ in rows], [a or 0 for _, a, _ in rows], [COL[r] for r, _, _ in rows], xlabel="AP50 on held-out sites")
    ax.set_title("Distillation: slow-head labels on 11 unlabeled sites")
    return save(fig, "fig_distill")


def fig_training():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    any_ = False
    for r in ["sft_rgb", "sft_dual", "sft_fusion"]:
        p = f"{RUNS}/{r}/log.jsonl"
        if not os.path.exists(p):
            continue
        L = [json.loads(l) for l in open(p) if '"loss"' in l]
        if not L:
            continue
        st = np.array([d["step"] for d in L]); lo = np.array([d["loss"] for d in L])
        k = max(1, len(lo) // 30)
        sm = np.convolve(lo, np.ones(k) / k, mode="valid")
        axs[0].plot(st[len(st) - len(sm):], sm, color=COL[r], lw=2, label=NAME[r]); any_ = True
        if r == "sft_fusion":
            for key, c, lab in [("gate_main", "#3987e5", "visual tokens"), ("gate_deep0", "#9085e9", "DeepStack level 1"),
                                ("gate_deep1", "#d55181", "DeepStack level 2"), ("gate_deep2", "#c98500", "DeepStack level 3")]:
                if key in L[0]:
                    axs[1].plot(st, [d[key] for d in L], color=c, lw=2, label=lab)
    if not any_:
        plt.close(fig); return None
    axs[0].set_title("SFT loss"); axs[0].set_xlabel("optimizer step"); axs[0].legend(fontsize=9)
    axs[1].set_title("How much thermal the model lets in"); axs[1].set_xlabel("optimizer step"); axs[1].set_ylabel("|thermal add| / |RGB token|")
    axs[1].axhline(0, color=GRID, lw=1); axs[1].legend(fontsize=9)
    return save(fig, "fig_training")


def fig_latency():
    rows = []
    for r in FAST:
        d = jload(f"{RUNS}/{r}/deploy.json")
        if d:
            rows.append((r, d))
    if not rows:
        return None, None
    r, d = next(((r, d) for r, d in rows if r == "fast_distill"), rows[0])
    items = [(k, d[k]) for k in ["torch_fp32_ms", "torch_fp16_ms", "tensorrt_fp16_ms", "onnxruntime_cuda_ms"] if k in d]
    names = {"torch_fp32_ms": "PyTorch FP32", "torch_fp16_ms": "PyTorch FP16", "tensorrt_fp16_ms": "TensorRT FP16", "onnxruntime_cuda_ms": "ONNX Runtime"}
    fig, ax = plt.subplots(figsize=(8, 2.6))
    hbar(ax, [names[k] for k, _ in items], [v for _, v in items], ["#3987e5" if "tensorrt" in k else GRAY for k, _ in items],
         fmt="{:.2f} ms", xlabel=f"batch-1 latency, 640x512 RGB+T, {d.get('gpu', '')}")
    ax.set_xlim(0, max(v for _, v in items) * 1.3)
    ax.set_title("Fast head, decode included")
    return save(fig, "fig_latency"), d


def draw(img, boxes, color, width=4, dashed=False):
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle([float(v) for v in b], outline=color, width=width)
    return img


def fig_examples(root="data/raw/LLVIP", n=4):
    def load(path, cond="normal"):
        out = {}
        if not os.path.exists(path):
            return out
        for l in open(path):
            d = json.loads(l)
            if d["task"] == "detect" and d["cond"] == cond:
                out[d["id"]] = d
        return out

    fus, rgb = load(f"{RUNS}/sft_fusion/pred_test.jsonl"), load(f"{RUNS}/sft_rgb/pred_test.jsonl")
    fast = load(f"{RUNS}/fast_distill/pred_test.jsonl")
    thr = (jload(f"{RUNS}/fast_distill/train_summary.json") or {}).get("score_thr", 0.3)
    if not fus:
        return None
    recs = {r.id: r for r in load_index("data/index.json", ("test",))}
    lums = np.array([r.rgb_lum for r in recs.values()]); dark = np.percentile(lums, 25)
    cand = []
    for i, d in fus.items():
        r = recs[i]
        if r.rgb_lum > dark:
            continue
        f_f = T.f1_reward(d["boxes"], r.boxes) if r.boxes else 1
        f_r = T.f1_reward(rgb[i]["boxes"], r.boxes) if i in rgb else 0
        cand.append((f_f - f_r + 0.1 * len(r.boxes) / 10, i))
    cand.sort(reverse=True)
    picks, scenes = [], set()
    for _, i in cand:                     # diverse sites
        if recs[i].scene not in scenes:
            picks.append(i); scenes.add(recs[i].scene)
        if len(picks) == n:
            break
    tiles = []
    for i in picks:
        r = recs[i]
        vis = Image.open(r.rgb_path(root)).convert("RGB")
        ir = Image.open(r.ir_path(root)).convert("RGB")
        row = [draw(vis.copy(), r.boxes, "#f0efec", 3),
               draw(vis.copy(), rgb.get(i, {}).get("boxes", []), "#199e70", 6),
               draw(ir.copy(), fus[i]["boxes"], "#3987e5", 6)]
        if i in fast:
            fb = [b for b, s in zip(fast[i]["boxes"], fast[i]["scores"]) if s >= thr]
            row.append(draw(ir.copy(), fb, "#c98500", 6))
        tiles.append([t.resize((480, 384)) for t in row])
    heads = ["visible camera + ground truth", "slow head, RGB only", "slow head, RGB+T fusion", "fast head (distilled)"][:len(tiles[0])]
    fig, axs = plt.subplots(len(tiles), len(tiles[0]), figsize=(4.2 * len(tiles[0]), 3.4 * len(tiles)))
    axs = np.atleast_2d(axs)
    for a, row in enumerate(tiles):
        for b, t in enumerate(row):
            axs[a, b].imshow(t); axs[a, b].set_xticks([]); axs[a, b].set_yticks([]); axs[a, b].grid(False)
            for s in axs[a, b].spines.values():
                s.set_visible(False)
            if a == 0:
                axs[a, b].set_title(heads[b], fontsize=12, loc="center")
        axs[a, 0].set_ylabel(f"site {recs[picks[a]].scene}", color=INK2)
    fig.suptitle("Darkest 25% of held-out frames (thermal panels are the same scene through the IR camera)", color=INK2, y=1.0)
    fig.tight_layout()
    return save(fig, "fig_examples")


# ------------------------------------------------------------------ assemble
def summary_table():
    lines = ["| model | F1 all | F1 darkest 25% | F1 camera dead | F1 thermal dead | count exact | img/s |", "|---|---|---|---|---|---|---|"]
    for r in SLOW + FAST:
        m = {**(metrics(r, "pred_test_robust_metrics") or {}), **(metrics(r) or {})}
        if not m:
            continue
        tim = jload(f"{RUNS}/{r}/pred_test_timing.json") or []
        ips = next((t["n"] / t["seconds"] for t in tim if t["cond"] == "normal"), None)
        dep = jload(f"{RUNS}/{r}/deploy.json") or {}
        if "tensorrt_fp16_fps" in dep:
            ips = dep["tensorrt_fp16_fps"]
        f = lambda v: "–" if v is None else f"{v:.3f}"
        lines.append(f"| {NAME[r]} | {f(get(m, 'detect/normal', 'all', 'f1'))} | {f(get(m, 'detect/normal', 'darkest_25pct', 'f1'))} | "
                     f"{f(get(m, 'detect/rgb_off', 'all', 'f1'))} | {f(get(m, 'detect/thermal_off', 'all', 'f1'))} | "
                     f"{f(get(m, 'detect/normal', 'all', 'count_exact'))} | {'–' if ips is None else f'{ips:.0f}'} |")
    return "\n".join(lines)


def reason_table():
    lines = ["| model | count exact | count MAE | zone yes/no accuracy |", "|---|---|---|---|"]
    n = 0
    for r in SLOW:
        m = metrics(r, "pred_test_reason_metrics")
        if not m:
            continue
        z = [get(m, f"zone_{k}/normal", "all", "accuracy") for k in ("left", "center", "right")]
        z = [v for v in z if v is not None]
        f = lambda v: "–" if v is None else f"{v:.3f}"
        lines.append(f"| {NAME[r]} | {f(get(m, 'count/normal', 'all', 'count_exact'))} | {f(get(m, 'count/normal', 'all', 'count_mae'))} | "
                     f"{f(sum(z) / len(z) if z else None)} |")
        n += 1
    return "\n".join(lines) if n else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    figs = {}
    figs["headline"], head = fig_headline()
    figs["slices"] = fig_slices()
    figs["robust"] = fig_robust()
    figs["escalation"], esc = fig_escalation()
    figs["distill"] = fig_distill()
    figs["training"] = fig_training()
    figs["latency"], dep = fig_latency()
    try:
        figs["examples"] = fig_examples()
    except Exception as e:
        print("examples skipped:", e); figs["examples"] = None
    trisar_table = ""
    try:
        from nightwatch.report.trisar_report import build as build_trisar
        tfigs, trisar_table = build_trisar()
        figs.update(tfigs)
    except Exception as e:
        print("trisar section skipped:", e)
    pseudo = metrics("sft_fusion", "pred_unlabeled_metrics")
    gate = jload(f"{RUNS}/gate.json")
    res = {"headline_f1": head, "escalation": esc and {k: esc[k] for k in ("fast_only_f1", "slow_only_f1")},
           "pseudo_label_quality": pseudo and pseudo.get("detect/normal", {}).get("all"),
           "gate": gate and gate["verdict"], "deploy": dep, "figures": {k: v for k, v in figs.items() if v}}
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    md = ["# Night Watch: results", "", summary_table(), "", reason_table(), ""]
    if trisar_table:
        md += ["## TriSAR-3D: RGB + thermal + LiDAR", "", trisar_table, ""]
    for k, v in figs.items():
        if v:
            md += [f"![{k}]({v})", ""]
    if pseudo:
        md.append(f"Pseudo-label quality on the unlabeled sites (scored against their hidden GT): {json.dumps(res['pseudo_label_quality'])}")
    if gate:
        md.append(f"\nRegression gate (fusion vs dual-image baseline): **{gate['verdict']}**")
    open(f"{OUT}/results.md", "w").write("\n".join(md))
    # self-contained HTML (images inlined) for quick viewing / sharing
    html = ["<!doctype html><meta charset=utf-8><title>Night Watch results</title><style>body{background:#1a1a19;color:#f0efec;"
            "font:15px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px}img{max-width:100%}table{border-collapse:collapse}"
            "td,th{border-bottom:1px solid #3a3936;padding:6px 10px;text-align:right}td:first-child,th:first-child{text-align:left}</style>",
            "<h1>Night Watch: results</h1>"]
    import re
    tbl = summary_table().splitlines()
    html.append("<table>" + "".join("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in l.strip("|").split("|")) + "</tr>"
                                    for k, l in enumerate(tbl) if k != 1) + "</table>")
    for k, v in figs.items():
        if v:
            b64 = base64.b64encode(open(f"{OUT}/{v}", "rb").read()).decode()
            html.append(f"<h2>{k}</h2><img src='data:image/png;base64,{b64}'>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
