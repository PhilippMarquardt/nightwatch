"""LLVIP: per-site gain of the fused slow head (and the brightness router) over the RGB-only slow head, against the
site's mean camera brightness. Bars: 95 % bootstrap over the site's frames (frame resampling within one site).

python -m nightwatch.report.fusion_vs_lum
"""
from __future__ import annotations

import json
import os

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.data.llvip import load_index
from nightwatch.eval.site_bootstrap import load_preds

RUNS = os.environ.get("NW_RUNS", "runs")


def per_frame(preds, gts, ids):
    return np.array([T.greedy_match(preds.get(i, []), gts[i])[:3] for i in ids], float)   # tp, fp, fn


def f1(c):
    tp, fp, fn = c.sum(0)
    return 2 * tp / max(1e-9, 2 * tp + fp + fn)


def main():
    recs = load_index("data/index.json", ("test",))
    gts = {r.id: r.boxes for r in recs}
    raw = {r["id"]: r for r in json.load(open("data/index.json"))}
    runs = {k: load_preds(k) for k in ("sft_rgb", "sft_fusion", "router_lum") if os.path.exists(f"{RUNS}/{k}/pred_test.jsonl")}
    ids = sorted(set.intersection(*[set(v) for v in runs.values()]))
    sites = sorted({raw[i]["scene"] for i in ids})
    rng = np.random.default_rng(0)
    rows = []
    for s in sites:
        si = [i for i in ids if raw[i]["scene"] == s]
        C = {k: per_frame(v, gts, si) for k, v in runs.items()}
        boot = rng.integers(0, len(si), size=(2000, len(si)))
        row = {"site": s, "lum": float(np.mean([raw[i]["rgb_lum"] for i in si])), "n": len(si)}
        for k in ("sft_fusion", "router_lum"):
            if k in C:
                d = [f1(C[k][b]) - f1(C["sft_rgb"][b]) for b in boot]
                row[k] = (f1(C[k]) - f1(C["sft_rgb"]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        rows.append(row)
        print(s, round(row["lum"], 1), {k: tuple(round(x, 3) for x in row[k]) for k in ("sft_fusion", "router_lum") if k in row})
    json.dump(rows, open(f"{RUNS}/fusion_vs_lum.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK, INK2, save
    thr = float(np.median([r["rgb_lum"] for r in raw.values() if r["split"] == "labeled"]))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for k, c, lab, dx in (("sft_fusion", "#d95926", "fused head − RGB-only head", -0.6), ("router_lum", "#3987e5", "brightness router − RGB-only head", 0.6)):
        xs = [r["lum"] + dx for r in rows if k in r]
        ys = [r[k][0] for r in rows if k in r]
        lo = [r[k][0] - r[k][1] for r in rows if k in r]
        hi = [r[k][2] - r[k][0] for r in rows if k in r]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=c, ecolor=c, elinewidth=1.4, capsize=3, label=lab, ms=7)
    for r in rows:
        ax.annotate(f"site {r['site']}", (r["lum"], r["sft_fusion"][0]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(thr, color=INK2, lw=0.8, ls="--")
    ax.text(thr + 0.8, ax.get_ylim()[1] * 0.92, f"router threshold\n(train median {thr:.0f})", fontsize=8, color=INK2)
    ax.set_xlabel("site mean camera brightness (0-255)"); ax.set_ylabel("Δ person F1 vs RGB-only slow head")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="x", visible=False)
    ax.set_title("LLVIP: thermal fusion pays off in the dark (7 unseen sites, 95 % bootstrap within site)", loc="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    print(save(fig, "fig_llvip_fusion_vs_lum"))


if __name__ == "__main__":
    main()
