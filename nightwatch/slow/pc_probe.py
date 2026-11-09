"""Verification gates V4 + V5 of docs/PLAN_POINTCLOUD.md, before any VLM training.

V5 (decisive): does a frozen Utonia cell feature carry more "is there a person in this 32-px cell" signal than simple
statistics of the LiDAR range image we already feed the VLM? Logistic regression on train-split cells, average
precision on held-out lab and forest cells.
V4: PCA of the cell features as an RGB overlay on the thermal image (alignment / grid-order check, README figure).

python -m nightwatch.slow.pc_probe          (needs data/trisar_pc/*.npz from nightwatch.slow.pc_encode)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from nightwatch.slow.pc_encode import CELL, GH, GW, TH, TW

FUSED, PC = "data/trisar_fused", "data/trisar_pc"


def cell_labels(persons):
    """1 if a person box covers >= 30 % of the cell, or the cell holds the box centre."""
    y = np.zeros((GH, GW), np.float32)
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        cx, cy = int((x1 + x2) / 2 // CELL), int((y1 + y2) / 2 // CELL)
        if 0 <= cy < GH and 0 <= cx < GW:
            y[cy, cx] = 1
        for gy in range(max(0, int(y1 // CELL)), min(GH, int(y2 // CELL) + 1)):
            for gx in range(max(0, int(x1 // CELL)), min(GW, int(x2 // CELL) + 1)):
                ix = max(0, min(x2, (gx + 1) * CELL) - max(x1, gx * CELL))
                iy = max(0, min(y2, (gy + 1) * CELL) - max(y1, gy * CELL))
                if ix * iy >= 0.3 * CELL * CELL:
                    y[gy, gx] = 1
    return y.reshape(-1)


def range_pixels(fid):
    """Matched-capacity baseline: raw range-image pixels of each 32-px cell, 2x average-pooled to 16x16x3 = 768 values
    (the range image is already dilated by 3 px, so little is lost), PCA'd later."""
    a = np.asarray(Image.open(os.path.join(FUSED, f"{fid}_l.png")).convert("RGB"), np.float32) / 255.0
    a = a.reshape(TH // 2, 2, TW // 2, 2, 3).mean((1, 3))
    h = CELL // 2
    return a.reshape(GH, h, GW, h, 3).transpose(0, 2, 1, 3, 4).reshape(GH * GW, h * h * 3).astype(np.float16)


def range_stats(fid):
    """Baseline: per-cell stats of the range image channels (inverse range, intensity, hit) -> (320, 10)."""
    a = np.asarray(Image.open(os.path.join(FUSED, f"{fid}_l.png")).convert("RGB"), np.float32) / 255.0
    a = a.reshape(GH, CELL, GW, CELL, 3).transpose(0, 2, 1, 3, 4).reshape(GH * GW, CELL * CELL, 3)
    hit = a[..., 2] > 0.5
    inv = np.where(hit, a[..., 0], np.nan)
    with np.errstate(all="ignore"):
        f = [np.nan_to_num(np.nanmean(inv, 1)), np.nan_to_num(np.nanmax(inv, 1)), np.nan_to_num(np.nanstd(inv, 1)),
             np.nan_to_num(np.nanmin(inv, 1)), a[..., 1].mean(1), a[..., 1].max(1), hit.mean(1),
             a[..., 0].mean(1), a[..., 0].std(1), a[..., 1].std(1)]
    return np.stack(f, 1).astype(np.float32)


def fit_eval(Xtr, ytr, tests, epochs=300, wd=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device="cuda")
    yt = torch.tensor(ytr, dtype=torch.float32, device="cuda")
    lin = torch.nn.Linear(Xt.shape[1], 1).cuda()
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    pw = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6), device="cuda")
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(Xt).squeeze(1), yt, pos_weight=pw)
        loss.backward(); opt.step()
    out = {}
    from sklearn.metrics import average_precision_score
    for name, (X, y) in tests.items():
        with torch.no_grad():
            s = lin(torch.tensor((X - mu) / sd, dtype=torch.float32, device="cuda")).squeeze(1).cpu().numpy()
        out[name] = {"ap": float(average_precision_score(y, s)), "pos_rate": float(y.mean()), "n_cells": int(len(y))}
    return out


def _load(f):
    z = np.load(os.path.join(PC, f"{f['id']}.npz"))
    u = np.concatenate([z["feat"].astype(np.float32), np.log1p(z["count"].astype(np.float32))[:, None]], 1)
    return u, range_stats(f["id"]), range_pixels(f["id"]), cell_labels(f["persons"]), z["count"] > 0


def main(max_train_frames=2500):
    allf = json.load(open(os.path.join(FUSED, "frames.json")))
    frames = [f for f in allf if os.path.exists(os.path.join(PC, f"{f['id']}.npz"))]
    assert len(frames) == len(allf), f"feature cache incomplete: {len(frames)}/{len(allf)}"
    rng = np.random.default_rng(0)
    tr = [f for f in frames if f["split"] == "train"]
    tr = [tr[i] for i in sorted(rng.choice(len(tr), min(max_train_frames, len(tr)), replace=False))]
    # Gate on the lab VALIDATION sequences (deciding on test AP would make the test set development data).
    # Test lab / forest are computed and saved but reported only as information, after the decision.
    te = {"val_lab": [f for f in frames if f["split"] == "val" and f["location"] == "lab"],
          "test_lab": [f for f in frames if f["split"] == "test" and f["location"] == "lab"],
          "test_forest": [f for f in frames if f["split"] == "test" and f["location"] == "forest"]}
    print("frames with features:", len(frames), "train used", len(tr), {k: len(v) for k, v in te.items()}, flush=True)

    from multiprocessing import Pool

    def build(fs):
        with Pool(48) as pool:
            rows = pool.map(_load, fs, chunksize=16)
        print("  loaded", len(fs), "frames", flush=True)
        return tuple(np.concatenate([r[i] for r in rows]) for i in range(5))

    Utr, Btr, Ptr, ytr, _ = build(tr)
    tests = {k: build(v) for k, v in te.items()}
    # matched-capacity baseline: range-image cell pixels, PCA to the Utonia feature width
    q = min(Utr.shape[1], Ptr.shape[1])
    Pm = torch.tensor(Ptr, dtype=torch.float32, device="cuda")
    pmu = Pm.mean(0)
    sub = Pm[torch.randperm(len(Pm), device="cuda")[:60000]] - pmu
    _, _, Vp = torch.linalg.svd(sub, full_matrices=False)
    Vp = Vp[:q].T
    pca = lambda X: ((torch.tensor(X, dtype=torch.float32, device="cuda") - pmu) @ Vp).cpu().numpy()
    Ptr_p = pca(Ptr); del Pm, sub
    arms = {"utonia": (Utr, {k: t[0] for k, t in tests.items()}),
            "range_image_stats": (Btr, {k: t[1] for k, t in tests.items()}),
            f"range_image_pixels_pca{q}": (Ptr_p, {k: pca(t[2]) for k, t in tests.items()}),
            # control: point count alone, i.e. "is there a LiDAR return here and how dense"
            "point_count_only": (Utr[:, -1:], {k: t[0][:, -1:] for k, t in tests.items()})}
    res = {"all_cells": {}, "cells_with_return": {}}
    for name, (Xtr, Xte) in arms.items():
        res["all_cells"][name] = fit_eval(Xtr, ytr, {k: (Xte[k], tests[k][3]) for k in tests})
        res["cells_with_return"][name] = fit_eval(Xtr, ytr, {k: (Xte[k][tests[k][4]], tests[k][3][tests[k][4]]) for k in tests})
    os.makedirs("runs/pc_probe", exist_ok=True)
    G = "val_lab"
    best_base = max(v[G]["ap"] for n, v in res["all_cells"].items() if n != "utonia")
    gate_pre = res["all_cells"]["utonia"][G]["ap"] >= res["all_cells"]["range_image_stats"][G]["ap"] + 0.05
    gate_strict = res["all_cells"]["utonia"][G]["ap"] >= best_base + 0.05
    print("--- DECISION on val_lab (all cells)")
    for name, r in res["all_cells"].items():
        print(f"{name:26s} val_lab AP {r[G]['ap']:.3f} (pos {r[G]['pos_rate']:.3f}, n {r[G]['n_cells']})")
    print("V5 GATE (Utonia val-lab AP >= range-image stats + 0.05):", "PASS" if gate_pre else "FAIL")
    print("V5 strict (vs best baseline incl. pixels and point count + 0.05):", "PASS" if gate_strict else "FAIL")
    json.dump({"results": res, "gate_set": G, "gate_pass": bool(gate_pre), "gate_strict": bool(gate_strict),
               "train_frames": len(tr)}, open("runs/pc_probe/probe.json", "w"), indent=1)
    print("--- INFORMATION ONLY (test sets; not used for the decision)")
    for scope, rr in res.items():
        print(f"[{scope}]")
        for name, r in rr.items():
            print(f"{name:26s} " + "  ".join(f"{k}: AP {v['ap']:.3f} (pos {v['pos_rate']:.3f})" for k, v in r.items() if k != G))

    # V4: PCA overlay for one forest and one lab test frame with people
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ufit = Utr[:, :-1]
    mu = Ufit.mean(0)
    _, _, V = np.linalg.svd(Ufit[rng.choice(len(Ufit), min(20000, len(Ufit)), replace=False)] - mu, full_matrices=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, key in enumerate(["test_forest", "test_lab"]):
        f = max(te[key], key=lambda f: len(f["persons"]))
        z = np.load(os.path.join(PC, f"{f['id']}.npz"))
        p = (z["feat"].astype(np.float32) - mu) @ V[:3].T
        p = (p - p.min(0)) / (np.ptp(p, 0) + 1e-6)
        p[z["count"] == 0] = 0
        th = np.asarray(Image.open(os.path.join(FUSED, f"{f['id']}_t.png")).convert("L"))
        li = np.asarray(Image.open(os.path.join(FUSED, f"{f['id']}_l.png")).convert("RGB"))
        axes[row, 0].imshow(th, cmap="gray"); axes[row, 0].set_title(f"{key}: thermal")
        axes[row, 1].imshow(li[..., 0], cmap="turbo"); axes[row, 1].set_title("LiDAR range image (what the VLM got)")
        axes[row, 2].imshow(th, cmap="gray"); axes[row, 2].imshow(p.reshape(GH, GW, 3), alpha=0.6,
                                                                  extent=(0, th.shape[1], th.shape[0], 0), interpolation="nearest")
        axes[row, 2].set_title("Utonia features, PCA per 32-px token")
        for ax in axes[row]:
            for q in f["persons"]:
                x1, y1, x2, y2 = q["box"]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#38e07b", lw=1))
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("runs/pc_probe/fig_utonia_tokens.png", dpi=130)
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
