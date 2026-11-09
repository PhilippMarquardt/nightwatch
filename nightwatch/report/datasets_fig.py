"""One figure that shows the three datasets as the models see them, with ground-truth person boxes.

Row 1: LLVIP (camera + thermal street scenes, a dark held-out test site).
Row 2: TriSAR-3D (camera + thermal + LiDAR range image, the unseen night forest).
Row 3: LTDv2 (one harbour thermal camera: the unlabelled adaptation month and a test month a year later).

python -m nightwatch.report.datasets_fig    (on the pod: needs data/raw LLVIP, data/trisar_fused, data/ltd)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image

from nightwatch.data import llvip
from nightwatch.slow.data_trisar import FH, FW, _level_match, lidar_image, load_frames

BOX = "#38e07b"


def llvip_root():
    for c in [os.environ.get("NW_LLVIP_ROOT", ""), "data/raw/LLVIP", "data/raw/LLVIP/LLVIP", "data/LLVIP"]:
        if c and os.path.isdir(os.path.join(c, "visible")):
            return c
    hit = glob.glob("data/**/visible", recursive=True)
    return os.path.dirname(hit[0])


def pick_llvip():
    recs = [r for r in llvip.load_index("data/index.json", ("test",)) if 3 <= len(r.boxes) <= 8]
    recs.sort(key=lambda r: r.rgb_lum)
    r = recs[len(recs) // 20]                   # among the darkest 5 % with a few people
    root = llvip_root()
    return r, Image.open(r.rgb_path(root)).convert("RGB"), Image.open(r.ir_path(root)).convert("RGB")


def pick_trisar(d="data/trisar_fused"):
    fr = [f for f in load_frames(d, ("test",)) if f.get("night") and "lab" not in str(f.get("location", "")).lower()
          and f.get("persons")]
    # Typical forest people are 8-90 px tall; boxes of people within a few metres of the rig are projected 3D cuboids
    # and come out loose, so show a frame where every person is mid-range.
    h = lambda p: p["box"][3] - p["box"][1]
    fr = [f for f in fr if all(25 <= h(p) <= 110 for p in f["persons"])]
    f = max(fr, key=lambda f: (min(len(f["persons"]), 3), sum(h(p) for p in f["persons"])))
    base = os.path.join(d, f["id"])
    rgb = Image.open(base + "_rgb.jpg").convert("RGB").resize((FW, FH))
    th = _level_match(Image.open(base + "_t.png").convert("L").resize((FW, FH))).convert("RGB")
    li = lidar_image(Image.open(base + "_l.png").convert("RGB").resize((FW, FH), Image.NEAREST))
    return f, rgb, th, li


def pick_ltd(month, d="data/ltd"):
    fr = [f for f in json.load(open(f"{d}/frames.json")) if f["month"] == month]
    pp = lambda f: [b["box"] for b in f["boxes"] if b["cls"] == "person"]
    f = max(fr, key=lambda f: (min(len(pp(f)), 6), f["id"]))
    arr = np.load(f"{d}/frames.npy", mmap_mode="r")
    return f, Image.fromarray(np.asarray(arr[f["idx"]], np.uint8)).convert("RGB"), pp(f)


def show(ax, img, boxes, title, sx=1.0, sy=1.0, lw=1.4):
    import matplotlib.patches as mp
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for b in boxes:
        ax.add_patch(mp.Rectangle((b[0] * sx, b[1] * sy), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy,
                                  fill=False, ec=BOX, lw=lw))
    ax.set_title(title, fontsize=9, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nightwatch.report.make_report import INK2, save
    fig = plt.figure(figsize=(13, 11.2))
    sf = fig.subfigures(3, 1, height_ratios=[1, 1, 1])

    r, rgb, ir = pick_llvip()
    ax = sf[0].subplots(1, 2)
    show(ax[0], rgb, r.boxes, f"camera (site {r.scene}, mean brightness {r.rgb_lum:.0f}/255)")
    show(ax[1], ir, r.boxes, "thermal")
    sf[0].suptitle("LLVIP: street cameras, camera + thermal, 1280×1024. Train on 7 sites, test on 7 others",
                   x=0.01, ha="left", fontsize=11, fontweight="bold")

    f, rgb, th, li = pick_trisar()
    B = [p["box"] for p in f["persons"]]
    ax = sf[1].subplots(1, 3)
    show(ax[0], rgb, B, f"camera ({f.get('location', '')}, night)")
    show(ax[1], th, B, "thermal (level-matched)")
    show(ax[2], li, B, "LiDAR: 1 / range, turbo colormap (black = no return)")
    sf[1].suptitle("TriSAR-3D: robot rig, camera + thermal + 128-beam LiDAR. Train in a lab, test in an unseen forest at night",
                   x=0.01, ha="left", fontsize=11, fontweight="bold")

    ax = sf[2].subplots(1, 2)
    for a, (m, lab) in zip(ax, [("202005", "May 2020: adaptation month, labels never used"), ("202103", "March 2021: test month, a year later")]):
        f, img, pp = pick_ltd(m)
        show(a, img, pp, lab, lw=1.2)
    sf[2].suptitle("LTDv2: one harbour thermal camera in Denmark, 384×288, people ~16–19 px tall. Never seen in training",
                   x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.003, "Green: ground-truth person boxes. Frames chosen to show several people; every number in the README is "
             "on the full held-out test sets.", fontsize=8, color=INK2)
    print(save(fig, "fig_datasets"))


if __name__ == "__main__":
    main()
