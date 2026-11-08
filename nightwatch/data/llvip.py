"""LLVIP index: parse annotations, assign splits by *scene*, compute slice tags.

LLVIP file names encode the recording location: ``SSNNNN.jpg`` where ``SS`` is the
scene id. The official test split (scenes 19-24, 26) shares no scene with train,
so every test number in this project is a held-out-site number.

Our splits (all by scene, never by frame -- consecutive frames are near duplicates):

    labeled   scenes 01-07   slow head is fine-tuned on these labels only
    val       scene  25      checkpoint / threshold selection ("a val site")
    unlabeled scenes 08-18   treated as unlabeled field data; the slow head labels it
    test      scenes 19-24,26  official test split = held-out sites, touched only for reporting
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1280, 1024  # native LLVIP resolution (all pairs)

LABELED_SCENES = {f"{i:02d}" for i in range(1, 8)}
VAL_SCENES = {"25"}
UNLABELED_SCENES = {f"{i:02d}" for i in range(8, 19)}


@dataclass
class Record:
    id: str
    scene: str
    split: str                     # labeled | val | unlabeled | test
    folder: str                    # train | test (LLVIP folder)
    boxes: list = field(default_factory=list)  # [[x1,y1,x2,y2], ...] in 1280x1024 px
    rgb_lum: float = -1.0          # mean luminance of the visible image (0-255)
    ir_mean: float = -1.0

    def rgb_path(self, root):
        return os.path.join(root, "visible", self.folder, self.id + ".jpg")

    def ir_path(self, root):
        return os.path.join(root, "infrared", self.folder, self.id + ".jpg")


def _parse_xml(path: str) -> list:
    boxes = []
    for obj in ET.parse(path).getroot().iter("object"):
        if obj.findtext("name", "person").strip() != "person":
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _split_for(scene: str, folder: str) -> str:
    if folder == "test":
        return "test"
    if scene in LABELED_SCENES:
        return "labeled"
    if scene in VAL_SCENES:
        return "val"
    if scene in UNLABELED_SCENES:
        return "unlabeled"
    raise ValueError(f"unknown scene {scene}")


def _stats(rec: Record, root: str) -> Record:
    # luminance on a 1/8 thumbnail is plenty for slicing
    with Image.open(rec.rgb_path(root)) as im:
        im.draft("L", (W // 8, H // 8))
        rec.rgb_lum = float(np.asarray(im.convert("L"), dtype=np.float32).mean())
    with Image.open(rec.ir_path(root)) as im:
        im.draft("L", (W // 8, H // 8))
        rec.ir_mean = float(np.asarray(im.convert("L"), dtype=np.float32).mean())
    return rec


def build_index(root: str, out: str, workers: int = 16) -> list[Record]:
    recs = []
    for folder in ("train", "test"):
        for f in sorted(os.listdir(os.path.join(root, "visible", folder))):
            if not f.endswith(".jpg"):
                continue
            iid = f[:-4]
            scene = iid[:2]
            boxes = _parse_xml(os.path.join(root, "Annotations", iid + ".xml"))
            recs.append(Record(iid, scene, _split_for(scene, folder), folder, boxes))
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda r: _stats(r, root), recs))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump([asdict(r) for r in recs], fh)
    return recs


def load_index(path: str, splits: tuple[str, ...] | None = None) -> list[Record]:
    with open(path) as fh:
        recs = [Record(**d) for d in json.load(fh)]
    if splits:
        recs = [r for r in recs if r.split in splits]
    return recs


# ---------------------------------------------------------------- slices
# Slices are defined once here so every model is scored on identical subsets.

SMALL_AREA = 48 ** 2      # px^2 at 1280x1024 (COCO's 32^2 scaled to LLVIP's 2x resolution)
LARGE_AREA = 128 ** 2


def image_slices(recs: list[Record], dark_q: float | None = None) -> dict[str, list[str]]:
    """Image-level slices. Dark threshold = 25th percentile of test-set luminance."""
    if dark_q is None:
        lums = np.array([r.rgb_lum for r in recs])
        dark_q = float(np.percentile(lums, 25))
    sl = {"all": [r.id for r in recs]}
    sl["darkest_25pct"] = [r.id for r in recs if r.rgb_lum <= dark_q]
    sl["crowded_5plus"] = [r.id for r in recs if len(r.boxes) >= 5]
    for s in sorted({r.scene for r in recs}):
        sl[f"site_{s}"] = [r.id for r in recs if r.scene == s]
    return sl


def box_size_bucket(b) -> str:
    a = (b[2] - b[0]) * (b[3] - b[1])
    return "small" if a < SMALL_AREA else ("large" if a >= LARGE_AREA else "medium")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/index.json")
    a = ap.parse_args()
    recs = build_index(a.root, a.out)
    from collections import Counter

    print(Counter(r.split for r in recs))
    nb = Counter()
    for r in recs:
        nb[r.split] += len(r.boxes)
    print("boxes", nb)
    print("max persons/img", max(len(r.boxes) for r in recs))
