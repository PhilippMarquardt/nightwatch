"""Self-training control: a trained fast head labels the withheld-label pool itself.

Without this control, "the 9B slow head's labels help the student" could just be "any extra pseudo-labelled
data helps". fast_selfdistill = same student trained on labeled GT + its own (fast_gt_labeled) pseudo-labels.

python -m nightwatch.fast.label_split --run runs/fast_gt_labeled --split unlabeled --out runs/pseudo_labels_unlabeled_self.json
"""
from __future__ import annotations

import argparse
import json

import torch

from nightwatch.fast.data import load_split
from nightwatch.fast.model import CenterNetRGBT
from nightwatch.fast.train import IN_CH, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/fast_gt_labeled")
    ap.add_argument("--split", default="unlabeled")
    ap.add_argument("--cache", default="data/cache_fast")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=None, help="override the val-selected threshold (e.g. low thr for consensus)")
    a = ap.parse_args()
    summ = json.load(open(f"{a.run}/train_summary.json"))
    dev = torch.device("cuda")
    net = CenterNetRGBT(a.backbone, IN_CH[summ["modal"]], pretrained=False).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(f"{a.run}/best.pt", map_location=dev))
    x, _, _, ids = load_split(a.cache, a.split, dev, index=a.index)
    thr = a.thr if a.thr is not None else summ["score_thr"]
    preds = predict(net, x, summ["modal"], thr=thr)                    # val-selected threshold = hard labels
    labels = {i: p["boxes"] for i, p in zip(ids, preds)}
    json.dump(labels, open(a.out, "w"))
    print(f"self-labels: {len(labels)} frames, {sum(len(v) for v in labels.values())} boxes (thr {thr:.3f})")


if __name__ == "__main__":
    main()
