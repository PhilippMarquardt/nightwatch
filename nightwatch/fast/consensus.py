"""Consensus pseudo-labels: keep a slow-head (teacher) box only if the labelled-only fast head independently fires on
it (IoU >= 0.3 with any fast detection above a low threshold). Uses no ground truth. Optionally reports label quality
against the withheld unlabeled-split GT (diagnostic only; nothing is tuned on it).

python -m nightwatch.fast.consensus --teacher runs/pseudo_labels_unlabeled.json --student runs/pseudo_labels_unlabeled_self_lo.json \
    --out runs/pseudo_labels_unlabeled_consensus.json --report
"""
from __future__ import annotations

import argparse
import json

from nightwatch.data import tasks as T


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def quality(labels, gts, ids):
    tp = fp = fn = 0
    for i in ids:
        a, b, c, _ = T.greedy_match(labels.get(i, []), gts[i])
        tp, fp, fn = tp + a, fp + b, fn + c
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": P, "recall": R, "f1": 2 * P * R / max(1e-9, P + R)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    t, s = json.load(open(a.teacher)), json.load(open(a.student))
    out = {i: [b for b in tb if any(iou(b, c) >= a.iou for c in s.get(i, []))] for i, tb in t.items()}
    json.dump(out, open(a.out, "w"))
    nt, no = sum(map(len, t.values())), sum(map(len, out.values()))
    print(f"consensus: kept {no}/{nt} teacher boxes ({no / max(1, nt):.1%}) on {len(out)} frames")
    if a.report:
        from nightwatch.data.llvip import load_index
        recs = load_index("data/index.json", ("unlabeled",))
        gts = {r.id: r.boxes for r in recs}
        ids = [i for i in out if i in gts]
        by_site = {}
        for r in recs:
            by_site.setdefault(r.scene, []).append(r.id)
        rep = {name: {"all": quality(lab, gts, ids), **{f"site_{k}": quality(lab, gts, v) for k, v in sorted(by_site.items())}}
               for name, lab in [("teacher", t), ("student_lo", s), ("consensus", out)]}
        json.dump(rep, open(a.out.replace(".json", "_quality.json"), "w"), indent=1)
        for name, v in rep.items():
            print(f"{name:11s} P {v['all']['precision']:.3f} R {v['all']['recall']:.3f} F1 {v['all']['f1']:.3f} | site16 {v['site_16']['f1']:.2f} site17 {v['site_17']['f1']:.2f}")


if __name__ == "__main__":
    main()
