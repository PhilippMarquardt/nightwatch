"""Fast head sees every frame; only hard frames escalate to the slow head.

For an escalation budget r (fraction of frames sent to the slow head) we compare routers:
    uncertainty : escalate frames whose fast-head scores sit in the ambiguous band
    random      : escalate a random r of frames (the baseline a router must beat)
    oracle      : escalate exactly the frames where the slow head does better (upper bound)
and two ways to combine: 'replace' (slow head's answer wins) and 'merge' (union, IoU-deduped).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.data.llvip import load_index


def load_preds(path, cond="normal", task="detect"):
    out = {}
    for line in open(path):
        d = json.loads(line)
        if d["task"] == task and d["cond"] == cond:
            out[d["id"]] = d
    return out


def ambiguity(scores, lo=0.05):
    s = np.asarray([x for x in scores if x >= lo])
    return float(np.sum(4 * s * (1 - s))) if len(s) else 0.0


def merge(fast_b, slow_b, thr=0.5):
    out = list(slow_b)
    for b in fast_b:
        if not out or T.iou_matrix([b], out).max() < thr:
            out.append(b)
    return out


def pooled_f1(pred, gts, ids):
    tp = fp = fn = 0
    for i in ids:
        a, b, c, _ = T.greedy_match(pred[i], gts[i])
        tp, fp, fn = tp + a, fp + b, fn + c
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", default="runs/fast_distill")
    ap.add_argument("--slow", default="runs/sft_fusion/pred_test.jsonl")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--out", default="runs/escalation.json")
    a = ap.parse_args()

    thr = json.load(open(f"{a.fast}/train_summary.json"))["score_thr"]
    fast = load_preds(f"{a.fast}/pred_test.jsonl")
    slow = load_preds(a.slow)
    recs = [r for r in load_index(a.index, ("test",)) if r.id in fast and r.id in slow]
    ids = [r.id for r in recs]
    gts = {r.id: r.boxes for r in recs}

    fb = {}
    for i in ids:
        d = fast[i]
        order = sorted([k for k, s in enumerate(d["scores"]) if s >= thr], key=lambda k: -d["scores"][k])
        fb[i] = [d["boxes"][k] for k in order]
    sb = {i: slow[i]["boxes"] for i in ids}
    amb = {i: ambiguity(fast[i]["scores"]) for i in ids}
    img_f1 = lambda p, i: T.f1_reward(p, gts[i]) if (p or gts[i]) else 1.0
    gain = {i: img_f1(sb[i], i) - img_f1(fb[i], i) for i in ids}

    routers = {
        "uncertainty": sorted(ids, key=lambda i: -amb[i]),
        "random": random.Random(0).sample(ids, len(ids)),
        "oracle": sorted(ids, key=lambda i: -gain[i]),
    }
    rates = [round(x, 3) for x in np.linspace(0, 1, 21)]
    res = {"n_images": len(ids), "fast_thr": thr, "rates": rates, "curves": {}}
    for rname, order in routers.items():
        for comb in ["replace", "merge"]:
            ys = []
            for r in rates:
                esc = set(order[: int(round(r * len(ids)))])
                pred = {i: ((sb[i] if comb == "replace" else merge(fb[i], sb[i])) if i in esc else fb[i]) for i in ids}
                ys.append(pooled_f1(pred, gts, ids))
            res["curves"][f"{rname}/{comb}"] = ys
    res["fast_only_f1"] = pooled_f1(fb, gts, ids)
    res["slow_only_f1"] = pooled_f1(sb, gts, ids)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"fast-only F1 {res['fast_only_f1']:.4f}   slow-only F1 {res['slow_only_f1']:.4f}")
    for k, v in res["curves"].items():
        print(f"{k:22s}", " ".join(f"{x:.3f}" for x in v[::4]))


if __name__ == "__main__":
    main()
