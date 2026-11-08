"""Score prediction files on identical slices.

Prediction jsonl rows: {id, task, cond, boxes?[, scores?], text?}
Writes metrics.json: {"<task>/<cond>": {"<slice>": {...}}}

python -m nightwatch.eval.metrics --pred runs/x/pred_test.jsonl --out runs/x/metrics_test.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from nightwatch.data import tasks as T
from nightwatch.data.llvip import image_slices, load_index


def height_buckets(recs):
    hs = np.array([b[3] - b[1] for r in recs for b in r.boxes])
    lo, hi = np.percentile(hs, [33.3, 66.7])
    return lambda b: "far" if (b[3] - b[1]) < lo else ("near" if (b[3] - b[1]) >= hi else "mid")


def ap50(preds: dict, gts: dict, ids) -> float:
    """VOC-style all-point AP at IoU 0.5 over a set of images. Needs scores."""
    allp, npos = [], 0
    for i in ids:
        g = gts[i]
        npos += len(g)
        p = preds.get(i, {"boxes": [], "scores": []})
        order = np.argsort(-np.asarray(p["scores"], dtype=float)) if len(p["boxes"]) else []
        boxes = [p["boxes"][k] for k in order]
        m = T.iou_matrix(boxes, g)
        used = np.zeros(len(g), bool)
        for k, kk in enumerate(order):
            j = int(np.argmax(np.where(~used, m[k], -1))) if len(g) else -1
            hit = j >= 0 and m[k, j] >= 0.5 and not used[j]
            if hit:
                used[j] = True
            allp.append((p["scores"][kk], hit))
    if npos == 0:
        return float("nan")
    allp.sort(key=lambda x: -x[0])
    tp = np.cumsum([h for _, h in allp]); fp = np.cumsum([not h for _, h in allp])
    rec = tp / npos; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detect_metrics(preds, gts, ids, bucket, score_thr=None):
    tp = fp = fn = 0
    cnt_exact, cnt_abs, img_f1 = [], [], []
    per_b = defaultdict(lambda: [0, 0])  # bucket -> [hit, total]
    for i in ids:
        p = preds.get(i, {"boxes": [], "scores": None})
        boxes, scores = p["boxes"], p.get("scores")
        if scores is not None and score_thr is not None:
            keep = [k for k, s in enumerate(scores) if s >= score_thr]
            order = sorted(keep, key=lambda k: -scores[k])
            boxes = [boxes[k] for k in order]
        g = gts[i]
        a, b, c, used = T.greedy_match(boxes, g)
        tp, fp, fn = tp + a, fp + b, fn + c
        img_f1.append(2 * a / max(1, 2 * a + b + c) if (g or boxes) else 1.0)
        cnt_exact.append(len(boxes) == len(g)); cnt_abs.append(abs(len(boxes) - len(g)))
        for gb, u in zip(g, used):
            per_b[bucket(gb)][0] += int(u); per_b[bucket(gb)][1] += 1
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    out = {"n_images": len(ids), "precision": P, "recall": R, "f1": 2 * P * R / max(1e-9, P + R),
           "mean_image_f1": float(np.mean(img_f1)) if img_f1 else float("nan"),
           "count_exact": float(np.mean(cnt_exact)) if cnt_exact else float("nan"),
           "count_mae": float(np.mean(cnt_abs)) if cnt_abs else float("nan")}
    for k, (h, t) in per_b.items():
        out[f"recall_{k}"] = h / max(1, t)
    if any(preds.get(i, {}).get("scores") is not None for i in ids[:5]):
        out["ap50"] = ap50(preds, gts, ids)
    return out


def score_file(pred_path, index_path, split="test", score_thr=None):
    recs = load_index(index_path, (split,))
    gts = {r.id: r.boxes for r in recs}
    slices = image_slices(recs)
    bucket = height_buckets(recs)
    q_gt = {r.id: T.to_qwen(r.boxes, 1280, 1024) for r in recs}
    rows = defaultdict(dict)
    for line in open(pred_path):
        d = json.loads(line)
        rows[(d["task"], d["cond"])][d["id"]] = d
    res = {}
    for (task, cond), preds in sorted(rows.items()):
        key = f"{task}/{cond}"
        res[key] = {}
        for sname, ids in slices.items():
            ids = [i for i in ids if i in preds]
            if not ids:
                continue
            if task == "detect":
                res[key][sname] = detect_metrics(preds, gts, ids, bucket, score_thr)
            elif task == "count":
                v = [T.parse_int(preds[i]["text"]) for i in ids]
                g = [len(gts[i]) for i in ids]
                res[key][sname] = {"n_images": len(ids), "count_exact": float(np.mean([a == b for a, b in zip(v, g)])),
                                   "count_mae": float(np.mean([abs((a if a is not None else 0) - b) for a, b in zip(v, g)]))}
            elif task.startswith("zone_"):
                z = task.split("_", 1)[1]
                v = [T.parse_yesno(preds[i]["text"]) for i in ids]
                g = [T.zone_truth(q_gt[i], z) for i in ids]
                res[key][sname] = {"n_images": len(ids), "accuracy": float(np.mean([a == b for a, b in zip(v, g)]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = score_file(a.pred, a.index, a.split, a.score_thr)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        allm = v.get("all", {})
        print(k, {m: round(x, 4) for m, x in allm.items() if isinstance(x, float)})


if __name__ == "__main__":
    main()
