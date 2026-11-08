"""Combine the fast and slow heads on TriSAR test using only model outputs (no site labels), CPU only.

Routers, applied per frame:
  fast      : fast head alone (its own val-tuned threshold)
  slow      : slow head alone
  abstain   : slow head's report if it reports anyone, else the fast head ("slow says nobody -> trust fast")
  union     : slow persons + fast persons not overlapping them (IoU < 0.3)
The camera-dropout slow head (cd50) abstains where it can't see, unlike the earlier one, which hallucinates.
Comparing abstain/union for the two slow heads shows what that calibration buys at the system level.

python -m nightwatch.eval.router_trisar
"""
from __future__ import annotations

import json
import os

from nightwatch.eval.metrics_trisar import score_file

RUNS = os.environ.get("NW_RUNS", "runs")
OUT = f"{RUNS}/t_router"


def load(path):
    return {d["id"]: d for d in map(json.loads, open(path)) if d["task"] == "report" and d["cond"] == "normal"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def route(fast, slow, thr, kind):
    out = {}
    for i, s in slow.items():
        f = [dict(q) for q in fast.get(i, {}).get("persons", []) if q["score"] >= thr]
        sp = [{**q, "score": 1.0} for q in s.get("persons", [])]
        if kind == "fast":
            p = f
        elif kind == "slow":
            p = sp
        elif kind == "abstain":
            p = sp if sp else f
        elif kind == "union":
            p = sp + [q for q in f if all(iou(q["box"], r["box"]) < 0.3 for r in sp)]
        out[i] = {"id": i, "task": "report", "cond": "normal", "persons": p}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fast_run = "tf_rgbtl"
    thr = json.load(open(f"{RUNS}/{fast_run}/train_summary.json"))["score_thr"]
    fast = load(f"{RUNS}/{fast_run}/pred_test.jsonl")
    res = {}
    for slow_run in ["t_sft_fusion_l", "t_sft_fusion_l_cd50"]:
        p = f"{RUNS}/{slow_run}/pred_test.jsonl"
        if not os.path.exists(p):
            continue
        slow = load(p)
        for kind in ["fast", "slow", "abstain", "union"]:
            name = kind if kind == "fast" else f"{kind}__{slow_run}"
            if name in res:
                continue
            path = f"{OUT}/pred_{name}.jsonl"
            with open(path, "w") as fh:
                for d in route(fast, slow, thr, kind).values():
                    fh.write(json.dumps(d) + "\n")
            m = score_file(path, os.environ.get("NW_TRISAR_FUSED", "data/trisar_fused"))["report/normal"]
            res[name] = m
            row = {sl: round(m[sl]["f1"], 3) for sl in ["all", "night", "loc_forest", "loc_lab"] if sl in m}
            print(f"{name:40s} F1 {row}  P {m['all']['precision']:.3f} R {m['all']['recall']:.3f} "
                  f"pose {m['all'].get('pose_acc', float('nan')):.3f} dist {m['all'].get('dist_mae_m', float('nan')):.2f}")
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
