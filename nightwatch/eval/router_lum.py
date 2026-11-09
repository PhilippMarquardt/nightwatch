"""Sensor-aware routing on LLVIP: per frame, use the fused (RGB+thermal) slow head when the camera image is dark and the
RGB-only slow head when it is bright. Threshold = median camera luminance of the *training* (labelled) split, fixed
without looking at test. Both heads share the base model; switching is a LoRA/adapter swap, not a second model.

python -m nightwatch.eval.router_lum
"""
from __future__ import annotations

import json
import os

import numpy as np

RUNS = os.environ.get("NW_RUNS", "runs")


def load(p):
    return {d["id"]: d for d in map(json.loads, open(p)) if d["task"] == "detect" and d["cond"] == "normal"}


def main():
    idx = {r["id"]: r for r in json.load(open("data/index.json"))}
    thr = float(np.median([r["rgb_lum"] for r in idx.values() if r["split"] == "labeled"]))
    fused, rgb = load(f"{RUNS}/sft_fusion/pred_test.jsonl"), load(f"{RUNS}/sft_rgb/pred_test.jsonl")
    ids = sorted(set(fused) & set(rgb))
    os.makedirs(f"{RUNS}/router_lum", exist_ok=True)
    with open(f"{RUNS}/router_lum/pred_test.jsonl", "w") as fh:
        for i in ids:
            fh.write(json.dumps(fused[i] if idx[i]["rgb_lum"] < thr else rgb[i]) + "\n")
    n_dark = sum(idx[i]["rgb_lum"] < thr for i in ids)
    json.dump({"lum_threshold": thr, "n_frames": len(ids), "n_fused": n_dark},
              open(f"{RUNS}/router_lum/router.json", "w"), indent=1)
    print(f"luminance router: thr {thr:.1f} (train median), {n_dark}/{len(ids)} frames -> fused head")


if __name__ == "__main__":
    main()
