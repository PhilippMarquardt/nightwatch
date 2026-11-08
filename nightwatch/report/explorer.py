"""Build the data + image sprites for the interactive Night Watch explorer page (report/explorer/).

Each frame -> one JPEG sprite: [camera ; thermal ; thermal + LiDAR] stacked vertically (640 x 1536), plus a JSON record
with ground truth and every head's output under each sensor condition. CPU only, from saved predictions.

python -m nightwatch.report.explorer
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
from PIL import Image

from nightwatch.report.video import lidar_overlay, stamp_index, iou

RUNS = os.environ.get("NW_RUNS", "runs")
FUSED = os.environ.get("NW_TRISAR_FUSED", "data/trisar_fused")
OUT = "report/explorer"
CONDS = ["normal", "rgb_off", "thermal_off", "lidar_off"]
CLIPS = [  # (key, label, scene, start, n)
    ("forest", "Unseen forest, night", "rosbag2_2025_11_17-17_55_10_out", 19, 36),
    ("forest_far", "Unseen forest, far range", "rosbag2_2025_11_17-17_59_40_out", 0, 36),
    ("lab_night", "Held-out lab, evening", "rosbag2_2025_11_14-18_08_17_out", None, 36),
    ("lab_day", "Held-out lab, day", "rosbag2_2025_09_18-16_28_32_out", 76, 36),
]


def load_all(path, thr=None):
    out = {}
    if not os.path.exists(path):
        return out
    for d in map(json.loads, open(path)):
        if d["task"] != "report":
            continue
        p = d.get("persons", [])
        if thr is not None:
            p = [q for q in p if q["score"] >= thr]
        out[(d["id"], d["cond"])] = {"text": d.get("text"), "persons": p}
    return out


def nms(ps, t=0.5):
    keep = []
    for q in sorted(ps, key=lambda q: -q.get("score", 1)):
        if all(iou(q["box"], r["box"]) < t for r in keep):
            keep.append(q)
    return keep


def P(q):
    r = {"b": [int(round(v)) for v in q["box"]], "pose": q.get("pose")}
    if q.get("distance") is not None:
        r["r"] = round(float(q["distance"]), 1)
    if "score" in q:
        r["s"] = round(float(q["score"]), 2)
    return r


def main():
    os.makedirs(f"{OUT}/f", exist_ok=True)
    frames = {f["id"]: f for f in json.load(open(f"{FUSED}/frames.json"))}
    ts = stamp_index()
    thr = json.load(open(f"{RUNS}/tf_rgbtl/train_summary.json"))["score_thr"]
    fast = load_all(f"{RUNS}/tf_rgbtl/pred_test.jsonl", thr)
    slow = {}
    for run, key in [("t_sft_fusion_l", "vlm"), ("t_sft_fusion_l_cd50", "vlm_cd")]:
        slow[key] = {**load_all(f"{RUNS}/{run}/pred_test.jsonl"), **load_all(f"{RUNS}/{run}/pred_test_robust.jsonl")}
    robust_ids = sorted({i for (i, c) in slow["vlm_cd"] if c == "rgb_off"})

    picks = []
    for key, label, scene, start, n in CLIPS:
        ids = sorted([i for i, f in frames.items() if f["scene"] == scene], key=lambda i: ts[i])
        if start is None:
            cnt = np.convolve([len(frames[i]["persons"]) for i in ids], np.ones(n), "valid")
            start = int(np.argmax(cnt))
        picks += [(key, label, i) for i in ids[start:start + n]]
    rf = [i for i in robust_ids if frames[i]["location"] == "forest" and frames[i]["persons"]]
    rl = [i for i in robust_ids if frames[i]["location"] == "lab" and frames[i]["persons"]]
    rng = random.Random(0)
    sens = sorted(rng.sample(rf, 12) + rng.sample(rl, 12), key=lambda i: (frames[i]["location"], ts[i]))
    picks += [("sensors", "Sensor-failure test set", i) for i in sens]

    recs = []
    for k, (clip, label, i) in enumerate(picks):
        f = frames[i]
        rgb = Image.open(f"{FUSED}/{i}_rgb.jpg").convert("RGB")
        th = np.asarray(Image.open(f"{FUSED}/{i}_t.png"))
        li = np.asarray(Image.open(f"{FUSED}/{i}_l.png"))
        sprite = Image.new("RGB", (640, 1536))
        sprite.paste(rgb, (0, 0)); sprite.paste(Image.fromarray(np.stack([th] * 3, -1)), (0, 512))
        sprite.paste(lidar_overlay(th, li), (0, 1024))
        sprite.save(f"{OUT}/f/{k:03d}.jpg", quality=68, optimize=True, progressive=True)
        pred = {}
        for c in CONDS:
            e = {}
            if (i, c) in fast:
                e["fast"] = [P(q) for q in nms(fast[(i, c)]["persons"])]
            for key in ("vlm", "vlm_cd"):
                if (i, c) in slow[key]:
                    s = slow[key][(i, c)]
                    e[key] = {"p": [P(q) for q in s["persons"]], "txt": (s["text"] or "").strip()[:600]}
            if "fast" in e and "vlm_cd" in e:
                sp = e["vlm_cd"]["p"]
                e["router"] = sp + [q for q in e["fast"] if all(iou(q["b"], r["b"]) < 0.3 for r in sp)]
            if e:
                pred[c] = e
        recs.append({"k": k, "clip": clip, "label": label, "id": i, "loc": f["location"], "night": f["night"],
                     "dark": bool(np.asarray(rgb.convert("L")).mean() < 25),
                     "t": round((ts[i] - ts[picks[0][2]]) / 1e9, 1),
                     "gt": [{"b": [int(round(v)) for v in q["box"]], "pose": q["pose"], "r": round(q["range_m"], 1)}
                            for q in f["persons"]],
                     "pred": pred})
    res = json.load(open(f"{RUNS}/t_router/results.json"))
    summary = {k: {s: round(res[k][s]["f1"], 3) for s in ("all", "loc_forest", "loc_lab", "night")}
               for k in ("fast", "slow__t_sft_fusion_l", "slow__t_sft_fusion_l_cd50", "union__t_sft_fusion_l_cd50")}
    json.dump({"frames": recs, "summary": summary, "fast_thr": thr}, open(f"{OUT}/data.json", "w"), separators=(",", ":"))
    tot = sum(os.path.getsize(f"{OUT}/f/{x}") for x in os.listdir(f"{OUT}/f"))
    print(len(recs), "frames;", round(tot / 1e6, 1), "MB sprites;", round(os.path.getsize(f"{OUT}/data.json") / 1e3), "kB data")


if __name__ == "__main__":
    main()
