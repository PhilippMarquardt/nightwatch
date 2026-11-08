"""Instruction tasks for the slow head, built from box annotations.

Every answer is machine-checkable, so the same functions serve as SFT targets,
evaluation scorers and (later) RL rewards.

Coordinates follow Qwen3-VL's native grounding convention: integers in [0, 1000]
relative to the image the model sees.
"""
from __future__ import annotations

import json
import random
import re

import numpy as np

DETECT_PROMPTS = [
    'Locate every person in the scene. Output a JSON list of objects with keys "bbox_2d" and "label".',
    'Security check: find all people visible to the sensors. Return JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, ...].',
    'Detect all persons. Answer only with a JSON list of {"bbox_2d", "label"} objects; use [] if nobody is there.',
]
COUNT_PROMPTS = [
    "How many people are in the scene? Answer with a single number.",
    "Count the persons visible to the sensors. Reply with just the number.",
]
ZONE_PROMPTS = [
    "Is anyone in the {zone} third of the frame? Answer yes or no.",
    "Security zone check: is a person present in the {zone} third of the view? Reply yes or no.",
]
ZONES = {"left": (0, 1 / 3), "center": (1 / 3, 2 / 3), "right": (2 / 3, 1.0)}

SENSOR_PREAMBLE = {
    "rgb": "Sensors: visible camera.",
    "rgbt": "Sensors: visible camera and thermal camera, time-synchronised and pixel-aligned.",
}

EVAL_DETECT_PROMPT = DETECT_PROMPTS[0]


def to_qwen(boxes, w, h):
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append([int(round(x1 / w * 1000)), int(round(y1 / h * 1000)),
                    int(round(x2 / w * 1000)), int(round(y2 / h * 1000))])
    return out


def from_qwen(boxes, w, h):
    return [[b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h] for b in boxes]


def detect_answer(qboxes) -> str:
    qboxes = sorted(qboxes, key=lambda b: (b[0], b[1]))
    if not qboxes:
        return "[]"
    items = ",\n".join(f'\t{{"bbox_2d": [{b[0]}, {b[1]}, {b[2]}, {b[3]}], "label": "person"}}' for b in qboxes)
    return "```json\n[\n" + items + "\n]\n```"


_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_boxes(text: str) -> list[list[float]]:
    """Robust: grab every 4-number list. Works on truncated JSON too."""
    out = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 > 0 and y2 - y1 > 0:
            out.append([max(0, x1), max(0, y1), min(1000, x2), min(1000, y2)])
    return out


def parse_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def parse_yesno(text: str) -> bool | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def zone_truth(qboxes, zone: str) -> bool:
    lo, hi = ZONES[zone]
    return any(lo * 1000 <= (b[0] + b[2]) / 2 < hi * 1000 for b in qboxes)


def sample_task(qboxes, rng: random.Random, weights=(0.7, 0.15, 0.15)):
    """Returns (task, prompt, answer). qboxes are already in the model's 0-1000 frame."""
    task = rng.choices(["detect", "count", "zone"], weights=weights)[0]
    if task == "detect":
        return task, rng.choice(DETECT_PROMPTS), detect_answer(qboxes)
    if task == "count":
        return task, rng.choice(COUNT_PROMPTS), str(len(qboxes))
    zone = rng.choice(list(ZONES))
    return task, rng.choice(ZONE_PROMPTS).format(zone=zone), "yes" if zone_truth(qboxes, zone) else "no"


# ------------------------------------------------------------------ crop aug
def random_crop(boxes, w, h, rng: random.Random, min_frac=0.35, max_frac=0.8, keep_vis=0.5):
    """Aspect-preserving crop. Returns crop window and boxes inside it (crop pixel frame).
    Produces empty frames naturally, which LLVIP almost never has."""
    f = rng.uniform(min_frac, max_frac)
    cw, ch = w * f, h * f
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    out = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + cw), min(y2, y0 + ch)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        vis = (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1))
        if vis >= keep_vis:
            out.append([ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0])
    return (x0, y0, x0 + cw, y0 + ch), out, (cw, ch)


# ------------------------------------------------------------------ matching
def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def greedy_match(pred, gt, thr=0.5):
    """pred assumed sorted by confidence. Returns (tp, fp, fn, matched_gt_mask)."""
    m = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    tp = 0
    for i in range(len(pred)):
        if len(gt) == 0:
            break
        cand = np.where(~used, m[i], -1)
        j = int(cand.argmax())
        if cand[j] >= thr:
            used[j] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp, used


def f1_reward(pred_q, gt_q, thr=0.5) -> float:
    """Per-image F1 in the 0-1000 frame; 1.0 for correctly empty. Usable as an RL reward."""
    if not gt_q and not pred_q:
        return 1.0
    tp, fp, fn, _ = greedy_match(pred_q, gt_q, thr)
    return 2 * tp / max(1, 2 * tp + fp + fn)


if __name__ == "__main__":
    g = [[100, 200, 300, 600], [500, 100, 600, 400]]
    s = detect_answer(g)
    print(s)
    assert parse_boxes(s) == [[100.0, 200.0, 300.0, 600.0], [500.0, 100.0, 600.0, 400.0]]
    assert f1_reward(parse_boxes(s), g) == 1.0
    assert parse_boxes('[{"bbox_2d": [1, 2, 3') == []
    print(json.dumps(sample_task(g, random.Random(0))))
    print("tasks ok")
