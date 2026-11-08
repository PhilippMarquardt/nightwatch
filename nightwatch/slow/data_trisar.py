"""TriSAR-3D tasks + datasets for the slow head: incident-style reports over RGB + thermal + LiDAR.

Frames come pre-rendered by nightwatch.data.trisar into the thermal reference frame (640x512):
    <id>_rgb.jpg (warped RGB), <id>_t.png (thermal), <id>_l.png (lidar: inv-range, intensity, hit).
Every answer is verifiable from the 3D labels (box, posture tag, metric range).
"""
from __future__ import annotations

import json
import os
import random
import re

import torch
from PIL import Image
from torch.utils.data import Dataset

from nightwatch.data import tasks as T
from nightwatch.slow.model import IMG_H, IMG_W, lidar_tensor, thermal_tensor

FW, FH = 640, 512
POSE3 = {"standing": "upright", "walking": "upright", "sitting": "sitting", "lying": "lying"}

REPORT_PROMPT = ('Report every person for the security operator: a JSON list of objects with "bbox_2d", '
                 '"pose" (upright, sitting or lying) and "distance_m". Use [] if nobody is there.')
LYING_PROMPT = "Is anyone lying on the ground? Answer yes or no."
NEAREST_PROMPT = "How far away is the nearest person, in metres? Answer with one number, or none if nobody is visible."
COUNT_PROMPT = T.COUNT_PROMPTS[0]

PREAMBLE = {
    "rgb": "Sensors: visible camera.",
    "fusion": "Sensors: visible camera and thermal camera, time-synchronised and pixel-aligned. Fused sensor stream:",
    "fusion_l": ("Sensors: visible camera, thermal camera and 128-beam LiDAR, time-synchronised and projected into one "
                 "frame. Fused sensor stream:"),
}


def load_frames(fused_dir, splits=None):
    fr = json.load(open(os.path.join(fused_dir, "frames.json")))
    return [f for f in fr if splits is None or f["split"] in splits]


def report_answer(persons, crop=None):
    items = []
    for p in sorted(persons, key=lambda p: (p["box"][0], p["box"][1])):
        b = T.to_qwen([p["box"]], *(crop if crop else (FW, FH)))[0]
        items.append(f'\t{{"bbox_2d": [{b[0]}, {b[1]}, {b[2]}, {b[3]}], "pose": "{POSE3[p["pose"]]}", '
                     f'"distance_m": {p["range_m"]:.1f}}}')
    return "[]" if not items else "```json\n[\n" + ",\n".join(items) + "\n]\n```"


_OBJ = re.compile(r"\{[^{}]*\}")


def parse_report(text):
    """-> list of {box (0-1000), pose, distance}; tolerant to truncation and missing fields."""
    out = []
    for m in _OBJ.finditer(text):
        s = m.group()
        bx = T.parse_boxes(s)
        if not bx:
            continue
        pose = re.search(r'"pose"\s*:\s*"(\w+)"', s)
        dist = re.search(r'"distance_m"\s*:\s*(-?\d+(?:\.\d+)?)', s)
        out.append({"box": bx[0], "pose": pose.group(1).lower() if pose else None,
                    "distance": float(dist.group(1)) if dist else None})
    return out


def sample_task(persons, rng, crop_wh):
    task = rng.choices(["report", "lying", "nearest", "count"], weights=(0.6, 0.15, 0.15, 0.1))[0]
    if task == "report":
        return task, REPORT_PROMPT, report_answer(persons, crop_wh)
    if task == "lying":
        return task, LYING_PROMPT, "yes" if any(p["pose"] == "lying" for p in persons) else "no"
    if task == "nearest":
        return task, NEAREST_PROMPT, (f'{min(p["range_m"] for p in persons):.1f}' if persons else "none")
    return task, COUNT_PROMPT, str(len(persons))


def load_triplet(fused_dir, fid, crop=None):
    base = os.path.join(fused_dir, fid)
    rgb = Image.open(base + "_rgb.jpg").convert("RGB")
    th = Image.open(base + "_t.png").convert("L")
    li = Image.open(base + "_l.png").convert("RGB")
    if crop is not None:
        c = tuple(int(round(v)) for v in crop)
        rgb, th, li = rgb.crop(c), th.crop(c), li.crop(c)
    return (rgb.resize((IMG_W, IMG_H), Image.BILINEAR), th.resize((IMG_W, IMG_H), Image.BILINEAR),
            li.resize((IMG_W, IMG_H), Image.NEAREST))      # nearest: keep lidar values un-blended


def crop_persons(persons, crop, keep_vis=0.5):
    x0, y0, x1, y1 = crop
    out = []
    for p in persons:
        b = p["box"]
        ix1, iy1, ix2, iy2 = max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / max(1e-6, (b[2] - b[0]) * (b[3] - b[1])) >= keep_vis:
            out.append({**p, "box": [ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0]})
    return out


def messages(mode, prompt):
    return [{"role": "user", "content": [{"type": "text", "text": PREAMBLE[mode]}, {"type": "image"},
                                         {"type": "text", "text": prompt}]}]


def _apply_cond(rgb, th, li, cond):
    if cond == "rgb_off":
        rgb = Image.new("RGB", rgb.size)
    elif cond == "thermal_off":
        th = Image.new("L", th.size)
    elif cond == "lidar_off":
        li = Image.new("RGB", li.size)
    return rgb, th, li


class TriSARSFT(Dataset):
    def __init__(self, frames, fused_dir, proc, mode, seed=0, crop_p=0.25, drop=(0.2, 0.1, 0.1)):
        self.frames, self.dir, self.proc, self.mode, self.seed = frames, fused_dir, proc, mode, seed
        self.crop_p, self.drop = crop_p, drop
        self.epoch = 0
        self.end_ids = proc.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        persons, crop, wh = f["persons"], None, (FW, FH)
        if rng.random() < self.crop_p:
            s = rng.uniform(0.5, 0.85)
            cw, ch = FW * s, FH * s
            x0, y0 = rng.uniform(0, FW - cw), rng.uniform(0, FH - ch)
            crop, wh = (x0, y0, x0 + cw, y0 + ch), (cw, ch)
            persons = crop_persons(persons, crop)
        rgb, th, li = load_triplet(self.dir, f["id"], crop)
        if self.mode != "rgb":
            r = rng.random()
            d_rgb, d_t, d_l = self.drop
            if r < d_rgb:
                rgb, th, li = _apply_cond(rgb, th, li, "rgb_off")
            elif r < d_rgb + d_t:
                rgb, th, li = _apply_cond(rgb, th, li, "thermal_off")
            elif self.mode == "fusion_l" and r < d_rgb + d_t + d_l:
                rgb, th, li = _apply_cond(rgb, th, li, "lidar_off")
        task, prompt, answer = sample_task(persons, rng, wh)
        text = self.proc.apply_chat_template(messages(self.mode, prompt), add_generation_prompt=True, tokenize=False,
                                               enable_thinking=False)  # direct answers, no CoT preamble
        enc = self.proc(text=[text], images=[rgb], return_tensors="pt")
        ans = self.proc.tokenizer(answer, add_special_tokens=False)["input_ids"] + self.end_ids
        ids = torch.cat([enc["input_ids"][0], torch.tensor(ans)])
        labels = torch.cat([torch.full_like(enc["input_ids"][0], -100), torch.tensor(ans)])
        out = {}
        if "mm_token_type_ids" in enc:      # transformers>=5.2x: M-RoPE needs per-token modality ids
            out["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"][0], torch.zeros(len(ans), dtype=enc["mm_token_type_ids"].dtype)])
        return {**out, "input_ids": ids, "labels": labels, "pixel_values": enc["pixel_values"],
                "image_grid_thw": enc["image_grid_thw"], "thermal": thermal_tensor(th), "lidar": lidar_tensor(li),
                "task": task}


class TriSARInfer(Dataset):
    def __init__(self, frames, fused_dir, proc, mode, prompt, cond="normal"):
        self.frames, self.dir, self.proc, self.mode, self.prompt, self.cond = frames, fused_dir, proc, mode, prompt, cond

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        rgb, th, li = _apply_cond(*load_triplet(self.dir, f["id"]), self.cond)
        text = self.proc.apply_chat_template(messages(self.mode, self.prompt), add_generation_prompt=True, tokenize=False,
                                               enable_thinking=False)  # direct answers, no CoT preamble
        return {"id": f["id"], "text": text, "images": [rgb], "thermal": thermal_tensor(th), "lidar": lidar_tensor(li)}
