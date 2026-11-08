"""Dataset + collator for slow-head SFT and inference."""
from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from nightwatch.data import tasks as T
from nightwatch.data.llvip import H, W, Record
from nightwatch.slow.model import IMG_H, IMG_W, thermal_tensor


def load_pair(rec: Record, root: str, crop=None):
    rgb = Image.open(rec.rgb_path(root)).convert("RGB")
    ir = Image.open(rec.ir_path(root)).convert("L")
    if crop is not None:
        c = tuple(int(round(v)) for v in crop)
        rgb, ir = rgb.crop(c), ir.crop(c)
    return rgb.resize((IMG_W, IMG_H), Image.BILINEAR), ir.resize((IMG_W, IMG_H), Image.BILINEAR)


def build_messages(mode: str, prompt: str):
    if mode == "rgb":
        content = [{"type": "text", "text": T.SENSOR_PREAMBLE["rgb"]}, {"type": "image"}, {"type": "text", "text": prompt}]
    elif mode == "dual":
        content = [{"type": "text", "text": T.SENSOR_PREAMBLE["rgbt"] + "\nVisible camera:"}, {"type": "image"},
                   {"type": "text", "text": "Thermal camera:"}, {"type": "image"}, {"type": "text", "text": prompt}]
    elif mode == "fusion":
        # one token stream: the visual tokens carry both sensors
        content = [{"type": "text", "text": T.SENSOR_PREAMBLE["rgbt"] + " Fused sensor stream:"}, {"type": "image"},
                   {"type": "text", "text": prompt}]
    else:
        raise ValueError(mode)
    return [{"role": "user", "content": content}]


def images_for(mode, rgb, ir):
    if mode == "dual":
        return [rgb, ir.convert("RGB")]
    return [rgb]


class SFTDataset(Dataset):
    def __init__(self, recs: list[Record], root: str, proc, mode: str, crop_p=0.35, drop_rgb=0.25, drop_t=0.10,
                 seed=0, labels: dict | None = None):
        self.recs, self.root, self.proc, self.mode = recs, root, proc, mode
        self.crop_p, self.drop_rgb, self.drop_t = crop_p, drop_rgb, drop_t
        self.seed = seed
        self.labels = labels  # optional override {id: boxes_px} (pseudo labels)
        self.epoch = 0
        self.end_ids = proc.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        boxes = self.labels.get(rec.id, rec.boxes) if self.labels else rec.boxes
        crop, w, h = None, W, H
        if rng.random() < self.crop_p:
            crop, boxes, (w, h) = T.random_crop(boxes, W, H, rng)
        rgb, ir = load_pair(rec, self.root, crop)
        if self.mode != "rgb":
            r = rng.random()
            if r < self.drop_rgb:
                rgb = Image.new("RGB", rgb.size)           # camera blinded / pitch black
            elif r < self.drop_rgb + self.drop_t:
                ir = Image.new("L", ir.size)               # thermal sensor offline
        q = T.to_qwen(boxes, w, h)
        task, prompt, answer = T.sample_task(q, rng)
        msgs = build_messages(self.mode, prompt)
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.proc(text=[text], images=images_for(self.mode, rgb, ir), return_tensors="pt")
        ans = self.proc.tokenizer(answer, add_special_tokens=False)["input_ids"] + self.end_ids
        ids = torch.cat([enc["input_ids"][0], torch.tensor(ans)])
        labels = torch.cat([torch.full_like(enc["input_ids"][0], -100), torch.tensor(ans)])
        return {"input_ids": ids, "labels": labels, "pixel_values": enc["pixel_values"],
                "image_grid_thw": enc["image_grid_thw"], "thermal": thermal_tensor(ir), "task": task}


def sft_collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):                       # right padding for training
        n = len(b["input_ids"])
        ids[i, :n], lab[i, :n], att[i, :n] = b["input_ids"], b["labels"], 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att,
            "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
            "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
            "thermal": torch.stack([b["thermal"] for b in batch])}


class InferDataset(Dataset):
    """Yields prompts for generation. cond: normal | rgb_off | thermal_off."""

    def __init__(self, recs, root, proc, mode, prompt, cond="normal"):
        self.recs, self.root, self.proc, self.mode, self.prompt, self.cond = recs, root, proc, mode, prompt, cond

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        rgb, ir = load_pair(rec, self.root)
        if self.cond == "rgb_off":
            rgb = Image.new("RGB", rgb.size)
        elif self.cond == "thermal_off":
            ir = Image.new("L", ir.size)
        if self.mode == "rgb" and self.cond == "rgb_off":
            pass  # rgb model with dead camera: sees black (the honest failure mode)
        msgs = build_messages(self.mode, self.prompt)
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return {"id": rec.id, "text": text, "images": images_for(self.mode, rgb, ir), "thermal": thermal_tensor(ir)}


def infer_collate(batch, proc):
    proc.tokenizer.padding_side = "left"
    enc = proc(text=[b["text"] for b in batch], images=[im for b in batch for im in b["images"]],
               return_tensors="pt", padding=True)
    enc["thermal"] = torch.stack([b["thermal"] for b in batch])
    enc["ids"] = [b["id"] for b in batch]
    return enc
