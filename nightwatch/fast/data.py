"""Fast-head data: decode once into a uint8 cache (N,4,512,640) = RGB + thermal.

Training then runs entirely from GPU memory with GPU-side augmentation -- no dataloader
in the loop, which is what makes a few-minute epoch on 12k frames possible.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nightwatch.data.llvip import H, W, load_index

FW, FH = 640, 512
SCALE = FW / W  # 0.5
MAXB = 16


def _load(args):
    rgb_p, ir_p = args
    rgb = np.asarray(Image.open(rgb_p).convert("RGB").resize((FW, FH), Image.BILINEAR))
    ir = np.asarray(Image.open(ir_p).convert("L").resize((FW, FH), Image.BILINEAR))
    return np.concatenate([rgb.transpose(2, 0, 1), ir[None]], 0)


def build_cache(index, root, out_dir, workers=16):
    os.makedirs(out_dir, exist_ok=True)
    for split in ("labeled", "val", "unlabeled", "test"):
        recs = load_index(index, (split,))
        path = os.path.join(out_dir, f"{split}.npy")
        if os.path.exists(path):
            continue
        arr = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.uint8, shape=(len(recs), 4, FH, FW))
        with ProcessPoolExecutor(workers) as ex:
            for i, x in enumerate(ex.map(_load, [(r.rgb_path(root), r.ir_path(root)) for r in recs], chunksize=16)):
                arr[i] = x
        arr.flush(); del arr
        os.replace(path + ".tmp", path)
        json.dump([r.id for r in recs], open(os.path.join(out_dir, f"{split}_ids.json"), "w"))
        print("cached", split, len(recs), flush=True)


def boxes_tensor(ids, label_map):
    """(N, MAXB, 4) in 640x512 px + (N,) counts. label_map: id -> boxes in 1280x1024 px."""
    bx = torch.zeros(len(ids), MAXB, 4)
    n = torch.zeros(len(ids), dtype=torch.long)
    for i, iid in enumerate(ids):
        b = label_map.get(iid, [])[:MAXB]
        if b:
            bx[i, :len(b)] = torch.tensor(b, dtype=torch.float32) * SCALE
        n[i] = len(b)
    return bx, n


def load_split(cache_dir, split, device, label_map=None, index=None):
    x = torch.from_numpy(np.load(os.path.join(cache_dir, f"{split}.npy"), mmap_mode="r")[:])
    ids = json.load(open(os.path.join(cache_dir, f"{split}_ids.json")))
    if label_map is None:
        label_map = {r.id: r.boxes for r in load_index(index, (split,))}
    bx, n = boxes_tensor(ids, label_map)
    if os.environ.get("NW_FAST_CPU"):
        return x.pin_memory(), bx.to(device), n.to(device), ids
    try:
        x = x.to(device)
    except torch.OutOfMemoryError:
        x = x.pin_memory()
    return x, bx.to(device), n.to(device), ids


# ---------------------------------------------------------------- GPU augmentation
def augment(x, bx, n, gen, zoom=(1.0, 1.6), p_flip=0.5, p_drop_rgb=0.1, p_drop_t=0.05, modal="rgbt"):
    """x uint8 (B,4,H,W) on GPU -> float normalised + transformed boxes.
    Random zoom-in crop + h-flip via one affine_grid call; photometric night simulation."""
    B = x.shape[0]
    dev = x.device
    xf = x.float().div_(255.0)
    s = torch.empty(B, device=dev).uniform_(*zoom, generator=gen)
    a = 1.0 / s
    tx = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - a)
    ty = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - a)
    flip = torch.rand(B, device=dev, generator=gen) < p_flip
    ax = torch.where(flip, -a, a)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = ax, tx, a, ty
    grid = F.affine_grid(theta, xf.shape, align_corners=False)
    xf = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # boxes: px -> normalised [-1,1] -> output frame
    Hh, Ww = x.shape[2], x.shape[3]
    u = bx.clone()
    u[..., [0, 2]] = u[..., [0, 2]] / Ww * 2 - 1
    u[..., [1, 3]] = u[..., [1, 3]] / Hh * 2 - 1
    ox = (u[..., [0, 2]] - tx[:, None, None]) / ax[:, None, None]
    oy = (u[..., [1, 3]] - ty[:, None, None]) / a[:, None, None]
    ox = torch.sort(ox, -1).values
    area0 = (ox[..., 1] - ox[..., 0]) * (oy[..., 1] - oy[..., 0])
    ox, oy = ox.clamp(-1, 1), oy.clamp(-1, 1)
    area1 = (ox[..., 1] - ox[..., 0]).clamp(min=0) * (oy[..., 1] - oy[..., 0]).clamp(min=0)
    valid = (torch.arange(bx.shape[1], device=dev)[None] < n[:, None]) & (area1 > 0.4 * area0.clamp(min=1e-9))
    out = torch.stack([(ox[..., 0] + 1) / 2 * Ww, (oy[..., 0] + 1) / 2 * Hh, (ox[..., 1] + 1) / 2 * Ww, (oy[..., 1] + 1) / 2 * Hh], -1)
    # photometric: darken / gamma the visible camera (night), contrast jitter on thermal
    g = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.6, generator=gen)
    xf[:, :3] = xf[:, :3].clamp(0, 1).pow(g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.2, generator=gen)
    xf[:, 3:] = (xf[:, 3:] - 0.5) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.25, generator=gen) + 0.5
    if modal == "rgbt":
        r = torch.rand(B, device=dev, generator=gen)
        xf[r < p_drop_rgb, :3] = 0
        xf[(r >= p_drop_rgb) & (r < p_drop_rgb + p_drop_t), 3:] = 0
    return normalise(xf, modal), out, valid


MEAN = torch.tensor([0.485, 0.456, 0.406, 0.449]).view(1, 4, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.226]).view(1, 4, 1, 1)


def select(xf, modal):
    return {"rgbt": xf, "rgb": xf[:, :3], "t": xf[:, 3:]}[modal]


def normalise(xf, modal):
    m, s = MEAN.to(xf.device), STD.to(xf.device)
    return select((xf - m) / s, modal)


def prep_eval(x, modal, cond="normal"):
    xf = x.float().div(255.0)
    if cond == "rgb_off":
        xf[:, :3] = 0
    elif cond == "thermal_off":
        xf[:, 3:] = 0
    return normalise(xf, modal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--root", default="data/raw/LLVIP")
    ap.add_argument("--out", default="data/cache_fast")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    build_cache(a.index, a.root, a.out, a.workers)
