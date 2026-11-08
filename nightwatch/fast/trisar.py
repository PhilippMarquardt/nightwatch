"""Fast head on TriSAR-3D: early-fused RGB+T+LiDAR (7 ch) CenterNet with posture classes + metric distance.

    heatmap   3 channels = upright / sitting / lying (posture is detected, not a separate classifier)
    wh, off   as in the LLVIP head
    dist      log-distance regression at person centres (LiDAR makes this a lookup, RGB alone must guess)

Modal ablations: rgbtl (7ch), rgbt (4ch), rgb (3ch), t (1ch), tl (4ch: thermal + lidar).
Distillation: --labels <slow-head report jsonl on the train split> replaces GT boxes/poses/distances.

python -m nightwatch.fast.trisar --name tf_rgbtl --modal rgbtl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from nightwatch.fast.model import STRIDE, CenterNetRGBT, _head, focal_loss
from nightwatch.slow.data_trisar import FH, FW, POSE3, load_frames
from nightwatch.utils import RunLogger, seed_all

CLASSES = ["upright", "sitting", "lying"]
CH = {"rgbtl": [0, 1, 2, 3, 4, 5, 6], "rgbt": [0, 1, 2, 3], "rgb": [0, 1, 2], "t": [3], "tl": [3, 4, 5, 6]}
MEAN = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.10, 0.20, 0.45]).view(1, 7, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.15, 0.25, 0.50]).view(1, 7, 1, 1)
MAXB = 24


# ---------------------------------------------------------------- cache
def _load7(args):
    d, fid = args
    rgb = np.asarray(Image.open(f"{d}/{fid}_rgb.jpg").convert("RGB"))
    th = np.asarray(Image.open(f"{d}/{fid}_t.png").convert("L"))
    li = np.asarray(Image.open(f"{d}/{fid}_l.png").convert("RGB"))
    return np.concatenate([rgb.transpose(2, 0, 1), th[None], li.transpose(2, 0, 1)], 0)


def build_cache(fused_dir, split, out, workers=16, limit=0):
    frames = load_frames(fused_dir, {split})
    if limit:
        import random
        frames = sorted(random.Random(0).sample(frames, min(limit, len(frames))), key=lambda f: f["id"])
    path = f"{out}/{split}.npy"
    if not os.path.exists(path):
        os.makedirs(out, exist_ok=True)
        arr = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.uint8, shape=(len(frames), 7, FH, FW))
        with ThreadPoolExecutor(workers) as ex:
            for i, x in enumerate(ex.map(_load7, [(fused_dir, f["id"]) for f in frames])):
                arr[i] = x
        arr.flush(); del arr
        os.replace(path + ".tmp", path)
        json.dump([f["id"] for f in frames], open(f"{out}/{split}_ids.json", "w"))
    return frames


def targets_tensor(frames, label_map=None):
    """(N,MAXB,6): x1,y1,x2,y2,class,distance ; (N,) counts."""
    t = torch.zeros(len(frames), MAXB, 6)
    n = torch.zeros(len(frames), dtype=torch.long)
    for i, f in enumerate(frames):
        ps = label_map.get(f["id"], []) if label_map is not None else [
            {"box": p["box"], "pose": POSE3[p["pose"]], "distance": p["range_m"]} for p in f["persons"]]
        ps = [p for p in ps if p.get("pose") in CLASSES and p.get("distance") is not None][:MAXB]
        for k, p in enumerate(ps):
            t[i, k] = torch.tensor([*p["box"], CLASSES.index(p["pose"]), p["distance"]])
        n[i] = len(ps)
    return t, n


# ---------------------------------------------------------------- model
class CenterNetTriSAR(CenterNetRGBT):
    def __init__(self, in_chans, backbone="resnet18.a1_in1k", width=128, pretrained=True):
        super().__init__(backbone, in_chans, width, pretrained)
        self.hm = _head(width, len(CLASSES), bias=-math.log((1 - 0.01) / 0.01))
        self.dist = _head(width, 1)

    def forward(self, x):
        feats = self.body(x)
        y = self.lat[-1](feats[-1])
        for f, l in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + l(f)
        y = self.smooth(y)
        return self.hm(y), self.wh(y), self.off(y), self.dist(y)


@torch.no_grad()
def decode(hm, wh, off, dist, k=40):
    B, C, Hh, Ww = hm.shape
    heat = hm.sigmoid()
    heat = heat * (F.max_pool2d(heat, 3, 1, 1) == heat).to(heat.dtype)
    scores, idx = heat.reshape(B, -1).topk(k)
    cls = idx // (Hh * Ww)
    pix = idx % (Hh * Ww)
    ys, xs = (pix // Ww).float(), (pix % Ww).float()
    g = lambda t: t.reshape(B, t.shape[1], -1).gather(2, pix[:, None].expand(-1, t.shape[1], -1))
    whv, ofv, dv = g(wh), g(off), g(dist)
    cx, cy = xs + ofv[:, 0], ys + ofv[:, 1]
    w, h = whv[:, 0].clamp(min=0), whv[:, 1].clamp(min=0)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * STRIDE
    return boxes, scores, cls, dv[:, 0].exp()


def make_targets(t, valid, Hh, Ww, alpha=0.54):
    B = t.shape[0]
    dev = t.device
    bi, mi = valid.nonzero(as_tuple=True)
    bb = t[bi, mi]
    b = bb[:, :4] / STRIDE
    cls = bb[:, 4].long()
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w, h = (b[:, 2] - b[:, 0]).clamp(min=1e-3), (b[:, 3] - b[:, 1]).clamp(min=1e-3)
    ix, iy = cx.long().clamp(0, Ww - 1), cy.long().clamp(0, Hh - 1)
    sx, sy = (alpha * w / 6).clamp(min=0.5), (alpha * h / 6).clamp(min=0.5)
    xs = torch.arange(Ww, device=dev).float(); ys = torch.arange(Hh, device=dev).float()
    gx = torch.exp(-((xs[None] - ix[:, None].float()) ** 2) / (2 * sx[:, None] ** 2))
    gy = torch.exp(-((ys[None] - iy[:, None].float()) ** 2) / (2 * sy[:, None] ** 2))
    g = (gy[:, :, None] * gx[:, None, :]).view(len(bi), -1)
    hm = torch.zeros(B * len(CLASSES), Hh * Ww, device=dev)
    if len(bi):
        hm.scatter_reduce_(0, (bi * len(CLASSES) + cls)[:, None].expand(-1, Hh * Ww), g, reduce="amax")
    reg = {"b": bi, "pos": iy * Ww + ix, "wh": torch.stack([w, h], 1), "off": torch.stack([cx - ix, cy - iy], 1),
           "logd": bb[:, 5].clamp(min=0.3).log()}
    return hm.view(B, len(CLASSES), Hh, Ww), reg


def loss_fn(out, hm_t, reg):
    hm, wh, off, dist = (o.float() for o in out)
    l_hm = focal_loss(hm, hm_t)
    if len(reg["b"]) == 0:
        return l_hm, {"hm": l_hm.item()}
    B = hm.shape[0]
    pick = lambda t: t.reshape(B, t.shape[1], -1)[reg["b"], :, reg["pos"]]
    l_wh = F.l1_loss(pick(wh), reg["wh"])
    l_off = F.l1_loss(pick(off), reg["off"])
    l_d = F.smooth_l1_loss(pick(dist)[:, 0], reg["logd"], beta=0.1)
    return l_hm + 0.1 * l_wh + l_off + l_d, {"hm": l_hm.item(), "wh": l_wh.item(), "off": l_off.item(), "logd": l_d.item()}


# ---------------------------------------------------------------- aug / io
def prep(x, idx_ch, cond="normal"):
    xf = x.float().div(255.0)
    if cond == "rgb_off":
        xf[:, :3] = 0
    elif cond == "thermal_off":
        xf[:, 3:4] = 0
    elif cond == "lidar_off":
        xf[:, 4:] = 0
    xf = (xf - MEAN.to(xf.device)) / STD.to(xf.device)
    return xf[:, idx_ch]


def augment(x, t, n, gen, idx_ch, p_drop=(0.15, 0.1, 0.1)):
    """h-flip + mild zoom on GPU; sensor dropout. Boxes in px of 640x512."""
    B = x.shape[0]
    dev = x.device
    xf = x.float().div_(255.0)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    xf[flip] = xf[flip].flip(-1)
    t = t.clone()
    x1 = t[..., 0].clone()
    t[flip, :, 0] = FW - t[flip, :, 2]
    t[flip, :, 2] = FW - x1[flip]
    g = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.5, generator=gen)
    xf[:, :3] = xf[:, :3].clamp(0, 1).pow(g)
    r = torch.rand(B, device=dev, generator=gen)
    xf[r < p_drop[0], :3] = 0
    xf[(r >= p_drop[0]) & (r < p_drop[0] + p_drop[1]), 3:4] = 0
    xf[(r >= p_drop[0] + p_drop[1]) & (r < sum(p_drop)), 4:] = 0
    xf = (xf - MEAN.to(dev)) / STD.to(dev)
    valid = torch.arange(t.shape[1], device=dev)[None] < n[:, None]
    return xf[:, idx_ch], t, valid


@torch.no_grad()
def predict(net, x, idx_ch, cond="normal", bs=64, thr=0.05):
    net.eval()
    out = []
    dev = next(net.parameters()).device
    for i in range(0, len(x), bs):
        xb = prep(x[i:i + bs].to(dev), idx_ch, cond).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(xb)
        b, s, c, d = decode(*(t.float() for t in o))
        for bb, ss, cc, dd in zip(b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy(), d.cpu().numpy()):
            k = ss >= thr
            out.append([{"box": bx.tolist(), "score": float(sc), "pose": CLASSES[int(cl)], "distance": float(di)}
                        for bx, sc, cl, di in zip(bb[k], ss[k], cc[k], dd[k])])
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--modal", choices=list(CH), default="rgbtl")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--cache", default="data/cache_trisar")
    ap.add_argument("--labels", default=None, help="slow-head report jsonl (train split) to distil from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: frames per split (use a separate --cache)")
    a = ap.parse_args()
    out_dir = f"runs/{a.name}"
    seed_all(a.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    idx_ch = CH[a.modal]
    from nightwatch.eval.metrics_trisar import score_report

    tr_frames = build_cache(a.fused_dir, "train", a.cache, limit=a.limit)
    va_frames = build_cache(a.fused_dir, "val", a.cache, limit=a.limit)
    te_frames = build_cache(a.fused_dir, "test", a.cache, limit=a.limit)
    label_map = None
    if a.labels:
        label_map = {}
        for line in open(a.labels):
            d = json.loads(line)
            if d["task"] == "report" and d["cond"] == "normal":
                label_map[d["id"]] = d["persons"]
    X = torch.from_numpy(np.load(f"{a.cache}/train.npy", mmap_mode="r")[:])
    try:
        X = X.to(dev)
    except torch.OutOfMemoryError:
        X = X.pin_memory()
    TT, NN = targets_tensor(tr_frames, label_map)
    TT, NN = TT.to(dev), NN.to(dev)
    Xv = torch.from_numpy(np.load(f"{a.cache}/val.npy", mmap_mode="r")[:])
    fv = {f["id"]: f for f in va_frames}

    net = CenterNetTriSAR(len(idx_ch)).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    iters = a.epochs * (len(X) // a.bs)
    warm = min(300, iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * i / iters)))
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    log = RunLogger(out_dir, vars(a))
    print(f"{a.name}: {len(X)} frames, {iters} iters, modal={a.modal}, distilled={bool(a.labels)}", flush=True)
    best, it, t0 = -1.0, 0, time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), device=dev, generator=gen)
        for s in range(0, len(X) - a.bs + 1, a.bs):
            idx = perm[s:s + a.bs]
            xb, tb, valid = augment(X[idx.to(X.device)].to(dev, non_blocking=True), TT[idx], NN[idx], gen, idx_ch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xb.contiguous(memory_format=torch.channels_last))
            hm_t, reg = make_targets(tb, valid, out[0].shape[2], out[0].shape[3])
            loss, parts = loss_fn(out, hm_t, reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step(); sched.step(); it += 1
            if it % 50 == 0:
                log.log({"step": it, "epoch": ep, "loss": loss.item(), **parts, "img_per_s": it * a.bs / (time.time() - t0)})
        if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
            pv = dict(zip([f["id"] for f in va_frames], predict(net, Xv, idx_ch)))
            m = score_report({k: {"persons": v} for k, v in pv.items()}, fv, list(fv), score_thr=0.3)
            log.log({"step": it, "epoch": ep, "val_f1": m["f1"], "val_pose": m["pose_acc"], "val_dist_mae": m["dist_mae_m"]})
            if m["f1"] > best:
                best = m["f1"]
                torch.save(net.state_dict(), f"{out_dir}/best.pt")

    net.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=dev))
    pv = dict(zip([f["id"] for f in va_frames], predict(net, Xv, idx_ch)))
    thr, bf = 0.3, -1
    for t_ in np.arange(0.1, 0.9, 0.05):
        f1 = score_report({k: {"persons": v} for k, v in pv.items()}, fv, list(fv), score_thr=float(t_))["f1"]
        if f1 > bf:
            bf, thr = f1, float(t_)
    del X
    Xt = torch.from_numpy(np.load(f"{a.cache}/test.npy", mmap_mode="r")[:])
    test_ids = [f["id"] for f in te_frames]
    with open(f"{out_dir}/pred_test.jsonl", "w") as fh:
        for cond in ["normal", "rgb_off", "thermal_off", "lidar_off"]:
            for iid, ps in zip(test_ids, predict(net, Xt, idx_ch, cond)):
                fh.write(json.dumps({"id": iid, "task": "report", "cond": cond, "persons": ps}) + "\n")
    xb = prep(Xt[:1].to(dev), idx_ch).contiguous(memory_format=torch.channels_last)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(20):
            net(xb)
        torch.cuda.synchronize(); t1 = time.time()
        for _ in range(200):
            net(xb)
        torch.cuda.synchronize()
    summ = {"name": a.name, "modal": a.modal, "labels": a.labels, "val_f1": best, "score_thr": thr,
            "torch_latency_ms_bs1": (time.time() - t1) / 200 * 1000,
            "params_M": sum(p.numel() for p in net.parameters()) / 1e6, "train_minutes": (time.time() - t0) / 60}
    json.dump(summ, open(f"{out_dir}/train_summary.json", "w"), indent=1)
    log.log({"step": it, **{k: v for k, v in summ.items() if isinstance(v, (int, float))}})
    log.close()


if __name__ == "__main__":
    main()
