"""Fast head: anchor-free CenterNet with an early-fusion RGB+T backbone.

No NMS (3x3 max-pool peak picking), so the whole thing -- decode included -- exports
to a single ONNX graph / TensorRT engine.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4


def _head(c, out, bias=0.0):
    h = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, out, 1))
    nn.init.constant_(h[-1].bias, bias)
    return h


class CenterNetRGBT(nn.Module):
    def __init__(self, backbone="resnet18.a1_in1k", in_chans=4, width=128, pretrained=True):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=in_chans,
                                      out_indices=(1, 2, 3, 4))
        chs = self.body.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chs])
        self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.hm = _head(width, 1, bias=-math.log((1 - 0.01) / 0.01))
        self.wh = _head(width, 2)
        self.off = _head(width, 2)

    def forward(self, x):
        feats = self.body(x)
        y = self.lat[-1](feats[-1])
        for f, l in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + l(f)
        y = self.smooth(y)
        return self.hm(y), self.wh(y), self.off(y)


@torch.no_grad()
def decode(hm, wh, off, k=50, scale=1.0):
    """-> boxes (B,k,4) in input px * scale, scores (B,k)."""
    B, _, Hh, Ww = hm.shape
    heat = hm.sigmoid()
    peak = F.max_pool2d(heat, 3, 1, 1)
    heat = heat * (peak == heat).to(heat.dtype)
    scores, idx = heat.view(B, -1).topk(k)
    ys, xs = (idx // Ww).float(), (idx % Ww).float()
    whv = wh.view(B, 2, -1).gather(2, idx[:, None].expand(-1, 2, -1))
    ofv = off.view(B, 2, -1).gather(2, idx[:, None].expand(-1, 2, -1))
    cx, cy = xs + ofv[:, 0], ys + ofv[:, 1]
    w, h = whv[:, 0].clamp(min=0), whv[:, 1].clamp(min=0)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * STRIDE * scale
    return boxes, scores


class Exportable(nn.Module):
    def __init__(self, net, k=50):
        super().__init__()
        self.net, self.k = net, k

    def forward(self, x):
        return decode(*self.net(x), k=self.k)


# ---------------------------------------------------------------- targets + loss
def make_targets(boxes, valid, Hh, Ww, alpha=0.54):
    """boxes (B,M,4) input px, valid (B,M) -> heatmap (B,1,Hh,Ww), reg targets at centre pixels.
    Fully vectorised elliptical Gaussians (TTFNet-style sigma = alpha * size / 6)."""
    B, M, _ = boxes.shape
    dev = boxes.device
    b = boxes / STRIDE
    bi, mi = valid.nonzero(as_tuple=True)
    bb = b[bi, mi]                                              # (N,4)
    cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
    w, h = (bb[:, 2] - bb[:, 0]).clamp(min=1e-3), (bb[:, 3] - bb[:, 1]).clamp(min=1e-3)
    ix, iy = cx.long().clamp(0, Ww - 1), cy.long().clamp(0, Hh - 1)
    sx, sy = (alpha * w / 6).clamp(min=0.5), (alpha * h / 6).clamp(min=0.5)
    xs = torch.arange(Ww, device=dev).float(); ys = torch.arange(Hh, device=dev).float()
    gx = torch.exp(-((xs[None] - ix[:, None].float()) ** 2) / (2 * sx[:, None] ** 2))
    gy = torch.exp(-((ys[None] - iy[:, None].float()) ** 2) / (2 * sy[:, None] ** 2))
    g = (gy[:, :, None] * gx[:, None, :]).view(len(bi), -1)    # (N, Hh*Ww)
    hm = torch.zeros(B, Hh * Ww, device=dev)
    if len(bi):
        hm.scatter_reduce_(0, bi[:, None].expand(-1, Hh * Ww), g, reduce="amax")
    pos = iy * Ww + ix
    reg = {"b": bi, "pos": pos, "wh": torch.stack([w, h], 1), "off": torch.stack([cx - ix, cy - iy], 1)}
    return hm.view(B, 1, Hh, Ww), reg


def focal_loss(logits, gt, a=2, b=4):
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pl = torch.log(p) * (1 - p) ** a * pos
    nl = torch.log(1 - p) * p ** a * (1 - gt) ** b * neg
    npos = pos.sum().clamp(min=1)
    return -(pl.sum() + nl.sum()) / npos


def centernet_loss(out, hm_t, reg, w_wh=0.1, w_off=1.0):
    hm, wh, off = (o.float() for o in out)
    l_hm = focal_loss(hm, hm_t)
    B, _, Hh, Ww = hm.shape
    if len(reg["b"]) == 0:
        return l_hm, {"hm": l_hm.item(), "wh": 0.0, "off": 0.0}
    p_wh = wh.view(B, 2, -1)[reg["b"], :, reg["pos"]]
    p_off = off.view(B, 2, -1)[reg["b"], :, reg["pos"]]
    l_wh = F.l1_loss(p_wh, reg["wh"])
    l_off = F.l1_loss(p_off, reg["off"])
    loss = l_hm + w_wh * l_wh + w_off * l_off
    return loss, {"hm": l_hm.item(), "wh": l_wh.item(), "off": l_off.item()}
