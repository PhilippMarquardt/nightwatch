"""Slow head: an open VLM (Qwen3-VL) that reasons over a fused RGB+thermal token stream.

Three ways to feed the thermal camera, selected with ``mode``:

    rgb     : visible camera only (the stock VLM).
    dual    : thermal rendered as a second image through the VLM's own RGB-trained ViT.
              Simple and strong, but doubles the visual token budget.
    fusion  : (ours) a native thermal encoder (DINOv3 ConvNeXt, 1-channel stem) whose
              stride-32 feature map lands exactly on the VLM's 2x2-merged visual token grid.
              Thermal features are added onto the RGB tokens -- and onto every DeepStack
              level -- through zero-initialised gates. One aligned token stream, same token
              budget as RGB alone, and at step 0 the model is exactly the pretrained VLM.

Fusion is implemented with a forward hook on the VLM's vision tower, so it does not
fork any transformers code and works for generation (the vision tower runs once at prefill).
"""
from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_W, IMG_H = 960, 768          # model input; multiple of 32 -> 30 x 24 merged tokens
GRID_W, GRID_H = IMG_W // 32, IMG_H // 32
THERMAL_MEAN, THERMAL_STD = 0.449, 0.226


class ThermalEncoder(nn.Module):
    """DINOv3 ConvNeXt with a 1-channel stem (timm sums the RGB stem weights)."""

    def __init__(self, name: str = "convnext_tiny.dinov3_lvd1689m", pretrained: bool = True):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=1, features_only=True, out_indices=(2, 3))
        chs = self.net.feature_info.channels()
        self.out_dim = sum(chs)

    def forward(self, x):                       # x: (B,1,H,W) normalised
        f16, f32 = self.net(x)                  # strides 16 and 32
        f16 = F.avg_pool2d(f16, 2)              # -> stride 32
        f = torch.cat([f16, f32], 1)            # (B, C, H/32, W/32)
        return f.flatten(2).transpose(1, 2)     # (B, h*w, C), row-major == Qwen merge order


class GatedProj(nn.Module):
    def __init__(self, cin, d, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.norm = nn.LayerNorm(cin)
        self.mlp = nn.Sequential(nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.gate = nn.Parameter(torch.zeros(1))  # tanh(0) = 0 -> identity at init

    def forward(self, t):
        return torch.tanh(self.gate) * self.mlp(self.norm(t))


class ThermalFusion(nn.Module):
    def __init__(self, d_model: int, n_deepstack: int, encoder: str, pretrained=True):
        super().__init__()
        self.encoder = ThermalEncoder(encoder, pretrained)
        c = self.encoder.out_dim
        self.main = GatedProj(c, d_model, hidden=min(2048, d_model))
        self.deep = nn.ModuleList([GatedProj(c, d_model, hidden=512) for _ in range(n_deepstack)])
        self.sensor_emb = nn.Parameter(torch.zeros(c))   # thermal sensor-type embedding
        self._thermal = None

    def set_thermal(self, t):
        self._thermal = t

    def hook(self, module, args, output):
        if self._thermal is None:
            return output
        t = self._thermal
        feats = self.encoder(t.to(next(self.encoder.parameters()).dtype)) + self.sensor_emb
        feats = feats.reshape(-1, feats.shape[-1])                  # (B*h*w, C)
        po = output.pooler_output
        if feats.shape[0] != po.shape[0]:
            raise RuntimeError(f"thermal tokens {feats.shape[0]} != visual tokens {po.shape[0]} "
                               f"(fusion mode needs exactly one {IMG_W}x{IMG_H} RGB image per sample)")
        output.pooler_output = po + self.main(feats).to(po.dtype)
        if output.deepstack_features is not None:
            output.deepstack_features = [d + p(feats).to(d.dtype) for d, p in zip(output.deepstack_features, self.deep)]
        return output

    def gate_values(self):
        return {"main": float(torch.tanh(self.main.gate)), **{f"deep{i}": float(torch.tanh(p.gate)) for i, p in enumerate(self.deep)}}


def thermal_tensor(img_l):
    """PIL 'L' image (already resized) -> (1,H,W) normalised float tensor."""
    import numpy as np
    a = torch.from_numpy(np.asarray(img_l, dtype=np.uint8).copy()).float().div_(255.0)
    return ((a - THERMAL_MEAN) / THERMAL_STD).unsqueeze(0)


def load_slow_head(model_path: str, mode: str, lora: dict | None = None, adapter_dir: str | None = None,
                   thermal_encoder: str = "convnext_tiny.dinov3_lvd1689m", device="cuda", attn="sdpa",
                   grad_ckpt=False):
    """Returns (model, processor, fusion_or_None)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation=attn)
    model.to(device)
    model.requires_grad_(False)
    fusion = None
    if mode == "fusion":
        tc = model.config.text_config
        n_deep = len(getattr(model.config.vision_config, "deepstack_visual_indexes", []) or [])
        fusion = ThermalFusion(tc.hidden_size, n_deep, thermal_encoder, pretrained=adapter_dir is None).to(device)
        if adapter_dir:
            fusion.load_state_dict(torch.load(os.path.join(adapter_dir, "fusion.pt"), map_location=device))
        model.model.visual.register_forward_hook(fusion.hook)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()          # inference: fold LoRA into the weights
    elif lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora.get("alpha", 2 * lora["r"]), lora_dropout=lora.get("dropout", 0.05),
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        model = get_peft_model(model, cfg)
    return model, proc, fusion
