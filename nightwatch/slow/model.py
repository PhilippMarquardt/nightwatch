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

import math
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

    def __init__(self, name: str = "convnext_tiny.dinov3_lvd1689m", pretrained: bool = True, in_chans: int = 1):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, features_only=True, out_indices=(2, 3))
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
        # ControlNet-style zero init: output is exactly 0 at step 0 (model == pretrained VLM), but unlike a
        # zero *gate* the last layer still gets full-strength gradients, so fusion starts learning at once.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gate = nn.Parameter(torch.ones(1))   # learnable global scale, logged

    def forward(self, t):
        return self.gate * self.mlp(self.norm(t))


class SensorBranch(nn.Module):
    """One extra pixel-aligned sensor (e.g. LiDAR range/intensity/hit maps) -> additive tokens."""

    def __init__(self, d_model, n_deepstack, encoder, in_chans, pretrained=True):
        super().__init__()
        self.encoder = ThermalEncoder(encoder, pretrained, in_chans)
        c = self.encoder.out_dim
        self.main = GatedProj(c, d_model, hidden=min(2048, d_model))
        self.deep = nn.ModuleList([GatedProj(c, d_model, hidden=512) for _ in range(n_deepstack)])
        self.sensor_emb = nn.Parameter(torch.zeros(c))


class ThermalFusion(nn.Module):
    """Thermal (+ optional extra sensors) fused onto the VLM's visual token grid.

    extra_sensors={"lidar": 3} adds one branch per sensor; all branches are summed onto the RGB tokens.
    Thermal keeps its original parameter names, so LLVIP checkpoints still load.
    """

    def __init__(self, d_model: int, n_deepstack: int, encoder: str, pretrained=True, extra_sensors: dict | None = None):
        super().__init__()
        self.encoder = ThermalEncoder(encoder, pretrained)
        c = self.encoder.out_dim
        self.main = GatedProj(c, d_model, hidden=min(2048, d_model))
        self.deep = nn.ModuleList([GatedProj(c, d_model, hidden=512) for _ in range(n_deepstack)])
        self.sensor_emb = nn.Parameter(torch.zeros(c))   # thermal sensor-type embedding
        self.extra = nn.ModuleDict({k: SensorBranch(d_model, n_deepstack, encoder, ch, pretrained)
                                    for k, ch in (extra_sensors or {}).items()})
        self._thermal = None
        self._extra = {}

    def set_thermal(self, t):
        self._thermal = t

    def set_extra(self, tensors: dict):
        self._extra = tensors or {}

    def _branches(self):
        yield "thermal", self.encoder, self.main, self.deep, self.sensor_emb, self._thermal
        for k, br in self.extra.items():
            yield k, br.encoder, br.main, br.deep, br.sensor_emb, self._extra.get(k)

    def hook(self, module, args, output):
        po = output.pooler_output
        deep = list(output.deepstack_features) if getattr(output, "deepstack_features", None) else None
        add_main = torch.zeros_like(po)
        add_deep = [torch.zeros_like(d) for d in deep] if deep else None
        self._ratio = {}
        rgb_norm = po.float().norm(dim=-1).mean().clamp(min=1e-6)
        for name, enc, main, deepp, emb, x in self._branches():
            if x is None:
                continue
            feats = enc(x.to(next(enc.parameters()).dtype)) + emb
            feats = feats.reshape(-1, feats.shape[-1])                  # (B*h*w, C)
            if feats.shape[0] != po.shape[0]:
                raise RuntimeError(f"{name} tokens {feats.shape[0]} != visual tokens {po.shape[0]} "
                                   f"(fusion mode needs exactly one {IMG_W}x{IMG_H} RGB image per sample)")
            a = main(feats).to(po.dtype)
            add_main = add_main + a
            self._ratio["main" if name == "thermal" else name] = float(a.float().norm(dim=-1).mean() / rgb_norm)
            if deep:
                for i, p in enumerate(deepp):
                    add_deep[i] = add_deep[i] + p(feats).to(deep[i].dtype)
        if not self._ratio:
            return output
        output.pooler_output = po + add_main
        if deep:
            for i, (d, a) in enumerate(zip(deep, add_deep)):
                self._ratio[f"deep{i}"] = float(a.float().norm(dim=-1).mean() / d.float().norm(dim=-1).mean().clamp(min=1e-6))
            output.deepstack_features = [d + a for d, a in zip(deep, add_deep)]
        return output

    def gate_values(self):
        """Thermal contribution as a fraction of the RGB token norm (last forward pass)."""
        return dict(getattr(self, "_ratio", {}))


class PCTokens(nn.Module):
    """Mode tri_pc: the LiDAR image's 320 merged visual tokens are replaced by projected frozen Utonia features.
    Cells without a LiDAR return (and LiDAR switched off) get a learned "no return" token. The LiDAR slot's deepstack
    features are zeroed, so LiDAR reaches the LM only through these tokens. Token layout and M-RoPE positions are
    unchanged from mode `tri` (the black range image stays as a placeholder).

    tri_pc3 additions (all off by default, so tri_pc / tri_pc2 checkpoints load and behave as before):
      d_geo > 0   the grid carries per-cell [min range, mean range, mean height] after the 1225 feature channels;
                  they enter through their own small MLP (Fourier features of range), added to the projected token
      pos         a learned per-cell embedding in image coordinates (zero-init), like the ViT's own position signal
      aux         a linear head on the projected token: person-in-cell logit and person range. Its loss keeps a
                  gradient on the projector that does not depend on the LM choosing to read LiDAR."""

    D_FEAT = 1225
    GEO_WAVELEN = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)

    def __init__(self, d_model: int, d_in: int = 1225, hidden: int = 1024, n_img: int = 3, slot: int = 2, n_tok: int = 320,
                 d_geo: int | None = None):
        super().__init__()
        if d_geo is None:
            d_geo = 3 if os.environ.get("NW_PC_GEO", "0") != "0" else 0
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.no_return = nn.Parameter(torch.randn(d_model) * 0.02)
        self.n_img, self.slot, self.n_tok, self.d_geo = n_img, slot, n_tok, d_geo
        if d_geo:
            self.geo = nn.Sequential(nn.Linear(3 + 2 * len(self.GEO_WAVELEN), 256), nn.GELU(), nn.Linear(256, d_model))
            self.pos = nn.Parameter(torch.zeros(n_tok, d_model))
            self.aux = nn.Linear(d_model, 2)
        # tri_pc2: rescale LiDAR tokens to the per-sample RMS of the camera/thermal tokens (saved in fusion.pt)
        self.register_buffer("norm_match", torch.tensor(float(os.environ.get("NW_PC_NORMMATCH", "0"))))
        self.extra = {"lidar": None}           # train/infer pass the token grid via set_extra({"lidar": ...})
        self._g, self._ratio, self._aux = None, {}, None

    def set_thermal(self, t):
        pass

    def set_extra(self, tensors: dict):
        self._g = tensors.get("lidar")

    def _geo_feats(self, geo):
        r = geo[..., 0:1]
        w = torch.tensor(self.GEO_WAVELEN, device=geo.device, dtype=geo.dtype)
        ph = 2 * math.pi * r / w
        return torch.cat([r / 60.0, geo[..., 1:2] / 60.0, geo[..., 2:3] / 5.0, ph.sin(), ph.cos()], -1)

    def tokens(self, g, target_rms=None):
        """g (B, 320, 1225 [+3]) -> (tokens (B, 320, D) float32-or-param dtype, has_return (B, 320) bool).
        target_rms (B, 1, 1): per-sample RMS the tokens are rescaled to when norm_match is on."""
        dt = next(self.proj.parameters()).dtype
        g = g.to(dt)
        has = g[..., self.D_FEAT - 1] > 0                                           # log1p(count) > 0
        tok = self.proj(g[..., :self.D_FEAT])
        if self.d_geo:
            tok = tok + self.geo(self._geo_feats(g[..., self.D_FEAT:self.D_FEAT + 3]))
        tok = torch.where(has.unsqueeze(-1), tok, self.no_return.to(tok.dtype).expand_as(tok))
        if self.d_geo:
            tok = tok + self.pos.to(tok.dtype)
        unit = tok / tok.float().pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6).to(tok.dtype)
        if self.d_geo:
            self._aux = (self.aux(unit), has)                                       # (B, 320, 2), (B, 320)
        if float(self.norm_match) > 0 and target_rms is not None:
            tok = unit * target_rms.to(tok.dtype)
        return tok, has

    def aux_loss(self, cell_y, cell_r):
        """Person-in-cell BCE (cells with a LiDAR return) + smooth-L1 on person range / 10 m (person cells with a return)."""
        out, has = self._aux
        out = out.float()
        y, r = cell_y.to(out.device).float(), cell_r.to(out.device).float()
        if not has.any():
            return out.sum() * 0.0, {}
        pw = torch.tensor(8.0, device=out.device)
        bce = F.binary_cross_entropy_with_logits(out[..., 0][has], y[has], pos_weight=pw)
        m = has & (y > 0.5)
        reg = F.smooth_l1_loss(out[..., 1][m], r[m] / 10.0) if m.any() else out.sum() * 0.0
        return bce + reg, {"aux_bce": float(bce), "aux_range": float(reg)}

    def hook(self, module, args, output):
        g = self._g
        if g is None:
            return output
        po = output.pooler_output
        B = g.shape[0]
        per = self.n_img * self.n_tok
        if po.shape[0] != B * per:
            raise RuntimeError(f"tri_pc expects {B}x{per} visual tokens, got {po.shape[0]}")
        po4 = po.view(B, self.n_img, self.n_tok, -1).clone()
        target = po4[:, :self.slot].detach().float().pow(2).mean(-1).sqrt().mean(dim=(1, 2)).view(B, 1, 1)
        tok, _ = self.tokens(g, target)
        tok = tok.to(po.dtype)
        self._ratio = {"pc_norm_ratio": float(tok.detach().float().norm(dim=-1).mean() / po4[:, :self.slot].detach().float().norm(dim=-1).mean().clamp(min=1e-6))}
        po4[:, self.slot] = tok
        output.pooler_output = po4.view_as(po)
        deep = getattr(output, "deepstack_features", None)
        if deep:
            new = []
            for d in deep:
                d4 = d.view(B, self.n_img, self.n_tok, -1).clone()
                d4[:, self.slot] = 0
                new.append(d4.view_as(d))
            output.deepstack_features = new
        return output

    def gate_values(self):
        return dict(self._ratio)


LIDAR_MEAN = torch.tensor([0.10, 0.20, 0.45]).view(3, 1, 1)
LIDAR_STD = torch.tensor([0.15, 0.25, 0.50]).view(3, 1, 1)


def lidar_tensor(img_rgb):
    """PIL RGB lidar map (inverse range, intensity, hit mask), already resized -> (3,H,W)."""
    import numpy as np
    a = torch.from_numpy(np.asarray(img_rgb, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    return (a - LIDAR_MEAN) / LIDAR_STD


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
    if proc.tokenizer.pad_token_id is None:          # Qwen3.5 ships without a pad token
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation=attn)
    model.to(device)
    model.requires_grad_(False)
    fusion = None
    if mode in ("fusion", "fusion_l"):
        tc = model.config.text_config
        n_deep = len(getattr(model.config.vision_config, "deepstack_visual_indexes", []) or [])
        extra = {"lidar": 3} if mode == "fusion_l" else None
        fusion = ThermalFusion(tc.hidden_size, n_deep, thermal_encoder, pretrained=adapter_dir is None,
                               extra_sensors=extra).to(device)
        if adapter_dir:
            fusion.load_state_dict(torch.load(os.path.join(adapter_dir, "fusion.pt"), map_location=device))
        model.model.visual.register_forward_hook(fusion.hook)
    if mode == "tri_pc":
        sd = torch.load(os.path.join(adapter_dir, "fusion.pt"), map_location=device) if adapter_dir else None
        # a checkpoint decides its own layout (tri_pc3 has the geo / pos / aux parts); a new model follows NW_PC_GEO
        fusion = PCTokens(model.config.text_config.hidden_size,
                          d_geo=(3 if "pos" in sd else 0) if sd is not None else None).to(device)
        if sd is not None:
            sd.setdefault("norm_match", torch.tensor(0.0, device=device))   # tri_pc v1 checkpoints predate the buffer
            fusion.load_state_dict(sd)
        model.model.visual.register_forward_hook(fusion.hook)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()          # inference: fold LoRA into the weights
    elif lora:
        from peft import LoraConfig, get_peft_model
        target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
        if os.environ.get("NW_VISION_LORA"):     # 2026-11-09: also adapt the ViT (27 blocks + patch merger)
            target = (r"(" + target + r")|(.*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                      r"|merger\.linear_fc[12]))")
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora.get("alpha", 2 * lora["r"]), lora_dropout=lora.get("dropout", 0.05),
                         target_modules=target)
        model = get_peft_model(model, cfg)
    return model, proc, fusion
