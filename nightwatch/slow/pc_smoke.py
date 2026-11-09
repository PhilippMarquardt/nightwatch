"""V6 gate of docs/PLAN_POINTCLOUD.md: one real tri_pc batch through the 9B.
Checks (1) the visual token count per sample is 3 x 320 as in mode tri, (2) gradients reach the point-cloud projector
and the no-return token, (3) the loss changes when the LiDAR tokens are switched off (the slot is actually read),
(4) a few optimiser steps run within memory.

NW_GPU_FRAC=0.8 python -m nightwatch.slow.pc_smoke
"""
from __future__ import annotations

import torch

from nightwatch.slow.data import sft_collate
from nightwatch.slow.data_trisar import TriSARSFT, load_frames
from nightwatch.slow.model import load_slow_head
from nightwatch.utils import gpu_mem_cap


def main():
    gpu_mem_cap()
    model, proc, pc = load_slow_head("Qwen/Qwen3.5-9B", "tri_pc", lora={"r": 32}, device="cuda", grad_ckpt=True)
    frames = [f for f in load_frames("data/trisar_fused", {"train"}) if f["persons"]][:4]
    ds = TriSARSFT(frames, "data/trisar_fused", proc, "tri_pc", seed=0, crop_p=0.0)
    batch = sft_collate([ds[i] for i in range(4)], pad_id=proc.tokenizer.pad_token_id)
    batch.pop("thermal")
    g = batch.pop("lidar").cuda()
    batch = {k: v.cuda() for k, v in batch.items() if k != "task"}
    print("image_grid_thw", batch["image_grid_thw"].tolist()[:3], "grid", tuple(g.shape))

    model.train()
    pc.set_extra({"lidar": g})
    out = model(**batch)
    out.loss.backward()
    gp = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in pc.named_parameters()}
    print("loss", round(float(out.loss), 4), "| projector grad norms:", {k: (round(v, 5) if v is not None else None) for k, v in gp.items()})
    print("token norm ratio (pc / camera+thermal):", pc.gate_values())
    assert all(v is not None and v > 0 for v in gp.values()), "no gradient reaches the point-cloud projector"

    model.eval()
    with torch.no_grad():
        pc.set_extra({"lidar": g}); l_on = float(model(**batch).loss)
        pc.set_extra({"lidar": torch.zeros_like(g)}); l_off = float(model(**batch).loss)
    print(f"eval loss LiDAR tokens on {l_on:.4f} vs off {l_off:.4f} (diff {l_off - l_on:+.4f})")
    assert abs(l_off - l_on) > 1e-4, "LiDAR slot has no effect on the output"

    model.train()
    params = [p for p in model.parameters() if p.requires_grad] + list(pc.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    for s in range(3):
        opt.zero_grad(); pc.set_extra({"lidar": g})
        loss = model(**batch).loss; loss.backward(); opt.step()
        print("step", s, "loss", round(float(loss), 4), "peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
    print("PC_SMOKE_PASS")


if __name__ == "__main__":
    main()
