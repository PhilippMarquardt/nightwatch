"""tri_pc3 stage 0: pre-align the point-cloud projector to the VLM's own visual token space, without the LM.

The LM demonstrably reads the thermal image's ViT tokens for person localisation (thermal-off costs 0.05-0.2 F1).
tri_pc / tri_pc2 showed that a from-scratch projector trained only through the LM loss never becomes readable: its
gradient collapses ~50x within 100 steps because camera + thermal already solve the task.

Here the projector is trained so that the token it produces for a cell matches the frozen ViT's token for the same
cell of the *thermal* image (cosine loss, cells with a LiDAR return), plus the per-cell aux loss (person-in-cell,
person range). The LM is never run: a capture hook stops the forward pass right after the vision tower. Result:
LiDAR tokens that start in the token dialect the LM already reads, at the right cells.

python -m nightwatch.slow.pc_align --out runs/pc_align_s0 --seed 0
Diagnostics (val split, never trained on): cosine to the true thermal token vs to another frame's token at the same
cell (is the alignment frame-specific?), on person cells and on background cells; aux-head person AP.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

os.environ["NW_MULTI_DROP"] = "0"          # stage 0 needs all three real images and the real cloud
os.environ["NW_LIDAR_ONLY_P"] = "0"
os.environ.setdefault("NW_PC_GEO", "1")
os.environ.setdefault("NW_PC_NORMMATCH", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nightwatch.slow.data import sft_collate
from nightwatch.slow.data_trisar import TriSARSFT, load_frames
from nightwatch.slow.model import load_slow_head
from nightwatch.utils import gpu_mem_cap, seed_all


class _Stop(Exception):
    pass


class Capture:
    """Forward hook on the vision tower: keep its merged tokens and abort the forward pass (the LM is not needed)."""

    def __init__(self):
        self.po = None

    def __call__(self, module, args, output):
        self.po = output.pooler_output
        raise _Stop


@torch.no_grad()
def vit_tokens(model, cap, batch, dev):
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(**{k: v.to(dev, non_blocking=True) for k, v in batch.items()})
    except _Stop:
        pass
    return cap.po.float()


def step_loss(fusion, po, g, cell_y, cell_r, aux_w):
    B = g.shape[0]
    po4 = po.view(B, 3, 320, -1)
    tgt = po4[:, 1]                                                     # thermal image tokens
    rms = po4[:, :2].pow(2).mean(-1).sqrt().mean(dim=(1, 2)).view(B, 1, 1)
    tok, has = fusion.tokens(g, rms)
    cos = F.cosine_similarity(tok.float(), tgt, dim=-1)                 # (B, 320)
    l_cos = (1 - cos)[has].mean()
    aux, rec = fusion.aux_loss(cell_y, cell_r)
    return l_cos + aux_w * aux, cos, has, {"cos_loss": float(l_cos), **rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()

    gpu_mem_cap()
    dev = torch.device("cuda")
    seed_all(a.seed)
    model, proc, fusion = load_slow_head(a.model, "tri_pc", lora=None, device=dev)
    model.eval()
    fusion.set_extra({})                    # the PCTokens hook passes the ViT output through untouched
    cap = Capture()
    model.model.visual.register_forward_hook(cap)
    fusion.train()

    def loader(split, shuffle, crop_p):
        ds = TriSARSFT(load_frames(a.fused_dir, {split}), a.fused_dir, proc, "tri_pc", seed=a.seed, crop_p=crop_p)
        return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, num_workers=a.workers, drop_last=shuffle,
                          collate_fn=partial(sft_collate, pad_id=proc.tokenizer.pad_token_id),
                          persistent_workers=False, prefetch_factor=4)

    tr, va = loader("train", True, 0.25), loader("val", False, 0.0)
    total = a.max_steps or int(len(tr) * a.epochs)
    opt = torch.optim.AdamW(fusion.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    print(f"pc_align: train batches {len(tr)}, steps {total}, params {sum(p.numel() for p in fusion.parameters()) / 1e6:.1f}M",
          flush=True)

    def pop(batch):
        return (batch.pop("lidar").to(dev), batch.pop("cell_y").to(dev), batch.pop("cell_r").to(dev),
                {k: v for k, v in batch.items() if k not in ("thermal", "labels")})

    step, t0, ep = 0, time.time(), 0
    while step < total:
        tr.dataset.epoch = ep
        for batch in tr:
            g, cy, cr, mb = pop(batch)
            po = vit_tokens(model, cap, mb, dev)
            for grp in opt.param_groups:
                grp["lr"] = a.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / total))
            loss, _, _, rec = step_loss(fusion, po, g, cy, cr, a.aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "elapsed_s": time.time() - t0, **rec}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(" ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if step >= total:
                break
        ep += 1

    # ---- validation diagnostics (lab val sequences; no training on them)
    fusion.eval()
    acc = {k: [0.0, 0] for k in ("cos_person", "cos_bg", "cos_person_shuf", "cos_bg_shuf")}
    scores, labels = [], []
    with torch.no_grad():
        for batch in va:
            g, cy, cr, mb = pop(batch)
            po = vit_tokens(model, cap, mb, dev)
            B = g.shape[0]
            _, cos, has, _ = step_loss(fusion, po, g, cy, cr, a.aux)
            tgt = po.view(B, 3, 320, -1)[:, 1]
            tok, _ = fusion.tokens(g, None)
            cos_sh = F.cosine_similarity(tok.float(), tgt.roll(1, 0), dim=-1)     # another frame, same cell
            person = cy > 0.5
            for name, c, m in (("cos_person", cos, has & person), ("cos_bg", cos, has & ~person),
                               ("cos_person_shuf", cos_sh, has & person), ("cos_bg_shuf", cos_sh, has & ~person)):
                acc[name][0] += float(c[m].sum()); acc[name][1] += int(m.sum())
            out, _ = fusion._aux
            scores.append(out[..., 0][has].float().cpu()); labels.append(cy[has].cpu())
    from sklearn.metrics import average_precision_score
    s, y = torch.cat(scores).numpy(), torch.cat(labels).numpy()
    res = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
    res.update({"aux_person_ap_cells_with_return": float(average_precision_score(y, s)), "pos_rate": float(y.mean()),
                "n_cells": int(len(y)), "steps": step, "minutes": (time.time() - t0) / 60})
    print("PC_ALIGN_VAL", json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.out, "align_val.json"), "w"), indent=1)
    torch.save(fusion.state_dict(), os.path.join(a.out, "fusion.pt"))
    json.dump({"config": vars(a), "mode": "tri_pc", "stage": "pc_align"}, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
    print("PC_ALIGN_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
