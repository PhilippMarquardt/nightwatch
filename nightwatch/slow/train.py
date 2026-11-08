"""Slow-head SFT: LoRA on the VLM's language model (+ thermal encoder & fusion in fusion mode).

Single GPU:   python -m nightwatch.slow.train --mode fusion --out runs/slow_fusion
Multi GPU:    torchrun --nproc_per_node 8 -m nightwatch.slow.train --mode fusion ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from nightwatch.data.llvip import load_index
from nightwatch.slow.data import SFTDataset, sft_collate
from nightwatch.slow.model import load_slow_head
from nightwatch.utils import RunLogger, gpu_mem_cap, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mode", choices=["rgb", "dual", "fusion", "fusion_l"], required=True)
    ap.add_argument("--dataset", choices=["llvip", "trisar"], default="llvip")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--root", default="data/raw/LLVIP")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--splits", default="labeled")
    ap.add_argument("--labels", default=None, help="optional pseudo-label json {id: boxes}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--lr-fusion", type=float, default=2e-4)
    ap.add_argument("--lr-gate", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = 0
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    gpu_mem_cap()
    dev = torch.device("cuda", torch.cuda.current_device())
    seed_all(a.seed + rank)

    model, proc, fusion = load_slow_head(a.model, a.mode, lora={"r": a.lora_r}, device=dev, grad_ckpt=a.grad_ckpt)
    if fusion is not None:
        fusion.train()
    if a.dataset == "trisar":
        from nightwatch.slow.data_trisar import TriSARSFT, load_frames
        recs = load_frames(a.fused_dir, set(a.splits.split(",")))
        if a.max_samples:
            recs = random.Random(a.seed).sample(recs, min(a.max_samples, len(recs)))
        ds = TriSARSFT(recs, a.fused_dir, proc, a.mode, seed=a.seed)
    else:
        recs = load_index(a.index, tuple(a.splits.split(",")))
        if a.max_samples:
            recs = random.Random(a.seed).sample(recs, min(a.max_samples, len(recs)))
        labels = None
        if a.labels:
            labels = json.load(open(a.labels))
        ds = SFTDataset(recs, a.root, proc, a.mode, seed=a.seed, labels=labels)
    sampler = DistributedSampler(ds, shuffle=True, seed=a.seed) if ddp else None
    dl = DataLoader(ds, batch_size=a.bs, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                    collate_fn=partial(sft_collate, pad_id=proc.tokenizer.pad_token_id), drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)

    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr, "name": "lora"}]
    if fusion is not None:
        enc = list(fusion.encoder.parameters())
        gates = [fusion.main.gate] + [p.gate for p in fusion.deep]
        gid = {id(p) for p in enc + gates}
        rest = [p for p in fusion.parameters() if id(p) not in gid]
        groups += [{"params": enc, "lr": a.lr_enc, "name": "thermal_encoder"},
                   {"params": rest, "lr": a.lr_fusion, "name": "fusion"},
                   {"params": gates, "lr": a.lr_gate, "name": "gates", "weight_decay": 0.0}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    train_params = [p for g in groups for p in g["params"]]

    steps_per_epoch = len(dl) // a.grad_accum
    total = a.max_steps or max(1, int(steps_per_epoch * a.epochs))
    if rank == 0:
        n_train = sum(p.numel() for p in train_params)
        print(f"mode={a.mode} samples={len(ds)} steps={total} trainable={n_train/1e6:.1f}M", flush=True)
        log = RunLogger(a.out, vars(a), use_wandb=a.wandb, project="nightwatch")

    if ddp:
        # LoRA params live in `model`; fusion params are synced manually below (all-reduce grads)
        pass

    step, micro, t0 = 0, 0, time.time()
    epoch = 0
    model.train()
    while step < total:
        if sampler is not None:
            sampler.set_epoch(epoch)
        ds.epoch = epoch
        for batch in dl:
            th = batch.pop("thermal").to(dev, non_blocking=True)
            li = batch.pop("lidar", None)
            li = li.to(dev, non_blocking=True) if li is not None else None
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            if fusion is not None:
                fusion.set_thermal(th)
                fusion.set_extra({"lidar": li} if (li is not None and "lidar" in fusion.extra) else {})
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / a.grad_accum
            loss.backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            if ddp:
                for p in train_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            lr_scale = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 5 == 0 or step == 1):
                rec = {"step": step, "loss": loss.item() * a.grad_accum, "grad_norm": float(gn),
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0,
                       "samples_per_s": step * a.grad_accum * a.bs * (dist.get_world_size() if ddp else 1) / (time.time() - t0)}
                if fusion is not None:
                    rec.update({f"gate_{k}": v for k, v in fusion.gate_values().items()})
                log.log(rec)
            if step >= total:
                break
        epoch += 1

    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        model.save_pretrained(a.out)
        if fusion is not None:
            torch.save(fusion.state_dict(), os.path.join(a.out, "fusion.pt"))
        json.dump({"model": a.model, "mode": a.mode, "steps": step, "minutes": (time.time() - t0) / 60},
                  open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
        log.close()
        print("saved", a.out, flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
