"""Optional stage: GRPO on the fused slow head with a verifiable reward (per-image detection F1).

Starts from the SFT adapter, samples G answers per prompt, rewards each with F1@0.5 against the
labels, and takes one on-policy policy-gradient step with group-normalised advantages
(Dr.GRPO-style token-mean, no std division by default, no KL -- short runs from a good SFT init).

python -m nightwatch.slow.grpo --init runs/sft_fusion --out runs/grpo_fusion
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from PIL import Image

from nightwatch.data import tasks as T
from nightwatch.data.llvip import H, W, load_index
from nightwatch.slow.data import build_messages, images_for, load_pair
from nightwatch.slow.model import load_slow_head, thermal_tensor
from nightwatch.utils import RunLogger, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--init", required=True, help="SFT run dir (LoRA adapter + fusion.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/raw/LLVIP")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--splits", default="labeled")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--crop-p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    dev = torch.device("cuda")

    from peft import PeftModel
    model, proc, fusion = load_slow_head(a.model, "fusion", device=dev, grad_ckpt=True)
    fusion.load_state_dict(torch.load(os.path.join(a.init, "fusion.pt"), map_location=dev))
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    fusion.encoder.requires_grad_(False)          # RL only moves the LoRA + fusion projections
    params = [p for p in model.parameters() if p.requires_grad] + [p for n, p in fusion.named_parameters()
                                                                     if not n.startswith("encoder.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    recs = load_index(a.index, tuple(a.splits.split(",")))
    rng = random.Random(a.seed)
    log = RunLogger(a.out, vars(a))
    tok = proc.tokenizer
    t0 = time.time()

    for step in range(1, a.steps + 1):
        batch = rng.sample(recs, a.prompts)
        texts, imgs, thermals, gts = [], [], [], []
        for r in batch:
            crop, boxes, (w, h) = (None, r.boxes, (W, H))
            if rng.random() < a.crop_p:
                crop, boxes, (w, h) = T.random_crop(r.boxes, W, H, rng)
            rgb, ir = load_pair(r, a.root, crop)
            msgs = build_messages("fusion", T.EVAL_DETECT_PROMPT)
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
            imgs += images_for("fusion", rgb, ir)
            thermals.append(thermal_tensor(ir))
            gts.append(T.to_qwen(boxes, w, h))
        tok.padding_side = "left"
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(dev)
        th = torch.stack(thermals).to(dev)

        # ---- sample G completions per prompt
        model.eval()
        fusion.set_thermal(th.repeat_interleave(a.group, 0))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=1.0, max_new_tokens=a.max_new, use_cache=True,
                                 num_return_sequences=a.group)
        P = enc["input_ids"].shape[1]
        comp = out[:, P:]
        texts_out = tok.batch_decode(comp, skip_special_tokens=True)
        rewards = torch.tensor([T.f1_reward(T.parse_boxes(t), gts[k // a.group]) for k, t in enumerate(texts_out)],
                               device=dev)
        R = rewards.view(a.prompts, a.group)
        adv = (R - R.mean(1, keepdim=True)).view(-1)
        if adv.abs().sum() == 0:
            log.log({"step": step, "reward": rewards.mean().item(), "skipped": 1})
            continue

        # ---- one on-policy gradient step
        model.train()
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        cmask = torch.ones_like(comp, dtype=torch.float)
        for k in range(comp.shape[0]):                      # mask after first <|im_end|> / pad
            hits = ((comp[k] == eos) | (comp[k] == tok.pad_token_id)).nonzero()
            if len(hits):
                cmask[k, hits[0, 0] + 1:] = 0
        full_att = torch.cat([enc["attention_mask"].repeat_interleave(a.group, 0), cmask.long()], 1)
        pv = enc["pixel_values"]
        n_patch = pv.shape[0] // a.prompts
        pv = pv.view(a.prompts, n_patch, -1).repeat_interleave(a.group, 0).view(-1, pv.shape[-1])
        grid = enc["image_grid_thw"].repeat_interleave(a.group, 0)
        opt.zero_grad(set_to_none=True)
        mb = a.group                                        # micro-batch = one prompt's group
        total_tok = cmask.sum()
        loss_acc = 0.0
        for s in range(0, out.shape[0], mb):
            sl = slice(s, s + mb)
            fusion.set_thermal(th.repeat_interleave(a.group, 0)[sl])
            ppatch = pv.view(out.shape[0], n_patch, -1)[sl].reshape(-1, pv.shape[-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                extra = {}
                if "mm_token_type_ids" in enc:
                    extra["mm_token_type_ids"] = (out[sl] == model.config.image_token_id).long()
                logits = model(input_ids=out[sl], attention_mask=full_att[sl], pixel_values=ppatch,
                               image_grid_thw=grid[sl], **extra).logits[:, P - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, comp[sl, :, None]).squeeze(-1)
            loss = -(adv[sl, None] * lp * cmask[sl]).sum() / total_tok
            loss.backward()
            loss_acc += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.log({"step": step, "reward": rewards.mean().item(), "reward_max": R.max(1).values.mean().item(),
                 "loss": loss_acc, "grad_norm": float(gn), "len": cmask.sum(1).mean().item(),
                 "elapsed_s": time.time() - t0})

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    torch.save(fusion.state_dict(), os.path.join(a.out, "fusion.pt"))
    json.dump({"init": a.init, "steps": a.steps, "minutes": (time.time() - t0) / 60}, open(f"{a.out}/train_summary.json", "w"))
    log.close()


if __name__ == "__main__":
    main()
