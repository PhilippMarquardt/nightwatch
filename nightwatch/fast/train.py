"""Train the fast head, pick the checkpoint + score threshold on the val site, predict test.

python -m nightwatch.fast.train --name fast_distill --splits labeled,unlabeled --labels runs/pseudo_labels_unlabeled.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from nightwatch.data.llvip import load_index
from nightwatch.eval.metrics import ap50, detect_metrics, height_buckets
from nightwatch.fast.data import SCALE, augment, load_split, prep_eval
from nightwatch.fast.model import CenterNetRGBT, centernet_loss, decode, make_targets
from nightwatch.utils import RunLogger, gpu_mem_cap, seed_all

IN_CH = {"rgbt": 4, "rgb": 3, "t": 1}


@torch.no_grad()
def predict(net, x, modal, cond="normal", bs=64, thr=0.05):
    net.eval()
    outs = []
    for i in range(0, len(x), bs):
        xb = prep_eval(x[i:i + bs].to(next(net.parameters()).device, non_blocking=True), modal, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(xb.contiguous(memory_format=torch.channels_last))
        b, s = decode(*(t.float() for t in o), k=50, scale=1 / SCALE)
        for bb, ss in zip(b.cpu().numpy(), s.cpu().numpy()):
            k = ss >= thr
            outs.append({"boxes": bb[k].tolist(), "scores": ss[k].tolist()})
    net.train()
    return outs


def best_f1_threshold(preds, gts, ids, bucket):
    best = (0.0, 0.3)
    for t in np.arange(0.1, 0.9, 0.025):
        f = detect_metrics(preds, gts, ids, bucket, score_thr=float(t))["f1"]
        best = max(best, (f, float(t)))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", default="labeled")
    ap.add_argument("--labels", default=None, help="pseudo labels json {id: boxes}; replaces GT on non-labeled splits")
    ap.add_argument("--min-pseudo-coverage", type=float, default=0.95)
    ap.add_argument("--modal", choices=list(IN_CH), default="rgbt")
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--cache", default="data/cache_fast")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    out_dir = f"runs/{a.name}"
    seed_all(a.seed)
    gpu_mem_cap()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    pseudo = json.load(open(a.labels)) if a.labels else {}
    xs, bxs, ns = [], [], []
    for sp in a.splits.split(","):
        gt = {r.id: r.boxes for r in load_index(a.index, (sp,))}
        use_pseudo = bool(pseudo) and sp != "labeled"
        lab = {i: pseudo.get(i, []) for i in gt} if use_pseudo else gt
        x, bx, n, ids = load_split(a.cache, sp, dev, lab)
        if use_pseudo:   # never fall back to ground truth on the "unlabeled" sites: keep only slow-head-labelled frames
            keep = torch.tensor([i in pseudo for i in ids])
            cov = float(keep.float().mean())
            print(f"{sp}: pseudo-label coverage {cov:.1%} ({int(keep.sum())}/{len(ids)} frames kept, GT never used)", flush=True)
            if cov < a.min_pseudo_coverage:
                raise SystemExit(f"pseudo-label coverage {cov:.1%} < {a.min_pseudo_coverage:.0%} - refusing to train")
            x, bx, n = x[keep.to(x.device)], bx[keep.to(bx.device)], n[keep.to(n.device)]
        xs.append(x); bxs.append(bx); ns.append(n)
    X, BX, N = torch.cat(xs), torch.cat(bxs), torch.cat(ns)
    del xs
    val_recs = load_index(a.index, ("val",))
    Xv, _, _, val_ids = load_split(a.cache, "val", dev, index=a.index)
    gts_v = {r.id: r.boxes for r in val_recs}
    bucket_v = height_buckets(val_recs)

    net = CenterNetRGBT(a.backbone, IN_CH[a.modal]).to(dev).to(memory_format=torch.channels_last)
    fwd = torch.compile(net) if a.compile else net
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    iters = a.epochs * (len(X) // a.bs)
    warm = min(500, iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * i / iters)))
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    log = RunLogger(out_dir, vars(a), use_wandb=a.wandb)
    print(f"{a.name}: {len(X)} train frames, {iters} iters, on-gpu={X.is_cuda}", flush=True)

    best_ap, it, t0 = -1.0, 0, time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), device=dev, generator=gen)
        for s in range(0, len(X) - a.bs + 1, a.bs):
            idx = perm[s:s + a.bs]
            xb = X[idx.to(X.device)].to(dev, non_blocking=True)
            xb, boxes, valid = augment(xb, BX[idx], N[idx], gen, modal=a.modal)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd(xb.contiguous(memory_format=torch.channels_last))
            hm_t, reg = make_targets(boxes, valid, out[0].shape[2], out[0].shape[3])
            loss, parts = centernet_loss(out, hm_t, reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step(); sched.step(); it += 1
            if it % 50 == 0:
                log.log({"step": it, "epoch": ep, "loss": loss.item(), **parts, "lr": sched.get_last_lr()[0],
                         "img_per_s": it * a.bs / (time.time() - t0)})
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            pv = predict(net, Xv, a.modal)
            pv = dict(zip(val_ids, pv))
            v_ap = ap50(pv, gts_v, val_ids)
            log.log({"step": it, "epoch": ep, "val_ap50": v_ap})
            if v_ap > best_ap:
                best_ap = v_ap
                torch.save(net.state_dict(), f"{out_dir}/best.pt")

    # ---- final: best ckpt, val-selected threshold, test predictions under 3 sensor conditions
    net.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=dev))
    pv = dict(zip(val_ids, predict(net, Xv, a.modal)))
    thr, vf1 = best_f1_threshold(pv, gts_v, val_ids, bucket_v)
    del X
    Xt, _, _, test_ids = load_split(a.cache, "test", dev, index=a.index)
    with open(f"{out_dir}/pred_test.jsonl", "w") as fh:
        for cond in ["normal", "rgb_off", "thermal_off"]:
            if (a.modal == "rgb" and cond == "thermal_off") or (a.modal == "t" and cond == "rgb_off"):
                continue
            for iid, p in zip(test_ids, predict(net, Xt, a.modal, cond)):
                fh.write(json.dumps({"id": iid, "task": "detect", "cond": cond, **p}) + "\n")
    # latency of the pytorch model (TensorRT numbers come from export.py)
    xb = prep_eval(Xt[:1].to(dev), a.modal).contiguous(memory_format=torch.channels_last)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(20):
            net(xb)
        torch.cuda.synchronize(); t1 = time.time()
        for _ in range(200):
            net(xb)
        torch.cuda.synchronize()
    lat = (time.time() - t1) / 200 * 1000
    summ = {"name": a.name, "splits": a.splits, "labels": a.labels, "modal": a.modal, "val_ap50": best_ap,
            "score_thr": thr, "val_f1": vf1, "torch_latency_ms_bs1": lat, "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
            "train_minutes": (time.time() - t0) / 60}
    json.dump(summ, open(f"{out_dir}/train_summary.json", "w"), indent=1)
    log.log({"step": it, **{k: v for k, v in summ.items() if isinstance(v, (int, float))}})
    log.close()


if __name__ == "__main__":
    main()
