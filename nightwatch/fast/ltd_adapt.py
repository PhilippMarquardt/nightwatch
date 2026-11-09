"""Track B (docs/PLAN_SITE_ADAPT.md v2): the LTDv2 harbour camera as a genuinely unseen site for the LLVIP-trained
thermal fast head. Different country, camera, resolution (384x288), season and year.

Adaptation month 2020-05 (labels withheld except for the oracle arm); test months 2021-03 and 2021-04 (never used for
any choice). Every arm fine-tunes the same start model (`fast_oracle_t`) for the same number of steps on LLVIP GT plus
its own labels for the same 2,000 adaptation frames; thresholds come from the LLVIP val site (no target labels).

python -m nightwatch.fast.ltd_adapt cache
python -m nightwatch.fast.ltd_adapt label --kind self|vlm|consensus|gt
python -m nightwatch.fast.ltd_adapt train --arm B1_self --labels runs/ltd/labels_self.json --seed 0
python -m nightwatch.fast.ltd_adapt eval  --run runs/ltd_B1_self_s0
python -m nightwatch.fast.ltd_adapt report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from nightwatch.data import tasks as T
from nightwatch.data.llvip import load_index
from nightwatch.eval.metrics import ap50
from nightwatch.fast.data import MAXB, SCALE, augment, load_split
from nightwatch.fast.model import CenterNetRGBT, centernet_loss, make_targets
from nightwatch.fast.train import IN_CH, best_f1_threshold, predict
from nightwatch.utils import gpu_mem_cap, seed_all

LTD, CACHE, OUT = "data/ltd", "data/cache_ltd", "runs/ltd"
ADAPT_MONTH, TEST_MONTHS, N_ADAPT = "202005", ("202103", "202104"), 2000
SX, SY = 1280 / 384, 1024 / 288            # LTD px -> LLVIP 1280x1024 px (the fast head's label convention)
START = "runs/fast_oracle_t"


def frames():
    return json.load(open(f"{LTD}/frames.json"))


def persons(f):
    return [[b["box"][0] * SX, b["box"][1] * SY, b["box"][2] * SX, b["box"][3] * SY] for b in f["boxes"] if b["cls"] == "person"]


def cmd_cache(a):
    os.makedirs(CACHE, exist_ok=True)
    F_ = frames()
    adapt = sorted([f for f in F_ if f["month"] == ADAPT_MONTH], key=lambda f: f["id"])
    adapt = sorted(random.Random(0).sample(adapt, min(N_ADAPT, len(adapt))), key=lambda f: f["id"])
    test = sorted([f for f in F_ if f["month"] in TEST_MONTHS], key=lambda f: f["id"])
    arr = np.load(f"{LTD}/frames.npy", mmap_mode="r")
    for name, sel in (("adapt", adapt), ("test", test)):
        x = np.zeros((len(sel), 4, 512, 640), np.uint8)
        for k, f in enumerate(sel):
            t = torch.from_numpy(np.asarray(arr[f["idx"]], np.float32))[None, None]
            x[k, 3] = F.interpolate(t, size=(512, 640), mode="bilinear", align_corners=False)[0, 0].clamp(0, 255).byte().numpy()
        np.save(f"{CACHE}/{name}.npy", x)
        json.dump([f["id"] for f in sel], open(f"{CACHE}/{name}_ids.json", "w"))
        json.dump({f["id"]: {"boxes": persons(f), "clip": f["clip"], "month": f["month"]} for f in sel},
                  open(f"{CACHE}/{name}_gt.json", "w"))
        print(name, x.shape, sum(len(persons(f)) for f in sel), "persons", flush=True)


def load_ltd(split, dev):
    x = torch.from_numpy(np.load(f"{CACHE}/{split}.npy", mmap_mode="r")[:])
    ids = json.load(open(f"{CACHE}/{split}_ids.json"))
    gt = json.load(open(f"{CACHE}/{split}_gt.json"))
    return x, ids, gt


def load_net(run, dev):
    summ = json.load(open(f"{run}/train_summary.json"))
    net = CenterNetRGBT("resnet18.a1_in1k", IN_CH[summ["modal"]], pretrained=False).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev))
    return net, summ


def cmd_label(a):
    dev = torch.device("cuda"); gpu_mem_cap()
    os.makedirs(OUT, exist_ok=True)
    x, ids, gt = load_ltd("adapt", dev)
    out = f"{OUT}/labels_{a.kind}.json"
    if a.kind == "gt":
        lab = {i: gt[i]["boxes"] for i in ids}
    elif a.kind == "self":
        net, summ = load_net(START, dev)
        pr = predict(net, x, summ["modal"], thr=summ["score_thr"])
        lab = {i: p["boxes"] for i, p in zip(ids, pr)}
    elif a.kind == "self_lo":      # low threshold, only for the consensus filter
        net, summ = load_net(START, dev)
        pr = predict(net, x, summ["modal"], thr=0.15)
        lab = {i: p["boxes"] for i, p in zip(ids, pr)}
    elif a.kind == "vlm":
        lab = vlm_label(x, ids, dev)
    elif a.kind == "consensus":
        v, s = json.load(open(f"{OUT}/labels_vlm.json")), json.load(open(f"{OUT}/labels_self_lo.json"))
        lab = {i: [b for b in v.get(i, []) if any(_iou(b, c) >= 0.3 for c in s.get(i, []))] for i in ids}
    json.dump(lab, open(out, "w"))
    q = quality(lab, {i: gt[i]["boxes"] for i in ids}, ids)
    json.dump(q, open(out.replace(".json", "_quality.json"), "w"), indent=1)
    print(a.kind, f"{sum(map(len, lab.values()))} boxes; vs withheld GT P {q['p']:.3f} R {q['r']:.3f} F1 {q['f1']:.3f}", flush=True)


def vlm_label(x, ids, dev, bs=24):
    """Thermal-only 9B slow head (LLVIP sft_thermal) on the adaptation frames; boxes back in 1280x1024 px."""
    from PIL import Image
    from nightwatch.slow.data import build_messages
    from nightwatch.slow.model import IMG_H, IMG_W, load_slow_head
    model, proc, _ = load_slow_head("Qwen/Qwen3.5-9B", "thermal", adapter_dir="runs/sft_thermal", device=dev)
    model.eval()
    proc.tokenizer.padding_side = "left"
    text = proc.apply_chat_template(build_messages("thermal", T.EVAL_DETECT_PROMPT), add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    lab, t0 = {}, time.time()
    for s in range(0, len(ids), bs):
        imgs = [Image.fromarray(x[k, 3].numpy()).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
                for k in range(s, min(len(ids), s + bs))]
        enc = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=512, do_sample=False)
        for iid, t in zip(ids[s:s + bs], proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            lab[iid] = T.from_qwen(T.parse_boxes(t), 1280, 1024)
        print(f"vlm {s + len(imgs)}/{len(ids)} {(s + len(imgs)) / (time.time() - t0):.2f} img/s", flush=True)
    return lab


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def quality(pred, gts, ids, iou=0.5):
    tp = fp = fn = 0
    for i in ids:
        x, y, z, _ = T.greedy_match(pred.get(i, []), gts[i], iou)
        tp, fp, fn = tp + x, fp + y, fn + z
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"p": p, "r": r, "f1": 2 * p * r / max(1e-9, p + r), "tp": tp, "fp": fp, "fn": fn}


def cmd_train(a):
    """Fine-tune START on LLVIP GT (labeled+unlabeled) + LTD adaptation frames with this arm's labels; fixed iters."""
    seed_all(a.seed); gpu_mem_cap()
    dev = torch.device("cuda")
    out_dir = f"runs/ltd_{a.arm}_s{a.seed}"
    os.makedirs(out_dir, exist_ok=True)
    xs, bxs, ns = [], [], []
    for sp in ("labeled", "unlabeled"):
        x, bx, n, _ = load_split("data/cache_fast", sp, dev, index="data/index.json")
        xs.append(x); bxs.append(bx); ns.append(n)
    if a.labels != "none":
        lab = json.load(open(a.labels))
        xl, ids, _ = load_ltd("adapt", dev)
        bx = torch.zeros(len(ids), MAXB, 4); n = torch.zeros(len(ids), dtype=torch.long)
        for k, i in enumerate(ids):
            b = lab.get(i, [])[:MAXB]
            if b:
                bx[k, :len(b)] = torch.tensor(b, dtype=torch.float32) * SCALE
            n[k] = len(b)
        rep = max(1, a.ltd_repeat)                 # LTD frames appear rep times per epoch (2,000 vs 11,602 LLVIP)
        xs.append(xl.repeat(rep, 1, 1, 1).to(xs[0].device)); bxs.append(bx.repeat(rep, 1, 1).to(dev)); ns.append(n.repeat(rep).to(dev))
    X, BX, N = torch.cat(xs), torch.cat(bxs), torch.cat(ns)
    net, summ = load_net(START, dev)
    modal = summ["modal"]
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / a.iters)))
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    net.train(); t0 = time.time()
    for it in range(a.iters):
        idx = torch.randint(0, len(X), (a.bs,), device=dev, generator=gen)
        xb = X[idx.to(X.device)].to(dev, non_blocking=True)
        xb, boxes, valid = augment(xb, BX[idx], N[idx], gen, modal=modal)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(xb.contiguous(memory_format=torch.channels_last))
        hm_t, reg = make_targets(boxes, valid, o[0].shape[2], o[0].shape[3])
        loss, _ = centernet_loss(o, hm_t, reg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step(); sched.step()
        if (it + 1) % 200 == 0:
            print(f"step={it + 1} loss={loss.item():.4f} img_per_s={(it + 1) * a.bs / (time.time() - t0):.0f}", flush=True)
    torch.save(net.state_dict(), f"{out_dir}/best.pt")
    # threshold from the LLVIP val site (no target labels anywhere)
    from nightwatch.eval.metrics import height_buckets
    recs = load_index("data/index.json", ("val",))
    Xv, _, _, vids = load_split("data/cache_fast", "val", dev, index="data/index.json")
    pv = dict(zip(vids, predict(net, Xv, modal)))
    thr, vf1 = best_f1_threshold(pv, {r.id: r.boxes for r in recs}, vids, height_buckets(recs))
    json.dump({"name": out_dir, "arm": a.arm, "labels": a.labels, "seed": a.seed, "modal": modal, "iters": a.iters,
               "score_thr": thr, "val_f1_llvip": vf1, "minutes": (time.time() - t0) / 60}, open(f"{out_dir}/train_summary.json", "w"), indent=1)
    print("trained", out_dir, "thr", thr, flush=True)


def cmd_eval(a):
    dev = torch.device("cuda"); gpu_mem_cap()
    net, summ = load_net(a.run, dev)
    x, ids, gt = load_ltd("test", dev)
    pr = predict(net, x, summ["modal"], thr=0.05)
    with open(f"{a.run}/pred_ltd_test.jsonl", "w") as fh:
        for i, p in zip(ids, pr):
            fh.write(json.dumps({"id": i, **p}) + "\n")
    P = dict(zip(ids, pr)); G = {i: gt[i]["boxes"] for i in ids}
    thr = summ["score_thr"]
    kept = {i: {"boxes": [b for b, s in sorted(zip(p["boxes"], p["scores"]), key=lambda t: -t[1]) if s >= thr]} for i, p in P.items()}
    res = {"n_frames": len(ids), "thr": thr, "ap50": ap50(P, G, ids),
           "iou50": quality({i: v["boxes"] for i, v in kept.items()}, G, ids, 0.5),
           "iou30": quality({i: v["boxes"] for i, v in kept.items()}, G, ids, 0.3)}
    for m in TEST_MONTHS:
        mi = [i for i in ids if gt[i]["month"] == m]
        res[f"f1_{m}"] = quality({i: kept[i]["boxes"] for i in mi}, G, mi, 0.5)["f1"]
    json.dump(res, open(f"{a.run}/ltd_test_metrics.json", "w"), indent=1)
    # LLVIP regression check (home domain): same metrics module as every other LLVIP fast head
    Xt, _, _, tids = load_split("data/cache_fast", "test", dev, index="data/index.json")
    with open(f"{a.run}/pred_test.jsonl", "w") as fh:
        for i, p in zip(tids, predict(net, Xt, summ["modal"])):
            fh.write(json.dumps({"id": i, "task": "detect", "cond": "normal", **p}) + "\n")
    print(a.run, f"LTD AP50 {res['ap50']:.3f} F1@.5 {res['iou50']['f1']:.3f} F1@.3 {res['iou30']['f1']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    p = sub.add_parser("label"); p.add_argument("--kind", required=True, choices=["self", "self_lo", "vlm", "consensus", "gt"])
    p = sub.add_parser("train"); p.add_argument("--arm", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ltd-repeat", type=int, default=3)
    p = sub.add_parser("eval"); p.add_argument("--run", required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "label": cmd_label, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
