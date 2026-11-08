"""Site memory on LTDv2: a JEPA-style latent predictor over a frozen thermal encoder; anomaly = measured surprise.

Encoder (frozen): the same DINOv3 ConvNeXt-T 1-channel thermal encoder the slow head uses -> 12x9 grid x 1152-d tokens.
Predictor: small transformer that predicts the (LayerNorm'd) tokens of frame t from
    temporal   : the previous K frames of the clip              ("what happens next")
    memory     : hour-of-day, day-of-year and weather only        ("what this place looks like at this hour/weather")
    both       : previous frames + hour/weather
Surprise(t) = mean token error, z-normalised per context bucket (4-h bin x daylight) on a held-out calibration month,
because raw latent error also reacts to thermal auto-gain and weather (Harborfront paper, sec. 5.2).

Evaluation on held-out future months (2021-03/04), positives that exist without hand labels:
    GT-derived  : big vehicle (area > train p99), crowd (#objects > train p99), person at a quiet hour,
                  person at the water edge (hand-drawn quay strip)
    synthetic   : a person crop pasted onto the quay at night (labelled by construction)
    contextual  : the true frame scored with the WRONG hour/season -> only a model with site memory can notice
Baselines: pixel distance to the hour-bucket mean image; token distance to the hour-bucket mean embedding.

python -m nightwatch.memory.site_memory --data data/ltd --out runs/ltd_memory
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
import torch.nn as nn
import torch.nn.functional as F

from nightwatch.slow.model import THERMAL_MEAN, THERMAL_STD, ThermalEncoder
from nightwatch.utils import RunLogger, gpu_mem_cap, seed_all

K = 3                       # context frames
GH, GW = 9, 12              # token grid at stride 32 on 288x384
QUAY = [(40, 288), (140, 85), (175, 85), (90, 288)]   # strip right of the water edge (hand-drawn on a reference frame)


# ---------------------------------------------------------------- data
def cond_vec(f, hour_shift=0.0, doy_shift=0, cond_override=None):
    h = (f["hour"] + hour_shift) % 24
    doy = (f["doy"] + doy_shift) % 366
    t, hum, pr, sun, wind = cond_override if cond_override is not None else f["cond"]
    return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), math.sin(2 * math.pi * doy / 366),
            math.cos(2 * math.pi * doy / 366), t / 20.0, hum / 100.0, min(pr, 5.0), sun / 500.0, wind / 10.0]


def context_index(frames):
    """For each frame: indices of the previous K frames of the same clip (or None)."""
    prev = {}
    by_clip = {}
    for f in frames:
        by_clip.setdefault(f["clip"], []).append(f)
    for fs in by_clip.values():
        fs.sort(key=lambda f: f["t"])
        for j, f in enumerate(fs):
            prev[f["idx"]] = [fs[j - k]["idx"] for k in range(K, 0, -1)] if j >= K else None
    return prev


@torch.no_grad()
def encode(X, dev, bs=256):
    enc = ThermalEncoder(pretrained=True).to(dev).eval()
    out = torch.empty(len(X), GH * GW, enc.out_dim, dtype=torch.float16)
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.asarray(X[i:i + bs])).to(dev).float().div_(255).sub_(THERMAL_MEAN).div_(THERMAL_STD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[i:i + bs] = enc(x[:, None]).half().cpu()
    return out


# ---------------------------------------------------------------- model
class Predictor(nn.Module):
    def __init__(self, cin=1152, d=384, layers=6, heads=6, use_ctx=True, use_cond=True, ncond=9):
        super().__init__()
        self.use_ctx, self.use_cond = use_ctx, use_cond
        self.inp = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, d))
        self.cond = nn.Sequential(nn.Linear(ncond, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.cell = nn.Parameter(torch.randn(GH * GW, d) * 0.02)
        self.slot = nn.Parameter(torch.randn(K + 1, d) * 0.02)
        self.null_cond = nn.Parameter(torch.zeros(2, d))
        self.q = nn.Parameter(torch.randn(GH * GW, d) * 0.02)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(lay, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cin))

    def forward(self, ctx, cond):           # ctx (B,K,108,C) or None ; cond (B,ncond)
        B = cond.shape[0]
        parts = [self.cond(cond).view(B, 2, -1) if self.use_cond else self.null_cond.expand(B, -1, -1)]
        if self.use_ctx:
            c = self.inp(ctx.float()) + self.cell[None, None] + self.slot[None, :K, None]
            parts.append(c.flatten(1, 2))
        parts.append((self.q + self.cell + self.slot[K])[None].expand(B, -1, -1))
        h = self.tf(torch.cat(parts, 1))
        return self.out(h[:, -GH * GW:])


def target(tok):
    return F.layer_norm(tok.float(), tok.shape[-1:])


PEAK_K = 3   # frame surprise = mean of the top-3 token errors: a 30-px person covers ~1 of 108 tokens


def frame_err(pred, tgt, peak=False):
    e = F.smooth_l1_loss(pred, tgt, reduction="none", beta=0.5).mean(-1)     # (B,108)
    return (e.topk(PEAK_K, dim=-1).values.mean(-1) if peak else e.mean(-1)), e


# ---------------------------------------------------------------- train / score
def batches(ids, bs, shuffle, rng):
    ids = list(ids)
    if shuffle:
        rng.shuffle(ids)
    for i in range(0, len(ids), bs):
        yield ids[i:i + bs]


def make_batch(b, TOK, prev, frames_by_idx, dev, cond_fn=None, tgt_tok=None):
    ctx = torch.stack([TOK[prev[i]] for i in b]).to(dev, non_blocking=True)
    cond = torch.tensor([(cond_fn or cond_vec)(frames_by_idx[i]) for i in b], device=dev, dtype=torch.float32)
    tgt = target((tgt_tok if tgt_tok is not None else TOK[b]).to(dev, non_blocking=True))
    return ctx, cond, tgt


def train_variant(name, use_ctx, use_cond, TOK, prev, fb, train_ids, dev, steps, log_dir, bs=256, lr=3e-4):
    m = Predictor(use_ctx=use_ctx, use_cond=use_cond).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng, step, t0 = random.Random(0), 0, time.time()
    log = RunLogger(os.path.join(log_dir, name), {"variant": name, "use_ctx": use_ctx, "use_cond": use_cond, "steps": steps})
    while step < steps:
        for b in batches(train_ids, bs, True, rng):
            ctx, cond, tgt = make_batch(b, TOK, prev, fb, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(ctx, cond)
            loss = frame_err(pred.float(), tgt)[0].mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log.log({"step": step, "loss": loss.item(), "elapsed_s": time.time() - t0})
            if step >= steps:
                break
    log.close()
    return m.eval()


@torch.no_grad()
def score(m, ids, TOK, prev, fb, dev, cond_fn=None, tgt_override=None, bs=512):
    errs, maps = [], []
    for b in batches(ids, bs, False, None):
        tt = None if tgt_override is None else torch.stack([tgt_override[i] for i in b])
        ctx, cond, tgt = make_batch(b, TOK, prev, fb, dev, cond_fn, tt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m(ctx, cond)
        e, mp = frame_err(pred.float(), tgt, peak=True)
        errs.append(e.cpu()); maps.append(mp.half().cpu())
    return torch.cat(errs).numpy(), torch.cat(maps).numpy()


def bucket(f):
    return int(f["hour"] // 4) * 2 + int(f["cond"][3] > 0)


def zscore(err, ids, calib_err, calib_ids, fb):
    stats = {}
    for e, i in zip(calib_err, calib_ids):
        stats.setdefault(bucket(fb[i]), []).append(e)
    mu = {k: (np.mean(v), np.std(v) + 1e-6) for k, v in stats.items()}
    g = (np.mean(calib_err), np.std(calib_err) + 1e-6)
    return np.array([(e - mu.get(bucket(fb[i]), g)[0]) / mu.get(bucket(fb[i]), g)[1] for e, i in zip(err, ids)])


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def in_quay(x, y):
    from matplotlib.path import Path
    return Path(QUAY).contains_point((x, y))


def events(frames, train_frames):
    va = [(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for f in train_frames for b in f["boxes"] if b["cls"] == "vehicle"]
    big = np.percentile(va, 99) if va else 1e9
    crowd = np.percentile([len(f["boxes"]) for f in train_frames], 99)
    pres = {}
    for f in train_frames:
        pres.setdefault(int(f["hour"]), []).append(any(b["cls"] == "person" for b in f["boxes"]))
    quiet = {h for h, v in pres.items() if np.mean(v) < 0.05}
    ev = {"big_vehicle": set(), "crowd": set(), "person_quiet_hour": set(), "person_water_edge": set()}
    for f in frames:
        bx = f["boxes"]
        if any(b["cls"] == "vehicle" and (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) > big for b in bx):
            ev["big_vehicle"].add(f["idx"])
        if len(bx) > crowd:
            ev["crowd"].add(f["idx"])
        persons = [b for b in bx if b["cls"] == "person"]
        if persons and int(f["hour"]) in quiet:
            ev["person_quiet_hour"].add(f["idx"])
        if any(in_quay((b["box"][0] + b["box"][2]) / 2, b["box"][3]) for b in persons):
            ev["person_water_edge"].add(f["idx"])
    return ev, {"big_vehicle_area_p99": float(big), "crowd_p99": float(crowd), "quiet_hours": sorted(quiet)}


def paste_person(img, crop, rng):
    """Paste a warm person crop onto the quay strip (labelled-by-construction anomaly).
    Perspective-scaled: people lower in the frame are closer to the camera and appear larger."""
    from PIL import Image as _I
    out = img.copy()
    for _ in range(50):
        x, y = rng.randint(40, 170), rng.randint(120, 285)
        if not in_quay(x, y):
            continue
        scale = 0.6 + 2.2 * (y - 85) / 200.0                       # ~0.6x at the far end, ~2.8x at the bottom
        h = max(10, int(crop.shape[0] * scale)); w = max(4, int(crop.shape[1] * scale))
        c = np.asarray(_I.fromarray(crop).resize((w, h), _I.BILINEAR), dtype=np.float32)
        if y - h < 0 or x + w >= 384:
            continue
        reg = out[y - h:y, x:x + w].astype(np.float32)
        mask = (c > np.median(c)).astype(np.float32)
        hot = np.maximum(c, reg.mean() + 0.8 * (c - c.mean()) + 40)   # people are warm relative to the night background
        out[y - h:y, x:x + w] = np.clip(reg * (1 - mask) + hot * mask, 0, 255).astype(np.uint8)
        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ltd")
    ap.add_argument("--out", default="runs/ltd_memory")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-paste", type=int, default=600)
    a = ap.parse_args()
    seed_all(0); gpu_mem_cap()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    frames = json.load(open(os.path.join(a.data, "frames.json")))
    X = np.load(os.path.join(a.data, "frames.npy"), mmap_mode="r")
    fb = {f["idx"]: f for f in frames}
    tok_path = os.path.join(a.data, "tokens_dinov3.pt")
    if os.path.exists(tok_path):
        TOK = torch.load(tok_path)
    else:
        t0 = time.time(); TOK = encode(X, dev); torch.save(TOK, tok_path)
        print(f"encoded {len(X)} frames in {time.time() - t0:.0f}s", flush=True)
    TOK = TOK.pin_memory()
    prev = context_index(frames)
    ok = [f for f in frames if prev[f["idx"]] is not None]
    tr = [f["idx"] for f in ok if f["wm"] == "train"]
    ca = [f["idx"] for f in ok if f["wm"] == "calib"]
    te = [f["idx"] for f in ok if f["wm"] == "test"]
    print(f"windows: train {len(tr)} calib {len(ca)} test {len(te)}", flush=True)
    train_frames = [f for f in frames if f["wm"] == "train"]
    ev, ev_meta = events([fb[i] for i in te], train_frames)
    any_ev = set().union(*ev.values())
    normal_te = [i for i in te if i not in any_ev]

    # synthetic: paste person crops (from train frames) onto the quay of night test frames
    rng = random.Random(1)
    crops = []
    for f in train_frames:
        for b in f["boxes"]:
            x1, y1, x2, y2 = (int(v) for v in b["box"])
            if b["cls"] == "person" and 22 <= y2 - y1 <= 60 and x2 - x1 >= 6:
                crops.append(np.asarray(X[f["idx"]][y1:y2, x1:x2]))
        if len(crops) > 2000:
            break
    night = [i for i in normal_te if fb[i]["cond"][3] == 0]
    paste_ids = rng.sample(night, min(a.n_paste, len(night)))
    enc = ThermalEncoder(pretrained=True).to(dev).eval()
    pasted = {}
    with torch.no_grad():
        for j in range(0, len(paste_ids), 128):
            b = paste_ids[j:j + 128]
            imgs = np.stack([paste_person(np.asarray(X[i]), rng.choice(crops), rng) for i in b])
            x = torch.from_numpy(imgs).to(dev).float().div_(255).sub_(THERMAL_MEAN).div_(THERMAL_STD)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = enc(x[:, None]).half().cpu()
            pasted.update({i: t[k] for k, i in enumerate(b)})
    del enc

    # baselines: hour-bucket means of pixels and tokens (non-learned "site memory")
    pix_mean, tok_mean = {}, {}
    for f in train_frames:
        k = bucket(f)
        pix_mean.setdefault(k, []).append(f["idx"])
    for k, v in pix_mean.items():
        sel = v[:: max(1, len(v) // 400)]
        tok_mean[k] = target(TOK[sel]).mean(0)
        pix_mean[k] = np.asarray(X[sel], dtype=np.float32).mean(0)

    def base_scores(ids, tok_src=None, img_src=None):
        p, t = [], []
        for i in ids:
            k = bucket(fb[i])
            img = img_src[i] if img_src is not None else np.asarray(X[i], dtype=np.float32)
            tok = tok_src[i] if tok_src is not None else TOK[i]
            p.append(float(np.abs(img - pix_mean.get(k, img)).mean()))
            d = (target(tok[None])[0] - tok_mean.get(k, target(tok[None])[0])).abs().mean(-1)
            t.append(float(d.topk(PEAK_K).values.mean()))
        return np.array(p), np.array(t)

    results = {"events": {k: len(v) for k, v in ev.items()}, "event_def": ev_meta, "n_normal_test": len(normal_te),
               "n_paste": len(paste_ids), "variants": {}}
    variants = {"temporal": (True, False), "memory": (False, True), "both": (True, True)}
    per_frame = {}
    for name, (uc, ucond) in variants.items():
        m = train_variant(name, uc, ucond, TOK, prev, fb, tr, dev, a.steps, a.out)
        ce, _ = score(m, ca, TOK, prev, fb, dev)
        te_e, te_map = score(m, te, TOK, prev, fb, dev)
        z = dict(zip(te, zscore(te_e, te, ce, ca, fb)))
        raw = dict(zip(te, te_e))
        pe, _ = score(m, paste_ids, TOK, prev, fb, dev, tgt_override=pasted)
        zp = zscore(pe, paste_ids, ce, ca, fb)
        # contextual swap: real frame, conditions from the opposite season and 12 h shifted
        winter = [fb[i]["cond"] for i in ca][:1] or [[0, 90, 0, 0, 5]]
        swap = lambda f: cond_vec(f, hour_shift=12, doy_shift=183, cond_override=winter[0] if f["month"] >= "202103" else None)
        se, _ = score(m, normal_te, TOK, prev, fb, dev, cond_fn=swap)
        zs = zscore(se, normal_te, ce, ca, fb)
        neg = [z[i] for i in normal_te]
        r = {f"auroc_{k}": auroc([z[i] for i in v], neg) for k, v in ev.items()}
        r["auroc_pasted_person"] = auroc(zp, [z[i] for i in paste_ids])
        r["auroc_context_swap"] = auroc(zs, neg)
        r["auroc_any_event"] = auroc([z[i] for i in any_ev], neg)
        mean_by_month = {}
        for i in te + ca:
            pass
        r["raw_err_test_mean"] = float(np.mean(te_e))
        results["variants"][name] = r
        per_frame[name] = {"z": {str(i): float(z[i]) for i in te}, "map_top": {}}
        top = sorted(te, key=lambda i: -z[i])[:24]
        idx_of = {i: k for k, i in enumerate(te)}
        per_frame[name]["map_top"] = {str(i): te_map[idx_of[i]].astype(np.float32).tolist() for i in top}
        print(name, json.dumps(r), flush=True)
        torch.save(m.state_dict(), os.path.join(a.out, f"{name}.pt"))
    # baselines
    bp, bt = base_scores(te)
    zpix, ztok = dict(zip(te, bp)), dict(zip(te, bt))
    pp, pt = base_scores(paste_ids, tok_src=pasted,
                         img_src={i: paste_person(np.asarray(X[i]), random.Random(i).choice(crops), random.Random(i)).astype(np.float32)
                                  for i in paste_ids})
    for name, zz, pz in [("baseline_pixel_hourmean", zpix, pp), ("baseline_token_hourmean", ztok, pt)]:
        neg = [zz[i] for i in normal_te]
        r = {f"auroc_{k}": auroc([zz[i] for i in v], neg) for k, v in ev.items()}
        r["auroc_pasted_person"] = auroc(pz, [zz[i] for i in paste_ids])
        r["auroc_context_swap"] = 0.5
        r["auroc_any_event"] = auroc([zz[i] for i in any_ev], neg)
        results["variants"][name] = r
        print(name, json.dumps(r), flush=True)
    # drift: mean z on normal frames per month (calib + test) for the 'both' model is in per_frame; summarise by month
    for name in variants:
        months = {}
        for i in normal_te:
            months.setdefault(fb[i]["month"], []).append(per_frame[name]["z"][str(i)])
        results["variants"][name]["mean_z_normal_by_month"] = {k: float(np.mean(v)) for k, v in sorted(months.items())}
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    json.dump({k: v["map_top"] for k, v in per_frame.items()}, open(os.path.join(a.out, "top_maps.json"), "w"))
    json.dump({k: v["z"] for k, v in per_frame.items()}, open(os.path.join(a.out, "surprise_test.json"), "w"))
    print("SITE_MEMORY_DONE", flush=True)


if __name__ == "__main__":
    main()
