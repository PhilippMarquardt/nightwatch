"""Fast-head-guided zoom: the fast head proposes, the slow head looks closer.

Problem it fixes: the VLM sees 32x32-px tokens; a person 40 m away is 8-17 px tall (< half a token), so on the
unseen forest site the slow head falls back to its training prior ("two people in the middle").
Cascade, per frame:
    1. fast head (stride-4 CenterNet on RGB+T+LiDAR, sees every frame) -> person proposals (score >= prop_thr)
    2. no proposal  -> report [] without calling the slow head (also kills hallucinations on empty frames)
    3. otherwise    -> 5:4 crop around the proposals (zoom up to MAX_ZOOM), slow head writes its report on the crop,
                       boxes are mapped back to the full frame
Outputs (scored with metrics_trisar):
    runs/t_zoom/pred_test.jsonl         slow head on the zoomed crop
    runs/t_zoom/pred_test_union.jsonl   + confident fast-head detections the slow head did not report
python -m nightwatch.eval.zoom_cascade --fast runs/tf_rgbtl --slow runs/t_sft_fusion_l --mode fusion_l
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from nightwatch.data import tasks as T
from nightwatch.slow import data_trisar as DT
from nightwatch.slow.data import infer_collate
from nightwatch.slow.model import lidar_tensor, load_slow_head, thermal_tensor
from nightwatch.utils import gpu_mem_cap

FW, FH = DT.FW, DT.FH
MAX_ZOOM = 4.0          # smallest crop = 160 x 128 px of the 640 x 512 frame


def crop_for(props, margin=0.6):
    x1 = min(p["box"][0] for p in props); y1 = min(p["box"][1] for p in props)
    x2 = max(p["box"][2] for p in props); y2 = max(p["box"][3] for p in props)
    w, h = (x2 - x1) * (1 + 2 * margin), (y2 - y1) * (1 + 2 * margin)
    w = max(w, h * FW / FH, FW / MAX_ZOOM); h = w * FH / FW              # keep 5:4, enforce max zoom
    if w >= FW * 0.85:
        return None                                                     # proposals spread out: use the full frame
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx = min(max(cx, w / 2), FW - w / 2); cy = min(max(cy, h / 2), FH - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class ZoomInfer(Dataset):
    def __init__(self, items, fused_dir, proc, mode):
        self.items, self.dir, self.proc, self.mode = items, fused_dir, proc, mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fid, crop = self.items[i]
        rgb, th, li = DT.load_triplet(self.dir, fid, crop)
        text = self.proc.apply_chat_template(DT.messages(self.mode, DT.REPORT_PROMPT), add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        return {"id": fid, "text": text, "images": [rgb], "thermal": thermal_tensor(th), "lidar": lidar_tensor(li)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", default="runs/tf_rgbtl")
    ap.add_argument("--slow", default="runs/t_sft_fusion_l")
    ap.add_argument("--mode", default="fusion_l")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--prop-thr", type=float, default=0.2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/t_zoom")
    a = ap.parse_args()
    gpu_mem_cap()
    os.makedirs(a.out, exist_ok=True)
    keep_thr = json.load(open(os.path.join(a.fast, "train_summary.json")))["score_thr"]
    fast = {}
    for line in open(os.path.join(a.fast, "pred_test.jsonl")):
        d = json.loads(line)
        if d["task"] == "report" and d["cond"] == "normal":
            fast[d["id"]] = d["persons"]
    items, crops, n_skip = [], {}, 0
    for fid, ps in fast.items():
        props = [p for p in ps if p["score"] >= a.prop_thr]
        if not props:
            n_skip += 1
            continue
        c = crop_for(props)
        crops[fid] = c
        items.append((fid, c))
    print(f"{len(fast)} frames: {len(items)} escalated to the slow head, {n_skip} answered [] by the fast head", flush=True)

    dev = torch.device("cuda")
    model, proc, fusion = load_slow_head(a.model, a.mode, adapter_dir=a.slow, device=dev)
    model.eval()
    dl = DataLoader(ZoomInfer(items, a.fused_dir, proc, a.mode), batch_size=a.bs, num_workers=a.workers,
                    collate_fn=lambda b: infer_collate(b, proc))
    slow, t0 = {}, time.time()
    with torch.no_grad():
        for batch in dl:
            ids = batch.pop("ids"); th = batch.pop("thermal").to(dev); li = batch.pop("lidar", None)
            batch = {k: v.to(dev) for k, v in batch.items()}
            if fusion is not None:
                fusion.set_thermal(th)
                fusion.set_extra({"lidar": li.to(dev)} if (li is not None and "lidar" in fusion.extra) else {})
            out = model.generate(**batch, max_new_tokens=640, do_sample=False)
            texts = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            for fid, txt in zip(ids, texts):
                c = crops[fid] or (0, 0, FW, FH)
                cw, ch = c[2] - c[0], c[3] - c[1]
                ps = []
                for o in DT.parse_report(txt):
                    b = T.from_qwen([o["box"]], cw, ch)[0]
                    ps.append({"box": [b[0] + c[0], b[1] + c[1], b[2] + c[0], b[3] + c[1]], "pose": o["pose"],
                               "distance": o["distance"]})
                slow[fid] = {"text": txt, "persons": ps, "crop": c}
            print(f"progress {len(slow)}/{len(items)} {len(slow) / (time.time() - t0):.2f} img/s", flush=True)

    with open(os.path.join(a.out, "pred_test.jsonl"), "w") as f1, open(os.path.join(a.out, "pred_test_union.jsonl"), "w") as f2:
        for fid, ps in fast.items():
            s = slow.get(fid, {"persons": [], "text": "[] (fast head: no proposal)", "crop": None})
            f1.write(json.dumps({"id": fid, "task": "report", "cond": "normal", "text": s["text"], "persons": s["persons"],
                                 "crop": s["crop"]}) + "\n")
            union = list(s["persons"])
            for p in ps:                                   # confident fast detections the slow head did not report
                if p["score"] >= keep_thr and (not union or T.iou_matrix([p["box"]], [u["box"] for u in union]).max() < 0.3):
                    union.append({"box": p["box"], "pose": p["pose"], "distance": p["distance"]})
            f2.write(json.dumps({"id": fid, "task": "report", "cond": "normal", "persons": union}) + "\n")
    json.dump({"frames": len(fast), "escalated": len(items), "fast_only": n_skip, "prop_thr": a.prop_thr,
               "slow_seconds": time.time() - t0}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("ZOOM_DONE", flush=True)


if __name__ == "__main__":
    main()
