"""Batched slow-head inference -> predictions jsonl (boxes in 1280x1024 px).

python -m nightwatch.slow.infer --mode fusion --adapter runs/slow_fusion --split test --out runs/slow_fusion/pred_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

from nightwatch.data import tasks as T
from nightwatch.data.llvip import H, W, load_index
from nightwatch.slow.data import InferDataset, infer_collate
from nightwatch.slow.model import load_slow_head
from nightwatch.utils import gpu_mem_cap

TASK_PROMPTS = {
    "detect": (T.EVAL_DETECT_PROMPT, 512),
    "count": (T.COUNT_PROMPTS[0], 8),
    "zone_left": (T.ZONE_PROMPTS[0].format(zone="left"), 4),
    "zone_center": (T.ZONE_PROMPTS[0].format(zone="center"), 4),
    "zone_right": (T.ZONE_PROMPTS[0].format(zone="right"), 4),
}


TRISAR_TASKS = {"report": 640, "lying": 4, "nearest": 8, "count": 8}


@torch.no_grad()
def run(model, proc, fusion, recs, root, mode, task, cond, bs, workers, out_fh, dev, dataset="llvip", fused_dir=None):
    if dataset == "trisar":
        from nightwatch.slow import data_trisar as DT
        prompt = {"report": DT.REPORT_PROMPT, "lying": DT.LYING_PROMPT, "nearest": DT.NEAREST_PROMPT,
                  "count": DT.COUNT_PROMPT}[task]
        max_new = TRISAR_TASKS[task]
        ds = DT.TriSARInfer(recs, fused_dir, proc, mode, prompt, cond)
    else:
        prompt, max_new = TASK_PROMPTS[task]
        ds = InferDataset(recs, root, proc, mode, prompt, cond)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=partial(infer_collate, proc=proc))
    n, t0 = 0, time.time()
    for batch in dl:
        ids = batch.pop("ids")
        th = batch.pop("thermal").to(dev)
        li = batch.pop("lidar", None)
        batch = {k: v.to(dev) for k, v in batch.items()}
        if fusion is not None:
            fusion.set_thermal(th)
            fusion.set_extra({"lidar": li.to(dev)} if (li is not None and "lidar" in fusion.extra) else {})
        out = model.generate(**batch, max_new_tokens=max_new, do_sample=False)
        texts = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for iid, txt in zip(ids, texts):
            rec = {"id": iid, "task": task, "cond": cond, "text": txt}
            if task == "detect":
                rec["boxes"] = T.from_qwen(T.parse_boxes(txt), W, H)
            elif task == "report":
                from nightwatch.slow.data_trisar import FH, FW, parse_report
                rec["persons"] = [{"box": T.from_qwen([o["box"]], FW, FH)[0], "pose": o["pose"], "distance": o["distance"]}
                                  for o in parse_report(txt)]
            out_fh.write(json.dumps(rec) + "\n")
        n += len(ids)
        out_fh.flush()
        nb = n // max(1, len(ids))
        if nb % 10 == 0:
            print(f"progress {task}/{cond}: {n}/{len(ds)} {n / max(time.time() - t0, 1e-6):.2f} img/s", flush=True)
    dt = time.time() - t0
    print(f"{task}/{cond}: {n} imgs in {dt:.0f}s ({n / max(dt, 1e-6):.2f} img/s)", flush=True)
    return {"task": task, "cond": cond, "n": n, "seconds": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mode", choices=["rgb", "thermal", "dual", "fusion", "fusion_l", "thermal_i", "rgbt_i", "tri"], required=True)
    ap.add_argument("--dataset", choices=["llvip", "trisar"], default="llvip")
    ap.add_argument("--fused-dir", default="data/trisar_fused")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--root", default="data/raw/LLVIP")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="detect")
    ap.add_argument("--conds", default="normal")
    ap.add_argument("--limit", type=int, default=0, help="random subset (fixed seed) for expensive conditions")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gpu_mem_cap()
    dev = torch.device("cuda")
    model, proc, fusion = load_slow_head(a.model, a.mode, adapter_dir=a.adapter, device=dev)
    model.eval()
    if fusion is not None:
        fusion.eval()
        print("gates", fusion.gate_values())
    if a.dataset == "trisar":
        from nightwatch.slow.data_trisar import load_frames
        recs = load_frames(a.fused_dir, set(a.split.split(",")))
        if a.limit:
            recs = sorted(random.Random(0).sample(recs, min(a.limit, len(recs))), key=lambda r: r["id"])
    else:
        recs = load_index(a.index, tuple(a.split.split(",")))
        if a.limit:
            recs = sorted(random.Random(0).sample(recs, min(a.limit, len(recs))), key=lambda r: r.id)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    timings = []
    tmp = a.out + ".tmp"            # atomic: a crashed run never leaves a "finished-looking" partial file
    with open(tmp, "w") as fh:
        for task in a.tasks.split(","):
            for cond in a.conds.split(","):
                if (a.mode == "rgb" and cond in ("thermal_off", "lidar_off")) or (a.mode == "fusion" and cond == "lidar_off") \
                        or (a.mode == "thermal" and cond in ("rgb_off", "lidar_off", "thermal_shuffle")) \
                        or (a.mode == "rgb" and cond in ("thermal_shuffle", "sensors_shuffle")) \
                        or (a.mode == "thermal_i" and cond in ("rgb_off", "lidar_off", "sensors_shuffle")) \
                        or (a.mode == "rgbt_i" and cond == "lidar_off"):
                    continue
                timings.append(run(model, proc, fusion, recs, a.root, a.mode, task, cond, a.bs, a.workers, fh, dev,
                                   a.dataset, a.fused_dir))
    os.replace(tmp, a.out)
    json.dump(timings, open(a.out.replace(".jsonl", "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
