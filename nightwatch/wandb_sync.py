"""Mirror run directories to Weights & Biases (project "nightwatch"), next to the local jsonl logs.

- training curve: every record of <run>/log.jsonl, logged once (progress kept in <run>/wandb_synced.txt)
- results: every <run>/*_metrics.json, flattened into the run summary (e.g. report/normal/loc_forest/f1)
- the W&B run id lives in <run>/wandb_run.txt, so RunLogger, this tool and later metric uploads share one run

python -m nightwatch.wandb_sync runs/t_sft_tri                  # backfill / upload metrics once
python -m nightwatch.wandb_sync runs/t_sft_tri_vl0.2_s1 --follow # stream a run that is training now
python -m nightwatch.wandb_sync --all                           # every run directory with a log or metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

from nightwatch.utils import wandb_run


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}/{k}" if pre else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[key] = v
    return out


def records(run_dir):
    p = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass                       # a line being written right now
    return rows


def push_metrics(run, run_dir):
    for f in sorted(glob.glob(os.path.join(run_dir, "*_metrics.json")) + glob.glob(os.path.join(run_dir, "train_summary.json"))):
        tag = os.path.basename(f).replace("_metrics.json", "").replace(".json", "")
        try:
            run.summary.update(flatten(json.load(open(f)), tag))
        except Exception as e:
            print("skip", f, e)


def sync(run_dir, follow=False, poll=30, idle_exit=1800):
    meta = os.path.join(run_dir, "run_meta.json")
    cfg = json.load(open(meta)).get("config", {}) if os.path.exists(meta) else {}
    run = wandb_run(run_dir, cfg)
    if run is None:
        return
    donef = os.path.join(run_dir, "wandb_synced.txt")
    done = int(open(donef).read()) if os.path.exists(donef) else 0
    last_new = time.time()
    while True:
        rows = records(run_dir)
        for i, r in enumerate(rows[done:], start=done):
            step = r.get("step", i)
            run.log({k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, step=step)
        if len(rows) > done:
            done = len(rows); last_new = time.time()
            open(donef, "w").write(str(done))
        push_metrics(run, run_dir)
        finished = os.path.exists(os.path.join(run_dir, "train_summary.json"))
        if not follow or finished or time.time() - last_new > idle_exit:
            break
        time.sleep(poll)
    run.finish()
    print("synced", run_dir, done, "log rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    dirs = a.runs
    if a.all:
        dirs = sorted({os.path.dirname(p) for p in glob.glob("runs/*/log.jsonl") + glob.glob("runs/*/*_metrics.json")})
        live = lambda d: (not os.path.exists(f"{d}/train_summary.json") and os.path.exists(f"{d}/log.jsonl")
                          and time.time() - os.path.getmtime(f"{d}/log.jsonl") < 600)
        dirs = [d for d in dirs if not live(d)]          # runs training right now get their own --follow process
    for d in dirs:
        sync(d, follow=a.follow)


if __name__ == "__main__":
    main()
