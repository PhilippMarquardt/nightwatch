from __future__ import annotations

import json
import os
import random
import subprocess
import time

import numpy as np
import torch


def gpu_mem_cap():
    """NW_GPU_FRAC=0.47 caps this process's share of GPU memory so two jobs can share one GPU safely."""
    frac = os.environ.get("NW_GPU_FRAC")
    if frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"gpu memory cap: {float(frac):.2f} of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
              flush=True)


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


class RunLogger:
    """jsonl log next to the checkpoint (always) + optional W&B. Config and git rev saved for reproducibility."""

    def __init__(self, out: str, cfg: dict, use_wandb=False, project="nightwatch"):
        os.makedirs(out, exist_ok=True)
        self.fh = open(os.path.join(out, "log.jsonl"), "a")
        meta = {"config": cfg, "git": git_rev(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__}
        json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
        self.wb = None
        if use_wandb:
            try:
                import wandb
                self.wb = wandb.init(project=project, name=os.path.basename(out.rstrip("/")), config=cfg, dir=out)
            except Exception as e:  # never let tracking kill a run
                print("wandb disabled:", e)

    def log(self, rec: dict):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if self.wb:
            self.wb.log(rec, step=rec.get("step"))

    def close(self):
        self.fh.close()
        if self.wb:
            self.wb.finish()
