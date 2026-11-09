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


def wandb_group(name: str) -> str:
    """Experiment family of a run directory name, used as the W&B group."""
    for pre, g in (("t_sft_", "trisar-slow-head"), ("t_zs", "trisar-slow-head"), ("tf_", "trisar-fast-head"),
                   ("ltd_", "ltdv2-site-adaptation"), ("fast_", "llvip-fast-head"), ("sft_", "llvip-slow-head"),
                   ("zs_", "llvip-slow-head"), ("mem_", "site-memory")):
        if name.startswith(pre):
            return g
    return "other"


def wandb_run(out: str, cfg: dict | None = None, project="nightwatch"):
    """Start or resume the W&B run that belongs to a run directory (id kept in <out>/wandb_run.txt). Never raises."""
    try:
        import wandb
        name = os.path.basename(out.rstrip("/"))
        idf = os.path.join(out, "wandb_run.txt")
        rid = open(idf).read().strip() if os.path.exists(idf) else __import__("uuid").uuid4().hex[:12]
        run = wandb.init(project=project, name=name, id=rid, resume="allow", group=wandb_group(name),
                         config=cfg or {}, dir=out, settings=wandb.Settings(silent=True))
        open(idf, "w").write(rid)
        return run
    except Exception as e:  # never let tracking kill a run
        print("wandb disabled:", e)
        return None


class RunLogger:
    """jsonl log next to the checkpoint (always) + optional W&B. Config and git rev saved for reproducibility."""

    def __init__(self, out: str, cfg: dict, use_wandb=None, project="nightwatch"):
        os.makedirs(out, exist_ok=True)
        self.fh = open(os.path.join(out, "log.jsonl"), "a")
        meta = {"config": cfg, "git": git_rev(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__}
        json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
        self.wb = None
        if use_wandb is None:      # on by default whenever W&B credentials exist; NW_WANDB=0 turns it off
            use_wandb = os.environ.get("NW_WANDB", "1") != "0"
        if use_wandb:
            self.wb = wandb_run(out, cfg, project)

    def log(self, rec: dict):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if self.wb:
            self.wb.log(rec, step=rec.get("step"))
            self.n_synced = getattr(self, "n_synced", 0) + 1     # lets nightwatch.wandb_sync skip rows already sent
            open(os.path.join(os.path.dirname(self.fh.name), "wandb_synced.txt"), "w").write(str(self.n_synced))

    def close(self):
        self.fh.close()
        if self.wb:
            self.wb.finish()
