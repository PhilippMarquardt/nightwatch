#!/usr/bin/env bash
# Same rerun, seed 1, on pod 1 after its LiDAR-off job.
cd "$(dirname "$0")/.."
until grep -q "LIDAROFF_DONE t_sft_tri_s1" logs/lidaroff.log 2>/dev/null; do sleep 60; done
VIS_LR_MULT=0.2 TRAIN_FRAC=0.9 EVAL_FRAC=0.9 MODE=tri SEED=1 bash scripts/run_tri_vl.sh
