#!/usr/bin/env bash
# tri_pc seed 1 on pod 1, after the vl0.2 s1 LiDAR-off job.
cd "$(dirname "$0")/.."
until grep -q "LIDAROFF_DONE t_sft_tri_vl0.2_s1" logs/lidaroffvl1.log 2>/dev/null; do sleep 30; done
TRAIN_FRAC=0.9 EVAL_FRAC=0.9 MODE=tri_pc SEED=1 bash scripts/run_tri.sh
