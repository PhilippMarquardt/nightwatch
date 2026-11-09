#!/usr/bin/env bash
# Vision-LoRA rerun with the ViT LoRA at 0.2x lr (vision-LoRA audit: 94 % of steps were clipped at 1x). Pod 2, after the LiDAR-off queue.
cd "$(dirname "$0")/.."
until grep -q "LIDAROFF_DONE t_sft_tri_s2" logs/lidaroff.log 2>/dev/null; do sleep 60; done
VIS_LR_MULT=0.2 TRAIN_FRAC=0.9 EVAL_FRAC=0.9 MODE=tri SEED=0 bash scripts/run_tri_vl.sh
