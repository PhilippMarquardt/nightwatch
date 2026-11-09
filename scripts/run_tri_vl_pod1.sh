#!/usr/bin/env bash
# Three-image TriSAR VLM with LoRA on the vision encoder too (NW_VISION_LORA=1): can an adapted ViT read the LiDAR range
# image? Pre-registered test (stated before the run, 2026-11-09 12:35): LiDAR-off lab or forest F1 drop >= 0.03.
# Pod 1, after tri seed 1 finishes training. Retries once at BS2/ACC8 if BS4 OOMs.
cd "$(dirname "$0")/.."
while [ ! -s runs/t_sft_tri_s1/train_summary.json ]; do sleep 60; done
TRAIN_FRAC=0.55 EVAL_FRAC=0.45 MODE=tri SEED=0 bash scripts/run_tri_vl.sh
if [ ! -s runs/t_sft_tri_vl/train_summary.json ]; then
  rm -rf runs/t_sft_tri_vl
  TRAIN_FRAC=0.55 EVAL_FRAC=0.45 BS=2 ACC=8 MODE=tri SEED=0 bash scripts/run_tri_vl.sh
fi
echo TRI_VL_DONE
