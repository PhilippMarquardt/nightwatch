#!/usr/bin/env bash
# second seed of the no-camera-dropout fused TriSAR VLM (cd0). Same recipe as cd0 (BS4/ACC4, 1 epoch,
# drop_rgb 0); train.py called directly so run_trisar.sh (live in cd0c) is not edited. Starts after CD0_DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f STOP_cd0s1 ] && ! grep -q CD0_DONE logs/cd0c.log 2>/dev/null; do sleep 60; done
[ -f STOP_cd0s1 ] && exit 0
OUT=runs/t_sft_fusion_l_cd0_s1
[ -s $OUT/train_summary.json ] || NW_GPU_FRAC=0.55 python -m nightwatch.slow.train --dataset trisar --fused-dir data/trisar_fused \
    --model Qwen/Qwen3.5-9B --mode fusion_l --splits train --epochs 1 --bs 4 --grad-accum 4 --workers 12 --grad-ckpt \
    --drop-rgb 0 --seed 1 --out $OUT
[ -s $OUT/train_summary.json ] || { echo "CD0S1_TRAIN_FAILED"; exit 1; }
NW_GPU_FRAC=0.45 IBS=32 RUN_TAG=_cd0_s1 MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh
echo CD0S1_DONE
