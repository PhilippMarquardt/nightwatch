#!/usr/bin/env bash
# Multi-image TriSAR slow heads (2026-11-09 follow-up, docs/PLAN_SITE_ADAPT.md v2).
#   MODE=tri|rgbt_i|thermal_i  SEED=0  bash scripts/run_tri.sh
# Same recipe as cd0 (1 epoch, BS4/ACC4, lr defaults); sensors enter the ViT as real images at 640x512.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE=${MODE:?set MODE}; SEED=${SEED:-0}
TAG=$( [ "$SEED" = 0 ] && echo "" || echo "_s$SEED" )
OUT=runs/t_sft_${MODE}${TAG}
M=Qwen/Qwen3.5-9B; FUSED=data/trisar_fused
if [ ! -s $OUT/train_summary.json ]; then
  NW_GPU_FRAC=${TRAIN_FRAC:-0.9} python -m nightwatch.slow.train --dataset trisar --fused-dir $FUSED --model $M --mode $MODE \
      --splits train --epochs 1 --bs ${BS:-4} --grad-accum ${ACC:-4} --workers 12 --grad-ckpt --seed $SEED --out $OUT
fi
[ -s $OUT/train_summary.json ] || { echo "TRI_TRAIN_FAILED $MODE $SEED"; exit 1; }
INF="python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $M --bs ${IBS:-32} --workers 12 --mode $MODE --adapter $OUT --split test"
export NW_GPU_FRAC=${EVAL_FRAC:-0.9}
[ -s $OUT/pred_test.jsonl ]         || $INF --tasks report --conds normal --out $OUT/pred_test.jsonl
[ -s $OUT/pred_test_robust.jsonl ]  || $INF --tasks report --conds rgb_off,thermal_off,lidar_off,sensors_shuffle --limit 600 \
    --out $OUT/pred_test_robust.jsonl
[ -s $OUT/pred_test_qa.jsonl ]      || $INF --tasks lying,nearest --conds normal --limit 600 --out $OUT/pred_test_qa.jsonl
for f in pred_test pred_test_robust pred_test_qa; do
  python -m nightwatch.eval.metrics_trisar --pred $OUT/$f.jsonl --out $OUT/${f}_metrics.json > /dev/null
done
echo "TRI_DONE $MODE $SEED"
