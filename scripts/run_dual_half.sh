#!/usr/bin/env bash
# Review #11: token-matched dual. Same recipe as sft_dual (BS2/ACC8, 2 epochs, 428 steps) but each image resized to
# 672x544 (357 merged tokens each, ~714 total = the fused head's 720). Decides whether the side-branch fusion's only
# remaining advantage (half the visual tokens) is real. Also runs the missing reasoning eval for full-res sft_dual.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f STOP_dualhalf ] && [ ! -s runs/t_sft_fusion_l_cd0/train_summary.json ]; do sleep 60; done
[ -f STOP_dualhalf ] && exit 0
M=Qwen/Qwen3.5-9B
INF="python -m nightwatch.slow.infer --model $M --bs 32 --workers 12"
# full-res dual: reasoning tasks (missing from the overnight eval)
[ -s runs/sft_dual/pred_test_reason.jsonl ] || NW_GPU_FRAC=0.4 $INF --mode dual --adapter runs/sft_dual --split test \
    --tasks count,zone_left,zone_center,zone_right --conds normal --limit 800 --out runs/sft_dual/pred_test_reason.jsonl
python -m nightwatch.eval.metrics --pred runs/sft_dual/pred_test_reason.jsonl --out runs/sft_dual/pred_test_reason_metrics.json > /dev/null
export NW_DUAL_HALF=1
OUT=runs/sft_dual_half
[ -s $OUT/train_summary.json ] || NW_GPU_FRAC=0.5 python -m nightwatch.slow.train --model $M --mode dual --splits labeled \
    --epochs 2 --bs 2 --grad-accum 8 --workers 16 --grad-ckpt --out $OUT
[ -s $OUT/train_summary.json ] || { echo "DUAL_HALF_TRAIN_FAILED"; exit 1; }
A="--mode dual --adapter $OUT --split test"
[ -s $OUT/pred_test.jsonl ] || NW_GPU_FRAC=0.6 $INF $A --tasks detect --conds normal --limit 1500 --out $OUT/pred_test.jsonl
[ -s $OUT/pred_test_robust.jsonl ] || NW_GPU_FRAC=0.6 $INF $A --tasks detect --conds rgb_off,thermal_off --limit 800 \
    --out $OUT/pred_test_robust.jsonl
for f in pred_test pred_test_robust; do
  python -m nightwatch.eval.metrics --pred $OUT/$f.jsonl --out $OUT/${f}_metrics.json > /dev/null
done
echo DUAL_HALF_DONE
