#!/usr/bin/env bash
# One seed of the compute-matched labelled-only student.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${NW_GPU_FRAC:-0.12} NW_FAST_CPU=1
s=$1; n=fast_gt_labeled_long_s$s
[ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.train --name $n --epochs $(( 40 * 11602 / 3438 )) --splits labeled --eval-every 17 --seed $s
thr=$(python -c "import json;print(json.load(open('runs/$n/train_summary.json'))['score_thr'])")
python -m nightwatch.eval.metrics --pred runs/$n/pred_test.jsonl --score-thr $thr --out runs/$n/pred_test_metrics.json > /dev/null
echo LONG_SEED_${s}_DONE
