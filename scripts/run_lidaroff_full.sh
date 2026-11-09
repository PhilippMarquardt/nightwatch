#!/usr/bin/env bash
# Full-test LiDAR-off scoring of three-image slow heads (review #18): is the tri > rgbt_i gain read from LiDAR at test time?
#   WAIT="<log>:<marker>" RUNS="t_sft_tri t_sft_tri_s2" bash scripts/run_lidaroff_full.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NW_GPU_FRAC=${EVAL_FRAC:-0.85}
if [ -n "${WAIT:-}" ]; then
  until grep -q "${WAIT#*:}" "${WAIT%%:*}" 2>/dev/null; do sleep 60; done
fi
for r in $RUNS; do
  OUT=runs/$r
  [ -s $OUT/pred_test_lidaroff_full.jsonl ] || python -m nightwatch.slow.infer --dataset trisar --fused-dir data/trisar_fused \
      --model Qwen/Qwen3.5-9B --bs 32 --workers 12 --mode tri --adapter $OUT --split test --tasks report --conds lidar_off \
      --out $OUT/pred_test_lidaroff_full.jsonl
  python -m nightwatch.eval.metrics_trisar --pred $OUT/pred_test_lidaroff_full.jsonl --out $OUT/pred_test_lidaroff_full_metrics.json > /dev/null
  python -m nightwatch.wandb_sync $OUT > /dev/null 2>&1 || true
  echo "LIDAROFF_DONE $r"
done
