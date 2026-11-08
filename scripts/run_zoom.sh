#!/usr/bin/env bash
# Zoom cascade on TriSAR test: waits for the fast head, runs with a small memory cap next to the two lanes.
set -uo pipefail
cd "$(dirname "$0")/.."
while [ ! -s runs/tf_rgbtl/train_summary.json ] || [ ! -s runs/t_sft_fusion_l/train_summary.json ]; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.30
[ -s runs/t_zoom/pred_test.jsonl ] || python -m nightwatch.eval.zoom_cascade --fast runs/tf_rgbtl --slow runs/t_sft_fusion_l \
    --mode fusion_l --bs 24 --workers 12 --out runs/t_zoom
for f in runs/t_zoom/pred_test.jsonl runs/t_zoom/pred_test_union.jsonl; do
  python -m nightwatch.eval.metrics_trisar --pred $f --out ${f%.jsonl}_metrics.json
done
echo ZOOM_ALL_DONE
