#!/usr/bin/env bash
# Train-time thermal normalisation for the TriSAR fast head (3 seeds) + router per seed. Starts after the first seed
# batch so it never runs concurrently with run_seeds.sh training (memory caps). STOP_tnorm cuts it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${NW_GPU_FRAC:-0.12} NW_FAST_CPU=1
while [ ! -f STOP_tnorm ] && [ ! -s runs/fast_gt_labeled_s2/train_summary.json ]; do sleep 60; done
thr_of() { python -c "import json;print(json.load(open('runs/$1/train_summary.json'))['score_thr'])"; }
for s in 0 1 2; do
  [ -f STOP_tnorm ] && break
  n=tf_rgbtl_tnorm_s$s
  [ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.trisar --name $n --modal rgbtl --fused-dir data/trisar_fused \
      --thermal-norm --seed $s
  [ -s runs/$n/train_summary.json ] && python -m nightwatch.eval.metrics_trisar --pred runs/$n/pred_test.jsonl \
      --score-thr $(thr_of $n) --out runs/$n/pred_test_metrics.json > /dev/null
done
for f in tf_rgbtl_s1 tf_rgbtl_s2 tf_rgbtl_tnorm_s0 tf_rgbtl_tnorm_s1 tf_rgbtl_tnorm_s2; do
  [ -s runs/$f/pred_test.jsonl ] && python -m nightwatch.eval.router_trisar --fast $f --tag _$f > /dev/null
done
echo TNORM_DONE
