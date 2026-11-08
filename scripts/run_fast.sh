#!/usr/bin/env bash
# Fast-head experiments. Needs runs/pseudo_labels_unlabeled.json for the distilled variant
# (produced by `run_slow.sh` stage "pseudo"). Everything else can start immediately.
#   VARIANTS="oracle gt_labeled" bash scripts/run_fast.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${FAST_EPOCHS:-40}
VARIANTS=${VARIANTS:-"gt_labeled oracle oracle_rgb oracle_t distill"}

[ -f data/cache_fast/test.npy ] || python -m nightwatch.fast.data --out data/cache_fast --workers ${WORKERS:-16}

run() {  # name, extra args...
  local name=$1; shift
  [ -s runs/$name/train_summary.json ] || python -m nightwatch.fast.train --name $name --epochs $EPOCHS "$@"
  python -m nightwatch.fast.export --name $name > /dev/null 2>&1 || echo "export failed for $name (see runs/$name/deploy.json)"
  thr=$(python -c "import json;print(json.load(open('runs/$name/train_summary.json'))['score_thr'])")
  python -m nightwatch.eval.metrics --pred runs/$name/pred_test.jsonl --score-thr $thr --out runs/$name/pred_test_metrics.json
}

for v in ${VARIANTS//,/ }; do   # space- or comma-separated
  case $v in
    gt_labeled)  run fast_gt_labeled --splits labeled ;;                       # what the labels alone buy
    oracle)      run fast_oracle     --splits labeled,unlabeled ;;             # upper bound: every site hand-labeled
    oracle_rgb)  run fast_oracle_rgb --splits labeled,unlabeled --modal rgb ;; # sensor ablation
    oracle_t)    run fast_oracle_t   --splits labeled,unlabeled --modal t ;;
    distill)     run fast_distill    --splits labeled,unlabeled --labels runs/pseudo_labels_unlabeled.json ;;  # slow head labels the rest
  esac
done
echo "FAST DONE"
