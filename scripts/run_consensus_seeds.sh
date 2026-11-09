#!/usr/bin/env bash
# Seeds 1 and 2 of the consensus student, after the seed lane is done so memory stays bounded.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${NW_GPU_FRAC:-0.12} NW_FAST_CPU=1
while [ ! -f STOP_consensus ] && ! grep -q SEEDS_DONE logs/seeds.log 2>/dev/null; do sleep 120; done
for s in 1 2; do
  [ -f STOP_consensus ] && break
  n=fast_distill_consensus_s$s
  [ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.train --name $n --epochs 40 --splits labeled,unlabeled \
      --labels runs/pseudo_labels_unlabeled_consensus.json --seed $s
  thr=$(python -c "import json;print(json.load(open('runs/$n/train_summary.json'))['score_thr'])")
  python -m nightwatch.eval.metrics --pred runs/$n/pred_test.jsonl --score-thr $thr --out runs/$n/pred_test_metrics.json > /dev/null
done
echo CONSENSUS_SEEDS_DONE
