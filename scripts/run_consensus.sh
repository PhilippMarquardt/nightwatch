#!/usr/bin/env bash
# Consensus-filtered distillation student (review-free addition, 02:55): teacher labels confirmed by the labelled-only
# fast head. Runs next to the lanes with a small memory cap. STOP_consensus cuts it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${NW_GPU_FRAC:-0.12} NW_FAST_CPU=1
[ -f STOP_consensus ] && exit 0
[ -s runs/pseudo_labels_unlabeled_self_lo.json ] || python -m nightwatch.fast.label_split --run runs/fast_gt_labeled \
    --split unlabeled --thr 0.15 --out runs/pseudo_labels_unlabeled_self_lo.json
python -m nightwatch.fast.consensus --teacher runs/pseudo_labels_unlabeled.json --student runs/pseudo_labels_unlabeled_self_lo.json \
    --out runs/pseudo_labels_unlabeled_consensus.json --report
n=fast_distill_consensus     # same recipe as run_fast.sh's distill variant; called directly (run_fast.sh is live in lane B)
[ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.train --name $n --epochs 40 --splits labeled,unlabeled     --labels runs/pseudo_labels_unlabeled_consensus.json
thr=$(python -c "import json;print(json.load(open('runs/$n/train_summary.json'))['score_thr'])")
python -m nightwatch.eval.metrics --pred runs/$n/pred_test.jsonl --score-thr $thr --out runs/$n/pred_test_metrics.json > /dev/null
[ -s runs/sft_fusion/pred_test.jsonl ] && python -m nightwatch.eval.metrics --pred runs/$n/pred_test.jsonl --score-thr $thr     --ids-from runs/sft_fusion/pred_test.jsonl --out runs/$n/pred_test_sub1500_metrics.json > /dev/null
echo CONSENSUS_DONE
