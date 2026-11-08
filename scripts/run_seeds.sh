#!/usr/bin/env bash
# Stable baselines: extra seeds of the fast heads behind the headline numbers, so every comparison has a spread.
# Runs next to lanes A/B with a small GPU cap (fast heads are ResNet18 CenterNets, ~10 GB). Resumable; STOP_seeds cuts it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${NW_GPU_FRAC:-0.12} NW_FAST_CPU=1
SEEDS=${SEEDS:-"1 2"}
on() { [ ! -f STOP_seeds ]; }
thr_of() { python -c "import json;print(json.load(open('runs/$1/train_summary.json'))['score_thr'])"; }
llvip() {  # name, args...
  local n=$1; shift
  on || return 0
  [ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.train --name $n --epochs 40 "$@"
  [ -s runs/$n/train_summary.json ] && python -m nightwatch.eval.metrics --pred runs/$n/pred_test.jsonl --score-thr $(thr_of $n) \
      --out runs/$n/pred_test_metrics.json > /dev/null
}
trisar() {  # name, args...
  local n=$1; shift
  on || return 0
  [ -s runs/$n/train_summary.json ] || python -m nightwatch.fast.trisar --name $n --fused-dir data/trisar_fused "$@"
  [ -s runs/$n/train_summary.json ] && python -m nightwatch.eval.metrics_trisar --pred runs/$n/pred_test.jsonl \
      --score-thr $(thr_of $n) --out runs/$n/pred_test_metrics.json > /dev/null
}
for s in $SEEDS; do
  trisar tf_rgbtl_s$s --modal rgbtl --seed $s          # TriSAR headline fast head (forest F1 0.31)
  llvip fast_gt_labeled_s$s --splits labeled --seed $s  # LLVIP distillation baseline
done
# distilled student + self-training control seeds, once their pseudo-labels exist (lane B, steps B5 / B6b)
while on && [ ! -s runs/fast_distill/train_summary.json ]; do sleep 120; done
for s in $SEEDS; do
  llvip fast_distill_s$s --splits labeled,unlabeled --labels runs/pseudo_labels_unlabeled.json --seed $s
done
while on && [ ! -s runs/pseudo_labels_unlabeled_self.json ]; do sleep 120; done
for s in $SEEDS; do
  llvip fast_selfdistill_s$s --splits labeled,unlabeled --labels runs/pseudo_labels_unlabeled_self.json --seed $s
done
echo SEEDS_DONE
