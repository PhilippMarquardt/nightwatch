#!/usr/bin/env bash
# Track B, after the 3-wide driver hit the pod's 188 GB container memory limit (3 x ~60 GB -> OOM-killed, 11:47).
# Waits for the old driver to stop (STOP_ltdadapt), then runs every missing arm two at a time. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_FAST_CPU=1
while ! grep -q LTD_ADAPT_DONE logs/ltdadapt.log 2>/dev/null; do sleep 30; done
rm -f STOP_ltdadapt
M="python -m nightwatch.fast.ltd_adapt"
arm() {
  local r=runs/ltd_$1_s$3
  [ -s $r/train_summary.json ] || { rm -rf $r; NW_GPU_FRAC=0.12 $M train --arm $1 --labels $2 --seed $3 > logs/ltd_$1_s$3.log 2>&1; }
  [ -s $r/train_summary.json ] || { echo "ARM_FAILED $1 $3"; return 0; }
  [ -s $r/ltd_test_metrics.json ] || NW_GPU_FRAC=0.12 $M eval --run $r >> logs/ltd_$1_s$3.log 2>&1
}
JOBS=()
for s in 0 1 2; do
  for spec in "B0p none" "B1_self runs/ltd/labels_self.json" "B2_vlm runs/ltd/labels_vlm.json" \
              "B3_consensus runs/ltd/labels_consensus.json" "B4_gt runs/ltd/labels_gt.json"; do
    set -- $spec; JOBS+=("$1 $2 $s")
  done
done
i=0
for j in "${JOBS[@]}"; do
  set -- $j; arm $1 $2 $3 &
  i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
done
wait
echo LTD_ADAPT2_DONE
