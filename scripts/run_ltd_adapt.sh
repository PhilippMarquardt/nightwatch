#!/usr/bin/env bash
# Track B driver: labels -> 5 arms x 3 seeds (3 in parallel) -> eval incl. zero-shot start model. STOP_ltdadapt cuts it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NW_FAST_CPU=1
while ! grep -q CACHES_DONE logs/caches.log 2>/dev/null; do sleep 30; done
M="python -m nightwatch.fast.ltd_adapt"
for k in gt self self_lo; do [ -s runs/ltd/labels_$k.json ] || NW_GPU_FRAC=0.15 $M label --kind $k; done
[ -s runs/ltd/labels_vlm.json ] || NW_GPU_FRAC=0.4 $M label --kind vlm
[ -s runs/ltd/labels_consensus.json ] || $M label --kind consensus
[ -s runs/fast_oracle_t/ltd_test_metrics.json ] || NW_GPU_FRAC=0.15 $M eval --run runs/fast_oracle_t      # B0 zero-shot
arm() {  # arm labels seed
  local r=runs/ltd_$1_s$3
  [ -f STOP_ltdadapt ] && return 0
  [ -s $r/train_summary.json ] || NW_GPU_FRAC=0.12 $M train --arm $1 --labels $2 --seed $3 > logs/ltd_$1_s$3.log 2>&1
  [ -s $r/ltd_test_metrics.json ] || NW_GPU_FRAC=0.12 $M eval --run $r >> logs/ltd_$1_s$3.log 2>&1
}
for s in 0 1 2; do
  arm B0p none $s &
  arm B1_self runs/ltd/labels_self.json $s &
  arm B2_vlm runs/ltd/labels_vlm.json $s &
  wait
  arm B3_consensus runs/ltd/labels_consensus.json $s &
  arm B4_gt runs/ltd/labels_gt.json $s &
  wait
done
echo LTD_ADAPT_DONE
