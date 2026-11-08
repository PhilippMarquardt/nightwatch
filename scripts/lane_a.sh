#!/usr/bin/env bash
# Lane A (overnight): slow-head SFT runs in series. Resumable; cut a step with `touch STOP_<name>`.
# Priority (after review): fusion > rgb > thermal-only control > TriSAR rgb > dual > TriSAR rgb+t.
set -uo pipefail                      # no -e: one failed step must not kill the lane
cd "$(dirname "$0")/.."
mkdir -p logs runs; exec 9>logs/laneA.lock; flock -n 9 || { echo "lane A already running"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.47 WORKERS=16 BS=4 ACC=4
wait_for() { while [ ! -s "$1" ]; do [ -n "${2:-}" ] && [ -f "STOP_$2" ] && return 1; sleep 60; done; }
on() { [ ! -f "STOP_$1" ]; }
echo "lane A start $(date)"
while pgrep -f "nightwatch.slow.train" > /dev/null; do sleep 60; done     # hand-over: let a running SFT finish
wait_for runs/t_sft_fusion_l/train_summary.json
EPOCHS=2 MODES=fusion  STAGES=sft bash scripts/run_slow.sh                                  # A1 LLVIP fusion (ours)
EPOCHS=2 MODES=rgb     STAGES=sft bash scripts/run_slow.sh                                  # A2 LLVIP RGB only
EPOCHS=2 MODES=thermal STAGES=sft bash scripts/run_slow.sh                                  # A3 LLVIP thermal only (control)
EPOCHS=1 MODES=rgb     STAGES=sft bash scripts/run_trisar.sh                                # A4 TriSAR RGB only
on dual     && BS=2 ACC=8 EPOCHS=2 MODES=dual STAGES=sft bash scripts/run_slow.sh           # A5 LLVIP thermal as 2nd image
on t_fusion && EPOCHS=1 MODES=fusion STAGES=sft bash scripts/run_trisar.sh                  # A6 TriSAR RGB+T (middle rung)
on dual     && NW_GPU_FRAC=0.45 IBS=32 MODES=dual STAGES=eval EVAL_PARTS="test robust" TEST_LIMIT=1500 SUB=800 \
    bash scripts/run_slow.sh                                                                # A7 dual eval
echo done > runs/.laneA_done; echo "LANE_A_DONE $(date)"
