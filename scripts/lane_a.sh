#!/usr/bin/env bash
# Lane A (overnight): slow-head SFT runs in series. Resumable; cut a step with `touch STOP_<name>`.
# Order: LLVIP fusion > TriSAR fusion_l camera-dropout 0.5 (night fix) > LLVIP rgb > LLVIP thermal-only > TriSAR rgb > dual.
set -uo pipefail                      # no -e: one failed step must not kill the lane
cd "$(dirname "$0")/.."
mkdir -p logs runs; exec 9>logs/laneA.lock; echo "waiting for lane A lock"; flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.47 WORKERS=16 BS=4 ACC=4
wait_for() { while [ ! -s "$1" ]; do [ -n "${2:-}" ] && [ -f "STOP_$2" ] && return 1; sleep 60; done; }
on() { [ ! -f "STOP_$1" ]; }
echo "lane A start $(date)"
while pgrep -f "nightwatch.slow.train" > /dev/null; do sleep 60; done     # hand-over: let a running SFT finish
wait_for runs/t_sft_fusion_l/train_summary.json
EPOCHS=2 MODES=fusion  STAGES=sft bash scripts/run_slow.sh                                  # A1 LLVIP fusion (ours)
# A5b (replaces the RGB+T middle rung): same fusion_l recipe, camera-dropout 0.5 instead of 0.2.
# Hypothesis from the 23:15 results: the slow head fails at night because its localisation depends on RGB.
if on cd50; then
  RUN_TAG=_cd50 DROP_RGB=0.5 EPOCHS=1 MODES=fusion_l STAGES=sft bash scripts/run_trisar.sh
  NW_GPU_FRAC=0.45 IBS=32 RUN_TAG=_cd50 MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh
fi
EPOCHS=2 MODES=rgb     STAGES=sft bash scripts/run_slow.sh                                  # A2 LLVIP RGB only
EPOCHS=2 MODES=thermal STAGES=sft bash scripts/run_slow.sh                                  # A3 LLVIP thermal only (control)
EPOCHS=1 MODES=rgb     STAGES=sft bash scripts/run_trisar.sh                                # A4 TriSAR RGB only
on dual     && BS=2 ACC=8 EPOCHS=2 MODES=dual STAGES=sft bash scripts/run_slow.sh           # A5 LLVIP thermal as 2nd image
on dual     && NW_GPU_FRAC=0.45 IBS=32 MODES=dual STAGES=eval EVAL_PARTS="test robust" TEST_LIMIT=1500 SUB=800 \
    bash scripts/run_slow.sh                                                                # A7 dual eval
echo done > runs/.laneA_done; echo "LANE_A_DONE $(date)"
