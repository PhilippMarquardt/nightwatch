#!/usr/bin/env bash
# Lane B (overnight): generation (eval, pseudo-labels), fast heads, then finalize. Resumable; cut with `touch STOP_<name>`.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs; exec 9>logs/laneB.lock; flock -n 9 || { echo "lane B already running"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.45 WORKERS=16 IBS=64 NW_FAST_CPU=1
wait_for() { while [ ! -s "$1" ]; do [ -n "${2:-}" ] && [ -f "STOP_$2" ] && return 1; sleep 60; done; }
on() { [ ! -f "STOP_$1" ]; }
echo "lane B start $(date)"
wait_for runs/t_sft_fusion_l/train_summary.json
MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh                               # B1 TriSAR fusion_l eval
FAST_MODALS="rgbtl tl" STAGES=fast bash scripts/run_trisar.sh                               # B2 TriSAR fast heads
wait_for runs/sft_fusion/train_summary.json
MODES=fusion STAGES=eval EVAL_PARTS="val test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh   # B3
STAGES=pseudo PSEUDO_LIMIT=0 bash scripts/run_slow.sh                                       # B4 label ALL 8164 frames
VARIANTS=distill bash scripts/run_fast.sh                                                   # B5 distilled fast head
python -m nightwatch.eval.escalation || true
wait_for runs/sft_rgb/train_summary.json
MODES=rgb STAGES=eval EVAL_PARTS="test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh          # B6
wait_for runs/t_sft_rgb/train_summary.json
MODES=rgb STAGES=eval SUB=600 bash scripts/run_trisar.sh                                    # B7
on tf_more && FAST_MODALS="rgb rgbt" STAGES=fast bash scripts/run_trisar.sh                 # B8
on llvip_fast && VARIANTS=oracle_rgb,oracle_t bash scripts/run_fast.sh                      # B8b LLVIP single-sensor fast heads
on ltd && [ -f scripts/run_ltd.sh ] && bash scripts/run_ltd.sh                              # B9 LTDv2 (if ready)
if on t_fusion && wait_for runs/t_sft_fusion/train_summary.json t_fusion; then
  MODES=fusion STAGES=eval SUB=600 bash scripts/run_trisar.sh                               # B10
fi
wait_for runs/.laneA_done
bash scripts/finalize.sh
echo "LANE_B_DONE $(date)"
