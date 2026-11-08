#!/usr/bin/env bash
# Lane B (overnight): generation (eval, pseudo-labels), fast heads, LTDv2, then finalize. Resumable; cut with `touch STOP_<name>`.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs; exec 9>logs/laneB.lock; echo "waiting for lane B lock"; flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.45 WORKERS=16 IBS=64 NW_FAST_CPU=1
wait_for() { while [ ! -s "$1" ]; do [ -n "${2:-}" ] && [ -f "STOP_$2" ] && return 1; sleep 60; done; }
on() { [ ! -f "STOP_$1" ]; }
echo "lane B start $(date)"
while pgrep -f "scripts/run_trisar.sh|scripts/run_fast.sh" > /dev/null; do sleep 60; done   # hand-over from the old lane
wait_for runs/t_sft_fusion_l/train_summary.json
MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh                               # B1 TriSAR fusion_l eval
FAST_MODALS="rgbtl tl" STAGES=fast bash scripts/run_trisar.sh                               # B2 TriSAR fast heads
on ltd && [ -f scripts/run_ltd.sh ] && bash scripts/run_ltd.sh                              # B3 LTDv2 site memory (fills the gap)
wait_for runs/sft_fusion/train_summary.json
MODES=fusion STAGES=eval EVAL_PARTS="val test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh   # B4
STAGES=pseudo TEACHER=fusion PSEUDO_LIMIT=0 bash scripts/run_slow.sh                        # B5 fused teacher labels ALL 8164
VARIANTS=distill bash scripts/run_fast.sh                                                   # B6 distilled student
python -m nightwatch.eval.escalation || true
wait_for runs/sft_rgb/train_summary.json
MODES=rgb STAGES=eval EVAL_PARTS="test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh          # B7
if on rgb_teacher; then
  STAGES=pseudo TEACHER=rgb PSEUDO_LIMIT=0 bash scripts/run_slow.sh                         # B8 RGB teacher labels ALL 8164
  VARIANTS=distill_rgb bash scripts/run_fast.sh                                             # B9 student of the RGB teacher
fi
wait_for runs/sft_thermal/train_summary.json
MODES=thermal STAGES=eval EVAL_PARTS="test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh      # B10
wait_for runs/t_sft_rgb/train_summary.json
MODES=rgb STAGES=eval SUB=600 bash scripts/run_trisar.sh                                    # B11
on tf_more && FAST_MODALS="rgb rgbt" STAGES=fast bash scripts/run_trisar.sh                 # B12
on llvip_fast && VARIANTS=oracle_rgb,oracle_t bash scripts/run_fast.sh                      # B13
if on t_fusion && wait_for runs/t_sft_fusion/train_summary.json t_fusion; then
  MODES=fusion STAGES=eval SUB=600 bash scripts/run_trisar.sh                               # B14
fi
wait_for runs/.laneA_done
bash scripts/finalize.sh
echo "LANE_B_DONE $(date)"
