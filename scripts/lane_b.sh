#!/usr/bin/env bash
# Lane B (overnight): generation (eval, pseudo-labels), fast heads, LTDv2, review extras, then finalize.
# Resumable; cut a step with `touch STOP_<name>`. Every wait can be escaped with its STOP_ file (no deadlocks).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs; exec 9>logs/laneB.lock; echo "waiting for lane B lock"; flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=0.45 WORKERS=16 IBS=64 NW_FAST_CPU=1
wait_for() { while [ ! -s "$1" ]; do [ -n "${2:-}" ] && [ -f "STOP_$2" ] && return 1; sleep 60; done; }
on() { [ ! -f "STOP_$1" ]; }
echo "lane B start $(date)"
while pgrep -f "scripts/run_trisar.sh|scripts/run_fast.sh|scripts/run_ltd.sh" > /dev/null; do sleep 60; done   # hand-over
wait_for runs/t_sft_fusion_l/train_summary.json
MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh                               # B1 TriSAR fusion_l eval
FAST_MODALS="rgbtl tl" STAGES=fast bash scripts/run_trisar.sh                               # B2 TriSAR fast heads
on ltd && bash scripts/run_ltd.sh                                                           # B3 LTDv2 site memory
on stretch && EXTRAS=stretch IBS=48 bash scripts/run_extras.sh                              # B3b thermal-stretch hypothesis
if wait_for runs/sft_fusion/train_summary.json fusion; then
  MODES=fusion STAGES=eval EVAL_PARTS="val test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh   # B4
  STAGES=pseudo TEACHER=fusion PSEUDO_LIMIT=0 bash scripts/run_slow.sh                      # B5 fused teacher labels ALL 8164
  VARIANTS=distill bash scripts/run_fast.sh                                                 # B6 distilled student
  python -m nightwatch.eval.escalation || true
fi
on self && EXTRAS=self bash scripts/run_extras.sh                                           # B6b self-training control
on shuffle && EXTRAS=shuffle IBS=48 bash scripts/run_extras.sh                              # B6c aligned vs shuffled sensors
on matched && EXTRAS=matched bash scripts/run_extras.sh                                     # B6d compute-matched labelled-only student
if wait_for runs/sft_rgb/train_summary.json rgb; then
  MODES=rgb STAGES=eval EVAL_PARTS="test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh        # B7
  if on rgb_teacher; then
    STAGES=pseudo TEACHER=rgb PSEUDO_LIMIT=0 bash scripts/run_slow.sh                       # B8 RGB teacher labels ALL 8164
    VARIANTS=distill_rgb bash scripts/run_fast.sh                                           # B9 student of the RGB teacher
  fi
fi
if wait_for runs/sft_thermal/train_summary.json thermal; then
  MODES=thermal STAGES=eval EVAL_PARTS="test robust reason" TEST_LIMIT=1500 SUB=800 bash scripts/run_slow.sh    # B10
fi
on stretch && EXTRAS=stretch IBS=48 bash scripts/run_extras.sh                              # cd50 stretch (if cd50 is done)
if wait_for runs/t_sft_rgb/train_summary.json t_rgb; then
  MODES=rgb STAGES=eval SUB=600 bash scripts/run_trisar.sh                                  # B11
fi
on tf_more && FAST_MODALS="rgb rgbt" STAGES=fast bash scripts/run_trisar.sh                 # B12
on llvip_fast && VARIANTS=oracle_rgb,oracle_t bash scripts/run_fast.sh                      # B13
wait_for runs/.laneA_done laneA
bash scripts/finalize.sh
echo "LANE_B_DONE $(date)"
