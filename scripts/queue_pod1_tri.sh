#!/usr/bin/env bash
# Pod 1 queue for the follow-up: matched controls, then a second tri seed. Starts when the morning extras have
# finished training (cd0 seed 1 and dual_half); their evals may still be running beside it (memory caps).
cd "$(dirname "$0")/.."
while [ ! -f STOP_queue1 ] && { [ ! -s runs/t_sft_fusion_l_cd0_s1/train_summary.json ] || [ ! -s runs/sft_dual_half/train_summary.json ]; }; do sleep 60; done
[ -f STOP_queue1 ] && exit 0
TRAIN_FRAC=0.5 EVAL_FRAC=0.5 MODE=thermal_i SEED=0 bash scripts/run_tri.sh
TRAIN_FRAC=0.6 EVAL_FRAC=0.6 MODE=rgbt_i SEED=0 bash scripts/run_tri.sh
TRAIN_FRAC=0.6 EVAL_FRAC=0.6 MODE=tri SEED=1 bash scripts/run_tri.sh
echo QUEUE1_DONE
