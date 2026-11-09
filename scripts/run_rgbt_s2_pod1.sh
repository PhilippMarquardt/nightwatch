#!/usr/bin/env bash
# Review #16: rgbt_i seed 2 on pod 1 after tri seed 1 finishes its evals (third seed for the primary contrast).
# Uses run_tri_nostop.sh because pod 1 carries old STOP_rgbt_i / STOP_tri flags.
cd "$(dirname "$0")/.."
while ! grep -q "TRI_DONE tri 1" logs/queue1.log 2>/dev/null; do sleep 60; done
TRAIN_FRAC=0.4 EVAL_FRAC=0.4 MODE=rgbt_i SEED=2 bash scripts/run_tri_nostop.sh
echo RGBT_S2_DONE
