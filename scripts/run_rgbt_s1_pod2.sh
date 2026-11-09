#!/usr/bin/env bash
# second seed of the primary-contrast control (rgbt_i) on pod 2 once tri seed 0 has finished its evals.
cd "$(dirname "$0")/.."
while ! grep -q "TRI_DONE" logs/tri0.log 2>/dev/null; do sleep 60; done
TRAIN_FRAC=0.7 EVAL_FRAC=0.7 MODE=rgbt_i SEED=1 bash scripts/run_tri.sh
echo RGBT_S1_DONE
