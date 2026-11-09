#!/usr/bin/env bash
# Review #13: run rgbt_i in parallel with thermal_i on pod 1 once the morning evals are done; then mark it so the
# serial queue skips its copy (run_tri.sh checks STOP_<mode> only at start).
cd "$(dirname "$0")/.."
while ! grep -q CD0S1_DONE logs/cd0s1.log 2>/dev/null || ! grep -q DUAL_HALF_DONE logs/dualhalf.log 2>/dev/null; do sleep 60; done
( sleep 90; touch STOP_rgbt_i ) &
TRAIN_FRAC=0.45 EVAL_FRAC=0.45 MODE=rgbt_i SEED=0 bash scripts/run_tri.sh
echo RGBT_PAR_DONE
