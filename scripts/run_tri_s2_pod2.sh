#!/usr/bin/env bash
# Pod 2: third seed of the three-image TriSAR model, next to the rgbt_i s1 eval.
cd "$(dirname "$0")/.."
TRAIN_FRAC=0.6 EVAL_FRAC=0.7 MODE=tri SEED=2 bash scripts/run_tri.sh
