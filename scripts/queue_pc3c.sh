#!/usr/bin/env bash
# Start tri_pc3c (seed 0) as soon as the tri_pc3b pipeline has finished its evals.
cd "$(dirname "$0")/.."
until grep -q -E "PC3_ALL_DONE|FAILED" logs/pc3b.log 2>/dev/null; do sleep 30; done
ARM=C SEED=${SEED:-0} bash scripts/run_tri_pc3.sh
