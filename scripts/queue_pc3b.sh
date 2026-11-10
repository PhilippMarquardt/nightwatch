#!/usr/bin/env bash
# Start tri_pc3b (seed 0) as soon as the tri_pc3 seed-0 pipeline has finished its evals.
cd "$(dirname "$0")/.."
until grep -q -E "PC3_ALL_DONE|FAILED" logs/pc3s0.log 2>/dev/null; do sleep 30; done
ARM=B SEED=${SEED:-0} bash scripts/run_tri_pc3.sh
