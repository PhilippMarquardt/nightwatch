#!/usr/bin/env bash
# Full-test LiDAR-off for a tri_pc seed after this pod's dead-thermal re-inference (needed for P1).
cd "$(dirname "$0")/.."
until grep -q REINFER_ALL_DONE logs/reinfer.log 2>/dev/null; do sleep 60; done
RUNS="$1" bash scripts/run_lidaroff_full.sh
