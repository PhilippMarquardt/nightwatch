#!/usr/bin/env bash
# Wait for both pod-1 jobs (trivl2 eval, rgbt_i s2 eval) so two GPU caps never overlap.
cd "$(dirname "$0")/.."
until grep -q "TRI_DONE rgbt_i 2" logs/rgbts2.log 2>/dev/null; do sleep 60; done
WAIT="logs/trivl2.log:TRI_VL_DONE" RUNS="t_sft_tri_s1" bash scripts/run_lidaroff_full.sh
