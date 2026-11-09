#!/usr/bin/env bash
cd "$(dirname "$0")/.."
WAIT="logs/tris2.log:TRI_DONE tri 2" RUNS="t_sft_tri t_sft_tri_s2" bash scripts/run_lidaroff_full.sh
