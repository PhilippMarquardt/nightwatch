#!/usr/bin/env bash
cd "$(dirname "$0")/.."
WAIT="logs/tripc0b.log:TRI_DONE tri_pc 0" RUNS="t_sft_rgbt_i_s1 t_sft_tri t_sft_tri_s2 t_sft_tri_vl0.2" bash scripts/reinfer_deadT.sh
