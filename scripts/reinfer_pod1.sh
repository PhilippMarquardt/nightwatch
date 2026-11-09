#!/usr/bin/env bash
cd "$(dirname "$0")/.."
WAIT="logs/tripc1.log:TRI_DONE tri_pc 1" RUNS="t_sft_thermal_i t_sft_rgbt_i t_sft_rgbt_i_s2 t_sft_tri_s1 t_sft_tri_vl t_sft_tri_vl0.2_s1 t_sft_tri_pc_s1" bash scripts/reinfer_deadT.sh
