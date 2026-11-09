#!/usr/bin/env bash
# tri_pc2 (docs/PLAN_POINTCLOUD.md follow-up): LiDAR-only samples + norm-matched projector. One seed; run two in parallel.
#   SEED=0 bash scripts/run_tri_pc2.sh
cd "$(dirname "$0")/.."
export NW_LIDAR_ONLY_P=0.15 NW_PC_NORMMATCH=1
TRAIN_FRAC=${TRAIN_FRAC:-0.47} EVAL_FRAC=${EVAL_FRAC:-0.47} MODE=tri_pc RUN_SUFFIX=2 SEED=${SEED:-0} \
  ROBUST_CONDS=rgb_off,thermal_off,lidar_off,lidar_only,sensors_shuffle bash scripts/run_tri.sh
R=t_sft_tri_pc2$( [ "${SEED:-0}" = 0 ] && echo "" || echo "_s$SEED" )
EVAL_FRAC=0.47 RUNS=$R bash scripts/run_lidaroff_full.sh
echo "PC2_ALL_DONE $R"
