#!/usr/bin/env bash
# Pod 2: runs that exist only here (LTDv2 arms, tri s0/s2, rgbt_i s1). Copies of pod-1 runs are left to pod 1.
cd "$(dirname "$0")/.."
python -m nightwatch.wandb_sync runs/ltd_B* runs/t_sft_rgbt_i_s1
until grep -q "LIDAROFF_DONE t_sft_tri_s2" logs/lidaroff.log 2>/dev/null; do sleep 60; done
python -m nightwatch.wandb_sync runs/t_sft_tri runs/t_sft_tri_s2
echo BACKFILL_DONE
