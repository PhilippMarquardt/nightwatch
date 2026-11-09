#!/usr/bin/env bash
# tri seed 1 on pod 1 only after rgbt_i training ends (avoid 0.6 + 0.45 memory caps). queue1 skips it (STOP_tri).
cd "$(dirname "$0")/.."
while [ ! -s runs/t_sft_rgbt_i/train_summary.json ] || pgrep -f "mode tri --splits" > /dev/null; do sleep 60; done
rm -f STOP_tri
TRAIN_FRAC=0.55 EVAL_FRAC=0.45 MODE=tri SEED=1 bash scripts/run_tri.sh
echo TRI_S1_DONE
