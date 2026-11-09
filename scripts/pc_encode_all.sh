#!/usr/bin/env bash
# Frozen Utonia features for every TriSAR frame, 3 shards in parallel (pod 1 only: it has the raw point clouds).
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  NW_GPU_FRAC=0.05 python -m nightwatch.slow.pc_encode --out data/trisar_pc --shard $s --nshard 3 > logs/pcenc_$s.log 2>&1 &
done
wait
ls data/trisar_pc | wc -l
echo PC_ALL_DONE
