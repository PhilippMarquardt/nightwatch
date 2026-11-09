#!/usr/bin/env bash
# Frozen Utonia features for every TriSAR frame, NSHARD (default 6) shards in parallel (pod 1 only: it has the raw point clouds).
cd "$(dirname "$0")/.."
for s in $(seq 0 $((${NSHARD:-6}-1))); do
  NW_GPU_FRAC=0.05 python -m nightwatch.slow.pc_encode --out data/trisar_pc --shard $s --nshard ${NSHARD:-6} > logs/pcenc_$s.log 2>&1 &
done
wait
ls data/trisar_pc | wc -l
echo PC_ALL_DONE
# second pass: re-encode any file a killed earlier run left truncated (the encoder validates existing files)
NW_GPU_FRAC=0.05 python -m nightwatch.slow.pc_encode --out data/trisar_pc > logs/pcenc_verify.log 2>&1
tail -1 logs/pcenc_verify.log
echo PC_VERIFIED
