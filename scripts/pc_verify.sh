#!/usr/bin/env bash
# After the shards finish: re-encode any truncated file (the encoder validates existing files), then count.
cd "$(dirname "$0")/.."
until [ "$(pgrep -fc '^python -m nightwatch.slow.pc_encode')" = 0 ] && [ "$(ls data/trisar_pc | wc -l)" -ge 9000 ]; do sleep 60; done
NW_GPU_FRAC=0.05 python -m nightwatch.slow.pc_encode --out data/trisar_pc 2>&1 | tail -2
ls data/trisar_pc | wc -l
echo PC_VERIFIED
