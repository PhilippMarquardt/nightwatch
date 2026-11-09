#!/usr/bin/env bash
# After the consensus seeds finish, resume the remaining plain seeds (distill_s2, selfdistill_s1/s2). Review #5 reorder.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q CONSENSUS_SEEDS_DONE logs/cseeds.log 2>/dev/null; do sleep 120; done
rm -f STOP_seeds
bash scripts/run_seeds.sh
