#!/usr/bin/env bash
# Copy data + adapters from pod 1 to pod 2 (one-off key ~/.ssh/pod2_xfer, created 2026-11-09).
set -euo pipefail
. "$(dirname "$0")/pods.env"   # POD2_HOST / POD2_PORT (gitignored)
cd "$(dirname "$0")/.."
tar cf - data/trisar_fused data/trisar_index.json data/ltd runs/sft_thermal runs/fast_oracle_t runs/fast_gt_labeled \
  | ssh -i ~/.ssh/pod2_xfer -p "$POD2_PORT" "$POD2_HOST" "cd /root/nightwatch && tar xf -"
echo XFER_DONE
