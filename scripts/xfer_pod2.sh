#!/usr/bin/env bash
# Copy data + adapters from pod 1 to pod 2 (one-off key ~/.ssh/pod2_xfer, created 2026-11-09).
set -euo pipefail
cd "$(dirname "$0")/.."
tar cf - data/trisar_fused data/trisar_index.json data/ltd runs/sft_thermal runs/fast_oracle_t runs/fast_gt_labeled \
  | ssh -i ~/.ssh/pod2_xfer -p POD2_PORT root@POD2_IP "cd /root/nightwatch && tar xf -"
echo XFER_DONE
