#!/usr/bin/env bash
# Pod 1 -> pod 2: Utonia token features (data/trisar_pc, ~6.5 GB) so tri_pc seed 0 can train on pod 2.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/pods.env
until grep -q PC_VERIFIED logs/pcverify.log 2>/dev/null; do sleep 60; done
tar cf - data/trisar_pc | ssh -o StrictHostKeyChecking=no -i ~/.ssh/pod2_xfer -p "$POD2_PORT" "$POD2_HOST" "cd /root/nightwatch && tar xf -"
ssh -i ~/.ssh/pod2_xfer -p "$POD2_PORT" "$POD2_HOST" "ls /root/nightwatch/data/trisar_pc | wc -l"
echo XFER_PC_DONE
