#!/usr/bin/env bash
# LTDv2 site-memory stage (exploratory extension for the "World Model & Site Memory" direction).
# One fixed thermal harbour camera, 8 months; month-held-out; surprise = latent prediction error.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
[ -s data/ltd/frames.json ] || python -m nightwatch.data.ltd --raw data/raw/ltd --out data/ltd --workers 32
[ -s runs/ltd_memory/results.json ] || python -m nightwatch.memory.site_memory --data data/ltd --out runs/ltd_memory \
    --steps ${LTD_STEPS:-3000} --n-paste 600
echo "LTD DONE"
