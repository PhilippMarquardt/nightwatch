#!/usr/bin/env bash
# Review #9 confound control: fusion_l with NO camera-dropout (thermal/LiDAR dropout unchanged). Separates
# "side-branch OOD causes forest hallucination" from "camera-dropout with kept targets taught black-camera => report people".
# Starts when the dual SFT has finished training (its eval then runs beside this). STOP_cd0 cuts it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f STOP_cd0 ] && [ ! -s runs/sft_dual/train_summary.json ]; do sleep 60; done
[ -f STOP_cd0 ] && exit 0
NW_GPU_FRAC=0.6 BS=4 ACC=4 RUN_TAG=_cd0 DROP_RGB=0 EPOCHS=1 MODES=fusion_l STAGES=sft bash scripts/run_trisar.sh   # same effective batch (16) as cd50
NW_GPU_FRAC=0.45 IBS=32 RUN_TAG=_cd0 MODES=fusion_l STAGES=eval SUB=600 bash scripts/run_trisar.sh
echo CD0_DONE
