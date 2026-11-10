#!/usr/bin/env bash
# tri_pc3: stage 0 projector pre-alignment -> stage 1 LiDAR-only SFT ->
# gate G1 on val -> stage 2 joint SFT -> the usual test evals.
#   SEED=0 bash scripts/run_tri_pc3.sh
#   ARM=N : stage 1 without pre-alignment (ablation; stops after the val gate eval)
#   ARM=R : range-image LiDAR-only (mode tri; stops after the val gate eval)
#   ARM=B : tri_pc3b = stage 2 with a FRESH LoRA on the stage-1 projector, LiDAR-only p 0.15 (needs stage 1 done)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NW_WANDB=0
export NW_PC_GEO=1 NW_PC_NORMMATCH=1 NW_GPU_FRAC=${FRAC:-0.47}
SEED=${SEED:-0}; ARM=${ARM:-main}
TAG=$( [ "$SEED" = 0 ] && echo "" || echo "_s$SEED" )
M=Qwen/Qwen3.5-9B; FUSED=data/trisar_fused
A=runs/pc_align$TAG; S1=runs/t_sft_tri_pc3a$TAG; S2=runs/t_sft_tri_pc3$TAG
S2_LP=0.3; S2_EXTRA=""
if [ "$ARM" = B ]; then S2=runs/t_sft_tri_pc3b$TAG; S2_LP=0.15; S2_EXTRA="--init-no-lora"; fi
# ARM=C : tri_pc3c = as B but with NO LiDAR-only samples in stage 2 (LiDAR for range, boxes from thermal/camera)
if [ "$ARM" = C ]; then S2=runs/t_sft_tri_pc3c$TAG; S2_LP=0; S2_EXTRA="--init-no-lora"; fi
TRAIN="python -m nightwatch.slow.train --dataset trisar --fused-dir $FUSED --model $M --splits train --bs 4 --grad-accum 4 --workers 12 --grad-ckpt --seed $SEED"
valgate() {   # $1 run dir, $2 mode
  [ -s $1/pred_val.jsonl ] || python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $M --bs 32 --workers 12 \
      --mode $2 --adapter $1 --split val --tasks report --conds lidar_only --out $1/pred_val.jsonl
  python -m nightwatch.eval.metrics_trisar --pred $1/pred_val.jsonl --split val --out $1/pred_val_metrics.json > /dev/null
  python scripts/pc3_gate.py $1
}

if [ "$ARM" = N ]; then
  O=runs/t_sft_tri_pc3a_noalign$TAG
  [ -s $O/train_summary.json ] || NW_LIDAR_ONLY_P=1.0 $TRAIN --mode tri_pc --epochs 2 --pc-aux 0.5 --pc-wd 0.01 --lr-pc 3e-4 --out $O
  valgate $O tri_pc; echo "PC3_ARM_DONE N $SEED"; exit 0
fi
if [ "$ARM" = R ]; then
  O=runs/t_sft_tri_lonly$TAG
  [ -s $O/train_summary.json ] || NW_LIDAR_ONLY_P=1.0 $TRAIN --mode tri --epochs 2 --out $O
  valgate $O tri; echo "PC3_ARM_DONE R $SEED"; exit 0
fi

[ -s $A/fusion.pt ] || python -m nightwatch.slow.pc_align --out $A --seed $SEED
[ -s $A/fusion.pt ] || { echo "PC3_ALIGN_FAILED $SEED"; exit 1; }
[ -s $S1/train_summary.json ] || NW_LIDAR_ONLY_P=1.0 $TRAIN --mode tri_pc --epochs 2 --init $A --pc-aux 0.5 --pc-wd 0.01 --lr-pc 3e-4 --out $S1
[ -s $S1/train_summary.json ] || { echo "PC3_S1_FAILED $SEED"; exit 1; }
valgate $S1 tri_pc
if ! python scripts/pc3_gate.py $S1 --check > /dev/null && [ "${FORCE_S2:-0}" != 1 ]; then
  echo "PC3_G1_FAIL $SEED (stage 2 not run)"; exit 0
fi
echo "PC3_G1_PASS $SEED"
[ "${STOP_AFTER_G1:-0}" = 1 ] && { echo "PC3_STAGE1_DONE $SEED"; exit 0; }
[ -s $S2/train_summary.json ] || NW_LIDAR_ONLY_P=$S2_LP $TRAIN --mode tri_pc --epochs 1 --init $S1 $S2_EXTRA --pc-aux 0.5 --pc-wd 0.01 --lr-pc 1e-4 --out $S2
[ -s $S2/train_summary.json ] || { echo "PC3_S2_FAILED $SEED"; exit 1; }
INF="python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $M --bs 32 --workers 12 --mode tri_pc --adapter $S2 --split test"
[ -s $S2/pred_test_robust.jsonl ] || $INF --tasks report --conds rgb_off,thermal_off,lidar_off,lidar_only,sensors_shuffle --limit 600 --out $S2/pred_test_robust.jsonl
[ -s $S2/pred_test.jsonl ]        || $INF --tasks report --conds normal --out $S2/pred_test.jsonl
[ -s $S2/pred_test_qa.jsonl ]     || $INF --tasks lying,nearest --conds normal --limit 600 --out $S2/pred_test_qa.jsonl
[ -s $S2/pred_test_lidaroff_full.jsonl ] || $INF --tasks report --conds lidar_off --out $S2/pred_test_lidaroff_full.jsonl
for f in pred_test pred_test_robust pred_test_qa pred_test_lidaroff_full; do
  python -m nightwatch.eval.metrics_trisar --pred $S2/$f.jsonl --out $S2/${f}_metrics.json > /dev/null
done
bash scripts/score_criteria.sh $(basename $S2)
echo "PC3_ALL_DONE $SEED $ARM"
