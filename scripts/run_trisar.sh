#!/usr/bin/env bash
# TriSAR-3D stage: RGB + thermal + LiDAR. Resumable (finished stages are skipped).
#   bash scripts/run_trisar.sh                         # everything
#   STAGES="data zeroshot sft" bash scripts/run_trisar.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

MODEL=${NW_MODEL:-Qwen/Qwen3.5-9B}
# "pseudo" (+ tf_distill) is opt-in: the slow head is trained on all train scenes, so labelling them again is circular.
# The LLVIP stage carries the distillation story; here the point is the sensor ladder.
STAGES=${STAGES:-"data zeroshot sft eval fast"}
FUSED=${FUSED:-data/trisar_fused}
STRIDE=${STRIDE:-2}          # every 2nd frame (10 Hz -> 5 Hz); consecutive frames are near-duplicates
EPOCHS=${EPOCHS:-1}   # 6.4k frames at 5 Hz; 1 epoch is enough for LoRA
SUB=${SUB:-600}
W=${WORKERS:-12}
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 90000 ]; then BS=8; ACC=2; IBS=96; else BS=4; ACC=4; IBS=24; fi   # H200 / RTX PRO 6000 (96 GB) vs A100-80G
has() { [ -s "$1" ]; }

# ------------------------------------------------------------ 0. data
if [[ " $STAGES " == *" data "* ]]; then
  if [ ! -s $FUSED/frames.json ]; then
    mkdir -p data/raw/trisar
    if ! ls data/raw/trisar/*/dataset.json >/dev/null 2>&1 && [ ! -f data/raw/trisar/dataset.json ]; then
      [ -s data/raw/trisar/trisar.zip ] || curl -L -C - --retry 5 -o data/raw/trisar/trisar.zip \
        "https://tudatalib.ulb.tu-darmstadt.de/server/api/core/bitstreams/cbdb5411-3156-48fd-9d73-897925972fc0/content"
      (cd data/raw/trisar && unzip -q trisar.zip && rm -f trisar.zip)
    fi
    python -m nightwatch.data.trisar --root data/raw/trisar --index data/trisar_index.json --out $FUSED \
        --stride $STRIDE --workers ${PREP_WORKERS:-32}
  fi
fi

# ------------------------------------------------------------ 1. zero-shot (stock VLM, RGB only)
if [[ " $STAGES " == *" zeroshot "* ]]; then
  OUT=runs/t_zs_rgb
  has $OUT/pred_test.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $MODEL --mode rgb \
      --split test --tasks report,lying,nearest --conds normal --limit $SUB --bs $IBS --workers $W --out $OUT/pred_test.jsonl
fi

# ------------------------------------------------------------ 2. SFT: sensor ablation ladder
if [[ " $STAGES " == *" sft "* ]]; then
  for MODE in fusion_l fusion rgb; do
    OUT=runs/t_sft_$MODE
    has $OUT/train_summary.json || python -m nightwatch.slow.train --dataset trisar --fused-dir $FUSED --model $MODEL \
        --mode $MODE --splits train --epochs $EPOCHS --bs $BS --grad-accum $ACC --workers $W --grad-ckpt --out $OUT \
        ${WANDB:+--wandb}
  done
fi

# ------------------------------------------------------------ 3. evaluate (test split)
if [[ " $STAGES " == *" eval "* ]]; then
  for MODE in fusion_l fusion rgb; do
    OUT=runs/t_sft_$MODE
    has $OUT/pred_test.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $MODEL --mode $MODE \
        --adapter $OUT --split test --tasks report,lying,nearest --conds normal --bs $IBS --workers $W --out $OUT/pred_test.jsonl
    has $OUT/pred_test_robust.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $MODEL \
        --mode $MODE --adapter $OUT --split test --tasks report --conds rgb_off,thermal_off,lidar_off --limit $SUB \
        --bs $IBS --workers $W --out $OUT/pred_test_robust.jsonl
  done
fi

# ------------------------------------------------------------ 4. slow head labels the train split (distillation source)
if [[ " $STAGES " == *" pseudo "* ]]; then
  OUT=runs/t_sft_fusion_l
  has $OUT/pred_train.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir $FUSED --model $MODEL \
      --mode fusion_l --adapter $OUT --split train --tasks report --conds normal --bs $IBS --workers $W --out $OUT/pred_train.jsonl
fi

# ------------------------------------------------------------ 5. fast heads: modality ladder + distilled
if [[ " $STAGES " == *" fast "* ]]; then
  for M in rgbtl rgbt rgb tl; do
    has runs/tf_$M/train_summary.json || python -m nightwatch.fast.trisar --name tf_$M --modal $M --fused-dir $FUSED
  done
  [ -s runs/t_sft_fusion_l/pred_train.jsonl ] && { has runs/tf_distill/train_summary.json || \
      python -m nightwatch.fast.trisar --name tf_distill --modal rgbtl --fused-dir $FUSED --labels runs/t_sft_fusion_l/pred_train.jsonl; }
fi

# ------------------------------------------------------------ metrics
for f in runs/t_*/pred_test*.jsonl runs/tf_*/pred_test.jsonl; do
  [ -s "$f" ] || continue
  thr=""
  d=$(dirname $f); [ -f $d/train_summary.json ] && thr=$(python -c "import json;print(json.load(open('$d/train_summary.json')).get('score_thr',''))")
  python -m nightwatch.eval.metrics_trisar --pred $f --fused-dir $FUSED ${thr:+--score-thr $thr} --out ${f%.jsonl}_metrics.json \
      > /dev/null || echo "metrics failed: $f"
done
[ -s runs/t_sft_fusion_l/pred_train.jsonl ] && python -m nightwatch.eval.metrics_trisar --pred runs/t_sft_fusion_l/pred_train.jsonl \
    --fused-dir $FUSED --split train --out runs/t_sft_fusion_l/pred_train_metrics.json > /dev/null || true
echo "TRISAR DONE"
