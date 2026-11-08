#!/usr/bin/env bash
# Slow-head experiments. Each stage is skipped if its output already exists,
# so the script can be re-run after an interruption.
#   bash scripts/run_slow.sh            # everything
#   STAGES="zeroshot sft" bash scripts/run_slow.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${NW_MODEL:-Qwen/Qwen3.5-9B}
STAGES=${STAGES:-"zeroshot sft eval pseudo"}
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 100000 ]; then BS=8; ACC=2; IBS=48; else BS=4; ACC=4; IBS=24; fi   # H200 vs A100-80G
EPOCHS=${EPOCHS:-2}
SUB=${SUB:-800}      # subset size for the expensive robustness conditions
W=${WORKERS:-12}
echo "model=$MODEL gpu_mem=${GPU_MEM}MiB bs=$BS acc=$ACC infer_bs=$IBS"

has() { [ -s "$1" ]; }

# ---------------------------------------------------------------- 1. zero-shot baselines
if [[ " $STAGES " == *" zeroshot "* ]]; then
  for MODE in rgb dual; do
    OUT=runs/zs_$MODE
    has $OUT/pred_test.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --split test \
        --tasks detect --conds normal --bs $IBS --workers $W --out $OUT/pred_test.jsonl
    has $OUT/pred_test_robust.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --split test \
        --tasks detect --conds rgb_off,thermal_off --limit $SUB --bs $IBS --workers $W --out $OUT/pred_test_robust.jsonl
    has $OUT/pred_test_reason.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --split test         --tasks count,zone_left,zone_center,zone_right --conds normal --limit $SUB --bs $IBS --workers $W --out $OUT/pred_test_reason.jsonl
  done
fi

# ---------------------------------------------------------------- 2. SFT on the labeled sites only
if [[ " $STAGES " == *" sft "* ]]; then
  for MODE in fusion dual rgb; do
    OUT=runs/sft_$MODE
    has $OUT/train_summary.json || python -m nightwatch.slow.train --model $MODEL --mode $MODE --splits labeled \
        --epochs $EPOCHS --bs $BS --grad-accum $ACC --workers $W --grad-ckpt --out $OUT ${WANDB:+--wandb}
  done
fi

# ---------------------------------------------------------------- 3. evaluate on held-out sites
if [[ " $STAGES " == *" eval "* ]]; then
  for MODE in fusion dual rgb; do
    OUT=runs/sft_$MODE
    has $OUT/pred_val.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --adapter $OUT --split val \
        --tasks detect --conds normal --bs $IBS --workers $W --out $OUT/pred_val.jsonl
    has $OUT/pred_test.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --adapter $OUT --split test \
        --tasks detect --conds normal --bs $IBS --workers $W --out $OUT/pred_test.jsonl
    has $OUT/pred_test_robust.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --adapter $OUT --split test \
        --tasks detect --conds rgb_off,thermal_off --limit $SUB --bs $IBS --workers $W --out $OUT/pred_test_robust.jsonl
    has $OUT/pred_test_reason.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode $MODE --adapter $OUT --split test \
        --tasks count,zone_left,zone_center,zone_right --conds normal --limit $SUB --bs $IBS --workers $W --out $OUT/pred_test_reason.jsonl
  done
fi

# ---------------------------------------------------------------- 4. slow head labels the unlabeled sites
if [[ " $STAGES " == *" pseudo "* ]]; then
  OUT=runs/sft_fusion
  has $OUT/pred_unlabeled.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode fusion --adapter $OUT --split unlabeled \
      --tasks detect --conds normal --bs $IBS --workers $W --out $OUT/pred_unlabeled.jsonl
  python - <<'EOF'
import json
lab = {}
for l in open("runs/sft_fusion/pred_unlabeled.jsonl"):
    d = json.loads(l)
    lab[d["id"]] = d["boxes"]
json.dump(lab, open("runs/pseudo_labels_unlabeled.json", "w"))
print("pseudo labels:", len(lab), "images,", sum(len(v) for v in lab.values()), "boxes")
EOF
fi

# ---------------------------------------------------------------- 5. optional: GRPO on top of the fusion SFT
if [[ " $STAGES " == *" grpo "* ]]; then
  OUT=runs/grpo_fusion
  has $OUT/train_summary.json || python -m nightwatch.slow.grpo --model $MODEL --init runs/sft_fusion --out $OUT       --steps ${GRPO_STEPS:-120}
  has $OUT/pred_test.jsonl || python -m nightwatch.slow.infer --model $MODEL --mode fusion --adapter $OUT --split test       --tasks detect --conds normal --bs $IBS --workers $W --out $OUT/pred_test.jsonl
fi

for f in runs/*/pred_*.jsonl; do
  python -m nightwatch.eval.metrics --pred $f --split $( [[ $f == *val* ]] && echo val || echo test ) \
      --out ${f%.jsonl}_metrics.json > /dev/null 2>&1 || true
done
[ -f runs/pseudo_labels_unlabeled.json ] && python -m nightwatch.eval.metrics --pred runs/sft_fusion/pred_unlabeled.jsonl \
    --split unlabeled --out runs/sft_fusion/pred_unlabeled_metrics.json || true
echo "SLOW DONE"
