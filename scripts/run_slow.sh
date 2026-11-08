#!/usr/bin/env bash
# Slow-head experiments on LLVIP. Each step is skipped if its output already exists (resumable).
#   bash scripts/run_slow.sh                                   # everything
#   STAGES="sft" MODES="fusion" bash scripts/run_slow.sh       # one model
#   STAGES=none bash scripts/run_slow.sh                       # only re-score
# Env: MODES, EPOCHS, BS, ACC, IBS, TEST_LIMIT, SUB, PSEUDO_LIMIT, EVAL_PARTS, WORKERS, NW_GPU_FRAC
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${NW_MODEL:-Qwen/Qwen3.5-9B}
STAGES=${STAGES:-"zeroshot sft eval pseudo"}
MODES=${MODES:-"fusion dual rgb"}
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 90000 ]; then dBS=8; dACC=2; dIBS=96; else dBS=4; dACC=4; dIBS=24; fi   # 96+ GB vs A100-80G
BS=${BS:-$dBS}; ACC=${ACC:-$dACC}; IBS=${IBS:-$dIBS}
EPOCHS=${EPOCHS:-2}
TEST_LIMIT=${TEST_LIMIT:-1500}     # fixed-seed test subset (consecutive frames are near-duplicates); 0 = full test
SUB=${SUB:-800}                    # subset for robustness + reasoning tasks
PSEUDO_LIMIT=${PSEUDO_LIMIT:-0}    # 0 = label ALL unlabeled frames (distillation must not fall back to GT)
EVAL_PARTS=${EVAL_PARTS:-"val test robust reason"}
W=${WORKERS:-12}
echo "model=$MODEL gpu_mem=${GPU_MEM}MiB bs=$BS acc=$ACC infer_bs=$IBS modes=$MODES stages=$STAGES"

has() { [ -s "$1" ]; }
part() { [[ " $EVAL_PARTS " == *" $1 "* ]]; }
INF="python -m nightwatch.slow.infer --model $MODEL --bs $IBS --workers $W"

# ---------------------------------------------------------------- 1. zero-shot baselines
if [[ " $STAGES " == *" zeroshot "* ]]; then
  for MODE in rgb dual; do
    OUT=runs/zs_$MODE
    has $OUT/pred_test.jsonl || $INF --mode $MODE --split test --tasks detect --conds normal --limit $TEST_LIMIT --out $OUT/pred_test.jsonl
    has $OUT/pred_test_robust.jsonl || $INF --mode $MODE --split test --tasks detect --conds rgb_off,thermal_off --limit $SUB \
        --out $OUT/pred_test_robust.jsonl
    has $OUT/pred_test_reason.jsonl || $INF --mode $MODE --split test --tasks count,zone_left,zone_center,zone_right \
        --conds normal --limit $SUB --out $OUT/pred_test_reason.jsonl
  done
fi

# ---------------------------------------------------------------- 2. SFT on the labeled sites only
if [[ " $STAGES " == *" sft "* ]]; then
  for MODE in $MODES; do
    OUT=runs/sft_$MODE
    has $OUT/train_summary.json || python -m nightwatch.slow.train --model $MODEL --mode $MODE --splits labeled \
        --epochs $EPOCHS --bs $BS --grad-accum $ACC --workers $W --grad-ckpt --out $OUT ${WANDB:+--wandb}
  done
fi

# ---------------------------------------------------------------- 3. evaluate on held-out sites
if [[ " $STAGES " == *" eval "* ]]; then
  for MODE in $MODES; do
    OUT=runs/sft_$MODE
    [ -s $OUT/train_summary.json ] || { echo "skip eval $MODE: not trained"; continue; }
    A="--mode $MODE --adapter $OUT"
    if part val;    then has $OUT/pred_val.jsonl || $INF $A --split val --tasks detect --conds normal --out $OUT/pred_val.jsonl; fi
    if part test;   then has $OUT/pred_test.jsonl || $INF $A --split test --tasks detect --conds normal --limit $TEST_LIMIT \
        --out $OUT/pred_test.jsonl; fi
    if part robust; then has $OUT/pred_test_robust.jsonl || $INF $A --split test --tasks detect --conds rgb_off,thermal_off \
        --limit $SUB --out $OUT/pred_test_robust.jsonl; fi
    if part reason; then has $OUT/pred_test_reason.jsonl || $INF $A --split test --tasks count,zone_left,zone_center,zone_right \
        --conds normal --limit $SUB --out $OUT/pred_test_reason.jsonl; fi
  done
fi

# ---------------------------------------------------------------- 4. slow head labels ALL unlabeled sites
if [[ " $STAGES " == *" pseudo "* ]]; then
  OUT=runs/sft_fusion
  has $OUT/pred_unlabeled.jsonl || $INF --mode fusion --adapter $OUT --split unlabeled --tasks detect --conds normal \
      --limit $PSEUDO_LIMIT --out $OUT/pred_unlabeled.jsonl
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
  has $OUT/train_summary.json || python -m nightwatch.slow.grpo --model $MODEL --init runs/sft_fusion --out $OUT --steps ${GRPO_STEPS:-120}
  has $OUT/pred_test.jsonl || $INF --mode fusion --adapter $OUT --split test --tasks detect --conds normal --limit $TEST_LIMIT \
      --out $OUT/pred_test.jsonl
fi

# ---------------------------------------------------------------- metrics: LLVIP slow-head runs only
# (fast heads are scored with their val-selected threshold in run_fast.sh; TriSAR runs in run_trisar.sh)
for f in runs/zs_*/pred_*.jsonl runs/sft_*/pred_*.jsonl runs/grpo_*/pred_*.jsonl; do
  [ -s "$f" ] || continue
  case $f in *unlabeled*) continue ;; esac
  python -m nightwatch.eval.metrics --pred $f --split $( [[ $f == *pred_val* ]] && echo val || echo test ) \
      --out ${f%.jsonl}_metrics.json > /dev/null 2>&1 || echo "metrics failed: $f"
done
if [ -s runs/sft_fusion/pred_unlabeled.jsonl ]; then
  python -m nightwatch.eval.metrics --pred runs/sft_fusion/pred_unlabeled.jsonl --split unlabeled \
      --out runs/sft_fusion/pred_unlabeled_metrics.json > /dev/null || true
fi
echo "SLOW DONE"
