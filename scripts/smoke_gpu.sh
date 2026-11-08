#!/usr/bin/env bash
# 10-minute end-to-end check on the real GPU + real model before launching the long runs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${NW_MODEL:-Qwen/Qwen3.5-9B}
mkdir -p logs
python - <<'EOF'
import torch, transformers, importlib
print("torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name())
for m in ["fla", "causal_conv1d", "kernels"]:
    try:
        importlib.import_module(m); print(m, "ok")
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
EOF
python -m nightwatch.slow.train --model $MODEL --mode fusion --out runs/smoke_gpu_fusion --bs 8 --grad-accum 1 \
    --max-steps 12 --workers 8 --max-samples 200 --grad-ckpt 2>&1 | grep -E "^mode|step=|saved|Error|error|Traceback" | grep -v Warn
python -m nightwatch.slow.infer --model $MODEL --mode fusion --adapter runs/smoke_gpu_fusion --split val --limit 96 --bs 48 \
    --workers 8 --conds normal --out runs/smoke_gpu_fusion/pred_val.jsonl 2>&1 | grep -E "img/s|Error|Traceback"
python -m nightwatch.slow.infer --model $MODEL --mode dual --split val --limit 48 --bs 24 --workers 8 \
    --out runs/smoke_gpu_dual/pred_val.jsonl 2>&1 | grep -E "img/s|Error|Traceback"
python -m nightwatch.eval.metrics --pred runs/smoke_gpu_fusion/pred_val.jsonl --split val --out runs/smoke_gpu_fusion/m.json
echo SMOKE_DONE
