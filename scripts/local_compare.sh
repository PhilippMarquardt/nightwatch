#!/usr/bin/env bash
# Local sanity check on a 16 GB GPU with Qwen3-VL-2B: does fusion learn, vs zero-shot?
set -euo pipefail
M=models/Qwen3-VL-2B-Instruct
python -m nightwatch.slow.train --model $M --mode fusion --out runs/local2b_fusion_v2 --bs 2 --grad-accum 2 --max-steps 150 --workers 4 --grad-ckpt --lr 2e-4
python -m nightwatch.slow.infer --model $M --mode fusion --adapter runs/local2b_fusion_v2 --split val --limit 80 --bs 8 --workers 4 --conds normal,rgb_off --out runs/local2b_fusion_v2/pred_val.jsonl
for r in local2b_fusion local2b_fusion_v2 local2b_zs; do
  python -m nightwatch.eval.metrics --pred runs/$r/pred_val.jsonl --split val --out runs/$r/m_val.json
done
