#!/usr/bin/env bash
# Track B home-regression check: score every adapted head's LLVIP test predictions at its own (LLVIP-val) threshold,
# then print per-arm LLVIP F1 over seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
for d in runs/ltd_*_s[0-9]; do
  [ -f "$d/pred_llvip_test.jsonl" ] || continue
  [ -f "$d/llvip_regress_metrics.json" ] && continue
  thr=$(python -c "import json; print(json.load(open('$d/train_summary.json'))['score_thr'])")
  python -m nightwatch.eval.metrics --pred "$d/pred_llvip_test.jsonl" --score-thr "$thr" --out "$d/llvip_regress_metrics.json" > /dev/null
done
python - <<'EOF'
import json, os
import numpy as np
for a in ["B0p", "B1_self", "B2_vlm", "B3_consensus", "B4_gt"]:
    v = [json.load(open(p))["detect/normal"]["all"]["f1"] for s in (0, 1, 2)
         if os.path.exists(p := f"runs/ltd_{a}_s{s}/llvip_regress_metrics.json")]
    print(f"{a:13s} LLVIP F1 {np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f}) n={len(v)}")
EOF
