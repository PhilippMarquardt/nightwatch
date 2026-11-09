#!/usr/bin/env bash
# Re-infer the dead-thermal forest frames with the _level_match guard, then merge into pred_test_guard.jsonl and score.
#   WAIT="<log>:<marker>" RUNS="t_sft_tri ..." bash scripts/reinfer_deadT.sh
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NW_GPU_FRAC=${EVAL_FRAC:-0.85}
if [ -n "${WAIT:-}" ]; then until grep -q "${WAIT#*:}" "${WAIT%%:*}" 2>/dev/null; do sleep 60; done; fi
python - <<'PY'
import json, numpy as np
from PIL import Image
fr = [f for f in json.load(open("data/trisar_fused/frames.json")) if f["split"] == "test"]
ids = [f["id"] for f in fr if np.asarray(Image.open(f"data/trisar_fused/{f['id']}_t.png").convert("L"), np.float32).std() < 2]
json.dump(ids, open("data/trisar_deadT_ids.json", "w")); print("dead-thermal test frames:", len(ids))
PY
for r in $RUNS; do
  [ -f runs/$r/adapter_model.safetensors ] || { echo "skip $r"; continue; }
  mode=$(python -c "import json; print(json.load(open('runs/$r/run_meta.json'))['config']['mode'])")
  [ -s runs/$r/pred_test_deadT.jsonl ] || NW_ONLY_IDS=data/trisar_deadT_ids.json python -m nightwatch.slow.infer --dataset trisar \
      --fused-dir data/trisar_fused --model Qwen/Qwen3.5-9B --bs 32 --workers 12 --mode $mode --adapter runs/$r --split test \
      --tasks report --conds normal --out runs/$r/pred_test_deadT.jsonl
  python - "$r" <<'PY'
import json, sys
r = sys.argv[1]
new = {d["id"]: d for d in map(json.loads, open(f"runs/{r}/pred_test_deadT.jsonl"))}
with open(f"runs/{r}/pred_test_guard.jsonl", "w") as fh:
    for d in map(json.loads, open(f"runs/{r}/pred_test.jsonl")):
        if d.get("task") == "report" and d.get("cond") == "normal" and d["id"] in new:
            d = new[d["id"]]
        fh.write(json.dumps(d) + "\n")
PY
  python -m nightwatch.eval.metrics_trisar --pred runs/$r/pred_test_guard.jsonl --out runs/$r/pred_test_guard_metrics.json > /dev/null
  echo "GUARD_DONE $r"
done
echo REINFER_ALL_DONE
