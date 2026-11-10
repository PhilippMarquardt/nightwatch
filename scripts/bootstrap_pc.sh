#!/usr/bin/env bash
# Fresh pod -> ready for the point-cloud (tri_pc) experiments: deps, Qwen3.5-9B, TriSAR raw + render, Utonia features.
#   bash scripts/bootstrap_pc.sh        (resumable; each step skips itself when done)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs data/raw/trisar
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_ENABLE_HF_TRANSFER=1

# 1. raw data download in the background (the slow part)
(
  if ! ls data/raw/trisar/*/dataset.json >/dev/null 2>&1 && [ ! -f data/raw/trisar/dataset.json ]; then
    curl -sS -L -C - --retry 8 -o data/raw/trisar/trisar.zip \
      "https://tudatalib.ulb.tu-darmstadt.de/server/api/core/bitstreams/cbdb5411-3156-48fd-9d73-897925972fc0/content" \
      && (cd data/raw/trisar && unzip -q trisar.zip && rm -f trisar.zip)
  fi
  echo RAW_DONE
) > logs/boot_raw.log 2>&1 &
RAW=$!

# 2. python deps
pip install -q -r requirements.txt 2>&1 | tail -2
pip install -q hf_transfer scikit-learn 2>&1 | tail -1
pip install -q flash-linear-attention 2>&1 | tail -1 || echo "fla not installed - slow fallback"
pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -1 || echo "causal-conv1d not installed - torch fallback"
python -c "import torch, transformers, peft; print('deps', torch.__version__, transformers.__version__, peft.__version__)"
echo DEPS_DONE

# 3. model weights (in the background) + Utonia deps
( bash scripts/prefetch_models.sh > logs/boot_models.log 2>&1 ) &
MODELS=$!
bash scripts/install_utonia.sh 2>&1 | tail -3

# 4. render the fused frames
wait $RAW; tail -2 logs/boot_raw.log
if [ ! -s data/trisar_fused/frames.json ]; then
  python -m nightwatch.data.trisar --root data/raw/trisar --index data/trisar_index.json --out data/trisar_fused \
      --stride 2 --workers 48 2>&1 | tail -5
fi
python -c "import json; f=json.load(open('data/trisar_fused/frames.json')); print('frames', len(f))"
echo RENDER_DONE

# 5. frozen Utonia token grids
NSHARD=8 bash scripts/pc_encode_all.sh 2>&1 | tail -4
wait $MODELS; tail -1 logs/boot_models.log
echo BOOTSTRAP_DONE
