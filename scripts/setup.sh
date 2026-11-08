#!/usr/bin/env bash
# One-shot setup on a fresh Linux GPU box (A100 / H100 / H200).
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip install -q -r requirements.txt
pip install -q hf_transfer || true
# Qwen3.5 uses Gated DeltaNet linear attention: fast kernels (Triton) + causal-conv1d, else slow torch fallback
pip install -q flash-linear-attention || echo "fla not installed - slow fallback"
pip install -q causal-conv1d --no-build-isolation || echo "causal-conv1d not installed - torch fallback"
# TensorRT is optional (fast-head export); the pipeline falls back to ONNX Runtime + torch timings.
pip install -q tensorrt-cu12 || pip install -q tensorrt || echo "tensorrt not installed - will fall back"

mkdir -p data/raw
if [ ! -d data/raw/LLVIP/visible ]; then
  python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("UserNae3/LLVIP", "LLVIP.zip", repo_type="dataset", local_dir="data/raw")
EOF
  (cd data/raw && unzip -q LLVIP.zip && rm -f LLVIP.zip)
fi
[ -f data/index.json ] || python -m nightwatch.data.llvip --root data/raw/LLVIP --out data/index.json

python - <<'EOF'
from huggingface_hub import snapshot_download
import timm, os
m = os.environ.get("NW_MODEL", "Qwen/Qwen3.5-9B")
print(snapshot_download(m))
timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=True)
timm.create_model("resnet18.a1_in1k", pretrained=True)
print("models ready")
EOF
python -m nightwatch.data.tasks
echo "SETUP DONE"
