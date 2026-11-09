#!/usr/bin/env bash
# Download model weights in parallel with setup.sh's package build (pod 2, 2026-11-09).
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download
import timm
for m in ["Qwen/Qwen3.5-9B"]:
    print(snapshot_download(m))
timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=True)
print("PREFETCH_DONE")
PY
