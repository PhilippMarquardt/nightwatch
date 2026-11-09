#!/usr/bin/env bash
# Point-cloud encoder deps for the Utonia experiment. Never touches torch (--no-deps where it could pull it).
set -x
cd /root
pip install -q --break-system-packages spconv-cu126 2>&1 | tail -2
pip install -q --break-system-packages --no-deps torch-scatter -f https://data.pyg.org/whl/torch-2.13.0+cu129.html 2>&1 | tail -2
python -c "import torch_scatter" 2>/dev/null || pip install -q --break-system-packages --no-deps --no-build-isolation torch-scatter 2>&1 | tail -3
pip install -q --break-system-packages addict 2>&1 | tail -1
[ -d Utonia ] || git clone -q https://github.com/Pointcept/Utonia.git
python -c "import torch; import spconv.pytorch as s; import torch_scatter; print('deps ok', torch.__version__)"
echo INSTALL_DONE
