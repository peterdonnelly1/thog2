#!/usr/bin/env bash
# vvv THOG
set -euo pipefail

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
PY

python -m pytest -q \
    tests/test_plastic_depth_coarse_fine_gpu_smoke.py \
    tests/test_plastic_depth_cuda.py \
    tests/test_plastic_depth_full_radius_oom.py
# ^^^ THOG
