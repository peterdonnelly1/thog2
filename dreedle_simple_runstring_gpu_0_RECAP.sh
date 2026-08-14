#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Leader recapitulation on current development code.
# Shape: L32 H16 D1024 C1024 P16
# Batch: b16 A6 = 98,304 tokens/update
# DEPTH: Chebyshev order 16
# GPU: physical GPU0
# Power cap: explicit TITAN RTX maximum, 320 W
# Plastic capabilities present in codebase but not enabled
# Numerical safety (necessary for NVIDIA TITAN RTX) Non-finite update policy: current code defaults to skip with max_nonfinite_update_skips=99999.

export THOG2_DREEDLE_GPU=0
export THOG2_DREEDLE_POWER_LIMIT_W=320
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
export THOG2_NUM_GPUS=1
export THOG2_DTYPE=float16
export THOG2_ATTENTION_BACKEND=sdpa
export THOG2_TORCH_COMPILE=false

source ./dreedle_gpu_common.sh
dreedle_prepare_single_gpu_launch

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

./train_OWT.sh \
  -g DREEDLE_GPU${THOG2_DREEDLE_GPU}_${THOG2_DREEDLE_POWER_TAG}_CTX1024_L32_D1024_P16_LEADER_RECAP \
  -n 40000 \
  -b 16 \
  -A 6 \
  -G 1 \
  -S 4 \
  -u 10 \
  -e 250 \
  -l 10 \
  -w 100 \
  -k 250 \
  -y adamw \
  -c 90 \
  -f 9 \
  -L 32 \
  -H 16 \
  -D 1024 \
  -C 1024 \
  -P 16 \
  -E true \
  -r depth_scaled \
  -z dof_implied_depth \
  -I wandb \
  -F none \
  -T float16 \
  -K sdpa \
  -t "$THOG2_OWT_DATA_DIR" \
  --select-depth \
  --option DEPTH.compressor=chebyshev \
  --option DEPTH.order=16 \
  --no-depth-compress-layer-norm-and-bias \
  --depth-materialisation-matmul false \
  --materialisation-profiling false \
  --torch-compile false \
  -- \
  --host-label "$THOG2_HOST_LABEL"
