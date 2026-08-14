#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Model/run:
#    GPU0 proposal: context bump from the Scruffy-inspired L32 DEPTH run.
#    Shape: L32 H16 D1024 C1024 P16.
#    Batch schedule: b16 A6, 98,304 tokens/update.
#    No layer dropout.
#
# Useful overrides:
#    THOG2_DREEDLE_POWER_LIMIT_W=200 ./dreedle_simple_runstring_gpu_0.sh
#    THOG2_DREEDLE_APPLY_POWER_LIMIT=false ./dreedle_simple_runstring_gpu_0.sh
#    THOG2_MIN_FREE_GPU_MEMORY_MIB=21000 ./dreedle_simple_runstring_gpu_0.sh
#    THOG2_ALLOW_BUSY_GPU=true ./dreedle_simple_runstring_gpu_0.sh
#    THOG2_TORCH_COMPILE=regional ./dreedle_simple_runstring_gpu_0.sh
#
# Schedule/logging:
#    -e 250, -u 20 gives periodic validation without making the run validation-dominated.
#    -k 1000 gives ordinary checkpoint cadence.
#
# Geometry:
#    DEPTH chebyshev order 16.
#    LayerNorm and bias are not depth-compressed.
#
# Compilation:
#    THOG2_TORCH_COMPILE=false|true|regional        eager | whole-model compile | checkpoint-segment regional compile

# vvv THOG host profile consumed by the canonical train_OWT.sh wrapper
export THOG2_DREEDLE_GPU=0
export THOG2_DREEDLE_RUN_KIND="ctx1024_l32_d1024_p16"
export THOG2_DREEDLE_POWER_LIMIT_W="${THOG2_DREEDLE_POWER_LIMIT_W:-220}"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
export THOG2_NUM_GPUS=1
export THOG2_DTYPE="${THOG2_DTYPE:-float16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-sdpa}"
export THOG2_TORCH_COMPILE="${THOG2_TORCH_COMPILE:-false}"
# ^^^ THOG

source ./dreedle_gpu_common.sh
dreedle_prepare_single_gpu_launch

python -m run_thog2_owt --print-geometry-registry

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

./train_OWT.sh \
  -g DREEDLE_GPU${THOG2_DREEDLE_GPU}_${THOG2_DREEDLE_POWER_TAG}_CTX1024_L32_D1024_P16 \
  -n 40000 \
  -b 16 \
  -A 6 \
  -G "$THOG2_NUM_GPUS" \
  -S 4 \
  -u 20 \
  -e 250 \
  -l 10 \
  -w 100 \
  -k 500 \
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
  -T "$THOG2_DTYPE" \
  -K "$THOG2_ATTENTION_BACKEND" \
  -t "$THOG2_OWT_DATA_DIR" \
  --select-depth \
  --option DEPTH.compressor=chebyshev \
  --option DEPTH.order=16 \
  --no-depth-compress-layer-norm-and-bias \
  --depth-materialisation-matmul false \
  --materialisation-profiling    false \
  --torch-compile                false \
  -- \
  --host-label "$THOG2_HOST_LABEL"
