#!/usr/bin/env bash
# vvv THOG
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${THOG2_REPO_DIR:-$script_dir}"

data_dir="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
attention_backend="${PLASTIC_SMOKE_BACKEND:-flash2}"
dtype="${PLASTIC_SMOKE_DTYPE:-bfloat16}"
memory_budget_gib="${PLASTIC_SMOKE_MEMORY_BUDGET_GIB:-15.0}"
smoke_tag="$(date +%y%m%d-%H%M%S)"

if [[ -n "${THOG2_PYTHON:-}" ]]; then
  python_bin="$THOG2_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "ERROR: no Python interpreter found. Activate the THOG2 environment or set THOG2_PYTHON." >&2
  exit 1
fi

if ! "$python_bin" -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: $python_bin cannot import torch. Activate the THOG2 environment or set THOG2_PYTHON." >&2
  exit 1
fi

echo "plastic smoke python:  $python_bin"
echo "plastic smoke backend: $attention_backend"
echo "plastic smoke dtype:   $dtype"
echo "plastic smoke data:    $data_dir"

if [[ "${PLASTIC_SMOKE_PREFLIGHT_ONLY:-false}" == "true" ]]; then
  exit 0
fi

common_args=(
  --model-type sheet
  --select-depth
  --option DEPTH.compressor=chebyshev
  --option DEPTH.order=8
  --data-dir "$data_dir"
  --max-iters 8
  --eval-interval 10001
  --eval-iters 1
  --log-interval 1
  --checkpoint-interval 4
  --batch-size 4
  --gradient-accumulation-steps 2
  --block-size 256
  --n-layer 16
  --n-head 8
  --n-embd 512
  --o-depth 8
  --plastic-enabled
  --plastic-geometry-learning-rate-multiplier 0.1
  --plastic-freeze-geometry-during-warmup
  --attention-backend "$attention_backend"
  --residual-init-policy depth_scaled
  --residual-init-depth-source dof_implied_depth
  --learning-rate 1e-4
  --min-lr 1e-5
  --warmup-iters 1
  --weight-decay 0.1
  --beta1 0.9
  --beta2 0.95
  --grad-clip 1
  --device cuda
  --dtype "$dtype"
  --no-wandb
)

echo '=== 1/4 fixed count, random learned geometry ==='
"$python_bin" run_thog2_owt.py \
  "${common_args[@]}" \
  --run-name PLASTIC_FIXED_GPU_SMOKE \
  --artifact-suffix "${smoke_tag}_fixed" \
  --no-plastic-do-learn-layer-count \
  --plastic-layers-to-sample 16 \
  --plastic-layer-sampling-initialisation random

echo '=== 2/4 learned count, lowest loss ==='
"$python_bin" run_thog2_owt.py \
  "${common_args[@]}" \
  --run-name PLASTIC_LOWEST_LOSS_GPU_SMOKE \
  --artifact-suffix "${smoke_tag}_lowest_loss" \
  --plastic-do-learn-layer-count \
  --plastic-initial-layer-count 6 \
  --plastic-max-permitted-layers 12 \
  --plastic-layer-sampling-initialisation equidistant \
  --plastic-layer-count-objective lowest_loss \
  --plastic-layer-count-hold-updates 2

echo '=== 3/4 learned count, relative training wall time ==='
"$python_bin" run_thog2_owt.py \
  "${common_args[@]}" \
  --run-name PLASTIC_WALL_TIME_GPU_SMOKE \
  --artifact-suffix "${smoke_tag}_wall_time" \
  --plastic-do-learn-layer-count \
  --plastic-initial-layer-count 6 \
  --plastic-max-permitted-layers 12 \
  --plastic-layer-sampling-initialisation equidistant \
  --plastic-layer-count-objective relative_training_wall_time \
  --plastic-layer-count-hold-updates 2 \
  --plastic-layer-count-cost-weight 0.02

echo '=== 4/4 learned count, memory budget ==='
"$python_bin" run_thog2_owt.py \
  "${common_args[@]}" \
  --run-name PLASTIC_MEMORY_GPU_SMOKE \
  --artifact-suffix "${smoke_tag}_memory" \
  --plastic-do-learn-layer-count \
  --plastic-initial-layer-count 6 \
  --plastic-max-permitted-layers 12 \
  --plastic-layer-sampling-initialisation equidistant \
  --plastic-layer-count-objective memory_budget \
  --plastic-layer-count-hold-updates 2 \
  --plastic-layer-memory-budget-gib "$memory_budget_gib"
# ^^^ THOG
