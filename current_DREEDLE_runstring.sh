#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${THOG2_PYTHON:-}" ]]; then
    python_bin="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_bin=".venv/bin/python"
else
    python_bin="python"
fi

# vvv THOG expose the same two Dreedle CUDA devices recorded in the original KARITANE_LONG checkpoint RNG state
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="0,1"
# ^^^ THOG
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export THOG2_INSTRUMENTATION="wandb"
export THOG2_FAST_DISCARD="${THOG2_FAST_DISCARD:-true}"
export THOG2_BYPASS_SEMANTIC_QKV_ADAPTER="${THOG2_BYPASS_SEMANTIC_QKV_ADAPTER:-true}"
export THOG2_DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"
export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="${THOG2_VECTORISE_PER_HEAD_MATERIALISATION:-true}"
export THOG2_DEPTH_CURVE_PLOTS="${THOG2_DEPTH_CURVE_PLOTS:-none}"

exec "$python_bin" -m run_karitane_long_resume "$@"
