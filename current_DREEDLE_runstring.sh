#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

checkpoint_path="checkpoints/SHEET_dreedle__KARITANE_LONG_260706_145723__n_99999_b_12_d_owt_w_20_k_500_A_4_L_144_H_32_D_2048_C_256_P_80_Q_256_r_depth_scaled_z_dof_implied_depth_S_12/ckpt.pt"

if [[ -n "${THOG2_PYTHON:-}" ]]; then
    python_bin="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_bin=".venv/bin/python"
else
    python_bin="python"
fi

[[ -f "$checkpoint_path" ]] || {
    echo "Missing KARITANE_LONG checkpoint:"
    echo "  $checkpoint_path"
    exit 2
}

help_text="$("$python_bin" -m run_thog2_owt_residual --help 2>&1 || true)"
grep -q -- "--depth-order" <<<"$help_text" || {
    echo "This checkout does not expose the legacy KARITANE_LONG resume CLI."
    exit 2
}
grep -q -- "--nonfinite-update-policy" <<<"$help_text" || {
    echo "This checkout lacks bounded non-finite update recovery."
    exit 2
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export THOG2_INSTRUMENTATION="wandb"

run_args=(
    --model-type sheet
    --run-mode resume
    --host-label dreedle
    --run-name KARITANE_LONG_260706_145723
    --dataset openwebtext
    --data-dir data/openwebtext
    --checkpoint-root checkpoints
    --log-root logs
    --result-root results
    --wandb-root wandb
    --max-iters 99999
    --batch-size 12
    --gradient-accumulation-steps 4
    --block-size 256
    --n-layer 144
    --n-head 32
    --n-embd 2048
    --depth-order 80
    --base-row-order 256
    --learning-rate 6e-4
    --min-lr 6e-5
    --warmup-iters 20
    --weight-decay 0.1
    --beta1 0.9
    --beta2 0.95
    --grad-clip 1.0
    --eval-interval 100
    --eval-iters 10
    --log-interval 10
    --checkpoint-interval 500
    --residual-init-policy depth_scaled
    --residual-init-depth-source dof_implied_depth
    --activation-checkpointing
    --checkpoint-segment-size 12
    --dtype float16
    --device cuda
    --nonfinite-update-policy skip
    --max-nonfinite-update-skips 10
    --wandb
    --wandb-project thog
    --wandb-mode online
)

resolved_json="$("$python_bin" -m run_thog2_owt_residual "${run_args[@]}" --print-resolved-json)"
resolved_checkpoint="$(
    "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["paths"]["checkpoint_path"])' \
        <<<"$resolved_json"
)"

if [[ "$resolved_checkpoint" != "$checkpoint_path" ]]; then
    echo "Legacy resolver selected the wrong checkpoint path."
    echo "Expected: $checkpoint_path"
    echo "Resolved: $resolved_checkpoint"
    exit 2
fi

log_dir="logs/KARITANE_LONG_RESUME_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
log_path="$log_dir/train.log"

echo "Resuming KARITANE_LONG"
echo "  checkpoint: $checkpoint_path"
echo "  log:        $log_path"
echo "  batch/accum: 12 / 4"
echo "  context:     256"
echo "  LR schedule: 6e-4 cosine to 6e-5 through update 99999"
echo "  validation:  10 batches every 100 updates"
echo

set +e
"$python_bin" -m run_thog2_owt_residual "${run_args[@]}" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

exit "$status"
