#!/bin/bash
set -euo pipefail

# Model/run:
#    -b BATCH_SIZE                                  single integer, comma list, or quoted space list
#    -A GRADIENT_ACCUMULATION_STEPS                 single integer, comma list, or quoted space list in this runstring
#    -c LR_CODES                                    1..1000; 70 means 7.0e-04 and 1000 means 1.0e-02; comma or quoted space list
#    -f MIN_LR_CODE                                 1..100; 06 means 6.0e-05 and 100 means 1.0e-03
#    -G NUM_GPUS
#    -y OPTIMIZER                                   adamw | sgd | sgd_nesterov | adafactor | rmsprop
#    --optimizer-momentum VALUE                     momentum for SGD, Nesterov, and RMSprop; defaults: adamw 60/06, sgd 1000/100, sgd_nesterov 1000/100, adafactor 1000/100, rmsprop 100/10; explicit -c and -f override independently
#    -p PRESET                                      dense | legacy_sheet_col | depth | jpeg_like_v1 | head_aware_block | mlp_block | full_block; single value, comma list, or quoted space list
#    -q RUN_MODE                                    fresh | resume
#    -g RUN_NAME                                    auto when omitted
#    -n STEPS
# Schedule/logging:
#    -k CHECKPOINT_INTERVAL                         0 disables periodic saves
#    -V DEPTH_CURVE_LOCAL_HTML                      true | false
#    -F DEPTH_CURVE_PLOTS                           none | final | eval
#    -U DEPTH_CURVE_RENDERER                        matplotlib | plotly | both
#    -N DEPTH_CURVE_SAMPLE_ELEMENTS
#    -e EVAL_INTERVAL
#    -u EVAL_ITERS
#    -I INSTRUMENTATION                             tensorboard | wandb | wandb_offline | none
#    -l LOG_INTERVAL
#    -w WARMUP_ITERS
# Systematic geometry:
#    --explain-geometry                             resolve and report the geometry, then exit without loading data or constructing a model
#    --option TARGET.PROPERTY=VALUE                 repeat as needed
#    --select-depth
#    --select-element SELECTOR                      repeat as needed
# Compact geometry:
#    -a ATTENTION_GEOMETRY                          preset default when omitted
#    -B BASIS_FAMILY                                canonical: chebyshev | dct | haar | lapped_cosine; single value, comma list, or quoted space list
#    -B chebyshev aliases                           cheby | chebyshev_first_kind_qr
#    -B dct aliases                                 dct_ii | dct_ii_orthonormal
#    -B haar aliases                                balanced_haar | haar_balanced
#    -B lapped_cosine aliases                       lapped | local_cosine | lapped_local_cosine
#    -v BASIS_VERSION                               auto recommended | chebyshev_first_kind_qr_v1 | dct_ii_orthonormal_v1 | haar_balanced_binary_orthonormal_v1 | lapped_cosine_dc_preserving_orthonormal_v1
#    -i LAPPED_COSINE_OVERLAP_FRACTION              currently 0.5 only
#    -W LAPPED_COSINE_WINDOW_LENGTH
#    -m MLP_GEOMETRY                                preset default when omitted
#    --mlp-hidden-compressor NAME                   registered local compressor; comma or quoted space list allowed
#    --mlp-hidden-group-size N                      physical MLP_HIDDEN segment size; comma or quoted space list allowed
# Shape/runtime:
#    -K ATTENTION_BACKEND                           auto | flash2 | sdpa | math
#    -C BLOCK_SIZE
#    -S CHECKPOINT_SEGMENT_SIZE                     single integer, comma list, or quoted space list in this runstring
#    --depth-compress-layer-norm-and-bias           DEPTH only; default false
#    --no-depth-compress-layer-norm-and-bias        DEPTH only; explicit default
#    -T DTYPE                                       float32 | float16 | bfloat16
#    -E FAST_DISCARD                                true | false
#    -D N_EMBD
#    -H N_HEAD
#    -L N_LAYER                                     single integer, comma list, or quoted space list in this runstring
#    -Q O_ATTN_D_MODEL
#    -O O_ATTN_OUT_PER_CHANNEL
#    -J O_ATTN_QKV_PER_CHANNEL
#    -P O_DEPTH                                     single integer, comma list, or quoted space list; ignored by dense
#    -X O_MLP_D_MODEL
#    -Y O_MLP_HIDDEN                                ignored by DEPTH
# Residual init:
#    -r RESIDUAL_INIT_POLICY                        depth_scaled | unscaled
#    -z RESIDUAL_INIT_DEPTH_SOURCE                  true_layer_depth | dof_implied_depth | user_forced_depth
#    -Z RESIDUAL_INIT_DEPTH_VALUE
# Paths:
#    -o CHECKPOINT_ROOT
#    -t DATA_DIR
#    -d DATASET_NAME
#    -x DRY_RUN
#    -j LOG_ROOT
#    -R RESULT_ROOT


THOG2_GENERAL_GRADIENT_ACCUMULATION_STEPS="8"
THOG2_GENERAL_CHECKPOINT_SEGMENT_SIZES="4"
THOG2_GENERAL_N_LAYERS="96,128,256"


# vvv THOG host profile consumed by the canonical train_OWT.sh wrapper
export THOG2_HOST_LABEL="scruffy"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}"
export THOG2_NUM_GPUS="${THOG2_NUM_GPUS:-1}"
export THOG2_DTYPE="${THOG2_DTYPE:-bfloat16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-flash2}"
# ^^^ THOG

# vvv THOG temporary outer-grid axes not yet handled by train_OWT.sh
gradient_accumulation_steps_values="${THOG2_GENERAL_GRADIENT_ACCUMULATION_STEPS:-1}"
checkpoint_segment_size_values="${THOG2_GENERAL_CHECKPOINT_SEGMENT_SIZES:-12}"
n_layer_values="${THOG2_GENERAL_N_LAYERS:-144}"


normalize_grid_values() {
  printf '%s\n' "${1//,/ }"
}

read -r -a gradient_accumulation_values <<< "$(normalize_grid_values "$gradient_accumulation_steps_values")"
read -r -a checkpoint_segment_values <<< "$(normalize_grid_values "$checkpoint_segment_size_values")"
read -r -a layer_values <<< "$(normalize_grid_values "$n_layer_values")"

for value in "${gradient_accumulation_values[@]}" "${checkpoint_segment_values[@]}" "${layer_values[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Grid values for -A, -S, and -L must be positive integers: $value" >&2; exit 2; }
done

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

for gradient_accumulation_steps in "${gradient_accumulation_values[@]}"; do
  for checkpoint_segment_size in "${checkpoint_segment_values[@]}"; do
    for n_layer in "${layer_values[@]}"; do
      ./train_OWT.sh \
        -g "REVAMP__NLAYERS_SWEEP_L${n_layer}" \
        -n 1000 \
        -b 16 \
        -A "$gradient_accumulation_steps" \
        -S "$checkpoint_segment_size" \
        -u 1 \
        -e 99999 \
        -l 10 \
        -w 100 \
        -k 500 \
        -L "$n_layer" \
        -H 16 \
        -D 1024 \
        -C 768 \
        -Y 64 \
        -E true \
        -T bfloat16 \
        -K flash2 \
        -I wandb \
        -F none \
        -y adamw \
        --select-depth \
        --option DEPTH.compressor=chebyshev \
        --select-element MLP_UP.MLP_HIDDEN \
        --option DEPTH.order=16 \
        --option MLP_UP.compressor=jpeg_like \
        --option MLP_UP.MLP_HIDDEN.order=64 \
        --option MLP_UP.MLP_HIDDEN.group_size=256 \
        -- \
        --host-label "$THOG2_HOST_LABEL"
    done
  done
done