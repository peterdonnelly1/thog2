#!/bin/bash
set -euo pipefail

# Model/run:
#    -b BATCH_SIZE                                  single integer, comma list, or quoted space list
#    -A GRADIENT_ACCUMULATION_STEPS
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
#    -S CHECKPOINT_SEGMENT_SIZE
#    --depth-compress-layer-norm-and-bias           DEPTH only; default false
#    --no-depth-compress-layer-norm-and-bias        DEPTH only; explicit default
#    -T DTYPE                                       float32 | float16 | bfloat16
#    -E FAST_DISCARD                                true | false
#    -D N_EMBD
#    -H N_HEAD
#    -L N_LAYER
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

# vvv THOG host profile consumed by the canonical train_OWT.sh wrapper
export CUDA_VISIBLE_DEVICES="1"
export THOG2_HOST_LABEL="dreedle"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
export THOG2_NUM_GPUS="${THOG2_NUM_GPUS:-1}"
export THOG2_DTYPE="${THOG2_DTYPE:-float16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-sdpa}"
# ^^^ THOG

python -m run_thog2_owt --print-geometry-registry

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off


./train_OWT.sh \
  -g REVAMPv1_GPU1 \
  -n 10000 \
  -b 16 \
  -A 8 \
  -G "$THOG2_NUM_GPUS" \
  -S 4 \
  -u 1 \
  -e 10001 \
  -l 10 \
  -w 100 \
  -k 1000 \
  -y adamw \
  -c 90 \
  -f 9 \
  -L 32 \
  -H 16 \
  -D 1024 \
  -C 768 \
  -Y 64 \
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
  -- \
  --host-label "$THOG2_HOST_LABEL"

  #  --select-depth \
#  --option DEPTH.compressor=chebyshev \
#  --select-element MLP_UP.MLP_HIDDEN \
#  --option DEPTH.order=16 \
#  --option MLP_UP.compressor=jpeg_like \
#  --option MLP_UP.MLP_HIDDEN.order=64 \
#  --option MLP_UP.MLP_HIDDEN.group_size=256 \
#  -- \
#  --host-label "$THOG2_HOST_LABEL"