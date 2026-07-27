
#  Model/run:
#    -p PRESET=${GEOMETRY_PRESET}                       dense | legacy_sheet_col | depth | head_aware_block | mlp_block | full_block
#                                                     single value, comma list, or quoted space list
#    -q RUN_MODE=${RUN_MODE}                        fresh | resume
#    -g RUN_NAME=${RUN_NAME:-auto}
#    -n STEPS=${STEPS}
#    -b BATCH_SIZE=${BATCH_SIZE}                         single integer, comma list, or quoted space list
#    -c LR_CODES=${LEARNING_RATE_CODES}                    1..1000; 70 means 7.0e-04 and 1000 means 1.0e-02; comma or quoted space list
#    -f MIN_LR_CODE=${MIN_LR_CODE}                         1..100; 06 means 6.0e-05 and 100 means 1.0e-03
#    -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
#    -G NUM_GPUS=${NUM_GPUS}
#    -y OPTIMIZER=${OPTIMIZER}                         adamw | sgd | sgd_nesterov | adafactor | rmsprop
#    --optimizer-momentum VALUE=${OPTIMIZER_MOMENTUM}  momentum for SGD/Nesterov/RMSprop
#                                                       defaults: adamw 60/06; sgd 1000/100;
#                                                       sgd_nesterov 1000/100; adafactor 1000/100;
#                                                       rmsprop 100/10. Explicit -c/-f override independently.
#  
#  Schedule/logging:
#    -u EVAL_ITERS=${EVAL_ITERS}
#    -e EVAL_INTERVAL=${EVAL_INTERVAL}
#    -l LOG_INTERVAL=${LOG_INTERVAL}
#    -w WARMUP_ITERS=${WARMUP_ITERS}
#    -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}     0 disables periodic saves
#    -I INSTRUMENTATION=${INSTRUMENTATION}             tensorboard | wandb | wandb_offline | none
#    -F DEPTH_CURVE_PLOTS=${DEPTH_CURVE_PLOTS}         none | final | eval
#    -N DEPTH_CURVE_SAMPLE_ELEMENTS=${DEPTH_CURVE_SAMPLE_ELEMENTS}
#    -U DEPTH_CURVE_RENDERER=${DEPTH_CURVE_RENDERER}   matplotlib | plotly | both
#    -V DEPTH_CURVE_LOCAL_HTML=${DEPTH_CURVE_LOCAL_HTML}  true | false
#  
#  Compact geometry:
#    -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar | lapped_cosine; single, comma, or quoted space list
#                                                      Chebyshev aliases: cheby | chebyshev_first_kind_qr
#                                                      DCT aliases: dct_ii | dct_ii_orthonormal
#                                                      Haar aliases: balanced_haar | haar_balanced
#                                                       Lapped cosine aliases: lapped | local_cosine | lapped_local_cosine
#    -v BASIS_VERSION=${BASIS_VERSION}                 auto (recommended), or exact:
#                                                      chebyshev_first_kind_qr_v1
#                                                      dct_ii_orthonormal_v1
#                                                      haar_balanced_binary_orthonormal_v1
#                                                       lapped_cosine_dc_preserving_orthonormal_v1
#    -W LAPPED_COSINE_WINDOW_LENGTH=${LAPPED_COSINE_WINDOW_LENGTH}
#    -i LAPPED_COSINE_OVERLAP_FRACTION=${LAPPED_COSINE_OVERLAP_FRACTION}  currently 0.5 only
#    -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
#    -m MLP_GEOMETRY=${MLP_GEOMETRY:-preset default}
#  
#  Shape/runtime:
#    -L N_LAYER=${N_LAYER}
#    -H N_HEAD=${N_HEAD}
#    -D N_EMBD=${N_EMBD}
#    -C BLOCK_SIZE=${BLOCK_SIZE}
#    -P O_DEPTH=${O_DEPTH}                             single integer, comma list, or quoted space list; ignored by dense
#    -Q O_ATTN_D_MODEL=${O_ATTN_D_MODEL}
#    -J O_ATTN_QKV_PER_CHANNEL=${O_ATTN_QKV_PER_CHANNEL}
#    -O O_ATTN_OUT_PER_CHANNEL=${O_ATTN_OUT_PER_CHANNEL}
#    -X O_MLP_D_MODEL=${O_MLP_D_MODEL}
#    -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}
#    -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
#    -E FAST_DISCARD=${FAST_DISCARD}                   true | false
#    -T DTYPE=${DTYPE}                                 float32 | float16 | bfloat16
#    -K ATTENTION_BACKEND=${ATTENTION_BACKEND}         auto | flash2 | sdpa | math
#  
#  Residual init:
#    -r RESIDUAL_INIT_POLICY=${RESIDUAL_INIT_POLICY}                 depth_scaled | unscaled
#    -z RESIDUAL_INIT_DEPTH_SOURCE=${RESIDUAL_INIT_DEPTH_SOURCE}     true_layer_depth | dof_implied_depth | user_forced_depth
#    -Z RESIDUAL_INIT_DEPTH_VALUE=${RESIDUAL_INIT_DEPTH_VALUE}
#  
#  Paths:
#    -d DATASET_NAME=${DATASET_NAME}
#    -t DATA_DIR=${DATA_DIR}
#    -o CHECKPOINT_ROOT=${CHECKPOINT_ROOT}
#    -j LOG_ROOT=${LOG_ROOT}
#    -R RESULT_ROOT=${RESULT_ROOT}
#    -x DRY_RUN=${DRY_RUN}
#    -h show this help


for W in 12 24 36; do
  ./current_scruffy_train_OWT.sh \
    -g "LAPPED_WINDOW_SWEEP_1000_W${W}" \
    -p depth \
    -B lapped_cosine \
    -y sgd \
    -n 1000 \
    -b 16 \
    -A 1 \
    -l 5 \
    -u 5 \
    -e 50 \
    -w 30 \
    -k 0 \
    -L 144 \
    -H 12 \
    -D 768 \
    -C 256 \
    -P 32 \
    -Q 256 \
    -J 6 \
    -O 6 \
    -X 64 \
    -Y 256 \
    -S 12 \
    -E true \
    -I wandb \
    -F none \
    -W "$W" \
    -i 0.5
done