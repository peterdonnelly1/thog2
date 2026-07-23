from pathlib import Path

root = Path(__file__).resolve().parent
wrapper_path = root / "current_scruffy_train_OWT.sh"
test_path = root / "tests/test_sheet_stage6_runner_scripts.py"
depth_test_path = root / "tests/test_depth_layer_norm_bias_modes.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


text = wrapper_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "O_MLP_HIDDEN=256\n",
    "O_MLP_HIDDEN=256\nDEPTH_COMPRESS_LAYER_NORM_AND_BIAS=false                                                                                                                  # <<< THOG DEPTH-only repeated LayerNorm/bias participation switch\n",
    "default",
)
text = replace_once(
    text,
    "  -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}\n",
    "  -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}                    ignored by DEPTH\n  --depth-compress-layer-norm-and-bias                   DEPTH only; default false\n  --no-depth-compress-layer-norm-and-bias                DEPTH only; explicit default\n",
    "usage",
)
text = replace_once(
    text,
    "    --optimizer)\n",
    "    --depth-compress-layer-norm-and-bias)\n      DEPTH_COMPRESS_LAYER_NORM_AND_BIAS=true\n      shift\n      ;;\n    --no-depth-compress-layer-norm-and-bias)\n      DEPTH_COMPRESS_LAYER_NORM_AND_BIAS=false\n      shift\n      ;;\n    --optimizer)\n",
    "long options",
)
text = replace_once(
    text,
    "HAS_JPEG_LIKE_PRESET=false\n",
    "HAS_JPEG_LIKE_PRESET=false\nHAS_NON_DEPTH_COMPACT_PRESET=false                                                                                                                         # <<< THOG dead Q/J/O/X/Y controls must not constrain pure DEPTH runs\n",
    "preset flag",
)
text = replace_once(
    text,
    "      legacy_sheet_col|depth|head_aware_block|mlp_block|full_block) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true ;;\n      jpeg_like_v1) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true; HAS_JPEG_LIKE_PRESET=true ;;\n",
    "      depth) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true ;;\n      legacy_sheet_col|head_aware_block|mlp_block|full_block) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true ;;\n      jpeg_like_v1) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true; HAS_JPEG_LIKE_PRESET=true ;;\n",
    "preset parsing",
)
text = replace_once(
    text,
    "for setting in \"$STEPS\" \"$GRADIENT_ACCUMULATION_STEPS\" \"$NUM_GPUS\" \"$EVAL_ITERS\" \"$EVAL_INTERVAL\" \"$LOG_INTERVAL\" \"$N_LAYER\" \"$N_HEAD\" \"$N_EMBD\" \"$BLOCK_SIZE\" \"$O_ATTN_D_MODEL\" \"$O_ATTN_QKV_PER_CHANNEL\" \"$O_ATTN_OUT_PER_CHANNEL\" \"$O_MLP_D_MODEL\" \"$O_MLP_HIDDEN\" \"$CHECKPOINT_SEGMENT_SIZE\" \"$RESIDUAL_INIT_DEPTH_VALUE\" \"$DEPTH_CURVE_SAMPLE_ELEMENTS\" \"$LAPPED_COSINE_WINDOW_LENGTH\"; do validate_positive_uint \"$setting\" \"numeric setting\"; done\n",
    "for setting in \"$STEPS\" \"$GRADIENT_ACCUMULATION_STEPS\" \"$NUM_GPUS\" \"$EVAL_ITERS\" \"$EVAL_INTERVAL\" \"$LOG_INTERVAL\" \"$N_LAYER\" \"$N_HEAD\" \"$N_EMBD\" \"$BLOCK_SIZE\" \"$CHECKPOINT_SEGMENT_SIZE\" \"$RESIDUAL_INIT_DEPTH_VALUE\" \"$DEPTH_CURVE_SAMPLE_ELEMENTS\" \"$LAPPED_COSINE_WINDOW_LENGTH\"; do validate_positive_uint \"$setting\" \"numeric setting\"; done\nif [[ \"$HAS_NON_DEPTH_COMPACT_PRESET\" == true ]]; then\n  for setting in \"$O_ATTN_D_MODEL\" \"$O_ATTN_QKV_PER_CHANNEL\" \"$O_ATTN_OUT_PER_CHANNEL\" \"$O_MLP_D_MODEL\" \"$O_MLP_HIDDEN\"; do validate_positive_uint \"$setting\" \"non-DEPTH compact order\"; done\nfi\n",
    "positive validation",
)
text = replace_once(
    text,
    "validate_true_false \"$VECTORISE_PER_HEAD_MATERIALISATION\" \"VECTORISE_PER_HEAD_MATERIALISATION\"                                                         # <<< THOG validate per-head option                                                                          # <<< THOG validate wrapper-only exact MLP application switch\n",
    "validate_true_false \"$VECTORISE_PER_HEAD_MATERIALISATION\" \"VECTORISE_PER_HEAD_MATERIALISATION\"                                                         # <<< THOG validate per-head option                                                                          # <<< THOG validate wrapper-only exact MLP application switch\nvalidate_true_false \"$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS\" \"DEPTH_COMPRESS_LAYER_NORM_AND_BIAS\"                                                           # <<< THOG validate DEPTH vector participation switch\n",
    "bool validation",
)
text = replace_once(
    text,
    "if [[ \"$HAS_COMPACT_PRESET\" == true ]]; then\n  for value in \"${O_DEPTH_VALUES[@]}\"; do (( value <= N_LAYER )) || { echo \"O_DEPTH must not exceed N_LAYER: P=${value}, L=${N_LAYER}.\" >&2; exit 2; }; done\n  (( O_ATTN_D_MODEL <= N_EMBD )) || { echo \"O_ATTN_D_MODEL must not exceed N_EMBD.\" >&2; exit 2; }\n  (( O_ATTN_QKV_PER_CHANNEL <= HEAD_DIM )) || { echo \"O_ATTN_QKV_PER_CHANNEL must not exceed N_EMBD/N_HEAD.\" >&2; exit 2; }\n  (( O_ATTN_OUT_PER_CHANNEL <= HEAD_DIM )) || { echo \"O_ATTN_OUT_PER_CHANNEL must not exceed N_EMBD/N_HEAD.\" >&2; exit 2; }\n  (( O_MLP_D_MODEL <= N_EMBD )) || { echo \"O_MLP_D_MODEL must not exceed N_EMBD.\" >&2; exit 2; }\n  (( O_MLP_HIDDEN <= 4 * N_EMBD )) || { echo \"O_MLP_HIDDEN must not exceed 4*N_EMBD.\" >&2; exit 2; }\n  if [[ \"$HAS_JPEG_LIKE_PRESET\" == true ]]; then\n",
    "if [[ \"$HAS_COMPACT_PRESET\" == true ]]; then\n  for value in \"${O_DEPTH_VALUES[@]}\"; do (( value <= N_LAYER )) || { echo \"O_DEPTH must not exceed N_LAYER: P=${value}, L=${N_LAYER}.\" >&2; exit 2; }; done\nfi\nif [[ \"$HAS_NON_DEPTH_COMPACT_PRESET\" == true ]]; then\n  (( O_ATTN_D_MODEL <= N_EMBD )) || { echo \"O_ATTN_D_MODEL must not exceed N_EMBD.\" >&2; exit 2; }\n  (( O_ATTN_QKV_PER_CHANNEL <= HEAD_DIM )) || { echo \"O_ATTN_QKV_PER_CHANNEL must not exceed N_EMBD/N_HEAD.\" >&2; exit 2; }\n  (( O_ATTN_OUT_PER_CHANNEL <= HEAD_DIM )) || { echo \"O_ATTN_OUT_PER_CHANNEL must not exceed N_EMBD/N_HEAD.\" >&2; exit 2; }\n  (( O_MLP_D_MODEL <= N_EMBD )) || { echo \"O_MLP_D_MODEL must not exceed N_EMBD.\" >&2; exit 2; }\n  (( O_MLP_HIDDEN <= 4 * N_EMBD )) || { echo \"O_MLP_HIDDEN must not exceed 4*N_EMBD.\" >&2; exit 2; }\n  if [[ \"$HAS_JPEG_LIKE_PRESET\" == true ]]; then\n",
    "limit validation",
)
text = replace_once(
    text,
    "fi\n(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || { echo \"GRADIENT_ACCUMULATION_STEPS must be divisible by NUM_GPUS.\" >&2; exit 2; }\n",
    "fi\nif [[ \"$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS\" == true && ( \"$HAS_NON_DEPTH_COMPACT_PRESET\" == true || \"$HAS_DENSE_PRESET\" == true ) ]]; then\n  echo \"--depth-compress-layer-norm-and-bias may be used only when every selected preset is depth.\" >&2\n  exit 2\nfi\n(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || { echo \"GRADIENT_ACCUMULATION_STEPS must be divisible by NUM_GPUS.\" >&2; exit 2; }\n",
    "scope validation",
)
text = replace_once(
    text,
    "  local -a compact_args optional_args train_args command\n",
    "  local -a compact_args compact_order_args optional_args train_args command\n",
    "local arrays",
)
text = replace_once(
    text,
    "  optional_args=(); compact_args=()\n",
    "  optional_args=(); compact_args=(); compact_order_args=()\n",
    "array init",
)
text = replace_once(
    text,
    "    orders_summary=\"P${o_depth_value} Q${O_ATTN_D_MODEL} J${O_ATTN_QKV_PER_CHANNEL} O${O_ATTN_OUT_PER_CHANNEL} X${O_MLP_D_MODEL} Y${O_MLP_HIDDEN}\"\n    [[ \"$geometry_preset_value\" == jpeg_like_v1 ]] && orders_summary=\"${orders_summary} MHG${mlp_hidden_group_size_value}\"\n",
    "    if [[ \"$geometry_preset_value\" == depth ]]; then\n      compact_order_args=(--o-depth \"$o_depth_value\" --o-attn-d-model 1 --o-attn-qkv-per-channel 1 --o-attn-out-per-channel 1 --o-mlp-d-model 1 --o-mlp-hidden 1)\n      orders_summary=\"P${o_depth_value} DLB=${DEPTH_COMPRESS_LAYER_NORM_AND_BIAS}\"\n      if [[ \"$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS\" == true ]]; then\n        optional_args+=(--depth-compress-layer-norm-and-bias)\n      else\n        optional_args+=(--no-depth-compress-layer-norm-and-bias)\n      fi\n    else\n      compact_order_args=(--o-depth \"$o_depth_value\" --o-attn-d-model \"$O_ATTN_D_MODEL\" --o-attn-qkv-per-channel \"$O_ATTN_QKV_PER_CHANNEL\" --o-attn-out-per-channel \"$O_ATTN_OUT_PER_CHANNEL\" --o-mlp-d-model \"$O_MLP_D_MODEL\" --o-mlp-hidden \"$O_MLP_HIDDEN\")\n      orders_summary=\"P${o_depth_value} Q${O_ATTN_D_MODEL} J${O_ATTN_QKV_PER_CHANNEL} O${O_ATTN_OUT_PER_CHANNEL} X${O_MLP_D_MODEL} Y${O_MLP_HIDDEN}\"\n      [[ \"$geometry_preset_value\" == jpeg_like_v1 ]] && orders_summary=\"${orders_summary} MHG${mlp_hidden_group_size_value}\"\n    fi\n",
    "order args",
)
text = replace_once(
    text,
    "    --o-depth \"$o_depth_value\" --o-attn-d-model \"$O_ATTN_D_MODEL\" --o-attn-qkv-per-channel \"$O_ATTN_QKV_PER_CHANNEL\" --o-attn-out-per-channel \"$O_ATTN_OUT_PER_CHANNEL\" --o-mlp-d-model \"$O_MLP_D_MODEL\" --o-mlp-hidden \"$O_MLP_HIDDEN\"\n",
    "    \"${compact_order_args[@]}\"\n",
    "train args",
)
wrapper_path.write_text(text, encoding="utf-8")

test_text = test_path.read_text(encoding="utf-8")
test_text = replace_once(
    test_text,
    "                \"--\",\n                \"--depth-compress-layer-norm-and-bias\",\n",
    "                \"--depth-compress-layer-norm-and-bias\",\n",
    "runner direct long option test",
)
test_path.write_text(test_text, encoding="utf-8")

depth_test_text = depth_test_path.read_text(encoding="utf-8")
depth_test_text = depth_test_text.replace(
    'float(trajectory.materialize_vector("ln_1_weight", 2)[3])',
    'float(trajectory.materialize_vector("ln_1_weight", 2)[3].detach())',
).replace(
    'float(trajectory.materialize_vector("ln_1_weight", 1)[3])',
    'float(trajectory.materialize_vector("ln_1_weight", 1)[3].detach())',
)
depth_test_path.write_text(depth_test_text, encoding="utf-8")

(root / ".thog_commit_message").write_text(
    "Make DEPTH wrapper ignore non-depth orders\n",
    encoding="utf-8",
)
print("updated DEPTH wrapper, display, direct option, and regression tests")
