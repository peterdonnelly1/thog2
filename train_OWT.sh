#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG reject non-canonical PLASTIC aliases before any wrapper parses them
thog2_normalize_nonplastic_long_option() {
  local option_name="$1"
  while [[ "$option_name" == *"__"* ]]; do
    option_name="${option_name//__/_}"
  done
  printf '%s' "${option_name//_/-}"
}
THOG2_STRICT_LONG_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --plastic__*|--no-plastic__*)
      THOG2_STRICT_LONG_ARGS+=("$1")
      shift
      ;;
    --plastic-*|--no-plastic-*|--plastic_[!_]*|--no-plastic_[!_]*)
      echo "Non-canonical PLASTIC option rejected: $1; use the exact --plastic__... or --no-plastic__... spelling." >&2
      exit 2
      ;;
    --*=*)
      THOG2_LONG_NAME="${1%%=*}"
      THOG2_LONG_VALUE="${1#*=}"
      THOG2_LONG_NAME="$(thog2_normalize_nonplastic_long_option "$THOG2_LONG_NAME")"
      THOG2_STRICT_LONG_ARGS+=("${THOG2_LONG_NAME}=${THOG2_LONG_VALUE}")
      shift
      ;;
    --*)
      THOG2_STRICT_LONG_ARGS+=("$(thog2_normalize_nonplastic_long_option "$1")")
      shift
      ;;
    *)
      THOG2_STRICT_LONG_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${THOG2_STRICT_LONG_ARGS[@]}"
unset THOG2_STRICT_LONG_ARGS THOG2_LONG_NAME THOG2_LONG_VALUE
unset -f thog2_normalize_nonplastic_long_option
# ^^^ THOG

# vvv THOG expose exact PLASTIC lookahead controls through the one canonical wrapper
source ./plastic_depth_lookahead_wrapper_options.sh
# ^^^ THOG

# vvv THOG expose granular pure-DEPTH materialisation controls; matmul and profiling are explicit default-off options
# THOG2_DEPTH_MATERIALISATION_MATMUL="${THOG2_DEPTH_MATERIALISATION_MATMUL:-true}"                                                # <<< THOG preserved previous default-on policy
THOG2_DEPTH_MATERIALISATION_MATMUL="${THOG2_DEPTH_MATERIALISATION_MATMUL:-false}"
THOG2_MATERIALISATION_PROFILING="${THOG2_MATERIALISATION_PROFILING:-false}"
case "$THOG2_DEPTH_MATERIALISATION_MATMUL" in
  true|false) ;;
  *) echo "THOG2_DEPTH_MATERIALISATION_MATMUL must be true or false; got: $THOG2_DEPTH_MATERIALISATION_MATMUL" >&2; exit 2 ;;
esac
case "$THOG2_MATERIALISATION_PROFILING" in
  true|false) ;;
  *) echo "THOG2_MATERIALISATION_PROFILING must be true or false; got: $THOG2_MATERIALISATION_PROFILING" >&2; exit 2 ;;
esac
THOG2_DEPTH_MATERIALISATION_FILTERED_ARGS=()
THOG2_DEPTH_MATERIALISATION_HELP=false
while (( $# > 0 )); do
  case "$1" in
    --depth-materialisation-matmul)
      (( $# >= 2 )) || { echo "--depth-materialisation-matmul requires true or false" >&2; exit 2; }
      case "$2" in true|false) ;; *) echo "--depth-materialisation-matmul requires true or false; got: $2" >&2; exit 2 ;; esac
      THOG2_DEPTH_MATERIALISATION_MATMUL="$2"
      shift 2
      ;;
    --depth-materialisation-matmul=*)
      THOG2_DEPTH_MATERIALISATION_MATMUL="${1#*=}"
      case "$THOG2_DEPTH_MATERIALISATION_MATMUL" in true|false) ;; *) echo "--depth-materialisation-matmul requires true or false; got: $THOG2_DEPTH_MATERIALISATION_MATMUL" >&2; exit 2 ;; esac
      shift
      ;;
    --materialisation-profiling)
      (( $# >= 2 )) || { echo "--materialisation-profiling requires true or false" >&2; exit 2; }
      case "$2" in true|false) ;; *) echo "--materialisation-profiling requires true or false; got: $2" >&2; exit 2 ;; esac
      THOG2_MATERIALISATION_PROFILING="$2"
      shift 2
      ;;
    --materialisation-profiling=*)
      THOG2_MATERIALISATION_PROFILING="${1#*=}"
      case "$THOG2_MATERIALISATION_PROFILING" in true|false) ;; *) echo "--materialisation-profiling requires true or false; got: $THOG2_MATERIALISATION_PROFILING" >&2; exit 2 ;; esac
      shift
      ;;
    --print-geometry-registry)
      # vvv THOG make the complete registry/help surface directly reachable from the canonical training wrapper
      if [[ -n "${THOG2_PYTHON:-}" ]]; then
        THOG2_REGISTRY_PYTHON="$THOG2_PYTHON"
      elif [[ -x .venv/bin/python ]]; then
        THOG2_REGISTRY_PYTHON=.venv/bin/python
      else
        THOG2_REGISTRY_PYTHON=python
      fi
      "$THOG2_REGISTRY_PYTHON" -m run_thog2_owt --print-geometry-registry
      printf '\ncanonical train_OWT.sh options\n------------------------------\n'
      bash ./train_OWT_core.sh -h
      # vvv THOG the direct registry entrypoint already prints complete parser and descriptor help once
      # printf '\nregistered runner hyperparameters\n---------------------------------\n'
      # "$THOG2_REGISTRY_PYTHON" -c 'from run_thog2_owt_core import build_parser; print(build_parser().format_help(), end="")'
      # ^^^ THOG
      printf '\nTorch compilation:\n  --torch-compile false|true|regional        false=eager, true=whole-model, regional=checkpoint-segment compile\n'
      exit 0
      # ^^^ THOG
      ;;
    -h|--help)
      THOG2_DEPTH_MATERIALISATION_HELP=true
      THOG2_DEPTH_MATERIALISATION_FILTERED_ARGS+=("$1")
      shift
      ;;
    *)
      THOG2_DEPTH_MATERIALISATION_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${THOG2_DEPTH_MATERIALISATION_FILTERED_ARGS[@]}"
unset THOG2_DEPTH_MATERIALISATION_FILTERED_ARGS
export THOG2_DEPTH_MATERIALISATION_MATMUL
export THOG2_MATERIALISATION_PROFILING
if [[ "$THOG2_DEPTH_MATERIALISATION_HELP" == true ]]; then
  printf '%s\n' \
    'DEPTH execution optimisation:' \
    '  --depth-materialisation-matmul true|false   DEPTH matrix materialisation; default false' \
    '' \
    'Profiling:' \
    '  --materialisation-profiling true|false      pure DEPTH timing; default false' \
    ''
fi
unset THOG2_DEPTH_MATERIALISATION_HELP
# ^^^ THOG

# vvv THOG expose eager, whole-model compile, and checkpoint-segment regional compile without changing the established true/false meanings
THOG2_TORCH_COMPILE="${THOG2_TORCH_COMPILE:-false}"
case "$THOG2_TORCH_COMPILE" in
  false|true|regional) ;;
  *) echo "THOG2_TORCH_COMPILE must be false, true, or regional; got: $THOG2_TORCH_COMPILE" >&2; exit 2 ;;
esac
THOG2_TORCH_COMPILE_FILTERED_ARGS=()
THOG2_TORCH_COMPILE_HELP=false
while (( $# > 0 )); do
  case "$1" in
    --torch-compile)
      (( $# >= 2 )) || { echo "--torch-compile requires false, true, or regional" >&2; exit 2; }
      case "$2" in false|true|regional) ;; *) echo "--torch-compile requires false, true, or regional; got: $2" >&2; exit 2 ;; esac
      THOG2_TORCH_COMPILE="$2"
      shift 2
      ;;
    --torch-compile=*)
      THOG2_TORCH_COMPILE="${1#*=}"
      case "$THOG2_TORCH_COMPILE" in false|true|regional) ;; *) echo "--torch-compile requires false, true, or regional; got: $THOG2_TORCH_COMPILE" >&2; exit 2 ;; esac
      shift
      ;;
    -h|--help)
      THOG2_TORCH_COMPILE_HELP=true
      THOG2_TORCH_COMPILE_FILTERED_ARGS+=("$1")
      shift
      ;;
    *)
      THOG2_TORCH_COMPILE_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${THOG2_TORCH_COMPILE_FILTERED_ARGS[@]}"
unset THOG2_TORCH_COMPILE_FILTERED_ARGS
export THOG2_TORCH_COMPILE
if [[ "$THOG2_TORCH_COMPILE_HELP" == true ]]; then
  printf '%s\n' \
    'Torch compilation:' \
    '  --torch-compile false|true|regional        false=eager, true=whole-model, regional=checkpoint-segment compile' \
    ''
fi
unset THOG2_TORCH_COMPILE_HELP
# ^^^ THOG

# vvv THOG give fresh, resume, fork and every DDP rank one collision-safe Ctrl-G request path
if [[ -z "${THOG2_CHECKPOINT_EXIT_FILE:-}" ]]; then
  export THOG2_CHECKPOINT_EXIT_FILE="/tmp/thog2_checkpoint_exit_${BASHPID}_$(date +%s%N)"
fi
rm -f "$THOG2_CHECKPOINT_EXIT_FILE"
# ^^^ THOG

# vvv THOG keep lifecycle options additive while ordinary runs execute the preserved master wrapper
THOG2_LIFECYCLE_DISPATCH=false
THOG2_LIFECYCLE_HELP=false
THOG2_PREVIOUS_ARGUMENT=""
for THOG2_ARGUMENT in "$@"; do
  case "$THOG2_ARGUMENT" in
    -h|--help)
      THOG2_LIFECYCLE_HELP=true
      ;;
    --resume|--fork|--resume=*|--fork=*|--resume-from|--resume-from=*|--fork-lr-mode|--fork-lr-mode=*|--fork-learning-rate|--fork-learning-rate=*|--fork-min-lr|--fork-min-lr=*|--fork-rewarm-iters|--fork-rewarm-iters=*|--wandb-continue-run|--no-wandb-continue-run)
      THOG2_LIFECYCLE_DISPATCH=true
      ;;
    -qresume|-qfork)
      THOG2_LIFECYCLE_DISPATCH=true
      ;;
  esac
  if [[ "$THOG2_PREVIOUS_ARGUMENT" == "-q" ]]; then
    case "$THOG2_ARGUMENT" in
      resume|fork) THOG2_LIFECYCLE_DISPATCH=true ;;
    esac
  fi
  THOG2_PREVIOUS_ARGUMENT="$THOG2_ARGUMENT"
done
unset THOG2_ARGUMENT THOG2_PREVIOUS_ARGUMENT
if [[ "$THOG2_LIFECYCLE_HELP" == true || "$THOG2_LIFECYCLE_DISPATCH" == true ]]; then
  exec ./resume_and_fork_OWT.sh "$@"
fi
unset THOG2_LIFECYCLE_DISPATCH THOG2_LIFECYCLE_HELP
# ^^^ THOG

# vvv THOG align the fresh-run summary on the longest optimisation label and show only controls that can affect the selected geometry
cat() {
  local first_line line label value optimisations_started geometry_preset effective_stratum_size effective_active_per_stratum
  if ! IFS= read -r first_line; then
    return 0
  fi
  if [[ "$first_line" != "scruffy OWT train" ]]; then
    printf '%s\n' "$first_line"
    command cat
    return 0
  fi

  printf '%s\n' "$first_line"
  optimisations_started=false
  geometry_preset=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]{2}([^:]+:)[[:space:]]*(.*)$ ]]; then
      label="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      if [[ "$label" == "model/preset/basis:" ]]; then
        geometry_preset="${value#* / }"
        geometry_preset="${geometry_preset%% / *}"
      fi
      case "$label" in
        "fast discard:")
          [[ "$geometry_preset" == dense ]] && continue
          [[ "$optimisations_started" == false ]] && { printf '  optimisations:\n'; optimisations_started=true; }
          printf '  %-35s %s\n' "$label" "$value"
          ;;
        "semantic adapter bypass:")
          [[ "$geometry_preset" == dense || "$geometry_preset" == depth ]] && continue
          printf '  %-35s %s\n' "$label" "$value"
          ;;
        "direct factorised MLP:")
          [[ "$geometry_preset" == mlp_block || "$geometry_preset" == full_block ]] || continue
          printf '  %-35s %s\n' "$label" "$value"
          ;;
        "vectorise per-head materialisation:")
          if [[ "$geometry_preset" == head_aware_block || "$geometry_preset" == full_block ]]; then
            printf '  %-35s %s\n' "$label" "$value"
          fi
          if [[ "$geometry_preset" == depth || "$geometry_preset" == jpeg_like_v1 || "$geometry_preset" == mlp_block || "$geometry_preset" == head_aware_block ]]; then
            printf '  %-35s %s\n' 'depth materialisation matmul:' "$THOG2_DEPTH_MATERIALISATION_MATMUL"
          fi
          ;;
        "instrumentation:")
          printf '  %-35s %s\n' "$label" "$value"
          [[ "$optimisations_started" == false ]] && { printf '  optimisations:\n'; optimisations_started=true; }
          printf '  %-35s %s\n' 'torch compile:' "$THOG2_TORCH_COMPILE"
          if [[ "$geometry_preset" == depth && "$THOG2_MATERIALISATION_PROFILING" == true ]]; then
            printf '  %-35s %s\n' 'materialisation profiling:' "$THOG2_MATERIALISATION_PROFILING"
          fi
          ;;
        "layer dropout:")
          effective_stratum_size="${LAYER_DROPOUT_STRATUM_SIZE:-${N_LAYER:-}}"
          effective_active_per_stratum="${LAYER_DROPOUT_ACTIVE_PER_STRATUM:-$effective_stratum_size}"
          [[ -n "$effective_stratum_size" && "$effective_active_per_stratum" == "$effective_stratum_size" ]] && continue
          printf '  %-35s %s\n' "$label" "$value"
          ;;
        "plastic fine:")
          printf '  %-35s %s same_batch=%s\n' "$label" "$value" "$THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES"
          ;;
        *)
          printf '  %-35s %s\n' "$label" "$value"
          ;;
      esac
    else
      printf '%s\n' "$line"
    fi
  done
}
# ^^^ THOG

# vvv THOG source rather than duplicate the established wrapper so its complete CLI remains authoritative
source ./train_OWT_core.sh "$@"
# ^^^ THOG
# vvv THOG preserved superseded source lines for exact history audit
# THOG2_DEPTH_MATERIALISATION_MATMUL="${THOG2_DEPTH_MATERIALISATION_MATMUL:-true}"
# '  --depth-materialisation-matmul true|false   DEPTH matrix materialisation; default true' \
# ^^^ THOG
