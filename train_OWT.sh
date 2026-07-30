#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG expose granular pure-DEPTH materialisation controls; profiling remains default-off so ordinary runs incur no timing-hook overhead
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
    '  --depth-materialisation-matmul true|false   pure DEPTH only; default false' \
    '' \
    'Profiling:' \
    '  --materialisation-profiling true|false      pure DEPTH timing; default false' \
    ''
fi
unset THOG2_DEPTH_MATERIALISATION_HELP
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

# vvv THOG align only the fresh-run startup summary; the longest label fixes the value column and optimisation controls form one visible block
cat() {
  local first_line line label value optimisations_started
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
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]{2}([^:]+:)[[:space:]]*(.*)$ ]]; then
      label="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      if [[ "$label" == "fast discard:" && "$optimisations_started" == false ]]; then
        printf '  optimisations:\n'
        optimisations_started=true
      fi
      printf '  %-35s %s\n' "$label" "$value"
      if [[ "$label" == "vectorise per-head materialisation:" ]]; then
        printf '  %-35s %s\n' 'depth materialisation matmul:' "$THOG2_DEPTH_MATERIALISATION_MATMUL"
        printf '  %-35s %s\n' 'materialisation profiling:' "$THOG2_MATERIALISATION_PROFILING"
      fi
    else
      printf '%s\n' "$line"
    fi
  done
}
# ^^^ THOG

# vvv THOG source rather than duplicate the established wrapper so its complete CLI remains authoritative
source ./train_OWT_core.sh "$@"
# ^^^ THOG
