#!/bin/bash
set -euo pipefail

# vvv THOG
# Public lifecycle dispatcher. The complete pre-enhancement wrapper is preserved
# unchanged in train_OWT_core.sh; ordinary fresh/grid commands are delegated there.
cd "$(dirname "$0")"

show_lifecycle_help=false
lifecycle_mode=false
saw_log_timestamp=false
saw_dry_run=false
args=("$@")

for (( index=0; index<${#args[@]}; index++ )); do
  argument="${args[index]}"
  case "$argument" in
    -h|--help)
      show_lifecycle_help=true
      ;;
    --resume|--fork|--resume-from|--fork-lr-mode|--fork-learning-rate|--fork-min-lr|--fork-rewarm-iters|--wandb-continue-run|--no-wandb-continue-run)
      lifecycle_mode=true
      ;;
    --resume=*|--fork=*|--resume-from=*|--fork-lr-mode=*|--fork-learning-rate=*|--fork-min-lr=*|--fork-rewarm-iters=*)
      lifecycle_mode=true
      ;;
    --log-timestamp|--log-timestamp=*)
      saw_log_timestamp=true
      ;;
    --dry-run)
      saw_dry_run=true
      ;;
    -q)
      if (( index + 1 < ${#args[@]} )); then
        case "${args[index + 1]}" in
          resume|fork) lifecycle_mode=true ;;
        esac
      fi
      ;;
    -qresume|-qfork)
      lifecycle_mode=true
      ;;
  esac
done

if [[ "$show_lifecycle_help" == true ]]; then
  cat <<'EOF_LIFECYCLE'
THOG2 fork and resume enhancement

No-brainer resume:
  ./train_OWT.sh --resume SELECTOR
  ./train_OWT.sh --resume SELECTOR -n TOTAL_STEPS

Fork:
  ./train_OWT.sh --fork SELECTOR -n TOTAL_STEPS \
    --fork-lr-mode restart_cosine \
    --fork-learning-rate VALUE \
    --fork-min-lr VALUE \
    --fork-rewarm-iters COUNT

Lifecycle telemetry:
  --wandb-continue-run / --no-wandb-continue-run
  Default: true for resume, false for fork when W&B is active.

Selectors may be an exact checkpoint file, checkpoint directory, artifact name,
or exact leading YYMMDD-HHMM timestamp. Ambiguous timestamps fail.

The ordinary fresh/grid options follow.
EOF_LIFECYCLE
  ./train_OWT_core.sh -h | sed 's/fresh | resume/fresh | resume | fork/'
  exit 0
fi

if [[ "$lifecycle_mode" != true ]]; then
  exec ./train_OWT_core.sh "$@"
fi

if [[ -n "${THOG2_PYTHON:-}" ]]; then
  PYTHON_BIN="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ "$saw_log_timestamp" != true ]]; then
  args+=(--log-timestamp "$(date +%Y%m%d_%H%M%S)")
fi

resolved_json="$($PYTHON_BIN -m run_thog2_owt "${args[@]}" --print-resolved-json)"
artifact_name="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
run_mode="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["run_mode"])')"
world_size="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["world_size"])')"
log_path="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"
append_log="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print("true" if json.load(sys.stdin)["append_log"] else "false")')"

if [[ "$saw_dry_run" == true ]]; then
  printf '%s\n' "$resolved_json"
  exit 0
fi

command=("$PYTHON_BIN" -m run_thog2_owt "${args[@]}")
if (( world_size > 1 )); then
  command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$world_size" -m run_thog2_owt "${args[@]}")
fi

mkdir -p "$(dirname "$log_path")"
cat <<EOF_RUN
THOG2 lifecycle run
  mode:               $run_mode
  artifact:           $artifact_name
  world size:         $world_size
  log:                file://$(realpath -m "$log_path")
EOF_RUN
printf '  command:            '; printf '%q ' "${command[@]}"; printf '\n\n'

set +e
if [[ "$append_log" == true ]]; then
  "${command[@]}" 2>&1 | tee -a "$log_path"
else
  "${command[@]}" 2>&1 | tee "$log_path"
fi
run_status=${PIPESTATUS[0]}
set -e

cat <<EOF_DONE
THOG2 lifecycle run finished
  status:             $run_status
  artifact:           $artifact_name
  log:                file://$(realpath -m "$log_path")
EOF_DONE
exit "$run_status"
# ^^^ THOG
