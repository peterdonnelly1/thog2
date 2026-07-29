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
    --dry-run|-xtrue)
      saw_dry_run=true
      ;;
    -x)
      if (( index + 1 < ${#args[@]} )) && [[ "${args[index + 1]}" == true ]]; then
        saw_dry_run=true
      fi
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

if [[ -z "${THOG2_LIFECYCLE_SESSION_ID:-}" ]]; then
  export THOG2_LIFECYCLE_SESSION_ID="$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4())')"                                                                   # <<< THOG make preflight and execution resolve one process-session identity
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
session_id="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')"
target_updates="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["target_updates"])')"

if [[ "$saw_dry_run" == true ]]; then
  printf '%s\n' "$resolved_json"
  exit 0
fi

command=("$PYTHON_BIN" -m run_thog2_owt "${args[@]}")
if (( world_size > 1 )); then
  command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$world_size" -m run_thog2_owt "${args[@]}")
fi

mkdir -p "$(dirname "$log_path")"
# vvv THOG align lifecycle summary values with the model-options value column and visually separate the heading
startup_summary="$(
  printf '\nTHOG2 lifecycle session\n'
  printf '  %-24s %s\n' 'mode:' "$run_mode"
  printf '  %-24s %s\n' 'session:' "$session_id"
  printf '  %-24s %s\n' 'artifact:' "$artifact_name"
  printf '  %-24s %s\n' 'target updates:' "$target_updates"
  printf '  %-24s %s\n' 'world size:' "$world_size"
  printf '  %-24s %s\n' 'log:' "file://$(realpath -m "$log_path")"
)"
printf '%s\n' "$startup_summary"
printf '  %-24s ' 'command:'; printf '%q ' "${command[@]}"; printf '\n\n'
# ^^^ THOG
if [[ "$append_log" == true ]]; then
  {
    printf '\n%s\n' "============================================================"
    printf '%s\n' "$startup_summary"
    printf '  %-24s ' 'command:'; printf '%q ' "${command[@]}"; printf '\n'
    printf '%s\n\n' "============================================================"
  } >> "$log_path"
fi

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
  session:            $session_id
  artifact:           $artifact_name
  log:                file://$(realpath -m "$log_path")
EOF_DONE
exit "$run_status"
# ^^^ THOG
