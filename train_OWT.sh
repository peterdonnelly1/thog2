#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

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

# vvv THOG align only the fresh-run startup summary; the longest label fixes the value column
cat() {
  local first_line line label value
  if ! IFS= read -r first_line; then
    return 0
  fi
  if [[ "$first_line" != "scruffy OWT train" ]]; then
    printf '%s\n' "$first_line"
    command cat
    return 0
  fi

  printf '%s\n' "$first_line"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]{2}([^:]+:)[[:space:]]*(.*)$ ]]; then
      label="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      printf '  %-35s %s\n' "$label" "$value"
    else
      printf '%s\n' "$line"
    fi
  done
}
# ^^^ THOG

# vvv THOG source rather than duplicate the established wrapper so its complete CLI remains authoritative
source ./train_OWT_core.sh "$@"
# ^^^ THOG
