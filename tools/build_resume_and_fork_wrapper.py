# vvv THOG
from __future__ import annotations

from pathlib import Path


SOURCE = Path("train_OWT_core.sh")
TARGET = Path("train_OWT.sh")
ANCHOR = 'cd "$(dirname "$0")"\n\n'

INJECTION = r'''# vvv THOG fork/resume lifecycle shortcut while retaining the complete current wrapper below
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

'''


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    if ANCHOR not in source:
        raise SystemExit("canonical train_OWT.sh cd anchor not found")
    generated = source.replace(ANCHOR, ANCHOR + INJECTION, 1)
    TARGET.write_text(generated, encoding="utf-8")
    TARGET.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
