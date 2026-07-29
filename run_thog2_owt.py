# vvv THOG
"""Public THOG2 OWT entry point with resume/fork lifecycle dispatch.

The complete pre-enhancement runner is preserved unchanged in
run_thog2_owt_core.py. Ordinary fresh and legacy-resume execution continues
through that implementation. Only the explicit enhanced lifecycle syntax is
routed through run_thog2_lifecycle.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

import run_thog2_owt_core as _core                                                                                                                          # <<< THOG keep the preserved runner as the implementation substrate
from sheet.run_naming import artifact_paths as _artifact_paths                                                                                              # <<< THOG lifecycle orchestration reuses the current artifact-path contract

_core.artifact_paths = _artifact_paths                                                                                                                       # <<< THOG expose the current naming helper to the preserved runner module used by lifecycle orchestration

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                        # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation
from run_thog2_lifecycle import main as _lifecycle_main                                                                                                     # <<< THOG enhanced lifecycle implementation is invoked only for explicit lifecycle syntax


_LIFECYCLE_ENVIRONMENT_KEYS = (
    "THOG2_INSTRUMENTATION",
    "THOG2_CURVE_ROOT",
    "THOG2_OPTIMIZER",
    "THOG2_OPTIMIZER_MOMENTUM",
    "WANDB_MODE",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
)
_CORE_FORWARD_NAMES = tuple(
    name
    for name in vars(_core)
    if not name.startswith("_") and name != "main"
)


def _enhanced_lifecycle_requested(argv: Sequence[str]) -> bool:
    previous = ""
    for argument in argv:
        if argument in (
            "--resume",
            "--fork",
            "--resume-from",
            "--fork-lr-mode",
            "--fork-learning-rate",
            "--fork-min-lr",
            "--fork-rewarm-iters",
            "--wandb-continue-run",
            "--no-wandb-continue-run",
        ):
            return True
        if argument.startswith((
            "--resume=",
            "--fork=",
            "--resume-from=",
            "--fork-lr-mode=",
            "--fork-learning-rate=",
            "--fork-min-lr=",
            "--fork-rewarm-iters=",
        )):
            return True
        if argument in ("-qresume", "-qfork"):
            return True
        if previous in ("-q", "--run-mode") and argument in ("resume", "fork"):
            return True
        previous = argument
    return False


def _call_preserved_core_main() -> int:
    # Existing callers and tests historically monkey-patch symbols on
    # run_thog2_owt. Forward any such public overrides into the preserved core
    # for the duration of the call rather than silently changing that API.
    restored = {}
    for name in _CORE_FORWARD_NAMES:
        if name not in globals():
            continue
        public_value = globals()[name]
        core_value = getattr(_core, name)
        if public_value is core_value:
            continue
        restored[name] = core_value
        setattr(_core, name, public_value)
    try:
        return int(_core.main())
    finally:
        for name, value in restored.items():
            setattr(_core, name, value)


def _call_lifecycle_main(argv: Sequence[str]) -> int:
    saved_environment = {name: os.environ.get(name) for name in _LIFECYCLE_ENVIRONMENT_KEYS}
    try:
        return int(_lifecycle_main(list(argv)))
    finally:
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main(argv: Optional[Sequence[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if _enhanced_lifecycle_requested(actual_argv):
        return _call_lifecycle_main(actual_argv)
    if argv is not None:
        # The preserved core main reads sys.argv directly. Programmatic callers
        # supplying argv use the lifecycle parser even for fresh mode so the
        # caller's explicit argument vector is respected without mutating
        # process-global sys.argv.
        return _call_lifecycle_main(actual_argv)
    return _call_preserved_core_main()


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
