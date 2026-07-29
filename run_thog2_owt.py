# vvv THOG
"""Public THOG2 OWT entry point with resume/fork lifecycle dispatch.

The complete pre-enhancement runner is preserved unchanged in
run_thog2_owt_core.py. Ordinary fresh and legacy-resume execution continues
through that implementation. Only explicit enhanced-lifecycle syntax is routed
through run_thog2_lifecycle.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Optional, Sequence

import run_thog2_lifecycle as _lifecycle                                                                                                                     # <<< THOG public lifecycle entry owns compatibility routing plus final material-policy registration
import run_thog2_owt_core as _core                                                                                                                          # <<< THOG keep the preserved runner as the implementation substrate
from sheet.interactive_interrupt import interactive_interrupt_checkpoint                                                                                   # <<< THOG turn the first Ctrl-C into a safe-boundary checkpoint choice
from sheet.owt_lifecycle_cli import normalize_lifecycle_wrapper_argv                                                                                         # <<< THOG preserve the established train_OWT.sh CLI in enhanced lifecycle mode
from sheet.run_naming import artifact_paths as _artifact_paths                                                                                              # <<< THOG lifecycle orchestration reuses the current artifact-path contract

_core.artifact_paths = _artifact_paths                                                                                                                       # <<< THOG expose the current naming helper to the preserved runner module used by lifecycle orchestration

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                        # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation


# vvv THOG non-finite controls alter future update acceptance; classify them as checkpoint-authoritative before lifecycle preflight
for _material_destination in (
    "nonfinite_update_policy",
    "max_nonfinite_update_skips",
):
    _lifecycle._OPERATIONAL_CONFIG_DESTINATIONS.discard(_material_destination)
_lifecycle._ARGUMENT_TO_CONFIG.update(
    {
        "nonfinite_update_policy": "nonfinite_update_policy",
        "max_nonfinite_update_skips": "max_nonfinite_update_skips",
    }
)
# ^^^ THOG


# vvv THOG align lifecycle schedule values with the established model-options console column

def _print_lifecycle_schedule(
    mode: str,
    lifecycle: Mapping[str, Any],
    training_config: Any,
    completed_updates: int,
) -> None:
    first_lr = _lifecycle.learning_rate_for_lifecycle(training_config, lifecycle, completed_updates)

    def row(label: str, value: Any) -> None:
        print(f"  {label:<24} {value}", flush=True)

    print(flush=True)
    print(f"{mode} schedule", flush=True)
    row("completed steps:", completed_updates)
    row("total steps:", int(lifecycle["target_updates"]))
    row("remaining steps:", int(lifecycle["target_updates"]) - completed_updates)
    phases = lifecycle.get("lr_phases", [])
    active_index = int(lifecycle.get("active_lr_phase_index", max(0, len(phases) - 1)))
    active = phases[active_index] if phases else {"phase_type": _lifecycle.COSINE_SCHEDULE}
    row("schedule:", active.get("phase_type", _lifecycle.COSINE_SCHEDULE))
    if active.get("phase_type") == _lifecycle.RESTART_COSINE_SCHEDULE:
        row("active phase end:", int(active["phase_end_update"]))
    else:
        row("original decay end:", training_config.decay_updates)
    row(f"first {mode} step LR:", f"{first_lr:.3e}")
    print(flush=True)


_lifecycle._print_schedule_startup = _print_lifecycle_schedule
_lifecycle_main = _lifecycle.main
# ^^^ THOG


_LIFECYCLE_ENVIRONMENT_KEYS = (
    "THOG2_INSTRUMENTATION",
    "THOG2_CURVE_ROOT",
    "THOG2_OPTIMIZER",
    "THOG2_OPTIMIZER_MOMENTUM",
    "THOG2_DEPTH_CURVE_PLOTS",
    "THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS",
    "THOG2_DEPTH_CURVE_RENDERER",
    "THOG2_DEPTH_CURVE_LOCAL_HTML",
    "THOG2_FAST_DISCARD",
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
            "--instrumentation",
            "--optimizer",
            "--optimizer-momentum",
            "-I",
            "-G",
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
            "--instrumentation=",
            "--optimizer=",
            "--optimizer-momentum=",
        )):
            return True
        if argument in ("-qfresh", "-qresume", "-qfork"):
            return True
        if argument.startswith("-I") and len(argument) > 2:
            return True
        if argument.startswith("-G") and len(argument) > 2:
            return True
        if previous in ("-q", "--run-mode") and argument in ("fresh", "resume", "fork"):
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


def _call_lifecycle_main(argv: Sequence[str], environment: Optional[Mapping[str, str]] = None) -> int:
    saved_environment = {name: os.environ.get(name) for name in _LIFECYCLE_ENVIRONMENT_KEYS}
    try:
        for name, value in (environment or {}).items():
            os.environ[name] = value
        return int(_lifecycle_main(list(argv)))
    finally:
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# vvv THOG keep all established dispatch paths inside the safe interactive Ctrl-C guard
def _main_without_interrupt_checkpoint(argv: Optional[Sequence[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if _enhanced_lifecycle_requested(actual_argv):
        try:
            normalized = normalize_lifecycle_wrapper_argv(actual_argv)
        except ValueError as error:
            print(f"train_OWT.sh: {error}", file=sys.stderr)
            return 2
        return _call_lifecycle_main(normalized.argv, normalized.environment)
    if argv is not None:
        # The preserved core main reads sys.argv directly. Programmatic callers
        # supplying argv use the lifecycle parser even for fresh mode so the
        # caller's explicit argument vector is respected without mutating
        # process-global sys.argv.
        return _call_lifecycle_main(actual_argv)
    return _call_preserved_core_main()


def main(argv: Optional[Sequence[str]] = None) -> int:
    with interactive_interrupt_checkpoint():
        return _main_without_interrupt_checkpoint(argv)
# ^^^ THOG


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
