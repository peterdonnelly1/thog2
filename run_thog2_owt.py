# vvv THOG
"""Public THOG2 OWT entry point with resume/fork lifecycle dispatch.

The complete pre-enhancement runner is preserved unchanged in
run_thog2_owt_core.py. Ordinary fresh and legacy-resume execution continues
through that implementation. Only explicit enhanced-lifecycle syntax is routed
through run_thog2_lifecycle.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


# vvv THOG make --print-geometry-registry the complete discoverability surface for registered geometry plus every Python runner option and wrapper-only execution control
def _print_complete_registry_help_if_requested() -> None:
    if "--print-geometry-registry" not in sys.argv[1:]:
        return
    original_argv = list(sys.argv)
    try:
        sys.argv[:] = [sys.argv[0], *[argument for argument in sys.argv[1:] if argument != "--print-geometry-registry"]]
        from sheet.geometry_registry import format_geometry_registry
        import run_thog2_owt_core as registry_core

        print(format_geometry_registry())
        print()
        print("training runner options")
        print("-----------------------")
        print(registry_core.build_parser().format_help().rstrip())
        print()
        print("registry / wrapper-only controls")
        print("--------------------------------")
        print("  --print-geometry-registry")
        print("  --initial-eval | --no-initial-eval")
        print("  -E true|false                                      fast discard")
        print("  THOG2_BYPASS_SEMANTIC_QKV_ADAPTER=true|false      semantic adapter bypass")
        print("  THOG2_DIRECT_FACTORISED_MLP=true|false             direct factorised MLP")
        print("  THOG2_VECTORISE_PER_HEAD_MATERIALISATION=true|false  vectorise per-head materialisation")
        print("  --depth-materialisation-matmul true|false          DEPTH matrix materialisation; default true")
        print("  --materialisation-profiling true|false             pure DEPTH timing; default false")
    finally:
        sys.argv[:] = original_argv
    raise SystemExit(0)


_print_complete_registry_help_if_requested()
# ^^^ THOG

import run_thog2_lifecycle as _lifecycle                                                                                                                     # <<< THOG public lifecycle entry owns compatibility routing plus final material-policy registration
import run_thog2_owt_core as _core                                                                                                                          # <<< THOG keep the preserved runner as the implementation substrate
import sheet.geometry_registry as _geometry_registry                                                                                                        # <<< THOG align geometry report value columns with the shared console label width
import sheet.stage6_trainer as _stage6                                                                                                                      # <<< THOG public console policy adjusts terminal colour semantics without altering the preserved trainer source
from sheet.depth_trajectory import DepthTrajectory                                                                                                          # <<< THOG identify geometries that actually retain DEPTH matrix materialisation
from sheet.interactive_interrupt import (                                                                                                                   # <<< THOG Ctrl-G requests one safe-boundary checkpoint and exit
    CheckpointExitController,
    CheckpointExitRequested,
)
from sheet.owt_lifecycle_cli import normalize_lifecycle_wrapper_argv                                                                                         # <<< THOG preserve the established train_OWT.sh CLI in enhanced lifecycle mode
from sheet.run_naming import artifact_paths as _artifact_paths                                                                                              # <<< THOG lifecycle orchestration reuses the current artifact-path contract

# vvv THOG matmul is now the default DEPTH matrix materialiser after the measured A/B win; profiling remains default-off
os.environ.setdefault("THOG2_DEPTH_MATERIALISATION_MATMUL", "true")
import sheet.depth_materialisation_runtime as _depth_materialisation_runtime
from sheet.depth_materialisation_runtime import install_depth_materialisation_runtime
install_depth_materialisation_runtime()
# ^^^ THOG

_core.artifact_paths = _artifact_paths                                                                                                                       # <<< THOG expose the current naming helper to the preserved runner module used by lifecycle orchestration

# vvv THOG use the longest established two-column label as one shared width across geometry, model options and lifecycle summaries
_CONSOLE_LABEL_WIDTH = len("vectorise per-head materialisation:")
_ORIGINAL_GEOMETRY_PLAN_LABEL_WIDTH = _geometry_registry._geometry_plan_label_width


def _geometry_plan_label_width_aligned(plan: Any) -> int:
    return max(_CONSOLE_LABEL_WIDTH, _ORIGINAL_GEOMETRY_PLAN_LABEL_WIDTH(plan))


def _format_geometry_field_aligned(label_width: int, label: str, value: Any) -> str:
    return f"  {label:<{label_width}} {value}"


def _print_model_option_aligned(label: str, value: str) -> None:
    print(f"  {label:<{_CONSOLE_LABEL_WIDTH}} {value}", flush=True)


_geometry_registry._geometry_plan_label_width = _geometry_plan_label_width_aligned
_geometry_registry._format_field = _format_geometry_field_aligned
_core._print_model_option = _print_model_option_aligned
# ^^^ THOG

# vvv THOG keep materialisation timing at five decimals and place the optional penalty at the absolute end of each progress row
_ORIGINAL_MATERIALISATION_PROGRESS_FORMAT = _stage6.format_progress_line


def _materialisation_interval_field_five_decimals(trainer: Any) -> Optional[str]:
    count = int(getattr(trainer, "_thog_materialisation_penalty_count", 0))
    if count <= 0:
        return None
    mean = float(trainer._thog_materialisation_penalty_mean)
    variance = max(0.0, float(trainer._thog_materialisation_penalty_m2) / count)
    standard_deviation = math.sqrt(variance)
    return f"{mean:.5f}±{standard_deviation:.5f}s/layer"


def _format_progress_line_with_materialisation_last(run_id: str, event: str, payload: Any) -> str:
    line = _ORIGINAL_MATERIALISATION_PROGRESS_FORMAT(run_id, event, payload)
    if event != "optimizer_progress" or "materialisation_penalty" not in payload:
        return line
    field = f"  materialisation penalty={payload['materialisation_penalty']}"
    return f"{line.replace(field, '')}{field}"


_depth_materialisation_runtime._materialisation_interval_field = _materialisation_interval_field_five_decimals
_stage6.format_progress_line = _format_progress_line_with_materialisation_last
# ^^^ THOG

# vvv THOG expose only optimisation controls that can affect the selected geometry and suppress inactive optional-run rows
def _depth_matrix_fallback(trajectory: Any) -> Optional[DepthTrajectory]:
    if isinstance(trajectory, DepthTrajectory):
        return trajectory
    nested = getattr(trajectory, "depth", None)
    if not isinstance(nested, DepthTrajectory):
        return None
    if any(parameter.ndim == 3 for parameter in nested.coefficients.values()):
        return nested
    return None


def _optimisation_fields(config: Any, trainer: Any) -> list[str]:
    fields: list[str] = []
    if config.model_type == "sheet":
        model = trainer.raw_model
        model_config = model.config
        trajectory = getattr(model, "trajectory", None)
        fields.append(f"fast_discard={model_config.fast_discard}")
        if config.geometry_preset != _core.GEOMETRY_PRESET_DEPTH:                                                                                             # <<< THOG suppress semantic-QKV bypass reporting only for DEPTH geometry
            fields.append(f"semantic_qkv_bypass={model_config.bypass_semantic_qkv_adapter}")
        if hasattr(trajectory, "vectorise_per_head_materialisation"):
            fields.append(f"vectorise_per_head={model_config.vectorise_per_head_materialisation}")
        if model._supports_direct_factorised_mlp():
            fields.append(f"direct_factorised_mlp={model_config.direct_factorised_mlp}")
        fields.append(f"activation_checkpointing={config.activation_checkpointing}")
        if isinstance(trajectory, DepthTrajectory):
            fields.append(f"depth_compress_layer_norm_and_bias={model_config.depth_compress_layer_norm_and_bias}")
        depth_fallback = _depth_matrix_fallback(trajectory)
        if depth_fallback is not None:
            fields.append(f"depth_materialisation_matmul={bool(depth_fallback.depth_materialisation_matmul)}")
        return fields
    fields.append(f"activation_checkpointing={config.activation_checkpointing}")
    return fields


def _print_model_parameters_and_optimisations(config: Any, trainer: Any) -> None:
    report = trainer.parameter_report
    persistent = int(report["persistent_parameters"])
    dense_equivalent = int(report["dense_equivalent_total_parameters"])
    sheet_coefficients = int(report["sheet_coefficients"])
    compression = (dense_equivalent / persistent) if persistent else 0.0
    print("model parameters and options", flush=True)
    _core._print_model_option("parameters:", f"persistent={persistent:,}  sheet coefficients={sheet_coefficients:,}  dense equivalent={dense_equivalent:,}  dense/persistent={compression:.2f}x")
    _core._print_model_option("optimiser:", f"lr={config.learning_rate:.3e}  min_lr={config.min_lr:.3e}  warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}")
    if int(config.max_wall_minutes) > 0:
        _core._print_model_option("wall time stop:", f"max_wall_minutes={config.max_wall_minutes}")
    _core._print_model_option("non-finite:", f"policy={config.nonfinite_update_policy}  max_skips={config.max_nonfinite_update_skips}")
    _core._print_model_option("batches:", f"micro={config.batch_size}  accumulation={config.gradient_accumulation_steps}  tokens/update={config.tokens_per_iter():,}")
    if config.layer_dropout_enabled:
        _core._print_model_option("layer dropout:", f"strata={config.layer_dropout_n_strata}  stratum_size={config.layer_dropout_stratum_size}  active/stratum={config.layer_dropout_active_per_stratum}  active_layers={config.n_active_layers}/{config.n_layer}  resample_steps={config.layer_dropout_resample_steps}")
    _core._print_model_option("optimisations:", "  ".join(_optimisation_fields(config, trainer)))
    print(flush=True)


_core.print_model_parameters_and_options = _print_model_parameters_and_optimisations
# ^^^ THOG

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                        # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation


# vvv THOG apply checkpoint-exit behaviour through one public trainer subclass used by fresh, resume and fork
_BASE_OWT_TRAINER = _core.OwtTrainer
_stage6._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;33m"                                                                                              # <<< THOG use terminal-portable bold yellow for the validation-loss field

# vvv THOG render progress timestamp as YYMMDD:HHMM and mean step seconds with four fixed-width decimals
_stage6._progress_timestamp = lambda: _stage6.datetime.now().strftime("%y%m%d:%H%M")
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _prepare_console_progress_payload_with_precise_step(self: Any, event: str, payload: Any):
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event in {"optimizer_progress", "evaluation_completed"}:
        mean_step_seconds = getattr(self, "_console_latest_mean_step_seconds", None)
        if mean_step_seconds is not None:
            values["mean_step_seconds"] = f"{float(mean_step_seconds):8.4f}"
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_with_precise_step
# ^^^ THOG

# vvv THOG explicitly report a successful Ctrl-G checkpoint exit as clean to W&B while retaining shell status 131
_CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = False
_ORIGINAL_WANDB_TELEMETRY_FINISH = _core.WandbTelemetry.finish


def _finish_telemetry_with_checkpoint_exit_policy(
    telemetry: Any,
    *,
    exit_code: Optional[int] = None,
) -> None:
    global _CHECKPOINT_EXIT_CLEAN_FINISH_PENDING
    clean_checkpoint_exit = _CHECKPOINT_EXIT_CLEAN_FINISH_PENDING
    resolved_exit_code = 0 if clean_checkpoint_exit and exit_code is None else exit_code
    try:
        _ORIGINAL_WANDB_TELEMETRY_FINISH(telemetry, exit_code=resolved_exit_code)
    finally:
        if clean_checkpoint_exit:
            _CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = False


_core.WandbTelemetry.finish = _finish_telemetry_with_checkpoint_exit_policy
# ^^^ THOG


class _CheckpointExitOwtTrainer(_BASE_OWT_TRAINER):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint_exit_controller = CheckpointExitController(
            is_primary=self.distributed.is_primary,
        )
        self._checkpoint_exit_controller.start()

    def _checkpoint_exit_requested(self) -> bool:
        local_request = self._checkpoint_exit_controller.requested()
        distributed = getattr(self, "distributed", None)
        if distributed is None or int(getattr(distributed, "world_size", 1)) <= 1:
            return local_request
        gathered = distributed.all_gather_object(local_request)
        return any(bool(value) for value in gathered)

    def _timed(self, function: Any):
        global _CHECKPOINT_EXIT_CLEAN_FINISH_PENDING
        result, elapsed = super()._timed(function)
        if not self._checkpoint_exit_requested():
            return result, elapsed

        checkpoint_path = Path(self.config.out_dir) / "ckpt.pt"
        completed_updates = int(self.state.completed_updates)
        if self.distributed.is_primary:
            print(
                f"Ctrl-G checkpoint exit started at completed update {completed_updates}: "
                f"{checkpoint_path}",
                flush=True,
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.save_checkpoint(checkpoint_path)
        _CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = True
        if self.distributed.is_primary:
            print(f"Ctrl-G checkpoint saved: {checkpoint_path}", flush=True)
        raise CheckpointExitRequested(checkpoint_path, completed_updates)

    def close(self) -> None:
        try:
            self._checkpoint_exit_controller.close()
        finally:
            super().close()


_core.OwtTrainer = _CheckpointExitOwtTrainer
OwtTrainer = _CheckpointExitOwtTrainer
# ^^^ THOG


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


# vvv THOG align lifecycle schedule values with the same 35-character console label field
def _print_lifecycle_schedule(
    mode: str,
    lifecycle: Mapping[str, Any],
    training_config: Any,
    completed_updates: int,
) -> None:
    first_lr = _lifecycle.learning_rate_for_lifecycle(training_config, lifecycle, completed_updates)

    def row(label: str, value: Any) -> None:
        print(f"  {label:<{_CONSOLE_LABEL_WIDTH}} {value}", flush=True)

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
    "THOG2_DEPTH_MATERIALISATION_MATMUL",                                                                                                                  # <<< THOG preserve granular DEPTH execution control across lifecycle dispatch
    "THOG2_MATERIALISATION_PROFILING",                                                                                                                     # <<< THOG preserve default-off materialisation diagnostics across lifecycle dispatch
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


# vvv THOG keep all established dispatch paths inside the shared Ctrl-G checkpoint-exit trainer policy
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
    try:
        return _main_without_interrupt_checkpoint(argv)
    except CheckpointExitRequested as request:
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"checkpoint exit completed at update {request.completed_updates}; "
                "telemetry finished cleanly",
                flush=True,
            )
        return 131
# ^^^ THOG


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG