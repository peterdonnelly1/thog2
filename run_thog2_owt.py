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
        print("  --depth-materialisation-matmul true|false          DEPTH matrix materialisation; default false")
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

# vvv THOG default DEPTH materialisation returns to the reference path; matmul remains an explicit opt-in
# os.environ.setdefault("THOG2_DEPTH_MATERIALISATION_MATMUL", "true")                                                                                     # <<< THOG preserved previous default-on matmul policy
os.environ.setdefault("THOG2_DEPTH_MATERIALISATION_MATMUL", "false")
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

# vvv THOG fixed-width PLASTIC progress tail columns keep probe losses, change gates and public layer coordinates readable as counts move
def _plastic_progress_probe_loss(value: Any) -> str:
    if value is None:
        return f"{'-':>7}"
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>7}"
    return f"{numeric:7.3f}"


def _plastic_progress_change_z(value: Any) -> str:
    if value is None:
        return f"{'-':>9}"
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>9}"
    magnitude = abs(numeric)
    if magnitude != 0.0 and (magnitude < 0.01 or magnitude >= 1000.0):
        return f"{numeric:+9.2e}"
    return f"{numeric:+9.2f}"


def _plastic_progress_layer_index(value: Any) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>5}"
    return f"{numeric:5.1f}"


def _plastic_change_z_field(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("plastic_change_z")
    if values is None:
        return None
    change_z = ", ".join(_plastic_progress_change_z(value) for value in values)
    return f"\tchange_z [L-1, L+1] = [{change_z}]"


def _plastic_progress_tail_tabs(line: str) -> str:
    return (
        line.replace("  probe_losses = ", "\tprobe_losses [L-1, L, L+1] = ")
        .replace("  layer indices = ", "\tlayer indices = ")
    )


_stage6._format_plastic_probe_loss = _plastic_progress_probe_loss
_stage6._format_depth_sample_point = _plastic_progress_layer_index
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


def _move_sampled_values_to_progress_tail(line: str) -> str:
    marker = "  sampled_values = ["
    start = line.find(marker)
    if start < 0:
        return line
    stop = line.find("]", start)
    if stop < 0:
        return line
    field = line[start : stop + 1]
    return f"{line[:start].rstrip()}{line[stop + 1:]}{field}"


def _format_progress_line_with_materialisation_last(run_id: str, event: str, payload: Any) -> str:
    line = _plastic_progress_tail_tabs(
        _ORIGINAL_MATERIALISATION_PROGRESS_FORMAT(run_id, event, payload)
    )
    change_z_field = _plastic_change_z_field(payload)
    if change_z_field is not None and "\tlayer indices = " in line:
        line = line.replace("\tlayer indices = ", f"{change_z_field}\tlayer indices = ", 1)
    elif change_z_field is not None:
        line = f"{line}{change_z_field}"
    if event == "optimizer_progress" and "materialisation_penalty" in payload:
        field = f"  materialisation penalty={payload['materialisation_penalty']}"
        line = f"{line.replace(field, '')}{field}"
    return _move_sampled_values_to_progress_tail(line)


_depth_materialisation_runtime._materialisation_interval_field = _materialisation_interval_field_five_decimals
_stage6.format_progress_line = _format_progress_line_with_materialisation_last
# ^^^ THOG

# vvv THOG restore the full startup report and give PLASTIC DEPTH its own untruncated hyper-parameter section
def _depth_matrix_fallback(trajectory: Any) -> Optional[DepthTrajectory]:
    if isinstance(trajectory, DepthTrajectory):
        return trajectory
    nested = getattr(trajectory, "depth", None)
    if not isinstance(nested, DepthTrajectory):
        return None
    if any(parameter.ndim == 3 for parameter in nested.coefficients.values()):
        return nested
    return None


def _optimizer_summary(trainer: Any) -> str:
    optimizer = trainer.optimizer
    group = optimizer.param_groups[0]
    name = str(group.get("thog2_optimizer_name", optimizer.__class__.__name__)).lower()
    learning_rate = float(group["lr"])
    if name == "adamw":
        fused = bool(optimizer.defaults.get("fused", False))
        betas = optimizer.defaults.get("betas", (0.9, 0.95))
        return f"{name} (lr={learning_rate:.3e}, fused={fused}, betas={betas})"
    if name in {"sgd", "sgd_nesterov"}:
        return f"{name} (lr={learning_rate:.3e}, momentum={optimizer.defaults.get('momentum', 0):g}, nesterov={bool(optimizer.defaults.get('nesterov', False))})"
    if name == "rmsprop":
        return f"{name} (lr={learning_rate:.3e}, momentum={optimizer.defaults.get('momentum', 0):g}, alpha={optimizer.defaults.get('alpha', 0.99):g})"
    return f"{name} (lr={learning_rate:.3e})"


def _lr_decay_summary(config: Any, trainer: Any) -> str:
    trainer_config = trainer.config
    decay_type = "cosine" if bool(trainer_config.decay_learning_rate) else "constant"
    decay_rate = float(config.min_lr) / float(config.learning_rate)
    return f"{decay_type} (decay_rate={decay_rate:.6g}, min_lr={config.min_lr:.3e}, fully_decayed_step={trainer_config.decay_updates})"


def _startup_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _startup_optional(value: Any) -> str:
    return "None" if value is None else str(value)


def _startup_float(value: Any) -> str:
    return "None" if value is None else f"{float(value):g}"


def _startup_public_indices(values: Any) -> str:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        return ""
    head, *tail = resolved
    items = [f"{head:.1f}"]
    items.extend(f"{value:5.1f}" for value in tail)
    return ", ".join(items)


_PLASTIC_STARTUP_LABELS = (
    "plastic__enabled:",
    "resolved count mode:",
    "current active layers:",
    "plastic__layers_to_sample:",
    "plastic__do_learn_layer_count:",
    "plastic__initial_layer_count:",
    "plastic__initial_active_layers:",
    "plastic__max_permitted_layers:",
    "plastic__layer_sampling_initialisation:",
    "plastic__layer_count_objective:",
    "plastic__layer_count_update_brake:",
    "plastic__layer_count_probe_noise_window:",
    "plastic__layer_count_probe_noise_min_observations:",
    "plastic__layer_count_probe_noise_lambda:",
    "plastic__layer_count_cost_weight:",
    "plastic__layer_memory_budget_gib:",
    "plastic__cuda_allocator_reserve_gib:",
    "plastic__geometry_learning_rate_multiplier:",
    "plastic__freeze_geometry_during_warmup:",
    "active sample_layer:",
    "capacity sample_layer:",
)
_PLASTIC_STARTUP_LABEL_WIDTH = max(len(label) for label in _PLASTIC_STARTUP_LABELS) + 3


def _print_plastic_option(label: str, value: str) -> None:
    print(f"  {label:<{_PLASTIC_STARTUP_LABEL_WIDTH}}{value}", flush=True)


def _print_plastic_depth_section(config: Any, trainer: Any) -> None:
    if not bool(config.plastic__enabled):
        return
    report = trainer.parameter_report.get("plastic_depth", {})
    current_layers = int(report.get("active_layers", config.plastic__initial_active_layers))
    public_coordinates = tuple(report.get("active_sample_layer_coordinates", report.get("active_public_coordinates", ())))
    full_coordinates = tuple(report.get("sample_layer_coordinates", report.get("public_coordinates", ())))
    print("plastic", flush=True)
    _print_plastic_option("plastic__enabled:", _startup_bool(config.plastic__enabled))
    _print_plastic_option("resolved count mode:", "learned" if config.plastic__do_learn_layer_count else "fixed")
    _print_plastic_option("current active layers:", f"{current_layers}/{config.n_layer}")
    _print_plastic_option("plastic__layers_to_sample:", _startup_optional(config.plastic__layers_to_sample))
    _print_plastic_option("plastic__do_learn_layer_count:", _startup_bool(config.plastic__do_learn_layer_count))
    _print_plastic_option("plastic__initial_layer_count:", _startup_optional(config.plastic__initial_layer_count))
    _print_plastic_option("plastic__initial_active_layers:", str(config.plastic__initial_active_layers))
    _print_plastic_option("plastic__max_permitted_layers:", _startup_optional(config.plastic__max_permitted_layers))
    _print_plastic_option("plastic__layer_sampling_initialisation:", str(config.plastic__layer_sampling_initialisation))
    _print_plastic_option("plastic__layer_count_objective:", str(config.plastic__layer_count_objective))
    _print_plastic_option("plastic__layer_count_update_brake:", str(config.plastic__layer_count_update_brake))
    _print_plastic_option("plastic__layer_count_probe_noise_window:", str(config.plastic__layer_count_probe_noise_window))
    _print_plastic_option("plastic__layer_count_probe_noise_min_observations:", str(config.plastic__layer_count_probe_noise_min_observations))
    _print_plastic_option("plastic__layer_count_probe_noise_lambda:", _startup_float(config.plastic__layer_count_probe_noise_lambda))
    _print_plastic_option("plastic__layer_count_cost_weight:", _startup_float(config.plastic__layer_count_cost_weight))
    _print_plastic_option("plastic__layer_memory_budget_gib:", _startup_float(config.plastic__layer_memory_budget_gib))
    _print_plastic_option("plastic__cuda_allocator_reserve_gib:", _startup_float(config.plastic__cuda_allocator_reserve_gib))
    _print_plastic_option("plastic__geometry_learning_rate_multiplier:", _startup_float(config.plastic__geometry_learning_rate_multiplier))
    _print_plastic_option("plastic__freeze_geometry_during_warmup:", _startup_bool(config.plastic__freeze_geometry_during_warmup))
    if public_coordinates:
        _print_plastic_option("active sample_layer:", _startup_public_indices(public_coordinates))
    if full_coordinates and full_coordinates != public_coordinates:
        _print_plastic_option("capacity sample_layer:", _startup_public_indices(full_coordinates))
    print(flush=True)


def _print_model_parameters_and_optimisations(config: Any, trainer: Any) -> None:
    report = trainer.parameter_report
    persistent = int(report["persistent_parameters"])
    dense_equivalent = int(report["dense_equivalent_total_parameters"])
    sheet_coefficients = int(report["sheet_coefficients"])
    compression = (dense_equivalent / persistent) if persistent else 0.0
    print("model parameters and options", flush=True)
    _core._print_model_option("parameters:", f"persistent={persistent:,}  sheet coefficients={sheet_coefficients:,}  dense equivalent={dense_equivalent:,}  dense/persistent={compression:.2f}x")
    # _core._print_model_option("optimiser:", f"lr={config.learning_rate:.3e}  min_lr={config.min_lr:.3e}  warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}")  # <<< THOG preserved pre-alignment optimiser row
    _core._print_model_option("optimiser:", _optimizer_summary(trainer))
    _core._print_model_option("LR decay:", _lr_decay_summary(config, trainer))
    _core._print_model_option("optimiser parms:", f"warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}")
    _core._print_model_option("wall stop:", f"max_wall_minutes={config.max_wall_minutes}")
    _core._print_model_option("non-finite:", f"policy={config.nonfinite_update_policy}  max_skips={config.max_nonfinite_update_skips}")
    _core._print_model_option("batches:", f"micro={config.batch_size}  accumulation={config.gradient_accumulation_steps}  tokens/update={config.tokens_per_iter():,}")
    _core._print_model_option("layer dropout:", f"strata={config.layer_dropout_n_strata}  stratum_size={config.layer_dropout_stratum_size}  active/stratum={config.layer_dropout_active_per_stratum}  active_layers={config.n_active_layers}/{config.n_layer}  resample_steps={config.layer_dropout_resample_steps}")
    if config.model_type == "sheet":
        model = trainer.raw_model
        model_config = model.config
        trajectory = getattr(model, "trajectory", None)
        depth_fallback = _depth_matrix_fallback(trajectory)
        depth_materialisation_matmul = None if depth_fallback is None else bool(depth_fallback.depth_materialisation_matmul)
        _core._print_model_option(
            "execution:",
            f"semantic_qkv_bypass={model_config.bypass_semantic_qkv_adapter}  "
            f"vectorise_per_head={getattr(model_config, 'vectorise_per_head_materialisation', False)}  "
            f"direct_factorised_mlp={getattr(model_config, 'direct_factorised_mlp', False)}  "
            f"direct_factorised_hyperblock_mlp={getattr(model_config, 'direct_factorised_hyperblock_mlp', False)}  "
            f"activation_checkpointing={config.activation_checkpointing}  "
            f"depth_compress_layer_norm_and_bias={model_config.depth_compress_layer_norm_and_bias}  "
            f"depth_materialisation_matmul={depth_materialisation_matmul}",
        )
        hyperblock = report.get("hyperblock")
        if isinstance(hyperblock, dict):
            plan = hyperblock["plan"]
            _core._print_model_option(
                "HYPERBLOCK:",
                f"topology={plan['topology']}  compressor={plan['compressor_family']}@{plan['compressor_version']}  "
                f"coefficients={plan['coefficient_counts']['total']:,}  matrix_dense/coefficient={hyperblock['compression_ratio']:.2f}x  "
                f"loops={hyperblock['loop_count']}  loop_decay={hyperblock['loop_decay']:.6g}",
            )
    print(flush=True)
    _print_plastic_depth_section(config, trainer)
# ^^^ THOG


_core.print_model_parameters_and_options = _print_model_parameters_and_optimisations
# ^^^ THOG

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                        # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation


# vvv THOG apply checkpoint-exit behaviour through one public trainer subclass used by fresh, resume and fork
_BASE_OWT_TRAINER = _core.OwtTrainer
# _stage6._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;33m"                                                                                           # <<< THOG preserved previous standard-yellow validation field
_stage6._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;93m"                                                                                              # <<< THOG bold bright yellow validation-loss field

# vvv THOG render progress timestamp as YYMMDD:HHMM and mean step seconds with four fixed-width decimals
_stage6._progress_timestamp = lambda: _stage6.datetime.now().strftime("%y%m%d:%H%M")
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _latest_plastic_change_z(trainer: Any) -> Optional[tuple[Optional[float], Optional[float]]]:
    if not bool(getattr(getattr(trainer, "config", None), "plastic__do_learn_layer_count", False)):
        return None
    for event in reversed(getattr(trainer, "events", ())):
        if event.name != "plastic_depth_count_decision":
            continue
        payload = event.payload
        values_by_direction = {}
        for item in payload.get("paired_evidence", ()):
            try:
                direction = int(item["direction"])
            except (KeyError, TypeError, ValueError):
                continue
            if direction not in (-1, 1):
                continue
            value = item.get("standardized_improvement")
            values_by_direction[direction] = None if value is None else float(value)
        if values_by_direction:
            return (values_by_direction.get(-1), values_by_direction.get(1))
        return None
    return None


def _prepare_console_progress_payload_with_precise_step(self: Any, event: str, payload: Any):
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event in {"optimizer_progress", "evaluation_completed"}:
        mean_step_seconds = getattr(self, "_console_latest_mean_step_seconds", None)
        if mean_step_seconds is not None:
            values["mean_step_seconds"] = f"{float(mean_step_seconds):8.4f}"
        change_z = _latest_plastic_change_z(self)
        if change_z is not None:
            values["plastic_change_z"] = change_z
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
# vvv THOG preserved superseded source lines for exact history audit
# print("  --depth-materialisation-matmul true|false          DEPTH matrix materialisation; default true")
# os.environ.setdefault("THOG2_DEPTH_MATERIALISATION_MATMUL", "true")
# _core._print_model_option("optimiser:", f"lr={config.learning_rate:.3e}  min_lr={config.min_lr:.3e}  warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}")
# _stage6._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;33m"                                                                                              # <<< THOG use terminal-portable bold yellow for the validation-loss field
# _print_plastic_option("initial layer indices:", _startup_public_indices(public_coordinates))                                                                 # <<< THOG preserve old relative-ruler startup label before absolute sample-layer ruler
# _print_plastic_option("capacity layer indices:", _startup_public_indices(full_coordinates))                                                                  # <<< THOG preserve old relative-ruler startup label before absolute sample-layer ruler
# ^^^ THOG
