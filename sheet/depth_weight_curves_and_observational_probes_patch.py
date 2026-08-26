# vvv THOG
"""Continuous DEPTH scalar-weight charts plus non-authoritative fixed-run layer-count probes."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
from collections import deque
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

import constants as _constants

from . import plastic_depth_probe_sampling_v0521_patch as _probe_sampling
from . import plastic_depth_wandb_probe_curves_patch as _probe_wandb
from . import trainer_step as _trainer_step
from . import wandb_telemetry as _wandb
from .basis import (
    chebyshev_first_kind_basis,
    stabilized_chebyshev_basis_at_coordinates,
)
from .depth_trajectory import DepthTrajectory
from .plastic_depth import public_to_internal_depth
from .semantic_materializer import (
    ATTENTION_QUERY_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from .training_model import TrainingSheetGPT


_ENV_PREFIX = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_"
_DEFAULT_SCALARS_PER_MATRIX = 3
_DEFAULT_DEPTH_POINTS = 256
_DEFAULT_TIME_MODE = "latest"
_DEFAULT_HISTORY_LENGTH = 20
_DEFAULT_LOG_EVERY_N_STEPS = 100
_DEFAULT_DESTINATION = "local"
_SAME_ALL_RUNS_SEED = 0x54484F4732
_DARKER_RHS_GREEN = "\033[38;2;0;180;0m"
_DEPTH_CHART_COLUMNS = (
    "depth_coordinate",
    "weight_value",
    "curve_id",
    "scalar_id",
    "optimizer_update",
    "matrix_family",
)
_CLI_INSTALLED_ATTRIBUTE = "_thog_depth_weight_curve_arguments_installed"


# vvv THOG instrumentation controls remain execution-only: CLI values are copied into environment state and never alter model/checkpoint identity
def _environment_name(suffix: str) -> str:
    return f"{_ENV_PREFIX}{suffix}"


def _bool_from_text(value: str, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be true or false; got {value!r}")


def _positive_int_from_environment(suffix: str, default: int) -> int:
    label = _environment_name(suffix)
    text = os.environ.get(label, str(default))
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a positive integer; got {text!r}") from error
    if value < 1:
        raise ValueError(f"{label} must be a positive integer; got {value!r}")
    return value


def _scalar_weights_per_matrix() -> int:
    return _positive_int_from_environment("SCALAR_WEIGHTS_PER_MATRIX", _DEFAULT_SCALARS_PER_MATRIX)


def _depth_evaluation_points() -> int:
    return _positive_int_from_environment("DEPTH_EVALUATION_POINTS", _DEFAULT_DEPTH_POINTS)


def _history_length() -> int:
    return _positive_int_from_environment("HISTORY_LENGTH", _DEFAULT_HISTORY_LENGTH)


def _log_every_n_steps() -> int:
    return _positive_int_from_environment("LOG_EVERY_N_STEPS", _DEFAULT_LOG_EVERY_N_STEPS)


def _optional_capture_step(suffix: str) -> Optional[int]:
    label = _environment_name(suffix)
    text = os.environ.get(label)
    if text is None or not str(text).strip():
        return None
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a non-negative integer; got {text!r}") from error
    if value < 0:
        raise ValueError(f"{label} must be a non-negative integer; got {value!r}")
    return value


def _weight_snapshot_due(optimizer_update: int) -> bool:
    """Return whether this successful update belongs to the capture window/cadence."""

    update = int(optimizer_update)
    if update < 1:
        return False
    start = _optional_capture_step("START_STEP")
    end = _optional_capture_step("END_STEP")
    if start is not None and end is not None and end < start:
        raise ValueError(
            f"{_environment_name('END_STEP')} must be greater than or equal to "
            f"{_environment_name('START_STEP')}"
        )
    if start is not None and update < start:
        return False
    if end is not None and update > end:
        return False

    cadence = _log_every_n_steps()
    if start is not None and start > 1:
        return (
            update == start
            or (update > start and (update - start) % cadence == 0)
            or (end is not None and update == end)
        )

    # Preserve the established unbounded cadence: update 1, then global cadence
    # multiples. A finite end is an inclusive boundary even when not aligned.
    return (
        update == 1
        or update % cadence == 0
        or (end is not None and update == end)
    )


def _time_mode() -> str:
    value = os.environ.get(_environment_name("TIME_MODE"), _DEFAULT_TIME_MODE).strip().lower()
    if value not in {"latest", "accumulate"}:
        raise ValueError(
            f"{_environment_name('TIME_MODE')} must be latest or accumulate; got {value!r}"
        )
    return value


def _same_coordinates_all_runs() -> bool:
    return _bool_from_text(
        os.environ.get(_environment_name("SAME_COORDINATES_ALL_RUNS"), "false"),
        label=_environment_name("SAME_COORDINATES_ALL_RUNS"),
    )


def _destination() -> str:
    value = os.environ.get(
        _environment_name("DESTINATION"),
        _DEFAULT_DESTINATION,
    ).strip().lower()
    if value not in {"wandb", "local", "none"}:
        raise ValueError(
            f"{_environment_name('DESTINATION')} must be wandb, local, or none; got {value!r}"
        )
    return value


def _ensure_cli_arguments(parser: argparse.ArgumentParser) -> None:
    if bool(getattr(parser, _CLI_INSTALLED_ATTRIBUTE, False)):
        return
    parser.add_argument(
        "--instrumentation__depth_weight_curves__scalar_weights_per_matrix",
        type=int,
        default=_DEFAULT_SCALARS_PER_MATRIX,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__depth_evaluation_points",
        type=int,
        default=_DEFAULT_DEPTH_POINTS,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__time_mode",
        choices=("latest", "accumulate"),
        default=_DEFAULT_TIME_MODE,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__history_length",
        type=int,
        default=_DEFAULT_HISTORY_LENGTH,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__log_every_n_steps",
        type=int,
        default=_DEFAULT_LOG_EVERY_N_STEPS,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__same_coordinates_all_runs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--instrumentation__depth_weight_curves__destination",
        choices=("wandb", "local", "none"),
        default=_DEFAULT_DESTINATION,
    )
    setattr(parser, _CLI_INSTALLED_ATTRIBUTE, True)


def _publish_cli_environment(namespace: argparse.Namespace) -> None:
    mapping = {
        "SCALAR_WEIGHTS_PER_MATRIX": "instrumentation__depth_weight_curves__scalar_weights_per_matrix",
        "DEPTH_EVALUATION_POINTS": "instrumentation__depth_weight_curves__depth_evaluation_points",
        "TIME_MODE": "instrumentation__depth_weight_curves__time_mode",
        "HISTORY_LENGTH": "instrumentation__depth_weight_curves__history_length",
        "LOG_EVERY_N_STEPS": "instrumentation__depth_weight_curves__log_every_n_steps",
        "SAME_COORDINATES_ALL_RUNS": "instrumentation__depth_weight_curves__same_coordinates_all_runs",
        "DESTINATION": "instrumentation__depth_weight_curves__destination",
    }
    for suffix, attribute in mapping.items():
        if not hasattr(namespace, attribute):
            continue
        value = getattr(namespace, attribute)
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        os.environ[_environment_name(suffix)] = text


_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _parse_known_args_with_depth_weight_curves(
    self: argparse.ArgumentParser,
    args: Optional[Sequence[str]] = None,
    namespace: Optional[argparse.Namespace] = None,
):
    _ensure_cli_arguments(self)
    parsed, remaining = _ORIGINAL_PARSE_KNOWN_ARGS(self, args=args, namespace=namespace)
    _publish_cli_environment(parsed)
    return parsed, remaining


def _format_help_with_depth_weight_curves(self: argparse.ArgumentParser) -> str:
    _ensure_cli_arguments(self)
    return _ORIGINAL_FORMAT_HELP(self)


argparse.ArgumentParser.parse_known_args = _parse_known_args_with_depth_weight_curves
argparse.ArgumentParser.format_help = _format_help_with_depth_weight_curves
# ^^^ THOG


# vvv THOG the old active-sample coefficient history is forensic-only and performs no sampling or chart construction unless DEBUG>9
def _legacy_coefficient_chart_enabled() -> bool:
    return int(getattr(_constants, "DEBUG", 0)) > 9


_ORIGINAL_CAPTURE_COEFFICIENT_RECORD = _probe_wandb._capture_coefficient_record
_ORIGINAL_SHOULD_REFRESH_COEFFICIENT_CHART = _probe_wandb._should_refresh_coefficient_chart


def _capture_coefficient_record_debug_gated(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    if not _legacy_coefficient_chart_enabled():
        return {}
    return _ORIGINAL_CAPTURE_COEFFICIENT_RECORD(*args, **kwargs)


def _should_refresh_coefficient_chart_debug_gated(*args: Any, **kwargs: Any) -> bool:
    if not _legacy_coefficient_chart_enabled():
        return False
    return bool(_ORIGINAL_SHOULD_REFRESH_COEFFICIENT_CHART(*args, **kwargs))


_probe_wandb._capture_coefficient_record = _capture_coefficient_record_debug_gated
_probe_wandb._should_refresh_coefficient_chart = _should_refresh_coefficient_chart_debug_gated
# ^^^ THOG


# vvv THOG only the favourable right/extrapolation negative delta becomes a slightly darker green; left-side and global loss colours are unchanged
def _render_probe_delta_values_with_darker_rhs(
    offsets: Sequence[Any],
    losses: Sequence[Any],
) -> Optional[str]:
    resolved_offsets = tuple(int(value) for value in offsets)
    resolved_losses = tuple(None if value is None else float(value) for value in losses)
    if len(resolved_offsets) != len(resolved_losses) or 0 not in resolved_offsets:
        return None
    current_index = resolved_offsets.index(0)
    current_loss = resolved_losses[current_index]
    if current_loss is None or not math.isfinite(current_loss):
        return None
    rendered = []
    for offset, loss in zip(resolved_offsets, resolved_losses):
        if offset == 0:
            rendered.append(
                f"{_constants.BOLD_WHITE}{_constants.UNDER}{_probe_sampling._format_probe_absolute(loss)}{_constants.R}"
            )
            continue
        delta = None if loss is None else float(loss) - float(current_loss)
        text = _probe_sampling._format_probe_delta(delta)
        if delta is not None and math.isfinite(delta) and delta < 0.0:
            if offset < 0:
                text = f"{_constants.BOLD_GREEN}{text}{_constants.R}"
            else:
                text = f"{_DARKER_RHS_GREEN}{text}{_constants.R}"
        rendered.append(text)
    return ", ".join(rendered)


_probe_sampling._render_probe_delta_values = _render_probe_delta_values_with_darker_rhs
# ^^^ THOG


# vvv THOG locate the actual DEPTH field even when a wrapper trajectory owns it
def _depth_trajectory_from_model(raw_model: Any) -> Optional[DepthTrajectory]:
    trajectory = getattr(raw_model, "trajectory", None)
    if isinstance(trajectory, DepthTrajectory):
        return trajectory
    nested = getattr(trajectory, "depth", None)
    return nested if isinstance(nested, DepthTrajectory) else None
# ^^^ THOG


# vvv THOG fixed scalar identities are deterministic for one run, with an explicit cross-run fixed-coordinate option
def _selection_seed(trainer: Any, telemetry: Any) -> int:
    if _same_coordinates_all_runs():
        return _SAME_ALL_RUNS_SEED
    identity = (
        f"{getattr(telemetry, 'name', '')}|"
        f"{int(getattr(trainer.config, 'model_seed', 0))}|"
        f"{getattr(telemetry, 'group', '')}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big", signed=False)


def _sample_unique_matrix_coordinates(
    *,
    seed: int,
    row_start: int,
    row_stop: int,
    column_count: int,
    count: int,
) -> Tuple[Tuple[int, int], ...]:
    row_count = int(row_stop) - int(row_start)
    population = row_count * int(column_count)
    if population < 1:
        raise ValueError("depth-weight diagnostic matrix region is empty")
    requested = min(int(count), population)
    generator = random.Random(int(seed))
    flat_indices = generator.sample(range(population), requested)
    return tuple(
        (
            int(row_start) + flat_index // int(column_count),
            flat_index % int(column_count),
        )
        for flat_index in flat_indices
    )


def _selected_scalar_coordinates(trainer: Any, telemetry: Any) -> Dict[str, Any]:
    cached = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    if cached is not None:
        return cached
    trajectory = _depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return {}
    width = int(trajectory.config.n_embd)
    n_head = int(trajectory.config.n_head)
    if width % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head for DEPTH attention diagnostics")
    seed = _selection_seed(trainer, telemetry)
    head = random.Random(seed ^ 0xA771).randrange(n_head)
    head_dim = width // n_head
    count = _scalar_weights_per_matrix()
    selection = {
        "seed": seed,
        "attention_head": head,
        "attn_q_head_N": _sample_unique_matrix_coordinates(
            seed=seed ^ 0x11,
            row_start=head * head_dim,
            row_stop=(head + 1) * head_dim,
            column_count=width,
            count=count,
        ),
        "mlp_up": _sample_unique_matrix_coordinates(
            seed=seed ^ 0x22,
            row_start=0,
            row_stop=4 * width,
            column_count=width,
            count=count,
        ),
        "mlp_down": _sample_unique_matrix_coordinates(
            seed=seed ^ 0x33,
            row_start=0,
            row_stop=width,
            column_count=4 * width,
            count=count,
        ),
    }
    setattr(telemetry, "_thog_depth_weight_curve_selection", selection)
    return selection
# ^^^ THOG


# vvv THOG evaluate one DEPTH coefficient vector on a dense continuous 1..100 ruler independently of the model's actual layer count
def _continuous_depth_basis(trajectory: DepthTrajectory, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    public = torch.linspace(
        1.0,
        100.0,
        _depth_evaluation_points(),
        dtype=torch.float64,
        device=reference.device,
    )
    internal = public_to_internal_depth(public).to(dtype=torch.float64)
    if trajectory.plastic_enabled:
        raw = chebyshev_first_kind_basis(internal, trajectory.config.depth_order)
        basis = raw @ trajectory.plastic_depth_inverse_r.to(
            device=internal.device,
            dtype=internal.dtype,
        )
        return public, basis.to(dtype=reference.dtype)
    basis = stabilized_chebyshev_basis_at_coordinates(
        internal,
        reference_sample_count=int(trajectory.config.n_layer),
        order=int(trajectory.config.depth_order),
        runtime_dtype=reference.dtype,
        version=trajectory.basis_version,
    )
    return public, basis


@torch.no_grad()
def _depth_weight_snapshot(trainer: Any, telemetry: Any, *, optimizer_update: int) -> Dict[str, Any]:
    trajectory = _depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return {}
    selection = _selected_scalar_coordinates(trainer, telemetry)
    if not selection:
        return {}
    family_by_chart = {
        "attn_q_head_N": ATTENTION_QUERY_WEIGHT,
        "mlp_up": MLP_EXPANSION_WEIGHT,
        "mlp_down": MLP_CONTRACTION_WEIGHT,
    }
    snapshot: Dict[str, Any] = {
        "optimizer_update": int(optimizer_update),
        "attention_head": int(selection["attention_head"]),
        "seed": int(selection["seed"]),
        "families": {},
    }
    for chart_name, family in family_by_chart.items():
        parameter = trajectory.coefficients[family]
        public, basis = _continuous_depth_basis(trajectory, parameter)
        curves = []
        for output_row, row_index in selection[chart_name]:
            coefficient = parameter[int(output_row), int(row_index)]
            values = basis.to(device=coefficient.device, dtype=coefficient.dtype) @ coefficient
            curves.append(
                {
                    "scalar_id": f"r{int(output_row)}_c{int(row_index)}",
                    "output_row": int(output_row),
                    "row_index": int(row_index),
                    "values": tuple(float(value) for value in values.detach().to(device="cpu", dtype=torch.float64).tolist()),
                }
            )
        snapshot["families"][chart_name] = {
            "semantic_family": family,
            "depth_coordinates": tuple(float(value) for value in public.detach().to(device="cpu").tolist()),
            "curves": tuple(curves),
        }
    return snapshot


def _depth_weight_history(telemetry: Any) -> deque:
    history = getattr(telemetry, "_thog_depth_weight_curve_history", None)
    desired = _history_length()
    if history is None or history.maxlen != desired:
        old = tuple(history) if history is not None else ()
        history = deque(old[-desired:], maxlen=desired)
        setattr(telemetry, "_thog_depth_weight_curve_history", history)
    return history


def _depth_chart_rows(snapshots: Iterable[Mapping[str, Any]], chart_name: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    snapshots_tuple = tuple(snapshots)
    include_update_in_curve = len(snapshots_tuple) > 1
    for snapshot in snapshots_tuple:
        update = int(snapshot["optimizer_update"])
        family = snapshot["families"][chart_name]
        semantic_family = str(family["semantic_family"])
        depth_coordinates = tuple(float(value) for value in family["depth_coordinates"])
        for curve in family["curves"]:
            scalar_id = str(curve["scalar_id"])
            curve_id = f"U{update}:{scalar_id}" if include_update_in_curve else scalar_id
            for depth, value in zip(depth_coordinates, curve["values"]):
                rows.append([depth, float(value), curve_id, scalar_id, update, semantic_family])
    return rows


def _log_depth_weight_snapshot(trainer: Any, telemetry: Any, *, optimizer_update: int) -> None:
    if int(getattr(_constants, "DEBUG", 0)) <= 2:
        return
    if telemetry.run is None or telemetry.module is None:
        return
    snapshot = _depth_weight_snapshot(trainer, telemetry, optimizer_update=optimizer_update)
    if not snapshot:
        return
    history = _depth_weight_history(telemetry)
    history.append(snapshot)
    snapshots = (snapshot,) if _time_mode() == "latest" else tuple(history)
    payload: Dict[str, Any] = {}
    for chart_name in ("attn_q_head_N", "mlp_up", "mlp_down"):
        rows = _depth_chart_rows(snapshots, chart_name)
        if not rows:
            continue
        table = telemetry.module.Table(data=rows, columns=list(_DEPTH_CHART_COLUMNS))
        title_name = (
            f"attn_q_head_{int(snapshot['attention_head'])}"
            if chart_name == "attn_q_head_N"
            else chart_name
        )
        payload[f"depth/{chart_name}"] = telemetry.module.plot.line(
            table=table,
            x="depth_coordinate",
            y="weight_value",
            stroke="curve_id",
            title=f"DEPTH generated scalar trajectories — {title_name}",
        )
    if not payload:
        return
    try:
        telemetry.run.log(payload, step=int(optimizer_update))
    except TypeError:
        telemetry.run.log(payload)
    if not bool(getattr(telemetry, "_thog_depth_weight_curve_selection_logged", False)):
        selection = _selected_scalar_coordinates(trainer, telemetry)
        metadata = {
            "depth/selection_seed": int(selection["seed"]),
            "depth/attention_head": int(selection["attention_head"]),
            "depth/same_coordinates_all_runs": bool(_same_coordinates_all_runs()),
            "depth/scalar_weights_per_matrix": int(_scalar_weights_per_matrix()),
            "depth/evaluation_points": int(_depth_evaluation_points()),
        }
        try:
            telemetry.run.log(metadata, step=int(optimizer_update))
        except TypeError:
            telemetry.run.log(metadata)
        telemetry._thog_depth_weight_curve_selection_logged = True
# ^^^ THOG


# vvv THOG W&B attachment layers the new depth group after the established PLASTIC probe charts while leaving TensorBoard untouched
_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def _attach_telemetry_with_depth_weight_curves(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    if int(getattr(_constants, "DEBUG", 0)) <= 2:
        return
    if _depth_trajectory_from_model(trainer.raw_model) is None:
        return
    original_timed = trainer._timed
    train_one_update = trainer.train_one_update

    def timed(function: Any):
        metrics, elapsed = original_timed(function)
        if function != train_one_update:
            return metrics, elapsed
        if not trainer.distributed.is_primary:
            return metrics, elapsed
        if bool(float(metrics.get("skipped_update", 0.0))):
            return metrics, elapsed
        update = int(trainer.state.completed_updates)
        if not _weight_snapshot_due(update):
            return metrics, elapsed
        try:
            _log_depth_weight_snapshot(
                trainer,
                telemetry,
                optimizer_update=update,
            )
        except Exception as error:
            print(
                "THOG2 WARNING: continuous DEPTH weight-curve logging failed; "
                f"continuing without this refresh: {error}",
                flush=True,
            )
        return metrics, elapsed

    trainer._timed = timed


_wandb.attach_telemetry = _attach_telemetry_with_depth_weight_curves
# ^^^ THOG


# vvv THOG observational probes temporarily view the same learned DEPTH field at a different count; authoritative parameters and optimizer state never change
_ORIGINAL_DEPTH_ROW = DepthTrajectory._depth_row
_ORIGINAL_DEPTH_MATERIALIZE = DepthTrajectory.materialize
_ORIGINAL_CONVENTIONAL_MATERIALIZE = DepthTrajectory._materialize_conventional_parameter
_ORIGINAL_LAYER_INDICES_FOR_FORWARD = TrainingSheetGPT._layer_indices_for_current_forward


def _observational_coordinates(trajectory: DepthTrajectory) -> Optional[torch.Tensor]:
    return getattr(trajectory, "_thog_observational_depth_coordinates", None)


def _depth_row_with_observational_coordinates(
    self: DepthTrajectory,
    layer_index: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    coordinates = _observational_coordinates(self)
    if coordinates is None:
        return _ORIGINAL_DEPTH_ROW(self, layer_index, reference)
    public = coordinates[int(layer_index) : int(layer_index) + 1].to(
        device=reference.device,
        dtype=torch.float64,
    )
    internal = public_to_internal_depth(public)
    if self.plastic_enabled:
        raw = chebyshev_first_kind_basis(internal, self.config.depth_order)
        row = raw @ self.plastic_depth_inverse_r.to(device=internal.device, dtype=internal.dtype)
        return row[0].to(device=reference.device, dtype=reference.dtype)
    row = stabilized_chebyshev_basis_at_coordinates(
        internal,
        reference_sample_count=int(self.config.n_layer),
        order=int(self.config.depth_order),
        runtime_dtype=reference.dtype,
        version=self.basis_version,
    )
    return row[0]


def _base_conventional_coordinates(self: DepthTrajectory, *, device: torch.device) -> torch.Tensor:
    if self.plastic_enabled and self.plastic_sampling is not None:
        return self.plastic_sampling.active_public_coordinates().detach().to(device=device, dtype=torch.float64)
    return torch.linspace(1.0, 100.0, int(self.config.n_layer), device=device, dtype=torch.float64)


def _materialize_conventional_parameter_observational(
    self: DepthTrajectory,
    name: str,
    layer_index: int,
) -> torch.Tensor:
    coordinates = _observational_coordinates(self)
    if coordinates is None:
        return _ORIGINAL_CONVENTIONAL_MATERIALIZE(self, name, layer_index)
    parameter = self.coefficients[name]
    target = coordinates[int(layer_index)].to(device=parameter.device, dtype=torch.float64)
    base = _base_conventional_coordinates(self, device=parameter.device)
    if base.numel() == 1:
        return parameter[0]
    right = int(torch.searchsorted(base, target, right=False).item())
    if right <= 0:
        return parameter[0]
    if right >= int(base.numel()):
        return parameter[-1]
    left = right - 1
    span = base[right] - base[left]
    fraction = ((target - base[left]) / span).to(dtype=parameter.dtype)
    return parameter[left] + fraction * (parameter[right] - parameter[left])


def _materialize_with_observational_coordinates(
    self: DepthTrajectory,
    name: str,
    layer_index: int,
) -> torch.Tensor:
    coordinates = _observational_coordinates(self)
    if coordinates is None:
        return _ORIGINAL_DEPTH_MATERIALIZE(self, name, layer_index)
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
    if layer_index < 0 or layer_index >= int(coordinates.numel()):
        raise IndexError(
            f"observational layer_index out of range: {layer_index}; candidate_count={int(coordinates.numel())}"
        )
    if name == LEGACY_ATTENTION_INPUT_WEIGHT:
        return torch.cat(
            (
                self._materialize_depth_parameter(ATTENTION_QUERY_WEIGHT, layer_index),
                self._materialize_depth_parameter("attention_key_weight", layer_index),
                self._materialize_depth_parameter("attention_value_weight", layer_index),
            ),
            dim=0,
        )
    item = self.family_metadata(name)
    representation = self._representation(item)
    if representation == "depth_coefficients":
        return self._materialize_depth_parameter(name, layer_index)
    if representation == "legacy_sheet_col":
        return self._materialize_legacy_vector(name, layer_index)
    return self._materialize_conventional_parameter(name, layer_index)


def _layer_indices_with_observational_candidate(self: TrainingSheetGPT) -> Tuple[int, ...]:
    candidate = getattr(self, "_thog_observational_depth_candidate_count", None)
    if candidate is not None:
        return tuple(range(int(candidate)))
    return _ORIGINAL_LAYER_INDICES_FOR_FORWARD(self)


DepthTrajectory._depth_row = _depth_row_with_observational_coordinates
DepthTrajectory._materialize_conventional_parameter = _materialize_conventional_parameter_observational
DepthTrajectory.materialize = _materialize_with_observational_coordinates
TrainingSheetGPT._layer_indices_for_current_forward = _layer_indices_with_observational_candidate
# ^^^ THOG


# vvv THOG an explicit probe cadence opts a fixed-count DEPTH run into read-only L-R..L+R probes; learned-count PLASTIC keeps its established inline path
def _observational_probe_enabled(trainer: Any) -> bool:
    config = getattr(trainer, "config", None)
    if config is None:
        return False
    if (
        bool(getattr(config, "plastic__enabled", False))
        and bool(getattr(config, "plastic__do_learn_layer_count", False))
    ):
        return False
    if _depth_trajectory_from_model(getattr(trainer, "raw_model", None)) is None:
        return False
    interval = getattr(config, "plastic__layer_count_probe__probe_every_n_steps", None)
    heatmap_mode = getattr(
        config,
        "instrumentation__delta_loss_v_layer_heatmap",
        None,
    )
    return (
        interval is not None and int(interval) >= 1
    ) or heatmap_mode == "linear"


def _observational_probe_due(trainer: Any, update: int) -> bool:
    configured_interval = getattr(
        trainer.config,
        "plastic__layer_count_probe__probe_every_n_steps",
        None,
    )
    interval = 1 if configured_interval is None else int(configured_interval)
    return int(update) == 1 or int(update) % interval == 0


def _masked_observational_targets(trainer: Any, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    sampled = trainer._plastic_depth_sampled_token_indices(targets)
    flattened_source = targets.reshape(-1)
    flattened = torch.full_like(flattened_source, -1)
    flattened.index_copy_(0, sampled, flattened_source.index_select(0, sampled))
    return flattened.reshape_as(targets), int(sampled.numel())


def _observational_probe_batch(trainer: Any, update: int) -> Any:
    from .batch_source import DeterministicBatchSource

    source = DeterministicBatchSource(
        trainer.batch_source.train_tokens,
        trainer.batch_source.validation_tokens,
        block_size=int(trainer.config.block_size),
        batch_size=int(trainer.config.batch_size),
        data_seed=int(trainer.config.data_seed) + 2_000_003 + int(update),
        rank=int(trainer.distributed.rank),
        world_size=int(trainer.distributed.world_size),
        trace_limit=0,
    )
    return source.get_batch("val", device=trainer.device)


def _current_observational_layer_count(trainer: Any, trajectory: DepthTrajectory) -> int:
    if trajectory.plastic_enabled and trajectory.plastic_sampling is not None:
        return int(trajectory.plastic_sampling.current_active_layers)
    return int(trajectory.config.n_layer)


@torch.no_grad()
def _run_observational_probe(trainer: Any, *, update: int) -> None:
    trajectory = _depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return
    current = _current_observational_layer_count(trainer, trajectory)
    radius = int(trainer.config.plastic__layer_count_probe_radius)
    candidates = tuple(range(max(1, current - radius), current + radius + 1))
    batch = _observational_probe_batch(trainer, update)
    probe_targets, sampled_token_count = _masked_observational_targets(trainer, batch.targets)
    was_training = trainer.model.training
    trainer.model.eval()
    measurements = []
    try:
        for candidate in candidates:
            coordinates = torch.linspace(
                1.0,
                100.0,
                int(candidate),
                dtype=torch.float64,
                device=trajectory.coefficients[ATTENTION_QUERY_WEIGHT].device,
            )
            trajectory._thog_observational_depth_coordinates = coordinates
            trainer.raw_model._thog_observational_depth_candidate_count = int(candidate)
            try:
                with trainer.autocast_context():
                    _, loss = trainer.model(batch.inputs, probe_targets)
                if loss is None or not bool(torch.isfinite(loss).item()):
                    value = float("inf")
                else:
                    value = trainer.distributed.mean_float(loss.detach())
            except torch.OutOfMemoryError:
                value = float("inf")
                if trainer.device.type == "cuda":
                    torch.cuda.empty_cache()
            measurements.append(
                {
                    "active_layers": int(candidate),
                    "validation_loss": float(value),
                    "training_time": None,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "feasible": math.isfinite(float(value)),
                    "score": float(value),
                    "observational_only": True,
                }
            )
    finally:
        if hasattr(trajectory, "_thog_observational_depth_coordinates"):
            delattr(trajectory, "_thog_observational_depth_coordinates")
        if hasattr(trainer.raw_model, "_thog_observational_depth_candidate_count"):
            delattr(trainer.raw_model, "_thog_observational_depth_candidate_count")
        trainer.model.train(was_training)
    current_measurement = next(
        (item for item in measurements if int(item["active_layers"]) == current),
        None,
    )
    if current_measurement is None or not math.isfinite(float(current_measurement["validation_loss"])):
        raise RuntimeError("observational DEPTH probe did not produce a finite current-count loss")
    trainer._record(
        "plastic_depth_count_decision",
        previous_active_layers=current,
        selected_active_layers=current,
        candidates=tuple(measurements),
        objective="observational_only",
        sampled_token_count=sampled_token_count,
        observational_only=True,
        probe_update=int(update),
        probe_radius=radius,
        public_coordinates=tuple(
            float(value)
            for value in torch.linspace(1.0, 100.0, current, dtype=torch.float64).tolist()
        ),
        transition={},
    )
# ^^^ THOG


_ORIGINAL_TRAIN_ONE_UPDATE = _trainer_step.TrainerStepMixin.train_one_update


def _train_one_update_with_observational_probe(self: Any) -> Dict[str, Any]:
    metrics = _ORIGINAL_TRAIN_ONE_UPDATE(self)
    if not _observational_probe_enabled(self):
        return metrics
    if bool(float(metrics.get("skipped_update", 0.0))):
        return metrics
    update = int(self.state.completed_updates)
    if not _observational_probe_due(self, update):
        return metrics
    _run_observational_probe(self, update=update)
    # vvv THOG this transient marker lets linear heatmap mode publish the completed observational probe even off the ordinary log interval
    self._depth_probe_optimizer_update = int(update)
    # ^^^ THOG
    return metrics


_trainer_step.TrainerStepMixin.train_one_update = _train_one_update_with_observational_probe
# ^^^ THOG


# vvv THOG W&B probe-history parser preserves the real optimizer update for post-update observational events
_ORIGINAL_PROBE_RECORD_FROM_EVENT = _probe_wandb._probe_record_from_event


def _probe_record_from_event_with_observational_update(event: Any) -> Optional[Dict[str, Any]]:
    record = _ORIGINAL_PROBE_RECORD_FROM_EVENT(event)
    if record is None:
        return None
    payload = getattr(event, "payload", None)
    if isinstance(payload, Mapping) and bool(payload.get("observational_only", False)):
        update = int(payload.get("probe_update", record["optimizer_update"]))
        record["optimizer_update"] = update
        record["probe_id"] = f"U{update}"
    return record


_probe_wandb._probe_record_from_event = _probe_record_from_event_with_observational_update
# ^^^ THOG


__all__ = [
    "_depth_weight_snapshot",
    "_legacy_coefficient_chart_enabled",
    "_observational_probe_enabled",
    "_render_probe_delta_values_with_darker_rhs",
    "_run_observational_probe",
    "_selected_scalar_coordinates",
]
# ^^^ THOG
