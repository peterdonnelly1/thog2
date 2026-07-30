# vvv THOG
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .depth_trajectory import DepthTrajectory
from .semantic_materializer import LEGACY_ATTENTION_INPUT_WEIGHT, MLP_CONTRACTION_WEIGHT
from . import stage6_trainer as _stage6


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_INSTALLED = False
_ORIGINAL_DEPTH_INIT = DepthTrajectory.__init__
_ORIGINAL_DEPTH_MATERIALIZE = DepthTrajectory.materialize
_ORIGINAL_DEPTH_MATERIALIZE_PARAMETER = DepthTrajectory._materialize_depth_parameter
_ORIGINAL_STAGE6_TIMED = _stage6.Stage6Trainer._timed
_ORIGINAL_STAGE6_PRINT_PROGRESS = _stage6.Stage6Trainer._print_progress
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


@dataclass
class _MaterialisationTimingRecord:
    layer_index: int
    name: str
    cpu_seconds: Optional[float] = None
    cuda_start: Any = None
    cuda_end: Any = None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in _TRUE_VALUES:
        return True
    if normalized_value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false; got {raw_value!r}")


def _depth_init_with_runtime_controls(self: DepthTrajectory, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_DEPTH_INIT(self, *args, **kwargs)
    requested_matmul = _env_bool("THOG2_DEPTH_MATERIALISATION_MATMUL", False)
    self.depth_materialisation_matmul = bool(requested_matmul and not self.legacy_sheet_col_vectors)
    self._thog_materialisation_profiling_enabled = False
    self._thog_materialisation_timing_records: List[_MaterialisationTimingRecord] = []


def _depth_materialize_parameter_with_matmul(self: DepthTrajectory, name: str, layer_index: int) -> Tensor:
    if not bool(getattr(self, "depth_materialisation_matmul", False)):
        return _ORIGINAL_DEPTH_MATERIALIZE_PARAMETER(self, name, layer_index)

    coefficient = self.coefficients[name]
    depth_row = self.depth_basis[layer_index].to(
        device=coefficient.device,
        dtype=coefficient.dtype,
    )
    item = self.family_metadata(name)
    with torch.autocast(device_type=coefficient.device.type, enabled=False):
        flattened_coefficient = coefficient.reshape(-1, coefficient.shape[-1])
        generated = torch.matmul(flattened_coefficient, depth_row).reshape(
            item.output_rows,
            item.row_width,
        )
    expected_shape = (item.output_rows, item.row_width)
    if tuple(generated.shape) != expected_shape:
        raise RuntimeError(
            f"depth parameter {name} has shape {tuple(generated.shape)}; expected {expected_shape}"
        )
    return generated


def _generated_materialisation(self: DepthTrajectory, name: str) -> bool:
    if self.legacy_sheet_col_vectors:
        return False
    if name == LEGACY_ATTENTION_INPUT_WEIGHT:
        return True
    item = self.family_metadata(name)
    return self._is_generated(item)


def _depth_materialize_with_timing(self: DepthTrajectory, name: str, layer_index: int) -> Tensor:
    profiling = bool(getattr(self, "_thog_materialisation_profiling_enabled", False)) and _generated_materialisation(self, name)
    cpu_started: Optional[float] = None
    cuda_start: Any = None
    cuda_end: Any = None
    if profiling:
        device = self.depth_basis.device
        if device.type == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record(torch.cuda.current_stream(device=device))
        else:
            cpu_started = time.perf_counter()

    generated = _ORIGINAL_DEPTH_MATERIALIZE(self, name, layer_index)

    if profiling:
        if cuda_start is not None:
            cuda_end.record(torch.cuda.current_stream(device=self.depth_basis.device))
            self._thog_materialisation_timing_records.append(
                _MaterialisationTimingRecord(
                    layer_index=layer_index,
                    name=name,
                    cuda_start=cuda_start,
                    cuda_end=cuda_end,
                )
            )
        else:
            assert cpu_started is not None
            self._thog_materialisation_timing_records.append(
                _MaterialisationTimingRecord(
                    layer_index=layer_index,
                    name=name,
                    cpu_seconds=max(0.0, time.perf_counter() - cpu_started),
                )
            )
    return generated


def _begin_materialisation_profiling(self: DepthTrajectory) -> None:
    if self.legacy_sheet_col_vectors:
        return
    self._thog_materialisation_timing_records = []
    self._thog_materialisation_profiling_enabled = True


def _end_materialisation_profiling(self: DepthTrajectory) -> None:
    self._thog_materialisation_profiling_enabled = False


def _record_seconds(record: _MaterialisationTimingRecord) -> float:
    if record.cpu_seconds is not None:
        return float(record.cpu_seconds)
    if record.cuda_start is None or record.cuda_end is None:
        raise RuntimeError("incomplete CUDA materialisation timing record")
    return float(record.cuda_start.elapsed_time(record.cuda_end)) / 1000.0


def _consume_materialisation_penalty_samples(self: DepthTrajectory) -> Tuple[float, ...]:
    records = tuple(self._thog_materialisation_timing_records)
    self._thog_materialisation_timing_records = []
    if not records:
        return ()

    samples: List[float] = []
    current_layer: Optional[int] = None
    current_seconds = 0.0
    for record in records:
        if current_layer is not None and record.layer_index != current_layer:
            samples.append(current_seconds)
            current_seconds = 0.0
        current_layer = record.layer_index
        current_seconds += _record_seconds(record)
        if record.name == MLP_CONTRACTION_WEIGHT:
            samples.append(current_seconds)
            current_layer = None
            current_seconds = 0.0
    if current_layer is not None:
        samples.append(current_seconds)
    return tuple(samples)


def _pure_depth_trajectory(trainer: Any) -> Optional[DepthTrajectory]:
    raw_model = getattr(trainer, "raw_model", None)
    trajectory = getattr(raw_model, "trajectory", None)
    if not isinstance(trajectory, DepthTrajectory):
        return None
    if trajectory.legacy_sheet_col_vectors:
        return None
    return trajectory


def _accumulate_materialisation_samples(trainer: Any, samples: Sequence[float]) -> None:
    count = int(getattr(trainer, "_thog_materialisation_penalty_count", 0))
    mean = float(getattr(trainer, "_thog_materialisation_penalty_mean", 0.0))
    m2 = float(getattr(trainer, "_thog_materialisation_penalty_m2", 0.0))
    for sample in samples:
        value = float(sample)
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError(f"invalid materialisation timing sample: {value!r}")
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    trainer._thog_materialisation_penalty_count = count
    trainer._thog_materialisation_penalty_mean = mean
    trainer._thog_materialisation_penalty_m2 = m2


def _reset_materialisation_interval(trainer: Any) -> None:
    trainer._thog_materialisation_penalty_count = 0
    trainer._thog_materialisation_penalty_mean = 0.0
    trainer._thog_materialisation_penalty_m2 = 0.0


def _materialisation_interval_field(trainer: Any) -> Optional[str]:
    count = int(getattr(trainer, "_thog_materialisation_penalty_count", 0))
    if count <= 0:
        return None
    mean = float(trainer._thog_materialisation_penalty_mean)
    variance = max(0.0, float(trainer._thog_materialisation_penalty_m2) / count)
    standard_deviation = math.sqrt(variance)
    return f"{mean:.4f}±{standard_deviation:.4f}s/layer"


def _stage6_timed_with_materialisation_profile(self: Any, function: Any):
    trajectory = _pure_depth_trajectory(self)
    profile_training = trajectory is not None and getattr(function, "__name__", "") == "train_one_update"
    if profile_training:
        trajectory.begin_materialisation_profiling()
    try:
        result, elapsed = _ORIGINAL_STAGE6_TIMED(self, function)
    finally:
        if profile_training:
            trajectory.end_materialisation_profiling()
    if profile_training:
        _accumulate_materialisation_samples(
            self,
            trajectory.consume_materialisation_penalty_samples(),
        )
    return result, elapsed


def _stage6_print_progress_with_materialisation(self: Any, run_id: str, event: str, **payload: Any) -> None:
    reset_interval = False
    if event == "optimizer_progress":
        materialisation_field = _materialisation_interval_field(self)
        if materialisation_field is not None:
            payload["materialisation_penalty"] = materialisation_field
            reset_interval = True
    try:
        _ORIGINAL_STAGE6_PRINT_PROGRESS(self, run_id, event, **payload)
    finally:
        if reset_interval:
            _reset_materialisation_interval(self)


def _format_progress_line_with_materialisation(run_id: str, event: str, payload: Any) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    if event != "optimizer_progress" or "materialisation_penalty" not in payload:
        return line
    field = f"materialisation penalty={payload['materialisation_penalty']}"
    for marker in ("  tok/s=", "  tokens=", "  training loss"):
        position = line.find(marker)
        if position >= 0:
            return f"{line[:position]}  {field}{line[position:]}"
    return f"{line}  {field}"


def install_depth_materialisation_runtime() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    DepthTrajectory.__init__ = _depth_init_with_runtime_controls
    DepthTrajectory._materialize_depth_parameter = _depth_materialize_parameter_with_matmul
    DepthTrajectory.materialize = _depth_materialize_with_timing
    DepthTrajectory.begin_materialisation_profiling = _begin_materialisation_profiling
    DepthTrajectory.end_materialisation_profiling = _end_materialisation_profiling
    DepthTrajectory.consume_materialisation_penalty_samples = _consume_materialisation_penalty_samples
    _stage6.Stage6Trainer._timed = _stage6_timed_with_materialisation_profile
    _stage6.Stage6Trainer._print_progress = _stage6_print_progress_with_materialisation
    _stage6.format_progress_line = _format_progress_line_with_materialisation
    _INSTALLED = True


__all__ = ["install_depth_materialisation_runtime"]
# ^^^ THOG
