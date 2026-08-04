# vvv THOG
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .model import SheetGPT
from .training_config import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_COMPATIBILITY_FIELDS,
    TrainingConfig,
)


def optimizer_group_names(optimizer: torch.optim.Optimizer) -> Tuple[Tuple[str, ...], ...]:
    result = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = group.get("parameter_names")
        if names is None:
            names = tuple(
                f"group_{group_index}_parameter_{position}"
                for position, _ in enumerate(group["params"])
            )
        result.append(tuple(names))
    return tuple(result)


def strip_compiled_prefix(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    prefix = "_orig_mod."
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# vvv THOG geometry option spelling/source is provenance; checkpoint identity is the resolved semantic plan
_GEOMETRY_PROVENANCE_FIELDS = frozenset({"parsed_options"})


def _semantic_geometry_plan(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: nested_value
        for key, nested_value in value.items()
        if key not in _GEOMETRY_PROVENANCE_FIELDS
    }


def _semantic_compatibility_value(name: str, value: Any) -> Any:
    if name == "resolved_geometry_plan":
        return _semantic_geometry_plan(value)
    return value


# vvv THOG canonical public spellings are provenance; checkpoint compatibility compares the resolved PLASTIC semantics

def _semantic_plastic_depth_identity(value: Any, *, maximum_layers: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if "plastic__enabled" not in value:
        return value
    return {
        "version": value.get("version"),
        "maximum_layers": maximum_layers,
        "initial_active_layers": value.get("plastic__initial_active_layers"),
        "learn_layer_count": value.get("plastic__do_learn_layer_count"),
        "sampling_initialisation": value.get("plastic__layer_sampling_initialisation"),
        "count_objective": value.get("plastic__layer_count_objective"),
        "count_update_brake": value.get("plastic__layer_count_update_brake"),
        "probe_noise_window": value.get("plastic__layer_count_probe_noise_window"),
        "probe_noise_min_observations": value.get("plastic__layer_count_probe_noise_min_observations"),
        "probe_noise_lambda": value.get("plastic__layer_count_probe_noise_lambda"),
        "count_cost_weight": value.get("plastic__layer_count_cost_weight"),
        "memory_budget_gib": value.get("plastic__layer_memory_budget_gib"),
        "cuda_allocator_reserve_gib": value.get("plastic__cuda_allocator_reserve_gib"),
        "geometry_lr_multiplier": value.get("plastic__geometry_learning_rate_multiplier"),
        "freeze_geometry_during_warmup": value.get("plastic__freeze_geometry_during_warmup"),
    }


# ^^^ THOG


def _semantic_compact_identity(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    if "resolved_geometry_plan" in normalized:
        normalized["resolved_geometry_plan"] = _semantic_geometry_plan(
            normalized["resolved_geometry_plan"]
        )
    # vvv THOG accept checkpoints written with the retired short PLASTIC aliases while all new writes use plastic__ names
    if "plastic_depth" in normalized:
        # normalized["plastic_depth"] = normalized["plastic_depth"]
        normalized["plastic_depth"] = _semantic_plastic_depth_identity(
            normalized["plastic_depth"],
            maximum_layers=normalized.get("n_layer"),
        )
    # ^^^ THOG
    return normalized
# ^^^ THOG


def validate_compatibility(payload: Mapping[str, Any], expected: TrainingConfig) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version mismatch: expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    checkpoint_signature = payload.get("compatibility_signature", {})
    expected_signature = expected.compatibility_signature()
    mismatches = []
    for name in MODEL_COMPATIBILITY_FIELDS:
        # if checkpoint_signature.get(name) != expected_signature[name]:
        # vvv THOG compare semantic geometry while retaining strict compatibility for every other model field
        checkpoint_value = _semantic_compatibility_value(
            name,
            checkpoint_signature.get(name),
        )
        expected_value = _semantic_compatibility_value(
            name,
            expected_signature[name],
        )
        if checkpoint_value != expected_value:
        # ^^^ THOG
            mismatches.append(
                f"{name}: checkpoint={checkpoint_signature.get(name)!r}, "
                f"expected={expected_signature[name]!r}"
            )
    checkpoint_identity = payload.get("compact_identity")
    expected_identity = expected.compact_identity_metadata()
    # if checkpoint_identity != expected_identity:
    # vvv THOG compact identity follows the same semantic geometry rule as the compatibility signature
    if _semantic_compact_identity(checkpoint_identity) != _semantic_compact_identity(expected_identity):
    # ^^^ THOG
        mismatches.append(
            "compact_identity: checkpoint does not match expected resolved compact metadata"
        )
    if mismatches:
        raise ValueError("incompatible checkpoint geometry or model state: " + "; ".join(mismatches))


def compact_model_state(model: torch.nn.Module, model_type: str) -> Mapping[str, Tensor]:
    state = model.state_dict()
    if model_type == "thog2_sheet":
        forbidden = [key for key in state if key.startswith("trajectory.bases.")]
        if forbidden:
            raise RuntimeError(f"fixed bases appeared in checkpoint state: {forbidden}")
        if isinstance(model, SheetGPT):
            violations = model.compact_state_violations()
            if violations:
                raise RuntimeError(f"compact-state violations: {violations}")
    return state


def save_payload(payload: Mapping[str, Any], path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), target)
    return target


def load_payload(path: Union[str, Path], *, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
# ^^^ THOG
