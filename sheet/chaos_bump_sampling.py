# vvv THOG
"""Deterministic, reversible sampling-coordinate chaos bumps for PLASTIC DEPTH."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor


CHAOS_BUMP_SAMPLING_VERSION = "chaos_bump_sampling_v1_3"
CHAOS_BUMP_SAMPLING_CONFIG_FIELDS = (
    "chaos_bump__sampling__enabled",
    "chaos_bump__sampling__initial_lockout__steps",
    "chaos_bump__sampling__maximum_bumps",
    "chaos_bump__sampling__interlude__min_steps",
    "chaos_bump__sampling__interlude__max_steps",
    "chaos_bump__sampling__duration__min_steps",
    "chaos_bump__sampling__duration__max_steps",
    "chaos_bump__sampling__duration__max_fraction_of_elapsed_steps",
    "chaos_bump__sampling__max_movement_fraction_of_local_gap",
)


@dataclass(frozen=True)
class ResolvedChaosBumpSamplingConfig:
    enabled: bool
    initial_lockout_steps: int
    maximum_bumps: int
    interlude_min_steps: int
    interlude_max_steps: int
    duration_min_steps: int
    duration_max_steps: int
    duration_max_fraction_of_elapsed_steps: float
    max_movement_fraction_of_local_gap: float

    def identity(self) -> Dict[str, Any]:
        return {
            "version": CHAOS_BUMP_SAMPLING_VERSION,
            "chaos_bump__sampling__enabled": self.enabled,
            "chaos_bump__sampling__initial_lockout__steps": self.initial_lockout_steps,
            "chaos_bump__sampling__maximum_bumps": self.maximum_bumps,
            "chaos_bump__sampling__interlude__min_steps": self.interlude_min_steps,
            "chaos_bump__sampling__interlude__max_steps": self.interlude_max_steps,
            "chaos_bump__sampling__duration__min_steps": self.duration_min_steps,
            "chaos_bump__sampling__duration__max_steps": self.duration_max_steps,
            "chaos_bump__sampling__duration__max_fraction_of_elapsed_steps": self.duration_max_fraction_of_elapsed_steps,
            "chaos_bump__sampling__max_movement_fraction_of_local_gap": self.max_movement_fraction_of_local_gap,
        }


@dataclass(frozen=True)
class SamplingCoordinateRattle:
    coordinates: Tuple[float, ...]
    signed_movements: Tuple[float, ...]
    movement_fractions: Tuple[float, ...]
    movement_directions: Tuple[int, ...]
    visit_order: Tuple[int, ...]

    @property
    def mean_absolute_movement(self) -> float:
        if not self.signed_movements:
            return 0.0
        return sum(abs(value) for value in self.signed_movements) / len(self.signed_movements)

    @property
    def maximum_absolute_movement(self) -> float:
        return max((abs(value) for value in self.signed_movements), default=0.0)


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer; got {value!r}")
    return int(value)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number; got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be a finite number; got {value!r}")
    return resolved


def resolve_chaos_bump_sampling_config(
    *,
    enabled: bool,
    plastic_enabled: bool,
    initial_lockout_steps: int,
    maximum_bumps: int,
    interlude_min_steps: int,
    interlude_max_steps: int,
    duration_min_steps: int,
    duration_max_steps: int,
    duration_max_fraction_of_elapsed_steps: float,
    max_movement_fraction_of_local_gap: float,
) -> ResolvedChaosBumpSamplingConfig:
    if not isinstance(enabled, bool):
        raise ValueError(f"chaos_bump__sampling__enabled must be bool; got {enabled!r}")
    if enabled and not bool(plastic_enabled):
        raise ValueError("chaos_bump__sampling__enabled requires plastic__enabled=true")
    lockout = _integer(
        initial_lockout_steps,
        label="chaos_bump__sampling__initial_lockout__steps",
        minimum=0,
    )
    bump_limit = _integer(
        maximum_bumps,
        label="chaos_bump__sampling__maximum_bumps",
        minimum=1,
    )
    interlude_min = _integer(
        interlude_min_steps,
        label="chaos_bump__sampling__interlude__min_steps",
        minimum=0,
    )
    interlude_max = _integer(
        interlude_max_steps,
        label="chaos_bump__sampling__interlude__max_steps",
        minimum=0,
    )
    if interlude_min > interlude_max:
        raise ValueError(
            "chaos_bump__sampling__interlude__min_steps must not exceed "
            "chaos_bump__sampling__interlude__max_steps"
        )
    duration_min = _integer(
        duration_min_steps,
        label="chaos_bump__sampling__duration__min_steps",
        minimum=1,
    )
    duration_max = _integer(
        duration_max_steps,
        label="chaos_bump__sampling__duration__max_steps",
        minimum=1,
    )
    if duration_min > duration_max:
        raise ValueError(
            "chaos_bump__sampling__duration__min_steps must not exceed "
            "chaos_bump__sampling__duration__max_steps"
        )
    duration_fraction = _finite_float(
        duration_max_fraction_of_elapsed_steps,
        label="chaos_bump__sampling__duration__max_fraction_of_elapsed_steps",
    )
    if duration_fraction <= 0.0:
        raise ValueError(
            "chaos_bump__sampling__duration__max_fraction_of_elapsed_steps "
            "must be positive"
        )
    movement_fraction = _finite_float(
        max_movement_fraction_of_local_gap,
        label="chaos_bump__sampling__max_movement_fraction_of_local_gap",
    )
    if not 0.0 < movement_fraction <= 1.0:
        raise ValueError(
            "chaos_bump__sampling__max_movement_fraction_of_local_gap "
            "must lie in (0, 1]"
        )
    return ResolvedChaosBumpSamplingConfig(
        enabled=enabled,
        initial_lockout_steps=lockout,
        maximum_bumps=bump_limit,
        interlude_min_steps=interlude_min,
        interlude_max_steps=interlude_max,
        duration_min_steps=duration_min,
        duration_max_steps=duration_max,
        duration_max_fraction_of_elapsed_steps=duration_fraction,
        max_movement_fraction_of_local_gap=movement_fraction,
    )


def chaos_bump_sampling_event_seed(
    model_seed: int,
    bump_number: int,
    purpose: str,
) -> int:
    payload = (
        f"{CHAOS_BUMP_SAMPLING_VERSION}:{int(model_seed)}:"
        f"{int(bump_number)}:{purpose}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _generator(model_seed: int, bump_number: int, purpose: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(
        chaos_bump_sampling_event_seed(model_seed, bump_number, purpose)
    )


def _inclusive_random_integer(
    low: int,
    high: int,
    *,
    generator: torch.Generator,
) -> int:
    if low > high:
        raise ValueError(f"invalid inclusive integer range [{low}, {high}]")
    if low == high:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def chaos_bump_sampling_duration_steps(
    config: ResolvedChaosBumpSamplingConfig,
    *,
    start_update: int,
    model_seed: int,
    bump_number: int,
) -> int:
    if start_update < 1:
        raise ValueError("start_update must be positive")
    duration_high = min(
        config.duration_max_steps,
        max(
            config.duration_min_steps,
            math.floor(
                config.duration_max_fraction_of_elapsed_steps * start_update
            ),
        ),
    )
    return _inclusive_random_integer(
        config.duration_min_steps,
        duration_high,
        generator=_generator(model_seed, bump_number, "duration"),
    )


def chaos_bump_sampling_interlude_steps(
    config: ResolvedChaosBumpSamplingConfig,
    *,
    model_seed: int,
    completed_bump_number: int,
) -> int:
    return _inclusive_random_integer(
        config.interlude_min_steps,
        config.interlude_max_steps,
        generator=_generator(model_seed, completed_bump_number, "interlude"),
    )


def rattle_sampling_coordinates(
    coordinates: Sequence[float] | Tensor,
    *,
    maximum_fraction_of_local_gap: float,
    model_seed: int,
    bump_number: int,
    lower_bound: float = 1.0,
    upper_bound: float = 100.0,
    minimum_separation: float = 1.0e-6,
) -> SamplingCoordinateRattle:
    if isinstance(coordinates, Tensor):
        base = coordinates.detach().to(device="cpu", dtype=torch.float64).clone()
    else:
        base = torch.tensor(tuple(float(value) for value in coordinates), dtype=torch.float64)
    if base.ndim != 1 or base.numel() < 1:
        raise ValueError("sampling coordinates must be a non-empty one-dimensional sequence")
    if not bool(torch.isfinite(base).all().item()):
        raise ValueError("sampling coordinates must be finite")
    if base.numel() > 1 and not bool(torch.all(base[1:] > base[:-1]).item()):
        raise ValueError("sampling coordinates must be strictly increasing")
    fraction_limit = _finite_float(
        maximum_fraction_of_local_gap,
        label="maximum_fraction_of_local_gap",
    )
    if not 0.0 < fraction_limit <= 1.0:
        raise ValueError("maximum_fraction_of_local_gap must lie in (0, 1]")
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound) or lower_bound >= upper_bound:
        raise ValueError("sampling-coordinate bounds must be finite and increasing")
    if not math.isfinite(minimum_separation) or minimum_separation <= 0.0:
        raise ValueError("minimum_separation must be finite and positive")
    tolerance = max(minimum_separation, 32.0 * torch.finfo(torch.float64).eps)
    if float(base[0].item()) < lower_bound - tolerance or float(base[-1].item()) > upper_bound + tolerance:
        raise ValueError("sampling coordinates lie outside the public depth bounds")

    rattled = base.clone()
    signed_movements = torch.zeros_like(base)
    movement_fractions = torch.zeros_like(base)
    movement_directions = torch.zeros(base.numel(), dtype=torch.long)
    anchored = torch.zeros(base.numel(), dtype=torch.bool)
    anchored[0] = math.isclose(float(base[0].item()), lower_bound, rel_tol=0.0, abs_tol=tolerance)
    anchored[-1] = bool(anchored[-1].item()) or math.isclose(
        float(base[-1].item()),
        upper_bound,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    movable = torch.nonzero(~anchored, as_tuple=False).flatten()
    generator = _generator(model_seed, bump_number, "coordinates")
    if movable.numel() == 0:
        visit_order: Tuple[int, ...] = ()
    else:
        permutation = torch.randperm(int(movable.numel()), generator=generator)
        visit_order = tuple(int(value) for value in movable.index_select(0, permutation).tolist())

    for index in visit_order:
        current = float(rattled[index].item())
        left = lower_bound if index == 0 else float(rattled[index - 1].item())
        right = upper_bound if index == rattled.numel() - 1 else float(rattled[index + 1].item())
        usable_left = current - left - minimum_separation
        usable_right = right - current - minimum_separation
        feasible_directions = []
        if usable_left > 0.0:
            feasible_directions.append(-1)
        if usable_right > 0.0:
            feasible_directions.append(1)
        if not feasible_directions:
            continue
        direction_index = _inclusive_random_integer(
            0,
            len(feasible_directions) - 1,
            generator=generator,
        )
        direction = feasible_directions[direction_index]
        drawn_fraction = float(torch.rand((), generator=generator, dtype=torch.float64).item()) * fraction_limit
        available = usable_left if direction < 0 else usable_right
        movement = float(direction) * drawn_fraction * available
        rattled[index] += movement
        signed_movements[index] = movement
        movement_fractions[index] = drawn_fraction
        movement_directions[index] = direction

    if float(rattled[0].item()) < lower_bound - tolerance or float(rattled[-1].item()) > upper_bound + tolerance:
        raise RuntimeError("sampling-coordinate rattle escaped the public depth bounds")
    if rattled.numel() > 1 and not bool(torch.all(rattled[1:] - rattled[:-1] >= minimum_separation).item()):
        raise RuntimeError("sampling-coordinate rattle violated minimum separation")
    return SamplingCoordinateRattle(
        coordinates=tuple(float(value) for value in rattled.tolist()),
        signed_movements=tuple(float(value) for value in signed_movements.tolist()),
        movement_fractions=tuple(float(value) for value in movement_fractions.tolist()),
        movement_directions=tuple(int(value) for value in movement_directions.tolist()),
        visit_order=visit_order,
    )


__all__ = [
    "CHAOS_BUMP_SAMPLING_CONFIG_FIELDS",
    "CHAOS_BUMP_SAMPLING_VERSION",
    "ResolvedChaosBumpSamplingConfig",
    "SamplingCoordinateRattle",
    "chaos_bump_sampling_duration_steps",
    "chaos_bump_sampling_event_seed",
    "chaos_bump_sampling_interlude_steps",
    "rattle_sampling_coordinates",
    "resolve_chaos_bump_sampling_config",
]
# ^^^ THOG
