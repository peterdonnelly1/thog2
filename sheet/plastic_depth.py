# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PLASTIC_DEPTH_VERSION = "plastic_depth_v0_1"
PLASTIC_LAYER_SAMPLING_INITIALISATIONS = ("equidistant", "random")
PLASTIC_LAYER_COUNT_OBJECTIVES = (
    "lowest_loss",
    "layer_efficiency",
    "relative_training_wall_time",
    "memory_budget",
)


@dataclass(frozen=True)
class ResolvedPlasticDepthCounts:
    maximum_layers: int
    initial_active_layers: int
    fixed_active_layers: Optional[int]


def resolve_plastic_depth_counts(
    *,
    n_layer: int,
    enabled: bool,
    layers_to_sample: Optional[int],
    do_learn_layer_count: bool,
    initial_layer_count: Optional[int],
    max_permitted_layers: Optional[int],
) -> ResolvedPlasticDepthCounts:
    if not enabled:
        return ResolvedPlasticDepthCounts(
            maximum_layers=n_layer,
            initial_active_layers=n_layer,
            fixed_active_layers=n_layer,
        )
    if do_learn_layer_count:
        if layers_to_sample is not None:
            raise ValueError(
                "plastic__layers_to_sample may not be supplied when "
                "plastic__do_learn_layer_count is true"
            )
        if max_permitted_layers is None:
            raise ValueError(
                "plastic__max_permitted_layers is required when "
                "plastic__do_learn_layer_count is true"
            )
        initial = n_layer if initial_layer_count is None else initial_layer_count
        maximum = max_permitted_layers
        if initial < 1 or initial > maximum:
            raise ValueError(
                "plastic__initial_layer_count must lie in "
                "[1, plastic__max_permitted_layers]; "
                f"got initial={initial}, maximum={maximum}"
            )
        return ResolvedPlasticDepthCounts(
            maximum_layers=maximum,
            initial_active_layers=initial,
            fixed_active_layers=None,
        )
    if initial_layer_count is not None or max_permitted_layers is not None:
        raise ValueError(
            "plastic__initial_layer_count and plastic__max_permitted_layers "
            "require plastic__do_learn_layer_count=true"
        )
    fixed = n_layer if layers_to_sample is None else layers_to_sample
    if fixed < 1:
        raise ValueError(
            "plastic__layers_to_sample must be a positive integer; "
            f"got {fixed!r}"
        )
    return ResolvedPlasticDepthCounts(
        maximum_layers=fixed,
        initial_active_layers=fixed,
        fixed_active_layers=fixed,
    )


def validate_plastic_layer_count_objective(value: str) -> str:
    if value not in PLASTIC_LAYER_COUNT_OBJECTIVES:
        raise ValueError(
            "plastic__layer_count_objective must be one of "
            f"{PLASTIC_LAYER_COUNT_OBJECTIVES}; got {value!r}"
        )
    return value


def validate_plastic_sampling_initialisation(value: str) -> str:
    if value not in PLASTIC_LAYER_SAMPLING_INITIALISATIONS:
        raise ValueError(
            "plastic__layer_sampling_initialisation must be one of "
            f"{PLASTIC_LAYER_SAMPLING_INITIALISATIONS}; got {value!r}"
        )
    return value


def public_to_internal_depth(public_coordinates: Tensor) -> Tensor:
    return 2.0 * (public_coordinates - 1.0) / 99.0 - 1.0


def internal_to_public_depth(internal_coordinates: Tensor) -> Tensor:
    return 1.0 + 99.0 * (internal_coordinates + 1.0) / 2.0


def evenly_distributed_active_ranks(maximum_layers: int, active_layers: int) -> Tuple[int, ...]:
    if isinstance(maximum_layers, bool) or not isinstance(maximum_layers, int) or maximum_layers <= 0:
        raise ValueError(f"maximum_layers must be a positive integer; got {maximum_layers!r}")
    if isinstance(active_layers, bool) or not isinstance(active_layers, int) or active_layers <= 0:
        raise ValueError(f"active_layers must be a positive integer; got {active_layers!r}")
    if active_layers > maximum_layers:
        raise ValueError(
            "active_layers must not exceed maximum_layers; "
            f"got active_layers={active_layers}, maximum_layers={maximum_layers}"
        )
    if active_layers == 1:
        return (0,)
    ranks = torch.linspace(0, maximum_layers - 1, active_layers, dtype=torch.float64)
    rounded = torch.round(ranks).to(dtype=torch.long).tolist()
    resolved = tuple(int(value) for value in rounded)
    if len(set(resolved)) != active_layers:
        raise RuntimeError(
            "evenly distributed active ranks were not unique; "
            f"maximum_layers={maximum_layers}, active_layers={active_layers}, ranks={resolved}"
        )
    if resolved[0] != 0 or resolved[-1] != maximum_layers - 1:
        raise RuntimeError(f"active ranks did not preserve endpoints: {resolved}")
    return resolved


class PlasticDepthSamplingLattice(nn.Module):
    """Persistent monotone sampling geometry and discrete active-count state."""

    def __init__(
        self,
        maximum_layers: int,
        *,
        initial_active_layers: int,
        initialisation: str,
        seed: int,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if isinstance(maximum_layers, bool) or not isinstance(maximum_layers, int) or maximum_layers <= 0:
            raise ValueError(f"maximum_layers must be a positive integer; got {maximum_layers!r}")
        if initial_active_layers < 1 or initial_active_layers > maximum_layers:
            raise ValueError(
                "initial_active_layers must lie in [1, maximum_layers]; "
                f"got initial_active_layers={initial_active_layers}, maximum_layers={maximum_layers}"
            )
        validate_plastic_sampling_initialisation(initialisation)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be finite and positive; got {epsilon!r}")
        self.maximum_layers = maximum_layers
        self.initialisation = initialisation
        self.epsilon = float(epsilon)

        if maximum_layers == 1:
            raw_intervals = torch.empty(0, dtype=torch.float32)
        elif initialisation == "equidistant":
            raw_intervals = torch.zeros(maximum_layers - 1, dtype=torch.float32)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            positive_intervals = torch.rand(maximum_layers - 1, generator=generator, dtype=torch.float64) + 0.25
            raw_intervals = self._inverse_softplus(positive_intervals).to(dtype=torch.float32)
        self.raw_intervals = nn.Parameter(raw_intervals)
        self.register_buffer(
            "initial_public_coordinates",
            self.public_coordinates().detach().to(dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "active_layer_count",
            torch.tensor(initial_active_layers, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer("last_count_decision_update", torch.tensor(-1, dtype=torch.long), persistent=True)
        self.register_buffer("count_decision_number", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("reference_training_time", torch.tensor(float("nan"), dtype=torch.float64), persistent=True)
        self.register_buffer(
            "training_time_ema",
            torch.full((maximum_layers + 1,), float("nan"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "training_time_observations",
            torch.zeros(maximum_layers + 1, dtype=torch.long),
            persistent=True,
        )
        # vvv THOG retain the count-independent optimiser-step component needed by relative training-wall-time probes
        self.register_buffer("optimizer_step_time_ema", torch.tensor(float("nan"), dtype=torch.float64), persistent=True)
        self.register_buffer("optimizer_step_time_observations", torch.tensor(0, dtype=torch.long), persistent=True)
        # ^^^ THOG
        # vvv THOG persist the latest representative training-memory probe for every candidate count
        self.register_buffer(
            "peak_allocated_gib",
            torch.full((maximum_layers + 1,), float("nan"), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "peak_reserved_gib",
            torch.full((maximum_layers + 1,), float("nan"), dtype=torch.float64),
            persistent=True,
        )
        # ^^^ THOG

    @staticmethod
    def _inverse_softplus(values: Tensor) -> Tensor:
        return values + torch.log(-torch.expm1(-values))

    def positive_intervals(self) -> Tensor:
        if self.maximum_layers == 1:
            return self.raw_intervals
        return F.softplus(self.raw_intervals) + self.epsilon

    def public_coordinates(self) -> Tensor:
        if self.maximum_layers == 1:
            return torch.ones(1, dtype=self.raw_intervals.dtype, device=self.raw_intervals.device)
        intervals = self.positive_intervals()
        cumulative = torch.cumsum(intervals, dim=0)
        interior = 1.0 + 99.0 * cumulative[:-1] / intervals.sum()
        return torch.cat(
            (
                interior.new_tensor([1.0]),
                interior,
                interior.new_tensor([100.0]),
            ),
            dim=0,
        )

    def active_ranks(self, active_layers: Optional[int] = None) -> Tuple[int, ...]:
        resolved_count = int(self.active_layer_count.item()) if active_layers is None else int(active_layers)
        return evenly_distributed_active_ranks(self.maximum_layers, resolved_count)

    def active_public_coordinates(self, active_layers: Optional[int] = None) -> Tensor:
        ranks = self.active_ranks(active_layers)
        index = torch.tensor(ranks, dtype=torch.long, device=self.raw_intervals.device)
        return self.public_coordinates().index_select(0, index)

    def set_active_layer_count(self, active_layers: int) -> None:
        evenly_distributed_active_ranks(self.maximum_layers, active_layers)
        self.active_layer_count.fill_(active_layers)

    def interval_report(self) -> Dict[str, object]:
        coordinates = self.public_coordinates()
        if coordinates.numel() <= 1:
            intervals = coordinates.new_empty(0)
        else:
            intervals = coordinates[1:] - coordinates[:-1]
        movement = coordinates.to(dtype=torch.float64) - self.initial_public_coordinates.to(coordinates.device)
        return {
            "maximum_layers": self.maximum_layers,
            "active_layers": int(self.active_layer_count.item()),
            "public_coordinates": tuple(float(value) for value in coordinates.detach().cpu().tolist()),
            "active_ranks": self.active_ranks(),
            "active_public_coordinates": tuple(
                float(value) for value in self.active_public_coordinates().detach().cpu().tolist()
            ),
            "minimum_interval": float(intervals.min().item()) if intervals.numel() else None,
            "maximum_interval": float(intervals.max().item()) if intervals.numel() else None,
            "mean_absolute_movement": float(movement.abs().mean().item()),
            "initialisation": self.initialisation,
            "version": PLASTIC_DEPTH_VERSION,
        }

    def record_training_time(
        self,
        active_layers: int,
        elapsed_seconds: float,
        *,
        smoothing: float = 0.9,
        update_reference: bool = False,
    ) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            return
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing!r}")
        previous = float(self.training_time_ema[active_layers].item())
        current = elapsed_seconds if not math.isfinite(previous) else smoothing * previous + (1.0 - smoothing) * elapsed_seconds
        self.training_time_ema[active_layers] = current
        self.training_time_observations[active_layers] += 1
        if update_reference:
            self.reference_training_time.fill_(current)

    def record_optimizer_step_time(self, elapsed_seconds: float, *, smoothing: float = 0.9) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            return
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing!r}")
        previous = float(self.optimizer_step_time_ema.item())
        current = elapsed_seconds if not math.isfinite(previous) else smoothing * previous + (1.0 - smoothing) * elapsed_seconds
        self.optimizer_step_time_ema.fill_(current)
        self.optimizer_step_time_observations.add_(1)

    def record_memory_probe(
        self,
        active_layers: int,
        *,
        peak_allocated_gib: float,
        peak_reserved_gib: float,
    ) -> None:
        if not math.isfinite(peak_allocated_gib) or peak_allocated_gib < 0.0:
            raise ValueError(f"peak_allocated_gib must be finite and non-negative; got {peak_allocated_gib!r}")
        if not math.isfinite(peak_reserved_gib) or peak_reserved_gib < 0.0:
            raise ValueError(f"peak_reserved_gib must be finite and non-negative; got {peak_reserved_gib!r}")
        self.peak_allocated_gib[active_layers] = peak_allocated_gib
        self.peak_reserved_gib[active_layers] = peak_reserved_gib


@dataclass(frozen=True)
class PlasticDepthCandidateMeasurement:
    active_layers: int
    validation_loss: float
    training_time: Optional[float] = None
    peak_allocated_gib: Optional[float] = None
    peak_reserved_gib: Optional[float] = None


def plastic_depth_candidate_score(
    measurement: PlasticDepthCandidateMeasurement,
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    reference_training_time: Optional[float],
    memory_budget_gib: Optional[float],
) -> Tuple[bool, float]:
    validate_plastic_layer_count_objective(objective)
    if not math.isfinite(measurement.validation_loss):
        return False, float("inf")
    if objective == "lowest_loss":
        return True, measurement.validation_loss
    if objective == "layer_efficiency":
        return True, measurement.validation_loss + cost_weight * measurement.active_layers / maximum_layers
    if objective == "relative_training_wall_time":
        if measurement.training_time is None or reference_training_time is None:
            return False, float("inf")
        if not math.isfinite(measurement.training_time) or not math.isfinite(reference_training_time) or reference_training_time <= 0.0:
            return False, float("inf")
        return True, measurement.validation_loss + cost_weight * measurement.training_time / reference_training_time
    if measurement.peak_allocated_gib is None or memory_budget_gib is None:
        return False, float("inf")
    feasible = measurement.peak_allocated_gib <= memory_budget_gib
    return feasible, measurement.validation_loss if feasible else float("inf")


def choose_plastic_depth_candidate(
    measurements: Iterable[PlasticDepthCandidateMeasurement],
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    reference_training_time: Optional[float],
    memory_budget_gib: Optional[float],
) -> Tuple[PlasticDepthCandidateMeasurement, Tuple[Dict[str, object], ...]]:
    scored = []
    for measurement in measurements:
        feasible, score = plastic_depth_candidate_score(
            measurement,
            objective=objective,
            maximum_layers=maximum_layers,
            cost_weight=cost_weight,
            reference_training_time=reference_training_time,
            memory_budget_gib=memory_budget_gib,
        )
        scored.append((measurement, feasible, score))
    feasible_scored = [item for item in scored if item[1]]
    if not feasible_scored:
        raise RuntimeError("no feasible PLASTIC DEPTH layer-count candidate")
    selected = min(feasible_scored, key=lambda item: (item[2], item[0].active_layers))[0]
    report = tuple(
        {
            "active_layers": item.active_layers,
            "validation_loss": item.validation_loss,
            "training_time": item.training_time,
            "peak_allocated_gib": item.peak_allocated_gib,
            "peak_reserved_gib": item.peak_reserved_gib,
            "feasible": feasible,
            "score": score,
        }
        for item, feasible, score in scored
    )
    return selected, report


__all__ = [
    "PLASTIC_DEPTH_VERSION",
    "PLASTIC_LAYER_COUNT_OBJECTIVES",
    "PLASTIC_LAYER_SAMPLING_INITIALISATIONS",
    "PlasticDepthCandidateMeasurement",
    "ResolvedPlasticDepthCounts",
    "PlasticDepthSamplingLattice",
    "choose_plastic_depth_candidate",
    "evenly_distributed_active_ranks",
    "internal_to_public_depth",
    "plastic_depth_candidate_score",
    "public_to_internal_depth",
    "resolve_plastic_depth_counts",
    "validate_plastic_layer_count_objective",
    "validate_plastic_sampling_initialisation",
]
# ^^^ THOG
