# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PLASTIC_DEPTH_VERSION = "plastic_depth_v0_3"
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


# vvv THOG new persisted identities use the exact canonical public control spellings; the sampling seed remains internal

def plastic_depth_identity_metadata(
    *,
    coarse_phase: str = "disabled",
    phase_1_n_steps: Optional[int] = None,
    phase_1_starting_layer_count: Optional[int] = None,
    phase_1_number_of_trials: Optional[int] = None,
    phase_1_evaluation_steps_count: Optional[int] = None,
    layer_count_probe_interval: Optional[int] = None,
    layer_count_probe_radius: int = 1,
    layer_count_max_step: int = 1,
    layers_to_sample: Optional[int],
    do_learn_layer_count: bool,
    initial_layer_count: Optional[int],
    max_permitted_layers: Optional[int],
    layer_sampling_initialisation: str,
    layer_count_objective: str,
    layer_count_update_brake: int,
    layer_count_probe_noise_window: int,
    layer_count_probe_noise_min_observations: int,
    layer_count_probe_noise_lambda: float,
    layer_count_cost_weight: float,
    layer_memory_budget_gib: Optional[float],
    cuda_allocator_reserve_gib: float,
    geometry_learning_rate_multiplier: float,
    freeze_geometry_during_warmup: bool,
    initial_active_layers: int,
) -> Dict[str, object]:
    return {
        "version": PLASTIC_DEPTH_VERSION,
        "plastic__enabled": True,
        "plastic__coarse_phase": coarse_phase,
        "plastic__phase_1_n_steps": phase_1_n_steps,
        "plastic__phase_1_starting_layer_count": phase_1_starting_layer_count,
        "plastic__phase_1__number_of_trials": phase_1_number_of_trials,
        "plastic__phase_1_evaluation_steps_count": phase_1_evaluation_steps_count,
        "plastic__layer_count_probe_interval": layer_count_probe_interval,
        "plastic__layer_count_probe_radius": int(layer_count_probe_radius),
        "plastic__layer_count_max_step": int(layer_count_max_step),
        "plastic__layers_to_sample": layers_to_sample,
        "plastic__do_learn_layer_count": bool(do_learn_layer_count),
        "plastic__initial_layer_count": initial_layer_count,
        "plastic__max_permitted_layers": max_permitted_layers,
        "plastic__layer_sampling_initialisation": layer_sampling_initialisation,
        "plastic__layer_count_objective": layer_count_objective,
        "plastic__layer_count_update_brake": int(layer_count_update_brake),
        "plastic__layer_count_probe_noise_window": int(layer_count_probe_noise_window),
        "plastic__layer_count_probe_noise_min_observations": int(layer_count_probe_noise_min_observations),
        "plastic__layer_count_probe_noise_lambda": float(layer_count_probe_noise_lambda),
        "plastic__layer_count_cost_weight": float(layer_count_cost_weight),
        "plastic__layer_memory_budget_gib": layer_memory_budget_gib,
        "plastic__cuda_allocator_reserve_gib": float(cuda_allocator_reserve_gib),
        "plastic__geometry_learning_rate_multiplier": float(geometry_learning_rate_multiplier),
        "plastic__freeze_geometry_during_warmup": bool(freeze_geometry_during_warmup),
        "plastic__initial_active_layers": int(initial_active_layers),
    }


# ^^^ THOG


# vvv THOG v0.3 discrete geometry transition prepared before any model state mutates
@dataclass(frozen=True)
class PlasticDepthGeometryTransition:
    previous_active_layers: int
    new_active_layers: int
    old_from_new_scale: float
    old_from_new_shift: float
    expected_raw_intervals: Tensor
    proposed_raw_intervals: Tensor


# ^^^ THOG
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
        learn_layer_count: bool = True,
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
        if not isinstance(learn_layer_count, bool):
            raise ValueError(f"learn_layer_count must be bool; got {learn_layer_count!r}")
        self.maximum_layers = maximum_layers
        self.initialisation = initialisation
        self.learn_layer_count = learn_layer_count
        self.epsilon = float(epsilon)

        # vvv THOG v0.3 preallocates capacity but initialises only the active-prefix geometry
        raw_intervals = torch.zeros(max(0, maximum_layers - 1), dtype=torch.float32)
        active_interval_count = max(0, initial_active_layers - 1)
        if initialisation == "random" and active_interval_count > 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            positive_intervals = (
                torch.rand(active_interval_count, generator=generator, dtype=torch.float64)
                + 0.25
            )
            raw_intervals[:active_interval_count] = self._inverse_softplus(
                positive_intervals
            ).to(dtype=torch.float32)
        # vvv THOG learned-count mode owns one explicit dormant probe gap after the active prefix
        if learn_layer_count and initial_active_layers < maximum_layers:
            probe_index = initial_active_layers - 1
            if probe_index > 0:
                raw_intervals[probe_index].copy_(raw_intervals[probe_index - 1])
        # ^^^ THOG
        self.raw_intervals = nn.Parameter(raw_intervals)
        self.register_buffer(
            "initial_public_coordinates",
            self.public_coordinates(
                initial_active_layers,
                include_probe=learn_layer_count,
            ).detach().to(dtype=torch.float64),
            persistent=True,
        )
        # ^^^ THOG
        # vvv THOG fixed-count mode owns geometry only; discrete controller, timing and memory state exist only when count learning is enabled
        if self.learn_layer_count:
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
            self.register_buffer("optimizer_step_time_ema", torch.tensor(float("nan"), dtype=torch.float64), persistent=True)
            self.register_buffer("optimizer_step_time_observations", torch.tensor(0, dtype=torch.long), persistent=True)
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

    # vvv THOG v0.3 constructs coordinates from only the active prefix; inactive capacity is mathematically inert
    def _positive_prefix(
        self,
        interval_count: int,
        *,
        raw_intervals: Optional[Tensor] = None,
    ) -> Tensor:
        source = self.raw_intervals if raw_intervals is None else raw_intervals
        if interval_count < 0 or interval_count > source.numel():
            raise ValueError(
                "interval_count must lie within allocated capacity; "
                f"got interval_count={interval_count}, capacity={source.numel()}"
            )
        if interval_count == 0:
            return source[:0]
        return F.softplus(source[:interval_count]) + self.epsilon

    def positive_intervals(self, active_layers: Optional[int] = None) -> Tensor:
        resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
        if resolved_count < 1 or resolved_count > self.maximum_layers:
            raise ValueError(
                "active_layers must lie in [1, maximum_layers]; "
                f"got active_layers={resolved_count}, maximum_layers={self.maximum_layers}"
            )
        return self._positive_prefix(max(0, resolved_count - 1))

    def _public_coordinates_from_raw(
        self,
        active_layers: int,
        *,
        include_probe: bool,
        raw_intervals: Tensor,
    ) -> Tensor:
        resolved_count = int(active_layers)
        if resolved_count < 1 or resolved_count > self.maximum_layers:
            raise ValueError(
                "active_layers must lie in [1, maximum_layers]; "
                f"got active_layers={resolved_count}, maximum_layers={self.maximum_layers}"
            )
        resolved_include_probe = bool(include_probe) and resolved_count < self.maximum_layers
        if resolved_count == 1:
            if resolved_include_probe:
                return raw_intervals.new_tensor([1.0, 100.0])
            return raw_intervals.new_tensor([50.5])

        active_intervals = self._positive_prefix(
            resolved_count - 1,
            raw_intervals=raw_intervals,
        )
        cumulative = torch.cat(
            (active_intervals.new_zeros(1), torch.cumsum(active_intervals, dim=0))
        )
        if resolved_include_probe:
            probe_gap = self._positive_prefix(
                resolved_count,
                raw_intervals=raw_intervals,
            )[-1]
            span = active_intervals.sum() + probe_gap
            active = 1.0 + 99.0 * cumulative / span
            return torch.cat((active, active.new_tensor([100.0])))

        span = active_intervals.sum()
        return 1.0 + 99.0 * cumulative / span

    def public_coordinates(
        self,
        active_layers: Optional[int] = None,
        *,
        include_probe: Optional[bool] = None,
    ) -> Tensor:
        resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
        resolved_include_probe = self.learn_layer_count if include_probe is None else bool(include_probe)
        return self._public_coordinates_from_raw(
            resolved_count,
            include_probe=resolved_include_probe,
            raw_intervals=self.raw_intervals,
        )

    def _chart_span(self, active_layers: int, *, raw_intervals: Tensor) -> Tensor:
        interval_count = (
            active_layers
            if self.learn_layer_count and active_layers < self.maximum_layers
            else max(0, active_layers - 1)
        )
        if interval_count == 0:
            return raw_intervals.new_tensor(1.0)
        return self._positive_prefix(
            interval_count,
            raw_intervals=raw_intervals,
        ).sum()

    def prepare_count_transition(
        self,
        new_active_layers: int,
    ) -> PlasticDepthGeometryTransition:
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no layer-count controller transition")
        previous_count = self.current_active_layers
        resolved_count = int(new_active_layers)
        if resolved_count < 1 or resolved_count > self.maximum_layers:
            raise ValueError(
                "new_active_layers must lie in [1, maximum_layers]; "
                f"got new_active_layers={resolved_count}, maximum_layers={self.maximum_layers}"
            )
        if abs(resolved_count - previous_count) > 1:
            raise ValueError(
                "PLASTIC DEPTH count transitions are limited to one layer; "
                f"got previous={previous_count}, new={resolved_count}"
            )

        expected = self.raw_intervals.detach().clone()
        proposed = expected.clone()
        if resolved_count == previous_count + 1 and resolved_count < self.maximum_layers:
            new_probe_index = resolved_count - 1
            previous_probe_index = new_probe_index - 1
            proposed[new_probe_index].copy_(proposed[previous_probe_index])

        old_span = self._chart_span(previous_count, raw_intervals=expected)
        new_span = self._chart_span(resolved_count, raw_intervals=proposed)
        scale = float((new_span / old_span).item())
        shift = scale - 1.0
        return PlasticDepthGeometryTransition(
            previous_active_layers=previous_count,
            new_active_layers=resolved_count,
            old_from_new_scale=scale,
            old_from_new_shift=shift,
            expected_raw_intervals=expected,
            proposed_raw_intervals=proposed,
        )

    def commit_count_transition(
        self,
        transition: PlasticDepthGeometryTransition,
    ) -> None:
        if self.current_active_layers != transition.previous_active_layers:
            raise RuntimeError(
                "PLASTIC DEPTH geometry changed after transition preparation; "
                f"expected count {transition.previous_active_layers}, "
                f"found {self.current_active_layers}"
            )
        if not torch.equal(
            self.raw_intervals.detach(),
            transition.expected_raw_intervals.to(self.raw_intervals.device),
        ):
            raise RuntimeError("PLASTIC DEPTH geometry changed after transition preparation")
        with torch.no_grad():
            self.raw_intervals.copy_(
                transition.proposed_raw_intervals.to(self.raw_intervals)
            )
            self.active_layer_count.fill_(transition.new_active_layers)
    # ^^^ THOG

    @property
    def current_active_layers(self) -> int:
        if not self.learn_layer_count:
            return self.maximum_layers
        return int(self.active_layer_count.item())

    # vvv THOG v0.3 executes contiguous active-prefix slots rather than ranks spread across maximum capacity
    def active_ranks(self, active_layers: Optional[int] = None) -> Tuple[int, ...]:
        resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
        if resolved_count < 1 or resolved_count > self.maximum_layers:
            raise ValueError(
                "active_layers must lie in [1, maximum_layers]; "
                f"got active_layers={resolved_count}, maximum_layers={self.maximum_layers}"
            )
        return tuple(range(resolved_count))

    def active_public_coordinates(self, active_layers: Optional[int] = None) -> Tensor:
        resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
        if resolved_count == self.current_active_layers:
            return self.public_coordinates()[:resolved_count]
        return self.public_coordinates(resolved_count, include_probe=False)

    def probe_public_coordinate(self) -> Tensor:
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no N+1 probe coordinate")
        if self.current_active_layers >= self.maximum_layers:
            raise RuntimeError("PLASTIC DEPTH is already at maximum active layers")
        return self.public_coordinates()[-1:]

    def set_active_layer_count(self, active_layers: int) -> None:
        # vvv THOG low-level controller/probe compatibility path; model commits use atomic re-gauge instead
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no layer-count controller")
        resolved_count = int(active_layers)
        if resolved_count < 1 or resolved_count > self.maximum_layers:
            raise ValueError(
                "active_layers must lie in [1, maximum_layers]; "
                f"got active_layers={resolved_count}, maximum_layers={self.maximum_layers}"
            )
        previous_count = self.current_active_layers
        if resolved_count > previous_count:
            with torch.no_grad():
                for count in range(previous_count + 1, resolved_count + 1):
                    if count < self.maximum_layers:
                        new_probe_index = count - 1
                        previous_probe_index = new_probe_index - 1
                        self.raw_intervals[new_probe_index].copy_(
                            self.raw_intervals[previous_probe_index]
                        )
        self.active_layer_count.fill_(resolved_count)
        # ^^^ THOG
    # ^^^ THOG

    def interval_report(self) -> Dict[str, object]:
        coordinates = self.public_coordinates()
        active_coordinates = coordinates[: self.current_active_layers]
        if active_coordinates.numel() <= 1:
            intervals = active_coordinates.new_empty(0)
        else:
            intervals = active_coordinates[1:] - active_coordinates[:-1]
        initial = self.initial_public_coordinates.to(coordinates.device)
        comparable = min(int(initial.numel()), int(coordinates.numel()))
        movement = (
            coordinates[:comparable].to(dtype=torch.float64) - initial[:comparable]
        )
        return {
            "maximum_layers": self.maximum_layers,
            "active_layers": self.current_active_layers,
            "public_coordinates": tuple(float(value) for value in coordinates.detach().cpu().tolist()),
            "active_ranks": self.active_ranks(),
            "active_public_coordinates": tuple(
                float(value) for value in active_coordinates.detach().cpu().tolist()
            ),
            "probe_public_coordinate": (
                float(coordinates[-1].detach().cpu().item())
                if self.learn_layer_count and coordinates.numel() > self.current_active_layers
                else None
            ),
            "minimum_interval": float(intervals.min().item()) if intervals.numel() else None,
            "maximum_interval": float(intervals.max().item()) if intervals.numel() else None,
            "mean_absolute_movement": float(movement.abs().mean().item()) if movement.numel() else 0.0,
            "initialisation": self.initialisation,
            "learn_layer_count": self.learn_layer_count,
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
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no training-time controller state")
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
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no optimiser-time controller state")
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
        if not self.learn_layer_count:
            raise RuntimeError("fixed-count PLASTIC DEPTH has no memory-probe controller state")
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
    "plastic_depth_identity_metadata",
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
