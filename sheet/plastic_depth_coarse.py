from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


PLASTIC_COARSE_PHASES = ("enabled", "disabled")
PLASTIC_COARSE_OBJECTIVE_HEADINGS = {
    "lowest_loss": "loss_score",
    "layer_efficiency": "efficiency_score",
    "relative_training_wall_time": "wall_time_score",
    "memory_budget": "budget_loss_score",
}
_WINNER_STYLE_START = "\033[1;92m"
_STYLE_END = "\033[0m"


@dataclass(frozen=True)
class ResolvedPlasticCoarseConfig:
    enabled: bool
    candidate_layers: Tuple[int, ...]
    n_steps: Optional[int]
    evaluation_steps_count: Optional[int]


@dataclass(frozen=True)
class PlasticCoarseTrialResult:
    trial_index: int
    layers: int
    status: str
    validation_losses: Tuple[float, ...] = ()
    training_elapsed_seconds: Optional[float] = None
    training_steps: int = 0
    tokens_per_update: int = 0
    peak_allocated_gib: Optional[float] = None
    peak_reserved_gib: Optional[float] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trial_index < 1:
            raise ValueError("trial_index must be positive")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.status not in {"success", "failed"}:
            raise ValueError("status must be success or failed")
        if self.status == "success":
            if not self.validation_losses:
                raise ValueError("successful COARSE trials require validation losses")
            if not all(math.isfinite(float(value)) for value in self.validation_losses):
                raise ValueError("COARSE validation losses must be finite")
            if self.training_elapsed_seconds is None or not math.isfinite(self.training_elapsed_seconds) or self.training_elapsed_seconds <= 0.0:
                raise ValueError("successful COARSE trials require positive finite training elapsed time")
            if self.training_steps < 1:
                raise ValueError("successful COARSE trials require positive training_steps")
            if self.tokens_per_update < 1:
                raise ValueError("successful COARSE trials require positive tokens_per_update")

    @property
    def mean_validation_loss(self) -> Optional[float]:
        if self.status != "success":
            return None
        return float(statistics.fmean(self.validation_losses))

    @property
    def validation_loss_std(self) -> Optional[float]:
        if self.status != "success":
            return None
        return float(statistics.pstdev(self.validation_losses))

    @property
    def seconds_per_step(self) -> Optional[float]:
        if self.training_elapsed_seconds is None or self.training_steps < 1:
            return None
        return float(self.training_elapsed_seconds) / float(self.training_steps)

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.training_elapsed_seconds is None or self.training_elapsed_seconds <= 0.0:
            return None
        return float(self.training_steps * self.tokens_per_update) / float(self.training_elapsed_seconds)

    def structured(self) -> Dict[str, object]:
        values = asdict(self)
        values.update(
            {
                "mean_validation_loss": self.mean_validation_loss,
                "validation_loss_std": self.validation_loss_std,
                "seconds_per_step": self.seconds_per_step,
                "tokens_per_second": self.tokens_per_second,
            }
        )
        return values


@dataclass(frozen=True)
class ScoredPlasticCoarseTrial:
    result: PlasticCoarseTrialResult
    objective: str
    objective_heading: str
    score: Optional[float]
    selectable: bool
    within_budget: Optional[bool]
    reference_training_elapsed_seconds: Optional[float]

    def structured(self) -> Dict[str, object]:
        return {
            **self.result.structured(),
            "objective": self.objective,
            "objective_heading": self.objective_heading,
            "score": self.score,
            "selectable": self.selectable,
            "within_budget": self.within_budget,
            "reference_training_elapsed_seconds": self.reference_training_elapsed_seconds,
        }


def _optional_positive_integer(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None; got {value!r}")
    return value


def validate_plastic_coarse_phase(value: str) -> str:
    if value not in PLASTIC_COARSE_PHASES:
        raise ValueError(
            "plastic__coarse_phase must be one of "
            f"{PLASTIC_COARSE_PHASES}; got {value!r}"
        )
    return value


def resolve_plastic_coarse_config(
    *,
    coarse_phase: str,
    plastic_enabled: bool,
    do_learn_layer_count: bool,
    n_steps: Optional[int],
    starting_layer_count: Optional[int],
    number_of_trials: Optional[int],
    evaluation_steps_count: Optional[int],
    max_permitted_layers: Optional[int],
) -> ResolvedPlasticCoarseConfig:
    phase = validate_plastic_coarse_phase(coarse_phase)
    resolved_n_steps = _optional_positive_integer(
        "plastic__phase_1_n_steps",
        n_steps,
    )
    resolved_start = _optional_positive_integer(
        "plastic__phase_1_starting_layer_count",
        starting_layer_count,
    )
    resolved_trials = _optional_positive_integer(
        "plastic__phase_1__number_of_trials",
        number_of_trials,
    )
    resolved_evaluation_steps = _optional_positive_integer(
        "plastic__phase_1_evaluation_steps_count",
        evaluation_steps_count,
    )

    if phase == "disabled":
        return ResolvedPlasticCoarseConfig(
            enabled=False,
            candidate_layers=(),
            n_steps=resolved_n_steps,
            evaluation_steps_count=resolved_evaluation_steps,
        )

    if not plastic_enabled:
        raise ValueError("plastic__coarse_phase=enabled requires plastic__enabled=true")
    if not do_learn_layer_count:
        raise ValueError(
            "plastic__coarse_phase=enabled requires "
            "plastic__do_learn_layer_count=true"
        )

    missing = [
        name
        for name, value in (
            ("plastic__phase_1_n_steps", resolved_n_steps),
            ("plastic__phase_1_starting_layer_count", resolved_start),
            ("plastic__phase_1__number_of_trials", resolved_trials),
            (
                "plastic__phase_1_evaluation_steps_count",
                resolved_evaluation_steps,
            ),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "plastic__coarse_phase=enabled is missing required controls: "
            + ", ".join(missing)
        )
    if max_permitted_layers is None:
        raise ValueError(
            "plastic__coarse_phase=enabled requires "
            "plastic__max_permitted_layers"
        )

    assert resolved_start is not None
    assert resolved_trials is not None
    candidates = tuple(resolved_start * (2**index) for index in range(resolved_trials))
    outside = tuple(value for value in candidates if value > max_permitted_layers)
    if outside:
        raise ValueError(
            "PLASTIC COARSE candidate layers exceed "
            "plastic__max_permitted_layers: "
            f"candidates={candidates}, maximum={max_permitted_layers}"
        )

    return ResolvedPlasticCoarseConfig(
        enabled=True,
        candidate_layers=candidates,
        n_steps=resolved_n_steps,
        evaluation_steps_count=resolved_evaluation_steps,
    )


def resolve_plastic_probe_interval(
    *,
    probe_interval: Optional[int],
    update_brake: int,
    enabled: bool,
    do_learn_layer_count: bool,
) -> Optional[int]:
    if probe_interval is not None:
        return _optional_positive_integer(
            "plastic__layer_count_probe_interval",
            probe_interval,
        )
    if not enabled or not do_learn_layer_count:
        return None
    if update_brake < 0:
        raise ValueError("plastic__layer_count_update_brake must be non-negative")
    # Version 0.3 used brake=0 to mean every update; preserve that exact path.
    return max(1, update_brake)


def validate_plastic_fine_count_controls(
    *,
    probe_radius: int,
    max_step: int,
) -> None:
    _optional_positive_integer("plastic__layer_count_probe_radius", probe_radius)
    _optional_positive_integer("plastic__layer_count_max_step", max_step)


def score_plastic_coarse_trials(
    results: Sequence[PlasticCoarseTrialResult],
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    memory_budget_gib: Optional[float],
) -> Tuple[Tuple[ScoredPlasticCoarseTrial, ...], ScoredPlasticCoarseTrial]:
    if objective not in PLASTIC_COARSE_OBJECTIVE_HEADINGS:
        raise ValueError(f"unsupported PLASTIC COARSE objective: {objective!r}")
    if maximum_layers < 1:
        raise ValueError("maximum_layers must be positive")
    if not math.isfinite(cost_weight) or cost_weight < 0.0:
        raise ValueError("cost_weight must be finite and non-negative")
    if objective == "memory_budget" and memory_budget_gib is None:
        raise ValueError("memory_budget requires memory_budget_gib")

    successful = tuple(result for result in results if result.status == "success")
    reference_elapsed = (
        successful[0].training_elapsed_seconds
        if objective == "relative_training_wall_time" and successful
        else None
    )
    scored = []
    for result in results:
        mean_loss = result.mean_validation_loss
        within_budget: Optional[bool] = None
        selectable = result.status == "success"
        score: Optional[float] = None
        if objective == "memory_budget":
            within_budget = (
                selectable
                and result.peak_allocated_gib is not None
                and float(result.peak_allocated_gib) <= float(memory_budget_gib)
            )
            selectable = bool(within_budget)
        if selectable and mean_loss is not None:
            if objective == "lowest_loss" or objective == "memory_budget":
                score = mean_loss
            elif objective == "layer_efficiency":
                score = mean_loss + cost_weight * float(result.layers) / float(maximum_layers)
            else:
                assert reference_elapsed is not None and result.training_elapsed_seconds is not None
                score = mean_loss + cost_weight * float(result.training_elapsed_seconds) / float(reference_elapsed)
        scored.append(
            ScoredPlasticCoarseTrial(
                result=result,
                objective=objective,
                objective_heading=PLASTIC_COARSE_OBJECTIVE_HEADINGS[objective],
                score=score,
                selectable=selectable and score is not None and math.isfinite(score),
                within_budget=within_budget,
                reference_training_elapsed_seconds=reference_elapsed,
            )
        )

    selectable_rows = tuple(row for row in scored if row.selectable)
    if not selectable_rows:
        if objective == "memory_budget":
            raise RuntimeError("all PLASTIC COARSE trials failed or were outside the memory budget")
        raise RuntimeError("all PLASTIC COARSE trials failed or were unselectable")
    winner = min(
        selectable_rows,
        key=lambda row: (float(row.score), row.result.layers, row.result.trial_index),
    )
    return tuple(scored), winner


def _format_optional(value: Optional[float], width: int, precision: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return f"{'-':>{width}}"
    return f"{float(value):>{width}.{precision}f}"


def render_plastic_coarse_report(
    scored_trials: Sequence[ScoredPlasticCoarseTrial],
    winner: ScoredPlasticCoarseTrial,
    *,
    training_steps: int,
    evaluation_steps_count: int,
    ansi: bool,
) -> str:
    if not scored_trials:
        raise ValueError("scored_trials must not be empty")
    objective = scored_trials[0].objective
    heading = scored_trials[0].objective_heading
    for row in scored_trials:
        if row.objective != objective or row.objective_heading != heading:
            raise ValueError("all scored trials must use one objective")

    headers = [
        f"PLASTIC COARSE RESULTS",
        f"{len(scored_trials)} trials x {training_steps} training steps",
        f"validation mean over final {evaluation_steps_count} batches",
        f"goal: {objective}",
    ]
    if objective == "relative_training_wall_time":
        reference = winner.reference_training_elapsed_seconds
        headers.append(f"reference training elapsed_s: {float(reference):.6f}")

    columns = (
        f"{'trial':>5} {'layers':>6} {'elapsed_s':>10} {'sec/step':>10} "
        f"{'tok/s':>9} {'mean_val':>10} {'val_std':>9} {'peak_GiB':>9} "
        + (f"{'within_budget':>13} " if objective == "memory_budget" else "")
        + f"{heading:>18} {'status':>9}"
    )
    lines = headers + [columns]
    for row in scored_trials:
        result = row.result
        marker = " <<< WINNER" if row is winner else ""
        within = ""
        if objective == "memory_budget":
            within = f"{('yes' if row.within_budget else 'no'):>13} "
        status = "failed" if result.status == "failed" else ("ok" if row.selectable else "unselectable")
        line = (
            f"{result.trial_index:5d} {result.layers:6d} "
            f"{_format_optional(result.training_elapsed_seconds, 10, 2)} "
            f"{_format_optional(result.seconds_per_step, 10, 5)} "
            f"{_format_optional(result.tokens_per_second, 9, 0)} "
            f"{_format_optional(result.mean_validation_loss, 10, 4)} "
            f"{_format_optional(result.validation_loss_std, 9, 4)} "
            f"{_format_optional(result.peak_allocated_gib, 9, 2)} "
            f"{within}{_format_optional(row.score, 18, 6)} {status:>9}{marker}"
        )
        if row is winner and ansi:
            line = f"{_WINNER_STYLE_START}{line}{_STYLE_END}"
        lines.append(line)
    return "\n".join(lines)


def coarse_results_payload(
    scored_trials: Sequence[ScoredPlasticCoarseTrial],
    winner: ScoredPlasticCoarseTrial,
) -> Mapping[str, object]:
    return {
        "objective": winner.objective,
        "objective_heading": winner.objective_heading,
        "selected_layers": winner.result.layers,
        "reference_training_elapsed_seconds": winner.reference_training_elapsed_seconds,
        "trials": [row.structured() for row in scored_trials],
    }
