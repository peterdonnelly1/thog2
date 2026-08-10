# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .plastic_depth_coarse import PlasticCoarseTrialResult


PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE = "coarse_trial"


def _as_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _as_non_negative_float(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}")
    return result


def _optional_finite_float(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or None; got {value!r}")
    return result


def _trial_result_payload(result: PlasticCoarseTrialResult) -> Mapping[str, object]:
    return {
        "trial_index": result.trial_index,
        "layers": result.layers,
        "status": result.status,
        "validation_losses": list(result.validation_losses),
        "training_losses": list(result.training_losses),
        "training_elapsed_seconds": result.training_elapsed_seconds,
        "training_steps": result.training_steps,
        "tokens_per_update": result.tokens_per_update,
        "peak_allocated_gib": result.peak_allocated_gib,
        "peak_reserved_gib": result.peak_reserved_gib,
        "error_class": result.error_class,
        "error_message": result.error_message,
    }


def _trial_result_from_payload(payload: Mapping[str, Any]) -> PlasticCoarseTrialResult:
    return PlasticCoarseTrialResult(
        trial_index=_as_positive_integer("completed trial index", payload["trial_index"]),
        layers=_as_positive_integer("completed trial layers", payload["layers"]),
        status=str(payload["status"]),
        validation_losses=tuple(float(value) for value in payload.get("validation_losses", ())),
        training_losses=tuple(float(value) for value in payload.get("training_losses", ())),
        training_elapsed_seconds=_optional_finite_float(
            "completed trial training_elapsed_seconds",
            payload.get("training_elapsed_seconds"),
        ),
        training_steps=int(payload.get("training_steps", 0)),
        tokens_per_update=int(payload.get("tokens_per_update", 0)),
        peak_allocated_gib=_optional_finite_float(
            "completed trial peak_allocated_gib",
            payload.get("peak_allocated_gib"),
        ),
        peak_reserved_gib=_optional_finite_float(
            "completed trial peak_reserved_gib",
            payload.get("peak_reserved_gib"),
        ),
        error_class=(
            None if payload.get("error_class") is None else str(payload["error_class"])
        ),
        error_message=(
            None if payload.get("error_message") is None else str(payload["error_message"])
        ),
    )


@dataclass(frozen=True)
class PlasticCoarseTrialCheckpointState:
    candidate_layers: Tuple[int, ...]
    n_steps: int
    evaluation_steps_count: int
    objective: str
    maximum_layers: int
    cost_weight: float
    memory_budget_gib: Optional[float]
    geometry_initialisation: str
    fine_max_updates: int
    current_trial_index: int
    current_trial_layers: int
    completed_steps: int
    training_losses: Tuple[float, ...]
    training_elapsed_seconds: float
    completed_trial_results: Tuple[PlasticCoarseTrialResult, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(
            _as_positive_integer("candidate layer count", value)
            for value in self.candidate_layers
        )
        if not candidates:
            raise ValueError("COARSE checkpoint candidate_layers must not be empty")
        if len(set(candidates)) != len(candidates):
            raise ValueError("COARSE checkpoint candidate_layers must be unique")
        _as_positive_integer("n_steps", self.n_steps)
        _as_positive_integer("evaluation_steps_count", self.evaluation_steps_count)
        _as_positive_integer("maximum_layers", self.maximum_layers)
        _as_positive_integer("fine_max_updates", self.fine_max_updates)
        trial_index = _as_positive_integer("current_trial_index", self.current_trial_index)
        if trial_index > len(candidates):
            raise ValueError("COARSE checkpoint current_trial_index exceeds candidate count")
        if int(self.current_trial_layers) != candidates[trial_index - 1]:
            raise ValueError(
                "COARSE checkpoint current trial layers do not match candidate schedule"
            )
        if isinstance(self.completed_steps, bool) or not isinstance(self.completed_steps, int):
            raise ValueError("COARSE checkpoint completed_steps must be an integer")
        if self.completed_steps < 0 or self.completed_steps >= self.n_steps:
            raise ValueError(
                "COARSE checkpoint completed_steps must lie in [0, n_steps)"
            )
        losses = tuple(float(value) for value in self.training_losses)
        if len(losses) != self.completed_steps:
            raise ValueError(
                "COARSE checkpoint loss history must match completed_steps"
            )
        if not all(math.isfinite(value) for value in losses):
            raise ValueError("COARSE checkpoint loss history contains non-finite values")
        _as_non_negative_float(
            "training_elapsed_seconds",
            self.training_elapsed_seconds,
        )
        if not math.isfinite(float(self.cost_weight)):
            raise ValueError("COARSE checkpoint cost_weight must be finite")
        if self.memory_budget_gib is not None:
            budget = float(self.memory_budget_gib)
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError(
                    "COARSE checkpoint memory_budget_gib must be positive and finite"
                )
        completed = tuple(self.completed_trial_results)
        expected_completed_indices = tuple(range(1, trial_index))
        actual_completed_indices = tuple(result.trial_index for result in completed)
        if actual_completed_indices != expected_completed_indices:
            raise ValueError(
                "COARSE checkpoint completed trial results must be a contiguous prefix"
            )
        for result, expected_layers in zip(completed, candidates):
            if result.layers != expected_layers:
                raise ValueError(
                    "COARSE checkpoint completed trial layers do not match candidate schedule"
                )

    def structured(self) -> Mapping[str, object]:
        return {
            "phase": PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE,
            "candidate_layers": list(self.candidate_layers),
            "n_steps": self.n_steps,
            "evaluation_steps_count": self.evaluation_steps_count,
            "objective": self.objective,
            "maximum_layers": self.maximum_layers,
            "cost_weight": self.cost_weight,
            "memory_budget_gib": self.memory_budget_gib,
            "geometry_initialisation": self.geometry_initialisation,
            "fine_max_updates": self.fine_max_updates,
            "current_trial": {
                "trial_index": self.current_trial_index,
                "layers": self.current_trial_layers,
                "completed_steps": self.completed_steps,
                "training_losses": list(self.training_losses),
                "training_elapsed_seconds": self.training_elapsed_seconds,
            },
            "completed_trial_results": [
                dict(_trial_result_payload(result))
                for result in self.completed_trial_results
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "PlasticCoarseTrialCheckpointState":
        if payload.get("phase") != PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE:
            raise ValueError(
                "checkpoint is not a PLASTIC mid-COARSE-trial checkpoint"
            )
        current = payload.get("current_trial")
        if not isinstance(current, Mapping):
            raise ValueError("COARSE checkpoint current_trial must be a mapping")
        completed_payloads = payload.get("completed_trial_results", ())
        if not isinstance(completed_payloads, Sequence) or isinstance(
            completed_payloads,
            (str, bytes),
        ):
            raise ValueError(
                "COARSE checkpoint completed_trial_results must be a sequence"
            )
        return cls(
            candidate_layers=tuple(int(value) for value in payload["candidate_layers"]),
            n_steps=_as_positive_integer("n_steps", payload["n_steps"]),
            evaluation_steps_count=_as_positive_integer(
                "evaluation_steps_count",
                payload["evaluation_steps_count"],
            ),
            objective=str(payload["objective"]),
            maximum_layers=_as_positive_integer(
                "maximum_layers",
                payload["maximum_layers"],
            ),
            cost_weight=float(payload["cost_weight"]),
            memory_budget_gib=_optional_finite_float(
                "memory_budget_gib",
                payload.get("memory_budget_gib"),
            ),
            geometry_initialisation=str(payload["geometry_initialisation"]),
            fine_max_updates=_as_positive_integer(
                "fine_max_updates",
                payload["fine_max_updates"],
            ),
            current_trial_index=_as_positive_integer(
                "current_trial.trial_index",
                current["trial_index"],
            ),
            current_trial_layers=_as_positive_integer(
                "current_trial.layers",
                current["layers"],
            ),
            completed_steps=int(current["completed_steps"]),
            training_losses=tuple(
                float(value) for value in current.get("training_losses", ())
            ),
            training_elapsed_seconds=_as_non_negative_float(
                "current_trial.training_elapsed_seconds",
                current.get("training_elapsed_seconds", 0.0),
            ),
            completed_trial_results=tuple(
                _trial_result_from_payload(item)
                for item in completed_payloads
                if isinstance(item, Mapping)
            ),
        )


def build_plastic_coarse_trial_checkpoint_state(
    *,
    candidate_layers: Sequence[int],
    n_steps: int,
    evaluation_steps_count: int,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    memory_budget_gib: Optional[float],
    geometry_initialisation: str,
    fine_max_updates: int,
    current_trial_index: int,
    current_trial_layers: int,
    completed_steps: int,
    training_losses: Sequence[float],
    training_elapsed_seconds: float,
    completed_trial_results: Sequence[PlasticCoarseTrialResult],
) -> PlasticCoarseTrialCheckpointState:
    return PlasticCoarseTrialCheckpointState(
        candidate_layers=tuple(int(value) for value in candidate_layers),
        n_steps=int(n_steps),
        evaluation_steps_count=int(evaluation_steps_count),
        objective=str(objective),
        maximum_layers=int(maximum_layers),
        cost_weight=float(cost_weight),
        memory_budget_gib=(
            None if memory_budget_gib is None else float(memory_budget_gib)
        ),
        geometry_initialisation=str(geometry_initialisation),
        fine_max_updates=int(fine_max_updates),
        current_trial_index=int(current_trial_index),
        current_trial_layers=int(current_trial_layers),
        completed_steps=int(completed_steps),
        training_losses=tuple(float(value) for value in training_losses),
        training_elapsed_seconds=float(training_elapsed_seconds),
        completed_trial_results=tuple(completed_trial_results),
    )


__all__ = [
    "PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE",
    "PlasticCoarseTrialCheckpointState",
    "build_plastic_coarse_trial_checkpoint_state",
]
# ^^^ THOG
