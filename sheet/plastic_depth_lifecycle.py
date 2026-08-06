from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO, Tuple

from .distributed import DistributedContext
from .plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    ResolvedPlasticCoarseConfig,
    ScoredPlasticCoarseTrial,
    coarse_results_payload,
    render_plastic_coarse_report,
    score_plastic_coarse_trials,
)
from .plastic_depth_coarse_runner import (
    coarse_trial_training_config,
    render_plastic_coarse_trial_header,
    run_fixed_plastic_coarse_trial,
)
from .plastic_depth_fresh_state import (
    PlasticFreshTrainingState,
    build_fresh_training_state,
    destroy_fresh_training_state,
)
from .plastic_depth_pause import (
    PLASTIC_COARSE_REVIEW_PAUSE_SECONDS,
    PlasticCoarsePauseResult,
    run_plastic_coarse_review_pause,
)


FreshStateBuilder = Callable[..., PlasticFreshTrainingState]
TrialRunner = Callable[..., PlasticCoarseTrialResult]
StateDestroyer = Callable[[PlasticFreshTrainingState], None]
PauseRunner = Callable[..., PlasticCoarsePauseResult]
CoordinatorFactory = Callable[..., DistributedContext]


@dataclass
class PlasticCoarseFineLifecycleOutcome:
    fine_state: Optional[PlasticFreshTrainingState]
    trial_results: Tuple[PlasticCoarseTrialResult, ...]
    scored_trials: Tuple[ScoredPlasticCoarseTrial, ...]
    selected_layers: int
    pause_result: PlasticCoarsePauseResult
    report: str
    provenance: Mapping[str, object]
    distributed_coordinator: Any

    def close_coordinator(self) -> None:
        coordinator = self.distributed_coordinator
        if coordinator is not None:
            coordinator.close()
            self.distributed_coordinator = None


def _emit(stream: TextIO, text: str) -> None:
    stream.write(text)
    if not text.endswith("\n"):
        stream.write("\n")
    stream.flush()


def _distributed_pause(
    coordinator: Any,
    *,
    pause_runner: PauseRunner,
    pause_duration_seconds: float,
    console_stream: TextIO,
    checkpoint_callback: Optional[Callable[[], None]],
) -> PlasticCoarsePauseResult:
    local_result: Optional[PlasticCoarsePauseResult] = None
    if coordinator.is_primary:
        local_result = pause_runner(
            duration_seconds=pause_duration_seconds,
            output=console_stream,
            checkpoint_callback=checkpoint_callback,
        )
    gathered = coordinator.all_gather_object(local_result)
    result = gathered[0]
    if not isinstance(result, PlasticCoarsePauseResult):
        raise RuntimeError("rank 0 did not provide a PLASTIC COARSE pause disposition")
    coordinator.barrier()
    return result


def run_plastic_coarse_fine_lifecycle(
    *,
    trainer_factory: Callable[[Any, Any, Any], Any],
    resolved_config: Any,
    train_tokens: Any,
    validation_tokens: Any,
    coarse_config: ResolvedPlasticCoarseConfig,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    memory_budget_gib: Optional[float],
    geometry_initialisation: str,
    pause_duration_seconds: float = PLASTIC_COARSE_REVIEW_PAUSE_SECONDS,
    console_stream: TextIO = sys.stdout,
    checkpoint_callback: Optional[Callable[[], None]] = None,
    fresh_state_builder: FreshStateBuilder = build_fresh_training_state,
    trial_runner: TrialRunner = run_fixed_plastic_coarse_trial,
    state_destroyer: StateDestroyer = destroy_fresh_training_state,
    pause_runner: PauseRunner = run_plastic_coarse_review_pause,
    coordinator_factory: CoordinatorFactory = DistributedContext.from_environment,
) -> PlasticCoarseFineLifecycleOutcome:
    if not coarse_config.enabled:
        raise ValueError("COARSE/FINE lifecycle requires enabled COARSE configuration")
    if coarse_config.n_steps is None or coarse_config.evaluation_steps_count is None:
        raise ValueError("resolved COARSE configuration is incomplete")

    coordinator = coordinator_factory(str(resolved_config.device))
    trial_results = []
    try:
        trial_count = len(coarse_config.candidate_layers)
        for trial_index, active_layers in enumerate(
            coarse_config.candidate_layers,
            start=1,
        ):
            trial_config = coarse_trial_training_config(
                resolved_config,
                active_layer_count=active_layers,
                n_steps=coarse_config.n_steps,
            )
            state = fresh_state_builder(
                trainer_factory=trainer_factory,
                resolved_config=trial_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                phase="coarse",
                active_layer_count=active_layers,
                instrumentation_namespace=f"coarse/trial_{trial_index}",
            )
            try:
                if coordinator.is_primary:
                    _emit(
                        console_stream,
                        render_plastic_coarse_trial_header(
                            trial_index=trial_index,
                            trial_count=trial_count,
                            layers=active_layers,
                            n_steps=coarse_config.n_steps,
                            evaluation_steps_count=coarse_config.evaluation_steps_count,
                            objective=objective,
                            geometry_initialisation=geometry_initialisation,
                        ),
                    )
                result = trial_runner(
                    state,
                    trial_index=trial_index,
                    n_steps=coarse_config.n_steps,
                    evaluation_steps_count=coarse_config.evaluation_steps_count,
                    progress_sink=(
                        (lambda line: _emit(console_stream, line))
                        if coordinator.is_primary
                        else None
                    ),
                )
                trial_results.append(result)
            finally:
                state_destroyer(state)

        scored_trials, winner = score_plastic_coarse_trials(
            tuple(trial_results),
            objective=objective,
            maximum_layers=maximum_layers,
            cost_weight=cost_weight,
            memory_budget_gib=memory_budget_gib,
        )
        selection_identity = {
            "selected_layers": winner.result.layers,
            "scores": tuple(
                (
                    row.result.trial_index,
                    row.result.layers,
                    row.status if hasattr(row, "status") else row.result.status,
                    row.score,
                    row.selectable,
                )
                for row in scored_trials
            ),
        }
        coordinator.assert_identical_object(
            selection_identity,
            "PLASTIC COARSE score table and winner",
        )
        report = render_plastic_coarse_report(
            scored_trials,
            winner,
            training_steps=coarse_config.n_steps,
            evaluation_steps_count=coarse_config.evaluation_steps_count,
            ansi=bool(getattr(console_stream, "isatty", lambda: False)()),
        )
        if coordinator.is_primary:
            _emit(console_stream, report)

        pause_result = _distributed_pause(
            coordinator,
            pause_runner=pause_runner,
            pause_duration_seconds=pause_duration_seconds,
            console_stream=console_stream,
            checkpoint_callback=checkpoint_callback,
        )
        payload = dict(coarse_results_payload(scored_trials, winner))
        payload.update(
            {
                "phase": "coarse_complete",
                "candidate_layers": list(coarse_config.candidate_layers),
                "pause": {
                    "disposition": pause_result.disposition,
                    "elapsed_seconds": pause_result.elapsed_seconds,
                    "remaining_seconds": pause_result.remaining_seconds,
                },
            }
        )
        if pause_result.disposition == "checkpoint_exit":
            return PlasticCoarseFineLifecycleOutcome(
                fine_state=None,
                trial_results=tuple(trial_results),
                scored_trials=scored_trials,
                selected_layers=winner.result.layers,
                pause_result=pause_result,
                report=report,
                provenance=payload,
                distributed_coordinator=coordinator,
            )

        fine_config = replace(
            resolved_config,
            plastic__coarse_phase="disabled",
            plastic__runtime_phase="fine",
            plastic__initial_layer_count=winner.result.layers,
        )
        fine_state = fresh_state_builder(
            trainer_factory=trainer_factory,
            resolved_config=fine_config,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            phase="fine",
            active_layer_count=winner.result.layers,
            instrumentation_namespace="fine",
        )
        fine_state.trainer.plastic_coarse_provenance = payload
        return PlasticCoarseFineLifecycleOutcome(
            fine_state=fine_state,
            trial_results=tuple(trial_results),
            scored_trials=scored_trials,
            selected_layers=winner.result.layers,
            pause_result=pause_result,
            report=report,
            provenance=payload,
            distributed_coordinator=coordinator,
        )
    except BaseException:
        coordinator.close()
        raise
