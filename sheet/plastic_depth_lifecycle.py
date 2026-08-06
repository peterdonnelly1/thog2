# vvv THOG
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, TextIO, Tuple

from .distributed import DistributedContext
from .plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    ResolvedPlasticCoarseConfig,
    ScoredPlasticCoarseTrial,
    coarse_results_payload,
    render_plastic_coarse_report,
    score_plastic_coarse_trials,
)
from .plastic_depth_coarse_checkpoint import (
    PlasticCoarseTrialCheckpointState,
    build_plastic_coarse_trial_checkpoint_state,
)
from .plastic_depth_coarse_runner import (
    PlasticCoarseTrialProgress,
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
    run_distributed_plastic_coarse_review_pause,
    run_plastic_coarse_review_pause,
)


FreshStateBuilder = Callable[..., PlasticFreshTrainingState]
TrialRunner = Callable[..., PlasticCoarseTrialResult]
StateDestroyer = Callable[[PlasticFreshTrainingState], None]
PauseRunner = Callable[..., PlasticCoarsePauseResult]
PauseCheckpointCallback = Callable[[Any, Mapping[str, object]], None]
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


def _build_fine_state(
    *,
    trainer_factory: Callable[[Any, Any, Any], Any],
    resolved_config: Any,
    train_tokens: Any,
    validation_tokens: Any,
    selected_layers: int,
    fresh_state_builder: FreshStateBuilder,
) -> PlasticFreshTrainingState:
    fine_config = replace(
        resolved_config,
        plastic__coarse_phase="disabled",
        plastic__runtime_phase="fine",
        plastic__initial_layer_count=int(selected_layers),
    )
    return fresh_state_builder(
        trainer_factory=trainer_factory,
        resolved_config=fine_config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        phase="fine",
        active_layer_count=int(selected_layers),
        instrumentation_namespace="fine",
    )


def _validate_resume_checkpoint(
    *,
    resume: PlasticCoarseTrialCheckpointState,
    coarse_config: ResolvedPlasticCoarseConfig,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    memory_budget_gib: Optional[float],
    geometry_initialisation: str,
    fine_max_updates: int,
    resume_state: Optional[PlasticFreshTrainingState],
) -> None:
    expected = {
        "candidate_layers": tuple(coarse_config.candidate_layers),
        "n_steps": int(coarse_config.n_steps or 0),
        "evaluation_steps_count": int(coarse_config.evaluation_steps_count or 0),
        "objective": str(objective),
        "maximum_layers": int(maximum_layers),
        "cost_weight": float(cost_weight),
        "memory_budget_gib": (
            None if memory_budget_gib is None else float(memory_budget_gib)
        ),
        "geometry_initialisation": str(geometry_initialisation),
        "fine_max_updates": int(fine_max_updates),
    }
    actual = {
        "candidate_layers": tuple(resume.candidate_layers),
        "n_steps": resume.n_steps,
        "evaluation_steps_count": resume.evaluation_steps_count,
        "objective": resume.objective,
        "maximum_layers": resume.maximum_layers,
        "cost_weight": resume.cost_weight,
        "memory_budget_gib": resume.memory_budget_gib,
        "geometry_initialisation": resume.geometry_initialisation,
        "fine_max_updates": resume.fine_max_updates,
    }
    if actual != expected:
        raise ValueError(
            "PLASTIC mid-COARSE checkpoint controls differ from the requested lifecycle; "
            f"checkpoint={actual!r}, requested={expected!r}"
        )
    if resume_state is None:
        raise ValueError("PLASTIC mid-COARSE resume requires the restored training state")
    if resume_state.phase != "coarse":
        raise ValueError("PLASTIC mid-COARSE restored state must have phase='coarse'")
    if int(resume_state.active_layer_count) != resume.current_trial_layers:
        raise ValueError(
            "PLASTIC mid-COARSE restored layer count differs from checkpoint state"
        )
    completed_updates = int(resume_state.trainer.state.completed_updates)
    if completed_updates != resume.completed_steps:
        raise ValueError(
            "PLASTIC mid-COARSE restored trainer step differs from checkpoint state; "
            f"trainer={completed_updates}, checkpoint={resume.completed_steps}"
        )


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
    checkpoint_callback: Optional[PauseCheckpointCallback] = None,
    coarse_checkpoint_interval: int = 0,
    resume_checkpoint_state: Optional[Mapping[str, Any]] = None,
    resume_state: Optional[PlasticFreshTrainingState] = None,
    distributed_coordinator: Optional[Any] = None,
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
    if coarse_checkpoint_interval < 0:
        raise ValueError("coarse_checkpoint_interval must be non-negative")
    if coarse_checkpoint_interval > 0 and checkpoint_callback is None:
        raise ValueError(
            "positive coarse_checkpoint_interval requires checkpoint_callback"
        )
    resume = (
        None
        if resume_checkpoint_state is None
        else PlasticCoarseTrialCheckpointState.from_mapping(resume_checkpoint_state)
    )
    if resume is None and resume_state is not None:
        raise ValueError("resume_state was supplied without a mid-COARSE checkpoint")
    if resume is not None:
        _validate_resume_checkpoint(
            resume=resume,
            coarse_config=coarse_config,
            objective=objective,
            maximum_layers=maximum_layers,
            cost_weight=cost_weight,
            memory_budget_gib=memory_budget_gib,
            geometry_initialisation=geometry_initialisation,
            fine_max_updates=int(resolved_config.max_updates),
            resume_state=resume_state,
        )

    coordinator = (
        distributed_coordinator
        if distributed_coordinator is not None
        else coordinator_factory(str(resolved_config.device))
    )
    trial_results = list(resume.completed_trial_results if resume is not None else ())
    try:
        trial_count = len(coarse_config.candidate_layers)
        if coordinator.is_primary:
            _emit(console_stream, "COARSE TRIALS")
            _emit(
                console_stream,
                "  layer counts: " + ", ".join(str(value) for value in coarse_config.candidate_layers),
            )
        for trial_index, active_layers in enumerate(
            coarse_config.candidate_layers,
            start=1,
        ):
            if resume is not None and trial_index < resume.current_trial_index:
                continue
            trial_config = coarse_trial_training_config(
                resolved_config,
                active_layer_count=active_layers,
                n_steps=coarse_config.n_steps,
            )
            is_resumed_trial = (
                resume is not None and trial_index == resume.current_trial_index
            )
            state = resume_state if is_resumed_trial else None
            if state is None:
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

                def checkpoint_trial_progress(
                    progress: PlasticCoarseTrialProgress,
                    trainer: Any,
                ) -> None:
                    if checkpoint_callback is None:
                        raise RuntimeError(
                            "PLASTIC COARSE periodic checkpoint callback is unavailable"
                        )
                    checkpoint_state = build_plastic_coarse_trial_checkpoint_state(
                        candidate_layers=coarse_config.candidate_layers,
                        n_steps=coarse_config.n_steps,
                        evaluation_steps_count=coarse_config.evaluation_steps_count,
                        objective=objective,
                        maximum_layers=maximum_layers,
                        cost_weight=cost_weight,
                        memory_budget_gib=memory_budget_gib,
                        geometry_initialisation=geometry_initialisation,
                        fine_max_updates=int(resolved_config.max_updates),
                        current_trial_index=progress.trial_index,
                        current_trial_layers=progress.layers,
                        completed_steps=progress.completed_steps,
                        training_losses=progress.training_losses,
                        training_elapsed_seconds=progress.training_elapsed_seconds,
                        completed_trial_results=tuple(trial_results),
                    )
                    payload = dict(checkpoint_state.structured())
                    trainer.plastic_coarse_fine_state = payload
                    checkpoint_callback(trainer, payload)

                runner_kwargs = {
                    "trial_index": trial_index,
                    "n_steps": coarse_config.n_steps,
                    "evaluation_steps_count": coarse_config.evaluation_steps_count,
                    "progress_sink": (
                        (lambda line: _emit(console_stream, line))
                        if coordinator.is_primary
                        else None
                    ),
                }
                if is_resumed_trial:
                    assert resume is not None
                    runner_kwargs.update(
                        {
                            "prior_training_losses": resume.training_losses,
                            "prior_training_elapsed_seconds": resume.training_elapsed_seconds,
                        }
                    )
                if coarse_checkpoint_interval > 0:
                    runner_kwargs.update(
                        {
                            "checkpoint_interval": coarse_checkpoint_interval,
                            "checkpoint_callback": checkpoint_trial_progress,
                        }
                    )
                result = trial_runner(state, **runner_kwargs)
                trial_results.append(result)
            finally:
                state_destroyer(state)
            if is_resumed_trial:
                resume_state = None
                resume = None

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
                    row.result.status,
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

        if bool(getattr(resolved_config, "plastic__coarse_phase_roll_through", False)):
            pause_result = PlasticCoarsePauseResult(
                disposition="roll_through",
                elapsed_seconds=0.0,
                remaining_seconds=0.0,
            )
            if coordinator.is_primary:
                _emit(console_stream, "COARSE roll-through enabled; starting FINE immediately.")
            coordinator.barrier()
        else:
            pause_result = run_distributed_plastic_coarse_review_pause(
                coordinator,
                duration_seconds=pause_duration_seconds,
                output=console_stream,
                pause_runner=pause_runner,
            )
        payload = dict(coarse_results_payload(scored_trials, winner))
        payload.update(
            {
                "phase": (
                    "review_pause"
                    if pause_result.disposition == "checkpoint_exit"
                    else "fine_start"
                ),
                "candidate_layers": list(coarse_config.candidate_layers),
                "pause": {
                    "disposition": pause_result.disposition,
                    "elapsed_seconds": pause_result.elapsed_seconds,
                    "remaining_seconds": pause_result.remaining_seconds,
                },
            }
        )

        if pause_result.disposition == "checkpoint_exit":
            if checkpoint_callback is None:
                raise RuntimeError(
                    "PLASTIC COARSE Ctrl-G requires a collective checkpoint callback"
                )
            checkpoint_state = _build_fine_state(
                trainer_factory=trainer_factory,
                resolved_config=resolved_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                selected_layers=winner.result.layers,
                fresh_state_builder=fresh_state_builder,
            )
            try:
                checkpoint_state.trainer.plastic_coarse_provenance = payload
                checkpoint_state.trainer.plastic_coarse_fine_state = payload
                checkpoint_callback(checkpoint_state.trainer, payload)
            finally:
                state_destroyer(checkpoint_state)
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

        fine_state = _build_fine_state(
            trainer_factory=trainer_factory,
            resolved_config=resolved_config,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            selected_layers=winner.result.layers,
            fresh_state_builder=fresh_state_builder,
        )
        fine_state.trainer.plastic_coarse_provenance = payload
        fine_state.trainer.plastic_coarse_fine_state = payload
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
# ^^^ THOG