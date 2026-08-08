from __future__ import annotations

from pathlib import Path


TARGET = Path("run_thog2_lifecycle.py")


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {description} replacement target; found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from sheet.plastic_depth_coarse import resolve_plastic_coarse_config\n"
        "from sheet.plastic_depth_fresh_state import build_fresh_training_state\n",
        "from sheet.distributed import DistributedContext\n"
        "from sheet.plastic_depth_coarse import (\n"
        "    ResolvedPlasticCoarseConfig,\n"
        "    resolve_plastic_coarse_config,\n"
        ")\n"
        "from sheet.plastic_depth_coarse_checkpoint import (\n"
        "    PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE,\n"
        "    PlasticCoarseTrialCheckpointState,\n"
        ")\n"
        "from sheet.plastic_depth_fresh_state import (\n"
        "    PlasticFreshTrainingState,\n"
        "    build_fresh_training_state,\n"
        ")\n",
        "PLASTIC COARSE/FINE import block",
    )

    helper_anchor = "_WANDB_BACKENDS = {\"wandb\", \"both\"}\n"
    helper = '''# vvv THOG route persisted mid-COARSE state before ordinary FINE-boundary resume handling
def _plastic_coarse_trial_checkpoint_from_payload(
    payload: Mapping[str, Any],
) -> Optional[PlasticCoarseTrialCheckpointState]:
    raw_state = payload.get("plastic_coarse_fine_state")
    if not isinstance(raw_state, Mapping):
        return None
    if raw_state.get("phase") != PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE:
        return None
    return PlasticCoarseTrialCheckpointState.from_mapping(raw_state)
# ^^^ THOG


'''
    text = replace_once(
        text,
        helper_anchor,
        helper + helper_anchor,
        "mid-COARSE payload helper anchor",
    )

    text = replace_once(
        text,
        '    completed_updates = int(payload.get("completed_updates", 0))\n',
        '''    checkpoint_completed_updates = int(payload.get("completed_updates", 0))
    plastic_coarse_trial_checkpoint = _plastic_coarse_trial_checkpoint_from_payload(payload)                                      # <<< THOG distinguish COARSE trial progress from FINE optimizer progress
    if plastic_coarse_trial_checkpoint is not None and mode == "fork":
        raise ValueError("fork is not permitted from a mid-COARSE-trial checkpoint")
    completed_updates = (                                                                                                         # <<< THOG FINE starts at optimizer step zero after resumed COARSE selection
        0
        if plastic_coarse_trial_checkpoint is not None
        else checkpoint_completed_updates
    )
''',
        "checkpoint completed-update assignment",
    )

    text = replace_once(
        text,
        "    fresh_state = None\n"
        "    coarse_fine_outcome = None\n"
        "    if context[\"mode\"] == \"fresh\":\n",
        '''    fresh_state = None
    coarse_fine_outcome = None

    # vvv THOG one callback persists both periodic COARSE progress and the fresh FINE review-pause boundary
    def checkpoint_plastic_coarse_fine(
        coarse_trainer: Any,
        state: Mapping[str, object],
    ) -> None:
        paused_lifecycle = {**lifecycle, "plastic_coarse_fine": dict(state)}
        coarse_trainer.lifecycle_metadata = paused_lifecycle
        coarse_trainer.plastic_coarse_fine_state = dict(state)
        coarse_trainer.save_checkpoint(checkpoint_path)
        if coarse_trainer.distributed.is_primary:
            write_run_manifest(paths["manifest_path"], paused_lifecycle)
    # ^^^ THOG

    if context["mode"] == "fresh":
''',
        "production lifecycle callback insertion",
    )

    old_callback = '''            def checkpoint_review_pause(coarse_trainer: Any, state: Mapping[str, object]) -> None:
                paused_lifecycle = {**lifecycle, "plastic_coarse_fine": dict(state)}
                coarse_trainer.lifecycle_metadata = paused_lifecycle
                coarse_trainer.plastic_coarse_fine_state = dict(state)
                coarse_trainer.save_checkpoint(checkpoint_path)
                if coarse_trainer.distributed.is_primary:
                    write_run_manifest(paths["manifest_path"], paused_lifecycle)
'''
    text = replace_once(
        text,
        old_callback,
        "",
        "old review-pause-only callback",
    )
    text = replace_once(
        text,
        "                checkpoint_callback=checkpoint_review_pause,\n"
        "            )\n",
        "                checkpoint_callback=checkpoint_plastic_coarse_fine,\n"
        "                coarse_checkpoint_interval=int(training_config.checkpoint_interval),\n"
        "            )\n",
        "fresh COARSE lifecycle checkpoint arguments",
    )

    old_resume = '''    else:
        resolved: ResolvedCheckpoint = context["resolved"]
        trainer = core.OwtTrainer.from_checkpoint(
            resolved.checkpoint_path,
            train_tokens,
            validation_tokens,
            expected_config=training_config,
            overrides=_resume_overrides(training_config),
        )
        resume_boundary = resume_plastic_coarse_fine_boundary(
            trainer,
            checkpoint_path,
        )
        if resume_boundary == PLASTIC_RESUME_CHECKPOINT_EXIT:
            trainer.close()
            return 0
        if resume_boundary == PLASTIC_RESUME_CONTINUE_FINE:
            lifecycle = {
                **lifecycle,
                "plastic_coarse_fine": dict(trainer.plastic_coarse_fine_state),
            }
            if trainer.distributed.is_primary:
                write_run_manifest(paths["manifest_path"], lifecycle)
'''
    new_resume = '''    else:
        resolved: ResolvedCheckpoint = context["resolved"]
        checkpoint_payload = context["checkpoint_payload"]
        plastic_coarse_resume = _plastic_coarse_trial_checkpoint_from_payload(                                                   # <<< THOG route mid-trial state before ordinary resume
            checkpoint_payload
        )
        coarse_resume_coordinator = (
            DistributedContext.from_environment(str(training_config.device))
            if plastic_coarse_resume is not None
            else None
        )
        try:
            trainer = core.OwtTrainer.from_checkpoint(
                resolved.checkpoint_path,
                train_tokens,
                validation_tokens,
                expected_config=training_config,
                overrides=_resume_overrides(training_config),
            )
        except BaseException:
            if coarse_resume_coordinator is not None:
                coarse_resume_coordinator.close()
            raise

        if plastic_coarse_resume is not None:
            # vvv THOG complete the interrupted COARSE trial and remaining candidates before constructing fresh FINE state
            resumed_coarse_state = PlasticFreshTrainingState(
                trainer=trainer,
                phase="coarse",
                active_layer_count=plastic_coarse_resume.current_trial_layers,
                instrumentation_namespace=(
                    f"coarse/trial_{plastic_coarse_resume.current_trial_index}"
                ),
                fingerprint={},
            )
            resumed_coarse_config = ResolvedPlasticCoarseConfig(
                enabled=True,
                candidate_layers=plastic_coarse_resume.candidate_layers,
                n_steps=plastic_coarse_resume.n_steps,
                evaluation_steps_count=(
                    plastic_coarse_resume.evaluation_steps_count
                ),
            )
            coarse_fine_outcome = run_plastic_coarse_fine_lifecycle(
                trainer_factory=core.OwtTrainer,
                resolved_config=training_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                coarse_config=resumed_coarse_config,
                objective=plastic_coarse_resume.objective,
                maximum_layers=plastic_coarse_resume.maximum_layers,
                cost_weight=plastic_coarse_resume.cost_weight,
                memory_budget_gib=plastic_coarse_resume.memory_budget_gib,
                geometry_initialisation=(
                    plastic_coarse_resume.geometry_initialisation
                ),
                checkpoint_callback=checkpoint_plastic_coarse_fine,
                coarse_checkpoint_interval=int(training_config.checkpoint_interval),
                resume_checkpoint_state=plastic_coarse_resume.structured(),
                resume_state=resumed_coarse_state,
                distributed_coordinator=coarse_resume_coordinator,
            )
            if coarse_fine_outcome.fine_state is None:
                coarse_fine_outcome.close_coordinator()
                return 0
            fresh_state = coarse_fine_outcome.fine_state
            trainer = fresh_state.trainer
            training_config = trainer.config
            lifecycle = {
                **lifecycle,
                "plastic_coarse_fine": dict(coarse_fine_outcome.provenance),
            }
            if trainer.distributed.is_primary:
                write_run_manifest(paths["manifest_path"], lifecycle)
            # ^^^ THOG
        else:
            resume_boundary = resume_plastic_coarse_fine_boundary(
                trainer,
                checkpoint_path,
            )
            if resume_boundary == PLASTIC_RESUME_CHECKPOINT_EXIT:
                trainer.close()
                return 0
            if resume_boundary == PLASTIC_RESUME_CONTINUE_FINE:
                lifecycle = {
                    **lifecycle,
                    "plastic_coarse_fine": dict(trainer.plastic_coarse_fine_state),
                }
                if trainer.distributed.is_primary:
                    write_run_manifest(paths["manifest_path"], lifecycle)
'''
    text = replace_once(
        text,
        old_resume,
        new_resume,
        "ordinary resume block",
    )

    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
