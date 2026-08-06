from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one production lifecycle anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    path = "run_thog2_lifecycle.py"
    replace_once(
        path,
        'from sheet.run_manifest import write_run_manifest\n'
        'from sheet.plastic_depth_fresh_state import build_fresh_training_state\n'
        'from sheet.training_config import TrainingConfig\n',
        'from sheet.run_manifest import write_run_manifest\n'
        '# vvv THOG PLASTIC COARSE/FINE one-shot discovery, fresh FINE reconstruction and review-pause resume\n'
        'from sheet.plastic_depth_coarse import resolve_plastic_coarse_config\n'
        'from sheet.plastic_depth_fresh_state import build_fresh_training_state\n'
        'from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle\n'
        'from sheet.plastic_depth_resume import (\n'
        '    PLASTIC_RESUME_CHECKPOINT_EXIT,\n'
        '    PLASTIC_RESUME_CONTINUE_FINE,\n'
        '    resume_plastic_coarse_fine_boundary,\n'
        ')\n'
        '# ^^^ THOG\n'
        'from sheet.training_config import TrainingConfig\n',
    )
    replace_once(
        path,
        '    train_tokens = core.load_tokens(dataset_dir / "train.bin")\n'
        '    validation_tokens = core.load_tokens(dataset_dir / "val.bin")\n'
        '    fresh_state = None\n'
        '    if context["mode"] == "fresh":\n'
        '        # vvv THOG PLASTIC fresh runs and every future COARSE trial share one deterministic step-zero constructor; the disabled path remains byte-for-byte on the established constructor\n'
        '        if training_config.plastic__enabled:\n'
        '            fresh_state = build_fresh_training_state(\n'
        '                trainer_factory=core.OwtTrainer,\n'
        '                resolved_config=training_config,\n'
        '                train_tokens=train_tokens,\n'
        '                validation_tokens=validation_tokens,\n'
        '                phase="fine",\n'
        '                active_layer_count=int(training_config.plastic__initial_active_layers),\n'
        '                instrumentation_namespace="fine",\n'
        '            )\n'
        '            trainer = fresh_state.trainer\n'
        '        else:\n'
        '            trainer = core.OwtTrainer(training_config, train_tokens, validation_tokens)\n'
        '        # ^^^ THOG\n'
        '    else:\n'
        '        resolved: ResolvedCheckpoint = context["resolved"]\n'
        '        trainer = core.OwtTrainer.from_checkpoint(\n'
        '            resolved.checkpoint_path,\n'
        '            train_tokens,\n'
        '            validation_tokens,\n'
        '            expected_config=training_config,\n'
        '            overrides=_resume_overrides(training_config),\n'
        '        )\n'
        '    trainer.lifecycle_metadata = dict(lifecycle)\n',
        '    train_tokens = core.load_tokens(dataset_dir / "train.bin")\n'
        '    validation_tokens = core.load_tokens(dataset_dir / "val.bin")\n'
        '    fresh_state = None\n'
        '    coarse_fine_outcome = None\n'
        '    if context["mode"] == "fresh":\n'
        '        # vvv THOG COARSE is a one-shot pre-FINE lifecycle; disabled PLASTIC remains on the exact established constructor\n'
        '        if training_config.plastic__enabled and training_config.plastic__coarse_phase == "enabled":\n'
        '            coarse_config = resolve_plastic_coarse_config(\n'
        '                coarse_phase=training_config.plastic__coarse_phase,\n'
        '                plastic_enabled=training_config.plastic__enabled,\n'
        '                do_learn_layer_count=training_config.plastic__do_learn_layer_count,\n'
        '                n_steps=training_config.plastic__phase_1_n_steps,\n'
        '                starting_layer_count=training_config.plastic__phase_1_starting_layer_count,\n'
        '                number_of_trials=training_config.plastic__phase_1__number_of_trials,\n'
        '                evaluation_steps_count=training_config.plastic__phase_1_evaluation_steps_count,\n'
        '                max_permitted_layers=training_config.plastic__max_permitted_layers,\n'
        '            )\n'
        '            def checkpoint_review_pause(coarse_trainer: Any, state: Mapping[str, object]) -> None:\n'
        '                paused_lifecycle = {**lifecycle, "plastic_coarse_fine": dict(state)}\n'
        '                coarse_trainer.lifecycle_metadata = paused_lifecycle\n'
        '                coarse_trainer.plastic_coarse_fine_state = dict(state)\n'
        '                coarse_trainer.save_checkpoint(checkpoint_path)\n'
        '                if coarse_trainer.distributed.is_primary:\n'
        '                    write_run_manifest(paths["manifest_path"], paused_lifecycle)\n'
        '            coarse_fine_outcome = run_plastic_coarse_fine_lifecycle(\n'
        '                trainer_factory=core.OwtTrainer,\n'
        '                resolved_config=training_config,\n'
        '                train_tokens=train_tokens,\n'
        '                validation_tokens=validation_tokens,\n'
        '                coarse_config=coarse_config,\n'
        '                objective=training_config.plastic__layer_count_objective,\n'
        '                maximum_layers=int(training_config.plastic__max_permitted_layers),\n'
        '                cost_weight=float(training_config.plastic__layer_count_cost_weight),\n'
        '                memory_budget_gib=training_config.plastic__layer_memory_budget_gib,\n'
        '                geometry_initialisation=training_config.plastic__layer_sampling_initialisation,\n'
        '                checkpoint_callback=checkpoint_review_pause,\n'
        '            )\n'
        '            if coarse_fine_outcome.fine_state is None:\n'
        '                coarse_fine_outcome.close_coordinator()\n'
        '                return 0\n'
        '            fresh_state = coarse_fine_outcome.fine_state\n'
        '            trainer = fresh_state.trainer\n'
        '            training_config = trainer.config\n'
        '            lifecycle = {\n'
        '                **lifecycle,\n'
        '                "plastic_coarse_fine": dict(coarse_fine_outcome.provenance),\n'
        '            }\n'
        '            if trainer.distributed.is_primary:\n'
        '                write_run_manifest(paths["manifest_path"], lifecycle)\n'
        '        elif training_config.plastic__enabled:\n'
        '            fresh_state = build_fresh_training_state(\n'
        '                trainer_factory=core.OwtTrainer,\n'
        '                resolved_config=training_config,\n'
        '                train_tokens=train_tokens,\n'
        '                validation_tokens=validation_tokens,\n'
        '                phase="fine",\n'
        '                active_layer_count=int(training_config.plastic__initial_active_layers),\n'
        '                instrumentation_namespace="fine",\n'
        '            )\n'
        '            trainer = fresh_state.trainer\n'
        '        else:\n'
        '            trainer = core.OwtTrainer(training_config, train_tokens, validation_tokens)\n'
        '        # ^^^ THOG\n'
        '    else:\n'
        '        resolved: ResolvedCheckpoint = context["resolved"]\n'
        '        trainer = core.OwtTrainer.from_checkpoint(\n'
        '            resolved.checkpoint_path,\n'
        '            train_tokens,\n'
        '            validation_tokens,\n'
        '            expected_config=training_config,\n'
        '            overrides=_resume_overrides(training_config),\n'
        '        )\n'
        '        resume_boundary = resume_plastic_coarse_fine_boundary(\n'
        '            trainer,\n'
        '            checkpoint_path,\n'
        '        )\n'
        '        if resume_boundary == PLASTIC_RESUME_CHECKPOINT_EXIT:\n'
        '            trainer.close()\n'
        '            return 0\n'
        '        if resume_boundary == PLASTIC_RESUME_CONTINUE_FINE:\n'
        '            lifecycle = {\n'
        '                **lifecycle,\n'
        '                "plastic_coarse_fine": dict(trainer.plastic_coarse_fine_state),\n'
        '            }\n'
        '            if trainer.distributed.is_primary:\n'
        '                write_run_manifest(paths["manifest_path"], lifecycle)\n'
        '    trainer.lifecycle_metadata = dict(lifecycle)\n',
    )
    replace_once(
        path,
        '        result["lifecycle"] = lifecycle\n'
        '        result["canonical_config"] = canonical\n',
        '        result["lifecycle"] = lifecycle\n'
        '        # vvv THOG keep COARSE trial evidence and selected-count provenance in the canonical result and telemetry payload\n'
        '        if hasattr(trainer, "plastic_coarse_provenance"):\n'
        '            result["plastic_coarse_fine"] = trainer.plastic_coarse_provenance\n'
        '        # ^^^ THOG\n'
        '        result["canonical_config"] = canonical\n',
    )
    replace_once(
        path,
        '        trainer.close()\n'
        '        if fresh_state is not None:\n'
        '            fresh_state.trainer = None\n',
        '        trainer.close()\n'
        '        if fresh_state is not None:\n'
        '            fresh_state.trainer = None\n'
        '        if coarse_fine_outcome is not None:\n'
        '            coarse_fine_outcome.close_coordinator()\n',
    )


if __name__ == "__main__":
    main()
