from __future__ import annotations

import io
import json
import os
from pathlib import Path

from sheet.plastic_depth_coarse import resolve_plastic_coarse_config
from sheet.plastic_depth_fresh_state import destroy_fresh_training_state
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


def _config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        max_updates=2,
        eval_interval=0,
        warmup_updates=0,
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="enabled",
        plastic__phase_1_n_steps=1,
        plastic__phase_1_starting_layer_count=2,
        plastic__phase_1__number_of_trials=2,
        plastic__phase_1_evaluation_steps_count=1,
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=None,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe_radius=2,
        plastic__layer_count__max_allowable_layer_change=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=4,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=8,
        plastic__layer_count_probe_noise_lambda=1.0e9,
    )


def main() -> None:
    config = _config()
    train_tokens, validation_tokens = token_splits(length=1024)
    coarse = resolve_plastic_coarse_config(
        coarse_phase=config.plastic__coarse_phase,
        plastic_enabled=config.plastic__enabled,
        do_learn_layer_count=config.plastic__do_learn_layer_count,
        n_steps=config.plastic__phase_1_n_steps,
        starting_layer_count=config.plastic__phase_1_starting_layer_count,
        number_of_trials=config.plastic__phase_1__number_of_trials,
        evaluation_steps_count=config.plastic__phase_1_evaluation_steps_count,
        max_permitted_layers=config.plastic__max_permitted_layers,
    )
    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=SharedTrainer,
        resolved_config=config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        coarse_config=coarse,
        objective="lowest_loss",
        maximum_layers=4,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        pause_duration_seconds=0.0,
        console_stream=io.StringIO(),
    )
    if outcome.fine_state is None:
        raise RuntimeError("DDP lifecycle did not construct FINE")
    fine_state = outcome.fine_state
    trainer = fine_state.trainer
    try:
        metrics = trainer.train_one_update()
        if float(metrics["skipped_update"]) != 0.0:
            raise RuntimeError("DDP FINE probe update was skipped")
        audit_rows = getattr(trainer, "plastic_depth_count_audit", ())
        if len(audit_rows) != 1:
            raise RuntimeError("DDP FINE update did not emit exactly one audit row")
        result = {
            "rank": trainer.distributed.rank,
            "world_size": trainer.distributed.world_size,
            "selected_layers": outcome.selected_layers,
            "trial_scores": [
                {
                    "trial_index": row.result.trial_index,
                    "layers": row.result.layers,
                    "status": row.result.status,
                    "score": row.score,
                }
                for row in outcome.scored_trials
            ],
            "pause_disposition": outcome.pause_result.disposition,
            "fine_completed_updates": trainer.state.completed_updates,
            "fine_active_layers": trainer.raw_model.trajectory.plastic_sampling.current_active_layers,
            "audit": audit_rows[0],
        }
        result_dir = Path(os.environ["THOG2_DDP_RESULT_DIR"])
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"rank_{trainer.distributed.rank}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trainer.distributed.barrier()
    finally:
        destroy_fresh_training_state(fine_state)
        outcome.close_coordinator()


if __name__ == "__main__":
    main()
