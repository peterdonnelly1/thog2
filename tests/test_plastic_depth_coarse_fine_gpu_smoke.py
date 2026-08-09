from __future__ import annotations

import io

import pytest
import torch

from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
from sheet.plastic_depth_audit_patch import replay_plastic_depth_count_audit
from sheet.plastic_depth_coarse import resolve_plastic_coarse_config
from sheet.plastic_depth_fresh_state import destroy_fresh_training_state
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real PLASTIC COARSE/FINE smoke requires CUDA",
)


@pytest.fixture
def _same_batch_runtime(monkeypatch):
    # vvv THOG make the external CUDA smoke exercise the v0.53 fixed-batch path without leaking process-global mode into sibling smoke tests
    monkeypatch.setenv(same_batch._RUNTIME_ENV, "true")
    monkeypatch.delenv(same_batch._EXPLICIT_ENV, raising=False)
    yield
    # ^^^ THOG


def _config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        device="cuda",
        dtype="float32",
        max_updates=4,
        warmup_updates=0,
        eval_interval=0,
        checkpoint_interval=0,
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="enabled",
        plastic__phase_1_n_steps=1,
        plastic__phase_1_starting_layer_count=2,
        plastic__phase_1__number_of_trials=1,
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
        plastic__layer_count_probe_noise_lambda=1.0e9,
        plastic__cuda_allocator_reserve_gib=0.125,
    )


def test_real_cuda_coarse_to_full_radius_fine_update(_same_batch_runtime) -> None:
    del _same_batch_runtime
    config = _config()
    train_tokens, validation_tokens = token_splits()
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

    torch.cuda.reset_peak_memory_stats()
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
    assert outcome.fine_state is not None
    fine_state = outcome.fine_state
    try:
        assert len(outcome.trial_results) == 1
        assert outcome.trial_results[0].status == "success"
        assert outcome.selected_layers == 2
        assert fine_state.trainer.state.completed_updates == 0

        # vvv THOG complete one strict four-probe same-batch window on real CUDA; no early count decision may change the architecture
        for expected_update in range(1, 5):
            metrics = fine_state.trainer.train_one_update()
            torch.cuda.synchronize()
            assert metrics["skipped_update"] == 0.0
            assert fine_state.trainer.state.completed_updates == expected_update
            assert metrics["plastic_active_layers"] == 2.0
            assert len(fine_state.trainer.plastic_depth_count_audit) == expected_update

        audits = fine_state.trainer.plastic_depth_count_audit
        assert {audit["probe_batch_digest"] for audit in audits} == {audits[0]["probe_batch_digest"]}
        assert [audit["probe_window_ordinal"] for audit in audits] == [1, 2, 3, 4]
        assert [audit["probe_window_complete"] for audit in audits] == [False, False, False, True]
        assert [tuple(audit["probe_window_provenance"]) for audit in audits] == [
            (1,),
            (1, 2),
            (1, 2, 3),
            (1, 2, 3, 4),
        ]
        assert audits[-1]["probe_window_disposition"] == "stay"
        assert audits[-1]["histories_after_window_retirement"] == {}
        assert same_batch._window_state(fine_state.trainer)["active"] is None
        # ^^^ THOG

        audit = audits[-1]
        assert audit["decision_candidate_counts"] == (1, 2, 3, 4)
        assert audit["execution_candidate_counts"] == (1, 2, 3, 4)
        replay_plastic_depth_count_audit(audit)
        assert torch.cuda.max_memory_allocated() > 0
    finally:
        destroy_fresh_training_state(fine_state)
        outcome.close_coordinator()
