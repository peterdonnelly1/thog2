from __future__ import annotations

import copy

import pytest

from sheet.plastic_depth_audit_patch import replay_plastic_depth_count_audit
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


def _synthetic_audit():
    return {
        "previous_count": 10,
        "winning_probe_count": 8,
        "committed_count": 9,
        "decision_reason": "max_step_limited",
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "robust_evidence": (
            {
                "candidate_count": 8,
                "feasible": True,
                "significant": True,
                "standardized_improvement": 5.0,
            },
            {
                "candidate_count": 12,
                "feasible": True,
                "significant": True,
                "standardized_improvement": 5.0,
            },
        ),
    }


def test_replay_separates_winning_probe_from_bounded_commit() -> None:
    replay = replay_plastic_depth_count_audit(_synthetic_audit())

    assert replay == {
        "winning_probe_count": 8,
        "committed_count": 9,
        "decision_reason": "max_step_limited",
    }


# vvv THOG reproduce the real warmup-held 4 -> winning probe 1 -> committed 4 audit path exposed by the forced-change CUDA run
def test_replay_preserves_winner_when_warmup_brake_holds_current_count() -> None:
    audit = {
        "previous_count": 4,
        "winning_probe_count": 1,
        "committed_count": 4,
        "decision_reason": "warmup_brake",
        "max_step": 5,
        "brake_active": False,
        "warmup_brake_active": True,
        "robust_evidence": (
            {
                "candidate_count": 1,
                "feasible": True,
                "significant": True,
                "standardized_improvement": 5.0,
            },
        ),
    }

    assert replay_plastic_depth_count_audit(audit) == {
        "winning_probe_count": 1,
        "committed_count": 4,
        "decision_reason": "warmup_brake",
    }
# ^^^ THOG


def test_replay_detects_tampered_commit() -> None:
    audit = _synthetic_audit()
    audit["committed_count"] = 8

    with pytest.raises(ValueError, match="audit replay mismatch"):
        replay_plastic_depth_count_audit(audit)


def _config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        max_updates=2,
        warmup_updates=0,
        eval_interval=0,
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="disabled",
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=2,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe_radius=2,
        plastic__layer_count__max_allowable_layer_change=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=4,
        plastic__layer_count_probe_noise_lambda=1.0e9,
    )


def test_real_fine_decision_audit_is_complete_replayable_and_checkpointed(tmp_path) -> None:
    config = _config()
    train_tokens, validation_tokens = token_splits()
    checkpoint_path = tmp_path / "fine_audit.pt"
    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        metrics = trainer.train_one_update()
        assert metrics["skipped_update"] == 0.0
        assert len(trainer.plastic_depth_count_audit) == 1
        audit = copy.deepcopy(trainer.plastic_depth_count_audit[0])
        required = {
            "phase",
            "update_number",
            "decision_number",
            "previous_count",
            "winning_probe_count",
            "committed_count",
            "decision_reason",
            "objective",
            "probe_every_n_steps",
            "probe_radius",
            "max_step",
            "update_brake",
            "brake_active",
            "warmup_brake_active",
            "decision_candidate_counts",
            "execution_candidate_counts",
            "sampled_tokens_by_rank",
            "sampled_token_count_global",
            "score_table",
            "robust_evidence",
            "histories_before",
            "histories_after",
            "active_public_coordinates_after",
            "transition",
        }
        assert required <= set(audit)
        # vvv THOG v0.521 paired-token SE is durable candidate diagnostic state and must not disappear before audit emission
        assert all(
            "paired_delta_standard_error" in item
            for item in audit["score_table"]
        )
        current_rows = [
            item
            for item in audit["score_table"]
            if int(item["active_layers"]) == int(audit["previous_count"])
        ]
        assert len(current_rows) == 1
        assert current_rows[0]["paired_delta_standard_error"] == pytest.approx(0.0)
        # ^^^ THOG
        assert audit["decision_candidate_counts"] == (1, 2, 3, 4)
        assert len(audit["sampled_tokens_by_rank"]) == 1
        rank_sample = audit["sampled_tokens_by_rank"][0]
        assert rank_sample["rank"] == 0
        assert rank_sample["sampled_token_count"] == len(
            rank_sample["sampled_token_positions"]
        )
        assert audit["sampled_token_count_global"] == rank_sample[
            "sampled_token_count"
        ]
        replay_plastic_depth_count_audit(audit)
        trainer.save_checkpoint(checkpoint_path)
    finally:
        trainer.close()

    resumed = SharedTrainer.from_checkpoint(
        checkpoint_path,
        train_tokens,
        validation_tokens,
        expected_config=config,
    )
    try:
        assert resumed.plastic_depth_count_audit == [audit]
        replay_plastic_depth_count_audit(
            resumed.plastic_depth_count_audit[0]
        )
    finally:
        resumed.close()
