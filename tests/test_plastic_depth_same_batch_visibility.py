# vvv THOG
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pytest

import run_thog2_owt as public_runner
from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
from sheet import plastic_depth_same_batch_visibility_patch as visibility
from sheet import stage6_trainer as stage6
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_training_config


@pytest.fixture(autouse=True)
def _isolated_same_batch_environment(monkeypatch):
    monkeypatch.delenv(same_batch._RUNTIME_ENV, raising=False)
    monkeypatch.delenv(same_batch._EXPLICIT_ENV, raising=False)
    try:
        yield
    finally:
        os.environ.pop(same_batch._RUNTIME_ENV, None)
        os.environ.pop(same_batch._EXPLICIT_ENV, None)


def _trainer(*, window_size: int = 2) -> SharedTrainer:
    same_batch._set_runtime_enabled(True)
    train_tokens, validation_tokens = token_splits(length=2048)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_objective="lowest_loss",
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=window_size,
        plastic__layer_count_probe_noise_lambda=1.0e9,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
        gradient_accumulation_steps=1,
        batch_size=4,
        block_size=8,
        max_updates=8,
        warmup_updates=0,
        checkpoint_segment_size=0,
        device="cpu",
        dtype="float32",
    )
    return SharedTrainer(config, train_tokens, validation_tokens)


def _progress_line_from_audit(row: dict) -> str:
    ordinal = int(row["probe_window_ordinal"])
    payload = {
        "completed_updates": f"{int(row['update_number']):6d}",
        "timestamp": "260809:1300",
        "cumulative_training_seconds": "       10",
        "mean_step_seconds": "  1.0000",
        "tok/s": "       10000",
        "consumed_tokens": "       10000",
        "training_loss": "  7.0000",
        "training_loss_delta": "  -0.001",
        "learning_rate": " 9.000e-04",
        "gradient_norm": "   1.000",
        "current_layer_count": int(row["previous_active_layers"]),
        "plastic_probe_offsets": (-1, 0, 1),
        "plastic_probe_edge_offsets": (-1, 1),
        "plastic_probe_losses": (7.001, 7.0, 6.999),
        "plastic_score_z": (None, None),
        "plastic_probe_sequence": ordinal,
        "plastic_probe_provenance": tuple(row["probe_window_provenance"]),
        "plastic_same_batch_window_id": int(row["probe_window_id"]),
        "plastic_same_batch_window_ordinal": ordinal,
        "plastic_same_batch_window_size": int(row["probe_window_size"]),
        "plastic_same_batch_batch_digest": str(row["probe_batch_digest"]),
    }
    return stage6.format_progress_line("visibility-test", "optimizer_progress", payload)


def test_same_batch_visible_probe_number_resets_with_fresh_batch() -> None:
    trainer = _trainer(window_size=2)
    try:
        trainer.train_one_update()
        first_audit = trainer.plastic_depth_count_audit[-1]
        first_line = _progress_line_from_audit(first_audit)

        trainer.train_one_update()
        second_audit = trainer.plastic_depth_count_audit[-1]
        second_line = _progress_line_from_audit(second_audit)

        trainer.train_one_update()
        third_audit = trainer.plastic_depth_count_audit[-1]
        third_line = _progress_line_from_audit(third_audit)

        assert "P   1" in first_line
        assert "same_batch W1:1/2 B=" in first_line
        assert "P   2" in second_line
        assert "(P1,2)" in second_line
        assert "same_batch W1:2/2 B=" in second_line
        assert "P   1" in third_line
        assert "(P1)" in third_line
        assert "same_batch W2:1/2 B=" in third_line

        audits = trainer.plastic_depth_count_audit[-3:]
        assert [row["probe_window_id"] for row in audits] == [1, 1, 2]
        assert [tuple(row["probe_window_provenance"]) for row in audits] == [
            (1,),
            (1, 2),
            (1,),
        ]
        assert [row["probe_global_sequence"] for row in audits] == [1, 2, 3]
        assert [tuple(row["probe_global_provenance"]) for row in audits] == [
            (1,),
            (1, 2),
            (3,),
        ]
    finally:
        trainer.close()


def test_plastic_startup_section_prints_resolved_same_batch_mode() -> None:
    same_batch._set_runtime_enabled(True)
    visibility._install_startup_visibility()
    stream = io.StringIO()
    with redirect_stdout(stream):
        public_runner._print_plastic_option(
            "plastic__layer_count_probe__window_size_as_number_of_probes:",
            "4",
        )
    rendered = stream.getvalue()
    assert "plastic__layer_count_probe__window_size_as_number_of_probes:" in rendered
    assert "plastic__layer_count__same_batch_all_probes:" in rendered
    assert "true" in rendered
# ^^^ THOG
