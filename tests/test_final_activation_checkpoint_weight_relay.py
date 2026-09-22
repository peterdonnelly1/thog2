# vvv THOG
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from run_thog2_owt_core import build_parser
from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig


def _config(*, relay: bool, compress_vectors: bool = False) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=1,
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_compress_layer_norm_and_bias=compress_vectors,
        checkpoint_segment_size=1,
        save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step=relay,
        batch_size=1,
        gradient_accumulation_steps=2,
        max_updates=1,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-3,
        decay_updates=1,
        decay_learning_rate=False,
        weight_decay=0.0,
        grad_clip=0.0,
        eval_interval=0,
        checkpoint_interval=0,
        model_seed=123,
        data_seed=456,
        device="cpu",
        dtype="float32",
    )


def _tokens() -> torch.Tensor:
    return torch.arange(96, dtype=torch.long).remainder(32)


def test_cli_flag_is_default_disabled_and_literal_spelling_enables_it() -> None:
    parser = build_parser()
    assert parser.parse_args([]).save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step is False
    arguments = parser.parse_args([
        "--save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step"
    ])
    assert arguments.save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step is True


def test_canonical_wrapper_preserves_literal_underscore_spelling() -> None:
    wrapper = Path("train_OWT.sh").read_text(encoding="utf-8")
    assert (
        "|--save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step)"
        in wrapper
    )


def test_relay_requires_compact_checkpointed_training() -> None:
    with pytest.raises(ValueError, match="model_type='thog2_sheet'"):
        replace(_config(relay=False), model_type="dense", save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step=True)
    with pytest.raises(ValueError, match="checkpoint_segment_size > 0"):
        replace(_config(relay=False), checkpoint_segment_size=0, save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step=True)
    with pytest.raises(ValueError, match="geometry_preset='depth'"):
        replace(
            _config(relay=False),
            geometry_preset="conventional",
            basis_family="conventional",
            save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step=True,
        )


@pytest.mark.parametrize(("compress_vectors", "expected_count"), ((False, 4), (True, 12)))
def test_relay_reuses_first_checkpoint_group_and_preserves_update(
    monkeypatch,
    compress_vectors: bool,
    expected_count: int,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", "true")
    monkeypatch.setenv("THOG2_DIRECT_FACTORISED_MLP", "false")
    tokens = _tokens()
    baseline = SharedTrainer(_config(relay=False, compress_vectors=compress_vectors), tokens, tokens)
    relay = SharedTrainer(_config(relay=True, compress_vectors=compress_vectors), tokens, tokens)
    try:
        baseline_result = baseline.train_one_update()
        relay_result = relay.train_one_update()
        assert relay_result["training_loss"] == pytest.approx(
            baseline_result["training_loss"], rel=0.0, abs=1.0e-7
        )
        baseline_state = baseline.raw_model.state_dict()
        relay_state = relay.raw_model.state_dict()
        assert baseline_state.keys() == relay_state.keys()
        for name in baseline_state:
            torch.testing.assert_close(
                relay_state[name],
                baseline_state[name],
                rtol=1.0e-5,
                atol=1.0e-7,
                msg=f"relay changed the optimizer update for {name}",
            )
        report = relay.raw_model.final_activation_checkpoint_weight_relay_report()
        assert report == {
            "enabled": True,
            "phase": "idle",
            "cached_count": 0,
            "captured": expected_count,
            "reused": expected_count,
            "discarded": 0,
            "boundaries": 1,
        }
        assert baseline.raw_model.final_activation_checkpoint_weight_relay_report()["enabled"] is False
    finally:
        baseline.close()
        relay.close()


def test_disabled_relay_is_omitted_from_persistent_checkpoint_identity() -> None:
    disabled = _config(relay=False).persistent_dict()
    enabled = _config(relay=True).persistent_dict()
    field = "save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step"
    assert field not in disabled
    assert enabled[field] is True
# ^^^ THOG
