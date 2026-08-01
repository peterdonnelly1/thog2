# vvv THOG
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor

from sheet.bases import DCT_BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION
from sheet.bases.haar import HAAR_BASIS_VERSION
from sheet.hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    CoupledFieldTrajectory,
)
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model


def _model_config(*, fast_discard: bool = False, bias: bool = True, compressor: str = "chebyshev") -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=bias,
        hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        hyperblock_compressor=compressor,
        hyperblock_compressor_version="auto",
        hyperblock_depth_order=2,
        hyperblock_d_model_order=4,
        hyperblock_mlp_hidden_order=4,
        hyperblock_attention_head_order=2,
        hyperblock_attention_head_channel_order=4,
        fast_discard=fast_discard,
    )


def _training_config(*, compressor: str = "chebyshev") -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        hyperblock_compressor=compressor,
        hyperblock_compressor_version="auto",
        hyperblock_depth_order=2,
        hyperblock_d_model_order=4,
        hyperblock_mlp_hidden_order=4,
        hyperblock_attention_head_order=2,
        hyperblock_attention_head_channel_order=4,
        checkpoint_segment_size=1,
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


def test_hyperblock_sheet_gpt_forward_backward_and_optimizer_groups() -> None:
    torch.manual_seed(123)
    model = SheetGPT(_model_config())
    assert isinstance(model.trajectory, CoupledFieldTrajectory)
    inputs = torch.randint(0, 32, (2, 4))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 4, 32)
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.trajectory.coefficients.values()
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1.0e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )
    groups = {float(group["weight_decay"]): group for group in optimizer.param_groups}
    coefficient_ids = {
        id(parameter) for parameter in model.trajectory.coefficients.values()
    }
    vector_ids = {
        id(parameter) for parameter in model.trajectory.vector_parameters.values()
    }
    assert coefficient_ids <= {id(parameter) for parameter in groups[0.1]["params"]}
    assert vector_ids <= {id(parameter) for parameter in groups[0.0]["params"]}
    assert model.compact_state_violations() == ()


def test_nonstandard_mlp_hidden_multiplier_runs_end_to_end() -> None:
    config = dataclasses.replace(
        _model_config(),
        hyperblock_mlp_hidden_multiplier=3,
        hyperblock_mlp_hidden_order=6,
    )
    model = SheetGPT(config)
    assert model.trajectory.materialize("mlp_expansion_weight", 0).shape == (24, 8)
    assert model.trajectory.materialize("mlp_contraction_weight", 0).shape == (8, 24)
    tokens = torch.randint(0, 32, (2, 4))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 4, 32)
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.trajectory.coefficients.values()
    )


def test_hyperblock_training_config_round_trip_and_identity() -> None:
    config = _training_config()
    restored = TrainingConfig(**dataclasses.asdict(config))
    assert restored.hyperblock_plan() == config.hyperblock_plan()
    assert restored.compatibility_signature() == config.compatibility_signature()
    identity = config.compact_identity_metadata()["hyperblock"]
    assert identity["topology"] == HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
    assert identity["coefficient_counts"]["total"] == 320
    model = build_training_model(config)
    assert isinstance(model.trajectory, CoupledFieldTrajectory)


def test_training_factory_resolves_residual_std_before_reduced_family_initialization() -> None:
    config = dataclasses.replace(
        _training_config(),
        hyperblock_common_family_order=3,
        hyperblock_attention_family_order=2,
        hyperblock_mlp_family_order=1,
        residual_init_policy="unscaled",
    )
    model = build_training_model(config)
    assert model.trajectory.family_metadata("attention_output_weight").target_weight_std == pytest.approx(0.02)
    assert model.trajectory.family_metadata("mlp_contraction_weight").target_weight_std == pytest.approx(0.02)

    with torch.random.fork_rng():
        torch.manual_seed(config.model_seed)
        expected = SheetGPT(SheetGPTConfig(**config.model_arguments()))
    for name, parameter in model.trajectory.coefficients.items():
        torch.testing.assert_close(parameter, expected.trajectory.coefficients[name])


def test_hyperblock_rejects_legacy_geometry_overlap() -> None:
    with pytest.raises(ValueError, match="legacy geometry"):
        SheetGPTConfig(
            **{
                **dataclasses.asdict(_model_config()),
                "geometry_preset": "depth",
            }
        )
    with pytest.raises(ValueError, match="legacy geometry"):
        TrainingConfig(
            **{
                **dataclasses.asdict(_training_config()),
                "basis_family": "chebyshev",
            }
        )


def _run_one_update(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fast_discard: bool,
) -> Tuple[Dict[str, float], Dict[str, Tensor], Dict[str, object]]:
    monkeypatch.setenv("THOG2_FAST_DISCARD", "true" if fast_discard else "false")
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    trainer = Stage4Trainer(_training_config(), tokens, tokens)
    try:
        metrics = trainer.train_one_update()
        state = {
            name: value.detach().clone()
            for name, value in trainer.raw_model.state_dict().items()
        }
        report = trainer.raw_model.update_retained_materialization_report()
        return metrics, state, report
    finally:
        trainer.close()


def test_hyperblock_retained_and_ephemeral_updates_remain_numerically_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_metrics, retained_state, retained_report = _run_one_update(
        monkeypatch,
        fast_discard=False,
    )
    ephemeral_metrics, ephemeral_state, ephemeral_report = _run_one_update(
        monkeypatch,
        fast_discard=True,
    )
    assert retained_metrics["training_loss"] == pytest.approx(
        ephemeral_metrics["training_loss"],
        rel=0.0,
        abs=1.0e-7,
    )
    assert retained_state.keys() == ephemeral_state.keys()
    maximum_difference = max(
        float((retained_state[name] - ephemeral_state[name]).abs().max().item())
        for name in retained_state
    )
    # Shared coefficients accumulate many cancelling contributions. The retained
    # projection and direct autograd paths differ only in floating-point reduction
    # order, but Adam's first normalized update magnifies near-zero sign changes.
    assert maximum_difference < 5.0e-4
    assert bool(retained_report["enabled"]) is True
    assert bool(retained_report["active"]) is False
    assert int(retained_report["retained_count"]) == 0
    assert int(retained_report["materialization_count"]) > 0
    assert bool(ephemeral_report["enabled"]) is False
    assert int(ephemeral_report["retained_count"]) == 0
# ^^^ THOG

# vvv THOG

def test_hyperblock_whole_model_is_fullgraph_compilable() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    torch.manual_seed(821)
    model = SheetGPT(_model_config(fast_discard=True))
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    tokens = torch.randint(0, 32, (1, 4))
    logits, loss = compiled(tokens, tokens)
    assert logits.shape == (1, 4, 32)
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.trajectory.coefficients.values()
    )
# ^^^ THOG

# vvv THOG

@pytest.mark.parametrize(
    ("compressor", "expected_version"),
    (
        ("dct", DCT_BASIS_VERSION),
        ("haar", HAAR_BASIS_VERSION),
        ("lapped_cosine", LAPPED_COSINE_BASIS_VERSION),
    ),
)
def test_registered_non_chebyshev_hyperblock_trains_checkpoints_and_resumes(
    compressor: str,
    expected_version: str,
) -> None:
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    config = _training_config(compressor=compressor)
    config.max_updates = 2
    config.decay_updates = 2
    trainer = Stage4Trainer(config, tokens, tokens)
    try:
        metrics = trainer.train_one_update()
        assert torch.isfinite(torch.tensor(metrics["training_loss"]))
        assert trainer.config.hyperblock_compressor_version == expected_version
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / f"hyperblock_{compressor}.pt"
            trainer.save_checkpoint(checkpoint)
            resumed = Stage4Trainer.from_checkpoint(
                checkpoint,
                tokens,
                tokens,
                overrides={"max_updates": 2},
            )
            try:
                assert resumed.config.hyperblock_compressor == compressor
                assert resumed.config.hyperblock_compressor_version == expected_version
                resumed.train_one_update()
                assert resumed.state.completed_updates == 2
            finally:
                resumed.close()
    finally:
        trainer.close()
# ^^^ THOG

# vvv THOG

def test_reduced_family_order_model_runs_end_to_end() -> None:
    config = dataclasses.replace(
        _model_config(),
        hyperblock_common_family_order=3,
        hyperblock_attention_family_order=2,
        hyperblock_mlp_family_order=1,
    )
    model = SheetGPT(config)
    tokens = torch.randint(0, 32, (2, 4))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 4, 32)
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.trajectory.coefficients.values()
    )
# ^^^ THOG
