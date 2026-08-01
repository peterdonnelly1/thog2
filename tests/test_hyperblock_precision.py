# vvv THOG
from __future__ import annotations

import math

import pytest
import torch

from sheet.hyperblock import HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig


def _config(*, device: str, dtype: str) -> TrainingConfig:
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
        hyperblock_compressor="chebyshev",
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
        model_seed=947,
        data_seed=749,
        device=device,
        dtype=dtype,
    )


def _one_update(config: TrainingConfig) -> None:
    tokens = torch.arange(128, dtype=torch.long).remainder(config.vocab_size)
    trainer = Stage4Trainer(config, tokens, tokens)
    try:
        metrics = trainer.train_one_update()
        assert math.isfinite(float(metrics["training_loss"]))
        assert float(metrics["skipped_update"]) == 0.0
        assert all(
            bool(torch.isfinite(parameter).all())
            for parameter in trainer.raw_model.parameters()
        )
    finally:
        trainer.close()


@pytest.mark.parametrize("fast_discard", (False, True))
def test_hyperblock_cpu_bfloat16_one_update(
    monkeypatch: pytest.MonkeyPatch,
    fast_discard: bool,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", str(fast_discard).lower())
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    _one_update(_config(device="cpu", dtype="bfloat16"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hyperblock_cuda_bfloat16_one_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", "false")
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    _one_update(_config(device="cuda", dtype="bfloat16"))
# ^^^ THOG
