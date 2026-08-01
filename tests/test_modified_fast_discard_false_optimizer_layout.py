# vvv THOG
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig
from sheet.update_retained_materializations import (
    attach_update_retained_materializations,
)


_CUDA_BFLOAT16_AVAILABLE = (
    torch.cuda.is_available()
    and bool(
        getattr(
            torch.cuda,
            "is_bf16_supported",
            lambda: False,
        )()
    )
)


class _AutocastProjectionTrajectory(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(
            torch.arange(
                12,
                dtype=torch.float32,
                device=device,
            ).view(4, 3)
        )
        self.register_buffer(
            "basis",
            torch.tensor(
                [[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]],
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if name != "weight":
            raise KeyError(name)
        if layer_index != 0:
            raise IndexError(layer_index)
        return torch.einsum(
            "ij,kj->ki",
            self.coefficient,
            self.basis,
        )


def _project_one_bfloat16_gradient(
    trajectory: _AutocastProjectionTrajectory,
    *,
    device_type: str,
) -> None:
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=True,
    )
    assert controller.begin() is True
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
    ):
        operational = trajectory.materialize("weight", 0)
    assert operational.dtype == torch.bfloat16
    operational.float().square().sum().backward()
    projected = controller.finalize()
    assert any(
        parameter is trajectory.coefficient
        for parameter in projected
    )


def test_bfloat16_projection_matches_compact_parameter_layout() -> None:
    trajectory = _AutocastProjectionTrajectory(torch.device("cpu"))
    _project_one_bfloat16_gradient(
        trajectory,
        device_type="cpu",
    )
    parameter = trajectory.coefficient
    gradient = parameter.grad
    assert gradient is not None
    assert gradient.dtype == parameter.dtype
    assert gradient.device == parameter.device
    assert gradient.layout == parameter.layout
    assert gradient.stride() == parameter.stride()
    assert gradient.is_contiguous() == parameter.is_contiguous()

    optimizer = torch.optim.AdamW(
        trajectory.parameters(),
        lr=1.0e-3,
    )
    optimizer.step()


@pytest.mark.skipif(
    not _CUDA_BFLOAT16_AVAILABLE,
    reason="CUDA bfloat16 is unavailable",
)
def test_bfloat16_projection_is_accepted_by_fused_adam() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    trajectory = _AutocastProjectionTrajectory(device)
    _project_one_bfloat16_gradient(
        trajectory,
        device_type="cuda",
    )
    parameter = trajectory.coefficient
    gradient = parameter.grad
    assert gradient is not None
    assert gradient.dtype == parameter.dtype
    assert gradient.device == parameter.device
    assert gradient.layout == parameter.layout
    assert gradient.stride() == parameter.stride()

    optimizer = torch.optim.AdamW(
        trajectory.parameters(),
        lr=1.0e-3,
        fused=True,
    )
    optimizer.step()


@pytest.mark.skipif(
    not _CUDA_BFLOAT16_AVAILABLE,
    reason="CUDA bfloat16 is unavailable",
)
def test_stage4_bfloat16_checkpointed_retained_update_uses_fused_adam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", "false")
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    monkeypatch.setenv("THOG2_OPTIMIZER", "adamw")
    config = TrainingConfig(
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
        checkpoint_segment_size=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        max_updates=1,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-3,
        decay_updates=1,
        decay_learning_rate=False,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_interval=0,
        checkpoint_interval=0,
        model_seed=123,
        data_seed=456,
        device="cuda",
        dtype="bfloat16",
    )
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    trainer = Stage4Trainer(config, tokens, tokens)
    try:
        metrics = trainer.train_one_update()
        assert metrics["completed_updates"] == 1.0
        assert metrics["skipped_update"] == 0.0
        report = (
            trainer.raw_model.update_retained_materialization_report()
        )
        assert bool(report["enabled"]) is True
        assert bool(report["active"]) is False
        assert int(report["retained_count"]) == 0
        assert int(report["materialization_count"]) > 0
    finally:
        trainer.close()
# ^^^ THOG
