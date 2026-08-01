# vvv THOG
from __future__ import annotations

import copy
from types import MethodType
from typing import Dict, List, Tuple

import pytest
import torch
from torch import Tensor, nn

from sheet.model import SheetGPTConfig
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig
from sheet.training_model import TrainingSheetGPT
from sheet.update_retained_materializations import (
    attach_update_retained_materializations,
)


class _ToyTrajectory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(
            torch.arange(6, dtype=torch.float32).view(2, 3)
        )
        self.calls = 0

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if name != "weight":
            raise KeyError(name)
        self.calls += 1
        return self.coefficient * float(layer_index + 1)


def _sheet_config(*, fast_discard: bool) -> SheetGPTConfig:
    return SheetGPTConfig(
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
        fast_discard=fast_discard,
    )


def _training_config() -> TrainingConfig:
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


def _microbatches() -> Tuple[Tuple[Tensor, Tensor], ...]:
    generator = torch.Generator().manual_seed(987)
    rows: List[Tuple[Tensor, Tensor]] = []
    for _ in range(2):
        inputs = torch.randint(0, 32, (1, 4), generator=generator)
        targets = torch.randint(0, 32, (1, 4), generator=generator)
        rows.append((inputs, targets))
    return tuple(rows)


def _run_manual_accumulation(
    model: TrainingSheetGPT,
    microbatches: Tuple[Tuple[Tensor, Tensor], ...],
) -> Tuple[
    Tuple[float, ...],
    Dict[str, Tensor],
    Dict[str, object],
    Dict[str, object],
]:
    model.train()
    model.zero_grad(set_to_none=True)
    retained_active = model.begin_optimizer_update()
    losses: List[float] = []
    first_forward_materializations = None
    try:
        for micro_step, (inputs, targets) in enumerate(microbatches):
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(float(loss.detach().item()))
            if micro_step == 0 and retained_active:
                first_forward_materializations = int(
                    model.update_retained_materialization_report()[
                        "materialization_count"
                    ]
                )
            loss.div(len(microbatches)).backward()
            if micro_step == 0 and retained_active:
                assert int(
                    model.update_retained_materialization_report()[
                        "materialization_count"
                    ]
                ) == first_forward_materializations
        report_before_finalize = (
            model.update_retained_materialization_report()
        )
        if retained_active:
            assert first_forward_materializations is not None
            assert int(
                report_before_finalize["materialization_count"]
            ) == first_forward_materializations
            controller = model._update_retained_materializations
            assert controller.request_count > controller.materialization_count
            projected_parameters = model.finalize_optimizer_update()
            assert projected_parameters
            retained_active = False
    finally:
        if retained_active:
            model.end_optimizer_update()
    report_after_finalize = model.update_retained_materialization_report()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return (
        tuple(losses),
        gradients,
        report_before_finalize,
        report_after_finalize,
    )


def test_retained_materialization_lifetime_projects_exact_gradient() -> None:
    trajectory = _ToyTrajectory()
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=True,
    )
    assert controller.begin() is True
    first = trajectory.materialize("weight", 1)
    second = trajectory.materialize("weight", 1)
    assert first is second
    assert first.is_leaf
    assert first.grad_fn is None
    assert trajectory.calls == 1
    assert controller.request_count == 2
    assert controller.materialization_count == 1
    assert controller.retained_count == 1

    upstream = torch.full_like(first, 3.0)
    (first * upstream).sum().backward()
    projected = controller.finalize()
    assert any(parameter is trajectory.coefficient for parameter in projected)
    torch.testing.assert_close(
        trajectory.coefficient.grad,
        upstream * 2.0,
        rtol=0.0,
        atol=0.0,
    )
    assert controller.active is False
    assert controller.retained_count == 0

    trajectory.coefficient.grad = None
    assert controller.begin() is True
    third = trajectory.materialize("weight", 1)
    assert third is not first
    assert trajectory.calls == 2
    controller.end()


def test_retained_materialization_deepcopy_and_state_are_isolated() -> None:
    trajectory = _ToyTrajectory()
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=True,
    )
    copied = copy.deepcopy(trajectory)
    with torch.no_grad():
        copied.coefficient.add_(100.0)

    copied_controller = (
        copied._update_retained_materializations_controller
    )
    assert copied_controller.begin() is True
    copied_value = copied.materialize("weight", 0)
    copied_controller.end()
    assert torch.equal(copied_value, copied.coefficient)
    assert not torch.equal(copied_value, trajectory.coefficient)
    assert copied.materialize.__self__ is copied
    assert controller.active is False
    assert tuple(trajectory.state_dict()) == ("coefficient",)
    assert tuple(copied.state_dict()) == ("coefficient",)


def test_disabled_retention_preserves_ephemeral_materialization() -> None:
    trajectory = _ToyTrajectory()
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=False,
    )
    assert controller.begin() is False
    first = trajectory.materialize("weight", 0)
    second = trajectory.materialize("weight", 0)
    assert first is not second
    assert trajectory.calls == 2
    assert controller.retained_count == 0
    assert controller.materialization_count == 0


def test_checkpointed_accumulation_matches_fast_discard_gradients() -> None:
    torch.manual_seed(1234)
    retained_model = TrainingSheetGPT(
        _sheet_config(fast_discard=False)
    )
    ephemeral_model = TrainingSheetGPT(
        _sheet_config(fast_discard=True)
    )
    ephemeral_model.load_state_dict(retained_model.state_dict())
    retained_model.set_checkpoint_segment_size(1)
    ephemeral_model.set_checkpoint_segment_size(1)
    microbatches = _microbatches()

    (
        retained_losses,
        retained_gradients,
        retained_before,
        retained_after,
    ) = _run_manual_accumulation(retained_model, microbatches)
    (
        ephemeral_losses,
        ephemeral_gradients,
        _,
        ephemeral_after,
    ) = _run_manual_accumulation(ephemeral_model, microbatches)

    assert retained_losses == pytest.approx(
        ephemeral_losses,
        rel=0.0,
        abs=1.0e-7,
    )
    assert retained_gradients.keys() == ephemeral_gradients.keys()
    for name in retained_gradients:
        torch.testing.assert_close(
            retained_gradients[name],
            ephemeral_gradients[name],
            rtol=1.0e-5,
            atol=1.0e-7,
            msg=f"gradient mismatch for {name}",
        )
    assert bool(retained_before["enabled"]) is True
    assert bool(retained_before["active"]) is True
    assert int(retained_before["retained_count"]) > 0
    assert int(retained_before["request_count"]) > int(
        retained_before["materialization_count"]
    )
    assert bool(retained_after["active"]) is False
    assert int(retained_after["retained_count"]) == 0
    assert bool(ephemeral_after["enabled"]) is False
    assert int(ephemeral_after["retained_count"]) == 0
    assert tuple(retained_model.state_dict()) == tuple(
        ephemeral_model.state_dict()
    )


def _run_trainer_update(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fast_discard: bool,
) -> Tuple[Dict[str, float], Dict[str, Tensor], Dict[str, object], int, int]:
    monkeypatch.setenv(
        "THOG2_FAST_DISCARD",
        "true" if fast_discard else "false",
    )
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    trainer = Stage4Trainer(_training_config(), tokens, tokens)
    try:
        metrics = trainer.train_one_update()
        state = {
            name: value.detach().clone()
            for name, value in trainer.raw_model.state_dict().items()
        }
        report = (
            trainer.raw_model.update_retained_materialization_report()
        )
        controller = trainer.raw_model._update_retained_materializations
        return (
            metrics,
            state,
            report,
            controller.request_count,
            controller.materialization_count,
        )
    finally:
        trainer.close()


def test_stage4_update_is_equivalent_and_releases_materializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _run_trainer_update(
        monkeypatch,
        fast_discard=False,
    )
    ephemeral = _run_trainer_update(
        monkeypatch,
        fast_discard=True,
    )
    (
        retained_metrics,
        retained_state,
        retained_report,
        requests,
        materializations,
    ) = retained
    (
        ephemeral_metrics,
        ephemeral_state,
        ephemeral_report,
        _,
        _,
    ) = ephemeral

    assert retained_metrics["training_loss"] == pytest.approx(
        ephemeral_metrics["training_loss"],
        rel=0.0,
        abs=1.0e-7,
    )
    assert retained_state.keys() == ephemeral_state.keys()
    for name in retained_state:
        torch.testing.assert_close(
            retained_state[name],
            ephemeral_state[name],
            rtol=1.0e-5,
            atol=1.0e-7,
            msg=f"updated state mismatch for {name}",
        )
    assert bool(retained_report["enabled"]) is True
    assert bool(retained_report["active"]) is False
    assert int(retained_report["retained_count"]) == 0
    assert requests > materializations > 0
    assert bool(ephemeral_report["enabled"]) is False
    assert int(ephemeral_report["retained_count"]) == 0


def test_nonfinite_loss_releases_update_retained_materializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", "false")
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    trainer = Stage4Trainer(_training_config(), tokens, tokens)
    original_forward = trainer.raw_model.forward

    def nonfinite_forward(
        model: TrainingSheetGPT,
        inputs: Tensor,
        targets: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor | None]:
        logits, loss = original_forward(inputs, targets)
        if loss is not None:
            loss = loss * torch.tensor(
                float("nan"),
                device=loss.device,
            )
        return logits, loss

    trainer.raw_model.forward = MethodType(
        nonfinite_forward,
        trainer.raw_model,
    )
    try:
        metrics = trainer.train_one_update()
        report = (
            trainer.raw_model.update_retained_materialization_report()
        )
        assert metrics["skipped_update"] == 1.0
        assert bool(report["active"]) is False
        assert int(report["retained_count"]) == 0
    finally:
        trainer.close()


@pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile unavailable",
)
def test_update_retention_survives_compiled_checkpointed_forward() -> None:
    torch.manual_seed(2468)
    retained_model = TrainingSheetGPT(
        _sheet_config(fast_discard=False)
    )
    ephemeral_model = TrainingSheetGPT(
        _sheet_config(fast_discard=True)
    )
    ephemeral_model.load_state_dict(retained_model.state_dict())
    retained_model.set_checkpoint_segment_size(1)
    ephemeral_model.set_checkpoint_segment_size(1)
    compiled_model = torch.compile(retained_model, backend="eager")
    microbatches = _microbatches()

    retained_model.train()
    retained_model.zero_grad(set_to_none=True)
    retained_active = retained_model.begin_optimizer_update()
    try:
        for inputs, targets in microbatches:
            _, loss = compiled_model(inputs, targets)
            assert loss is not None
            loss.div(len(microbatches)).backward()
        projected = retained_model.finalize_optimizer_update()
        assert projected
        retained_active = False
    finally:
        if retained_active:
            retained_model.end_optimizer_update()

    ephemeral_model.train()
    ephemeral_model.zero_grad(set_to_none=True)
    for inputs, targets in microbatches:
        _, loss = ephemeral_model(inputs, targets)
        assert loss is not None
        loss.div(len(microbatches)).backward()

    retained_gradients = {
        name: parameter.grad
        for name, parameter in retained_model.named_parameters()
    }
    ephemeral_gradients = {
        name: parameter.grad
        for name, parameter in ephemeral_model.named_parameters()
    }
    for name, retained_gradient in retained_gradients.items():
        ephemeral_gradient = ephemeral_gradients[name]
        if retained_gradient is None or ephemeral_gradient is None:
            assert retained_gradient is None
            assert ephemeral_gradient is None
            continue
        torch.testing.assert_close(
            retained_gradient,
            ephemeral_gradient,
            rtol=1.0e-5,
            atol=1.0e-7,
            msg=f"compiled gradient mismatch for {name}",
        )
    controller = retained_model._update_retained_materializations
    assert controller.request_count > controller.materialization_count > 0
    report = retained_model.update_retained_materialization_report()
    assert bool(report["active"]) is False
    assert int(report["retained_count"]) == 0
# ^^^ THOG
