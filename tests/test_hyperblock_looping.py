# vvv THOG
from __future__ import annotations

from unittest import mock

import pytest
import torch

from sheet.hyperblock import HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
from sheet.model import SheetGPT, SheetGPTConfig


def _config(*, loop_count: int = 1, loop_decay: float = 1.0) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=4,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        hyperblock_depth_order=1,
        hyperblock_d_model_order=4,
        hyperblock_mlp_hidden_order=4,
        hyperblock_attention_head_order=2,
        hyperblock_attention_head_channel_order=4,
        hyperblock_loop_count=loop_count,
        hyperblock_loop_decay=loop_decay,
        fast_discard=False,
    )


def test_loop_count_one_preserves_single_application_exactly() -> None:
    torch.manual_seed(11)
    model = SheetGPT(_config())
    inputs = torch.randn(2, 4, 8)
    materializations = model.trajectory.materialize_layer_matrices(0)
    expected = model._logical_block_once(inputs, 0, materializations, None)
    actual = model._logical_block(inputs, 0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_recurrence_matches_exponentially_decayed_manual_loop_and_materializes_once() -> None:
    torch.manual_seed(12)
    model = SheetGPT(_config(loop_count=4, loop_decay=0.5))
    reference = SheetGPT(_config(loop_count=1, loop_decay=1.0))
    reference.load_state_dict(model.state_dict())
    inputs = torch.randn(2, 4, 8)

    reference_materializations = reference.trajectory.materialize_layer_matrices(0)
    expected = inputs
    for loop_index in range(4):
        loop_input = expected
        block_output = reference._logical_block_once(
            loop_input,
            0,
            reference_materializations,
            None,
        )
        loop_gain = 0.5 ** loop_index
        expected = (
            block_output
            if loop_gain == 1.0
            else loop_input + loop_gain * (block_output - loop_input)
        )

    with mock.patch.object(
        model.trajectory,
        "materialize_layer_matrices",
        wraps=model.trajectory.materialize_layer_matrices,
    ) as materialize_spy:
        actual = model._logical_block(inputs, 0)

    assert materialize_spy.call_count == 1
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_looped_forward_backward_reaches_all_hyperblock_coefficients() -> None:
    torch.manual_seed(13)
    model = SheetGPT(_config(loop_count=2, loop_decay=0.8))
    tokens = torch.randint(0, 32, (2, 4))
    _, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.trajectory.coefficients.values()
    )


@pytest.mark.parametrize(
    ("loop_count", "loop_decay", "message"),
    [
        (0, 1.0, "hyperblock_loop_count"),
        (1, 0.0, "hyperblock_loop_decay"),
        (1, 1.01, "hyperblock_loop_decay"),
    ],
)
def test_invalid_loop_controls_are_rejected(
    loop_count: int,
    loop_decay: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(loop_count=loop_count, loop_decay=loop_decay)
# ^^^ THOG
