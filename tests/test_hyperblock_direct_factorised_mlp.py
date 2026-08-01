# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from sheet.hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    apply_factorised_hyperblock_mlp,
)
from sheet.model import SheetGPT, SheetGPTConfig


ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    direct: bool,
    fast_discard: bool = True,
) -> SheetGPTConfig:
    return SheetGPTConfig(
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
        hyperblock_common_family_order=3,
        hyperblock_attention_family_order=2,
        hyperblock_mlp_family_order=1,
        hyperblock_depth_order=2,
        hyperblock_d_model_order=4,
        hyperblock_mlp_hidden_order=5,
        hyperblock_attention_head_order=2,
        hyperblock_attention_head_channel_order=4,
        direct_factorised_hyperblock_mlp=direct,
        fast_discard=fast_discard,
    )


def _linear_case(
    model: SheetGPT,
    *,
    expansion: bool,
) -> Tuple[str, str, int, Tensor]:
    if expansion:
        return (
            "mlp_expansion_weight",
            "mlp_expansion_bias",
            0,
            torch.randn(2, 3, model.config.n_embd, dtype=torch.float64),
        )
    return (
        "mlp_contraction_weight",
        "mlp_contraction_bias",
        1,
        torch.randn(
            2,
            3,
            model.config.hyperblock_mlp_hidden_multiplier * model.config.n_embd,
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("expansion", (True, False))
def test_factorised_hyperblock_mlp_matches_materialised_value_and_gradients(
    expansion: bool,
) -> None:
    torch.manual_seed(901)
    dense_model = SheetGPT(_config(direct=False)).double()
    direct_model = SheetGPT(_config(direct=True)).double()
    direct_model.load_state_dict(dense_model.state_dict())

    weight_name, bias_name, family_index, dense_inputs = _linear_case(
        dense_model,
        expansion=expansion,
    )
    direct_inputs = dense_inputs.detach().clone()
    dense_inputs.requires_grad_(True)
    direct_inputs.requires_grad_(True)
    layer_index = 1

    dense_weight = dense_model.trajectory.materialize(weight_name, layer_index)
    dense_bias = dense_model.trajectory.materialize_vector(bias_name, layer_index)
    dense_output = F.linear(dense_inputs, dense_weight, dense_bias)

    direct_bias = direct_model.trajectory.materialize_vector(bias_name, layer_index)
    direct_output = apply_factorised_hyperblock_mlp(
        direct_inputs,
        direct_model.trajectory.factorised_mlp_layer(layer_index),
        family_index=family_index,
        expansion=expansion,
        bias=direct_bias,
    )
    torch.testing.assert_close(direct_output, dense_output, rtol=1.0e-9, atol=1.0e-10)

    upstream = torch.randn_like(dense_output)
    dense_targets = (
        dense_inputs,
        dense_model.trajectory.coefficients["common"],
        dense_model.trajectory.coefficients["mlp"],
        dense_model.trajectory.vector_parameters[bias_name],
    )
    direct_targets = (
        direct_inputs,
        direct_model.trajectory.coefficients["common"],
        direct_model.trajectory.coefficients["mlp"],
        direct_model.trajectory.vector_parameters[bias_name],
    )
    dense_gradients = torch.autograd.grad(
        dense_output,
        dense_targets,
        grad_outputs=upstream,
    )
    direct_gradients = torch.autograd.grad(
        direct_output,
        direct_targets,
        grad_outputs=upstream,
    )
    for dense_gradient, direct_gradient in zip(dense_gradients, direct_gradients):
        torch.testing.assert_close(
            direct_gradient,
            dense_gradient,
            rtol=2.0e-8,
            atol=2.0e-9,
        )


@pytest.mark.parametrize("fast_discard", (False, True))
def test_whole_model_direct_option_matches_dense_mlp_path(
    fast_discard: bool,
) -> None:
    torch.manual_seed(902)
    dense_model = SheetGPT(_config(direct=False, fast_discard=fast_discard))
    direct_model = SheetGPT(_config(direct=True, fast_discard=fast_discard))
    direct_model.load_state_dict(dense_model.state_dict())

    include_mlp_calls = []
    original_materialize_layer_matrices = direct_model.trajectory.materialize_layer_matrices

    def materialize_layer_matrices(
        layer_index: int,
        *,
        include_mlp: bool = True,
    ) -> Dict[str, Tensor]:
        include_mlp_calls.append(include_mlp)
        values = original_materialize_layer_matrices(
            layer_index,
            include_mlp=include_mlp,
        )
        if not include_mlp:
            assert "mlp_expansion_weight" not in values
            assert "mlp_contraction_weight" not in values
        return values

    direct_model.trajectory.materialize_layer_matrices = materialize_layer_matrices
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    dense_logits, dense_loss = dense_model(tokens, tokens)
    direct_logits, direct_loss = direct_model(tokens, tokens)
    assert dense_loss is not None and direct_loss is not None
    torch.testing.assert_close(direct_logits, dense_logits, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(direct_loss, dense_loss, rtol=2.0e-6, atol=2.0e-7)

    dense_loss.backward()
    direct_loss.backward()
    dense_gradients = {
        name: parameter.grad
        for name, parameter in dense_model.named_parameters()
    }
    direct_gradients = {
        name: parameter.grad
        for name, parameter in direct_model.named_parameters()
    }
    assert dense_gradients.keys() == direct_gradients.keys()
    for name in dense_gradients:
        assert dense_gradients[name] is not None
        assert direct_gradients[name] is not None
        torch.testing.assert_close(
            direct_gradients[name],
            dense_gradients[name],
            rtol=3.0e-4,
            atol=3.0e-6,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )
    assert include_mlp_calls == [False, False]


def test_direct_hyperblock_mlp_option_rejects_non_hyperblock_model() -> None:
    with pytest.raises(ValueError, match="requires HYPERBLOCK"):
        SheetGPTConfig(
            block_size=4,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            direct_factorised_hyperblock_mlp=True,
        )


def test_wrapper_exposes_default_off_boolean_option() -> None:
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            "train_OWT.sh",
            "--hyperblock",
            "--direct-factorised-hyperblock-mlp",
            "--hyperblock-depth-order",
            "2",
            "--hyperblock-d-model-order",
            "4",
            "--hyperblock-mlp-hidden-order",
            "4",
            "--hyperblock-attention-head-order",
            "2",
            "--hyperblock-attention-head-channel-order",
            "4",
            "-n",
            "2",
            "-w",
            "0",
            "-b",
            "1",
            "-A",
            "1",
            "-L",
            "2",
            "-H",
            "2",
            "-D",
            "8",
            "-C",
            "8",
            "-S",
            "1",
            "-I",
            "none",
            "-F",
            "none",
            "-T",
            "float32",
            "-K",
            "math",
            "-x",
            "true",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert re.search(
        r"direct factorised HYPERBLOCK MLP: +true",
        result.stdout,
    )
    help_result = subprocess.run(
        ["bash", "train_OWT.sh", "-h"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--direct-factorised-hyperblock-mlp" in help_result.stdout
    assert "--no-direct-factorised-hyperblock-mlp" in help_result.stdout
# ^^^ THOG
