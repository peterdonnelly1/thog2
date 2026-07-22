# vvv THOG
from __future__ import annotations

import copy
import os
from pathlib import Path
from unittest import mock

import pytest
import torch

from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK, GEOMETRY_PRESET_HEAD_AWARE_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import ATTENTION_KEY_WEIGHT, ATTENTION_OUTPUT_WEIGHT, ATTENTION_QUERY_WEIGHT, ATTENTION_VALUE_WEIGHT

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_NAMES = (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT, ATTENTION_OUTPUT_WEIGHT)


def _config(*, vectorised: bool) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=4,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=3,
        base_row_order=8,
        o_attn_d_model=8,
        o_attn_qkv_per_channel=4,
        o_attn_out_per_channel=4,
        o_mlp_d_model=8,
        o_mlp_hidden=20,
        geometry_preset=GEOMETRY_PRESET_FULL_BLOCK,
        vectorise_per_head_materialisation=vectorised,
    )


def test_vectorise_per_head_materialisation_defaults_true_and_environment_can_disable_it() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("THOG2_VECTORISE_PER_HEAD_MATERIALISATION", None)
        assert SheetGPTConfig().vectorise_per_head_materialisation is True
    with mock.patch.dict(os.environ, {"THOG2_VECTORISE_PER_HEAD_MATERIALISATION": "false"}):
        assert SheetGPTConfig().vectorise_per_head_materialisation is False


@pytest.mark.parametrize("name", ATTENTION_NAMES)
@pytest.mark.parametrize("layer_index", [0, 2, 3])
def test_vectorised_and_loop_materialisation_match_for_every_head_aware_family_and_multiple_layers(name: str, layer_index: int) -> None:
    torch.manual_seed(8801)
    vectorised = SheetGPT(_config(vectorised=True)).double().eval()
    looped = copy.deepcopy(vectorised)
    looped.trajectory.vectorise_per_head_materialisation = False
    torch.testing.assert_close(
        vectorised.trajectory.materialize(name, layer_index),
        looped.trajectory.materialize(name, layer_index),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_vectorised_and_loop_attention_forward_input_and_all_attention_coefficient_gradients_match() -> None:
    torch.manual_seed(8802)
    vectorised = SheetGPT(_config(vectorised=True)).double().eval()
    looped = copy.deepcopy(vectorised)
    looped.trajectory.vectorise_per_head_materialisation = False
    vectorised_inputs = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    looped_inputs = vectorised_inputs.detach().clone().requires_grad_(True)
    vectorised_output = vectorised._attention(vectorised_inputs, 2)
    looped_output = looped._attention(looped_inputs, 2)
    torch.testing.assert_close(vectorised_output, looped_output, rtol=1.0e-11, atol=1.0e-11)
    vectorised_output.square().mean().backward()
    looped_output.square().mean().backward()
    torch.testing.assert_close(vectorised_inputs.grad, looped_inputs.grad, rtol=1.0e-10, atol=1.0e-10)
    for name in ATTENTION_NAMES:
        torch.testing.assert_close(vectorised.trajectory.coefficients[name].grad, looped.trajectory.coefficients[name].grad, rtol=1.0e-10, atol=1.0e-10)


def test_loop_option_really_executes_one_depth_contraction_per_head_while_vectorised_path_executes_one() -> None:
    model = SheetGPT(_config(vectorised=True)).eval()
    coefficient = model.trajectory.coefficients[ATTENTION_QUERY_WEIGHT]
    original_einsum = torch.einsum
    with mock.patch("torch.einsum", wraps=original_einsum) as einsum_spy:
        model.trajectory.materialize(ATTENTION_QUERY_WEIGHT, 1)
    assert sum(call.args[0] == "p,hpab->hab" for call in einsum_spy.call_args_list) == 1
    model.trajectory.vectorise_per_head_materialisation = False
    with mock.patch("torch.einsum", wraps=original_einsum) as einsum_spy:
        model.trajectory.materialize(ATTENTION_QUERY_WEIGHT, 1)
    assert sum(call.args[0] == "p,pab->ab" for call in einsum_spy.call_args_list) == model.config.n_head
    assert coefficient.shape[0] == model.config.n_head


def test_training_wrappers_expose_default_on_vectorise_per_head_materialisation_without_getopts_letter() -> None:
    for wrapper_name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        source = (REPOSITORY_ROOT / wrapper_name).read_text(encoding="utf-8")
        assert 'VECTORISE_PER_HEAD_MATERIALISATION="${THOG2_VECTORISE_PER_HEAD_MATERIALISATION:-true}"' in source
        assert 'export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="$VECTORISE_PER_HEAD_MATERIALISATION"' in source
        assert "vectorise per-head materialisation:" in source
# ^^^ THOG
