# vvv THOG
from __future__ import annotations

import copy
import os
from pathlib import Path
from unittest import mock

import pytest
import torch

from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK, GEOMETRY_PRESET_MLP_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MLP_WEIGHT_NAMES = (MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT)


def _config(preset: str, *, direct: bool) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
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
        geometry_preset=preset,
        direct_factorised_mlp=direct,
    )


# def test_direct_thog_mlp_application_defaults_false_and_environment_can_enable_it() -> None:
#     ...
def test_direct_factorised_mlp_defaults_true_and_environment_can_disable_it() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("THOG2_DIRECT_FACTORISED_MLP", None)
        assert SheetGPTConfig().direct_factorised_mlp is True
    with mock.patch.dict(os.environ, {"THOG2_DIRECT_FACTORISED_MLP": "false"}):
        assert SheetGPTConfig().direct_factorised_mlp is False


@pytest.mark.parametrize("preset", [GEOMETRY_PRESET_MLP_BLOCK, GEOMETRY_PRESET_FULL_BLOCK])
def test_direct_factorised_mlp_matches_materialised_forward_input_gradients_and_coefficient_gradients(preset: str) -> None:
    torch.manual_seed(7701)
    materialised = SheetGPT(_config(preset, direct=False)).double().eval()
    direct = copy.deepcopy(materialised)
    direct.config.direct_factorised_mlp = True

    materialised_inputs = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    direct_inputs = materialised_inputs.detach().clone().requires_grad_(True)
    materialised_output = materialised._mlp(materialised_inputs, layer_index=2)
    direct_output = direct._mlp(direct_inputs, layer_index=2)
    torch.testing.assert_close(direct_output, materialised_output, rtol=1.0e-11, atol=1.0e-11)

    materialised_objective = materialised_output.square().mean() + 0.125 * materialised_output.sum()
    direct_objective = direct_output.square().mean() + 0.125 * direct_output.sum()
    materialised_objective.backward()
    direct_objective.backward()
    torch.testing.assert_close(direct_inputs.grad, materialised_inputs.grad, rtol=1.0e-10, atol=1.0e-10)
    for name in MLP_WEIGHT_NAMES:
        torch.testing.assert_close(
            direct.trajectory.coefficients[name].grad,
            materialised.trajectory.coefficients[name].grad,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


@pytest.mark.parametrize("preset", [GEOMETRY_PRESET_MLP_BLOCK, GEOMETRY_PRESET_FULL_BLOCK])
def test_direct_factorised_mlp_avoids_dense_mlp_weight_materialisation_for_both_mlp_matrices(preset: str) -> None:
    torch.manual_seed(7702)
    model = SheetGPT(_config(preset, direct=True)).eval()
    inputs = torch.randn(2, 4, 16)
    original_materialise = model.trajectory.materialize
    with mock.patch.object(model.trajectory, "materialize", wraps=original_materialise) as materialise_spy:
        output = model._mlp(inputs, layer_index=1)
    assert tuple(output.shape) == tuple(inputs.shape)
    requested_names = [call.args[0] for call in materialise_spy.call_args_list]
    assert MLP_EXPANSION_WEIGHT not in requested_names
    assert MLP_CONTRACTION_WEIGHT not in requested_names


def test_direct_factorised_mlp_falls_back_to_materialisation_when_geometry_has_no_mlp_block() -> None:
    config = SheetGPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=3,
        base_row_order=8,
        geometry_preset="depth",
        direct_factorised_mlp=True,
    )
    model = SheetGPT(config).eval()
    inputs = torch.randn(2, 4, 16)
    original_materialise = model.trajectory.materialize
    with mock.patch.object(model.trajectory, "materialize", wraps=original_materialise) as materialise_spy:
        model._mlp(inputs, layer_index=1)
    requested_names = [call.args[0] for call in materialise_spy.call_args_list]
    assert requested_names.count(MLP_EXPANSION_WEIGHT) == 1
    assert requested_names.count(MLP_CONTRACTION_WEIGHT) == 1


# def test_training_wrappers_expose_default_off_direct_thog_mlp_application_without_getopts_letter() -> None:
#     ...
def test_training_wrappers_expose_default_on_direct_factorised_mlp_without_getopts_letter() -> None:
    for wrapper_name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        source = (REPOSITORY_ROOT / wrapper_name).read_text(encoding="utf-8")
        assert 'DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"' in source
        assert 'export THOG2_DIRECT_FACTORISED_MLP="$DIRECT_FACTORISED_MLP"' in source
        assert "direct factorised MLP:" in source
        assert "THOG2_DIRECT_THOG_MLP_APPLICATION" in source  # retained only as commented source history
        active_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        assert not any("DIRECT_THOG_MLP_APPLICATION" in line for line in active_lines)
# ^^^ THOG
