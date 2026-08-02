# vvv THOG
from __future__ import annotations

from pathlib import Path

import pytest

from run_thog2_owt_core import build_parser, config_from_arguments
from sheet.hyperblock import HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
from sheet.run_config import OwtRunConfig


def _cli_arguments(*extra: str):
    return build_parser().parse_args(
        [
            "--hyperblock",
            "--model-type",
            "sheet",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--hyperblock-common-family-order",
            "3",
            "--hyperblock-attention-family-order",
            "2",
            "--hyperblock-mlp-family-order",
            "1",
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
            "--hyperblock-loop-count",
            "3",
            "--hyperblock-loop-decay",
            "0.8",
            "--max-iters",
            "2",
            "--warmup-iters",
            "0",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            *extra,
        ]
    )


def test_hyperblock_cli_resolves_topology_and_training_config() -> None:
    config = config_from_arguments(_cli_arguments())
    assert config.hyperblock_enabled is True
    assert config.hyperblock_topology == HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
    assert config.geometry_preset is None
    assert config.basis_family is None
    assert config.hyperblock_plan().coefficient_counts["total"] == 160
    assert config.hyperblock_common_family_order == 3
    assert config.hyperblock_attention_family_order == 2
    assert config.hyperblock_mlp_family_order == 1
    assert "HB_chebyshev" in config.artifact_name
    assert "HB_SMOKE" not in config.artifact_name
    assert "___HB_chebyshev" in config.run_descriptor()
    assert "HFC_3_HFA_2_HFM_1_HL_2_HD_4_HM_4_HH_2_HC_4_HLC_3_HLD_0p8" in config.artifact_name
    assert config.hyperblock_loop_count == 3
    assert config.hyperblock_loop_decay == pytest.approx(0.8)
    training = config.to_training_config(
        vocab_size=32,
        world_size=1,
        out_dir=Path("out-test"),
    )
    assert training.hyperblock_enabled is True
    assert training.hyperblock_plan() == config.hyperblock_plan()
    assert training.hyperblock_loop_count == 3
    assert training.hyperblock_loop_decay == pytest.approx(0.8)


def test_hyperblock_cli_implies_sheet_model_type() -> None:
    arguments = _cli_arguments()
    arguments.model_type = None
    config = config_from_arguments(arguments)
    assert config.model_type == "sheet"


def test_hyperblock_rejects_selector_geometry() -> None:
    with pytest.raises(ValueError, match="selector-based geometry"):
        config_from_arguments(_cli_arguments("--select-depth"))


def test_hyperblock_run_config_rejects_legacy_defaults_when_constructed_directly() -> None:
    with pytest.raises(ValueError, match="legacy geometry controls"):
        OwtRunConfig(
            model_type="sheet",
            hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        )
# ^^^ THOG
