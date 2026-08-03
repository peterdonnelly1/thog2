# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from run_thog2_owt_core import build_parser, config_from_arguments
from sheet.run_config import OwtRunConfig


ROOT = Path(__file__).resolve().parents[1]


def _plastic_cli(*extra: str):
    return build_parser().parse_args(
        [
            "--model-type",
            "sheet",
            "--select-depth",
            "--basis-family",
            "chebyshev",
            "--n-layer",
            "4",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--o-depth",
            "4",
            "--plastic-enabled",
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


def test_plastic_fixed_cli_resolves_and_preserves_public_identity() -> None:
    config = config_from_arguments(
        _plastic_cli(
            "--plastic-layers-to-sample",
            "6",
            "--plastic-layer-sampling-initialisation",
            "random",
            "--plastic-geometry-learning-rate-multiplier",
            "0.2",
            "--no-plastic-freeze-geometry-during-warmup",
        )
    )
    assert config.plastic__enabled is True
    assert config.n_layer == 6
    assert config.plastic__initial_active_layers == 6
    assert config.plastic__layer_sampling_initialisation == "random"
    assert config.plastic__geometry_learning_rate_multiplier == pytest.approx(0.2)
    assert config.plastic__freeze_geometry_during_warmup is False
    assert "PLN_6_PLM_6_PLI_random_PLO_lowest_loss" in config.artifact_name
    training = config.to_training_config(vocab_size=32, world_size=1, out_dir=Path("out-test"))
    assert training.n_layer == 6
    assert config.compact_identity()["plastic_depth"]["maximum_layers"] == 6
    assert training.compact_identity_metadata()["plastic_depth"]["maximum_layers"] == 6


def test_plastic_learned_count_cli_resolves_maximum_and_initial_count() -> None:
    config = config_from_arguments(
        _plastic_cli(
            "--plastic-do-learn-layer-count",
            "--plastic-initial-layer-count",
            "2",
            "--plastic-max-permitted-layers",
            "7",
            "--plastic-layer-count-objective",
            "layer_efficiency",
            "--plastic-layer-count-cost-weight",
            "0.25",
            "--plastic-layer-count-hold-updates",
            "50",
        )
    )
    assert config.n_layer == 7
    assert config.plastic__initial_active_layers == 2
    assert config.plastic__do_learn_layer_count is True
    assert config.plastic__layer_count_objective == "layer_efficiency"
    assert "PLN_2_PLM_7" in config.artifact_name
    assert "PLH_50" in config.artifact_name


def test_plastic_capability_guards_are_explicit() -> None:
    with pytest.raises(ValueError, match="Chebyshev"):
        OwtRunConfig(
            model_type="sheet",
            geometry_preset="depth",
            basis_family="dct",
            n_layer=4,
            o_depth=4,
            plastic__enabled=True,
        )
    with pytest.raises(ValueError, match="[Hh]yperblock"):
        config_from_arguments(_plastic_cli("--hyperblock"))
    with pytest.raises(ValueError, match="layer dropout"):
        OwtRunConfig(
            model_type="sheet",
            geometry_preset="depth",
            basis_family="chebyshev",
            n_layer=4,
            o_depth=4,
            plastic__enabled=True,
            layer_dropout_stratum_size=4,
            layer_dropout_active_per_stratum=2,
        )
    with pytest.raises(ValueError, match="memory_budget"):
        config_from_arguments(
            _plastic_cli(
                "--plastic-do-learn-layer-count",
                "--plastic-initial-layer-count",
                "2",
                "--plastic-max-permitted-layers",
                "4",
                "--plastic-layer-count-objective",
                "memory_budget",
            )
        )


def test_disabled_artifact_descriptor_is_unchanged() -> None:
    base = OwtRunConfig(
        model_type="sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        n_layer=4,
        o_depth=4,
        max_iters=2,
        warmup_iters=0,
        wandb_enabled=False,
    )
    explicit_disabled = OwtRunConfig(
        model_type="sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        n_layer=4,
        o_depth=4,
        max_iters=2,
        warmup_iters=0,
        wandb_enabled=False,
        plastic__enabled=False,
    )
    assert base.parameter_artifact_fragment() == explicit_disabled.parameter_artifact_fragment()
    assert base.compact_identity() == explicit_disabled.compact_identity()


def test_disabled_persistent_surfaces_omit_plastic_fields() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        n_layer=4,
        n_head=2,
        n_embd=8,
        block_size=8,
        o_depth=4,
        max_iters=2,
        warmup_iters=0,
        activation_checkpointing=False,
        device="cpu",
        dtype="float32",
        wandb_enabled=False,
        wandb_mode="disabled",
    )
    training = config.to_training_config(
        vocab_size=32,
        world_size=1,
        out_dir=Path("out-test"),
    )
    surfaces = (
        config.persistent_dict(),
        config.canonical_dict(world_size=1),
        training.persistent_dict(),
        training.model_arguments(),
    )
    for surface in surfaces:
        assert not any(name.startswith("plastic__") for name in surface), surface


def test_public_wrapper_dry_run_propagates_plastic_controls() -> None:
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            "train_OWT.sh",
            "--plastic-enabled",
            "--plastic-do-learn-layer-count",
            "--plastic-initial-layer-count",
            "2",
            "--plastic-max-permitted-layers",
            "4",
            "--plastic-layer-sampling-initialisation",
            "random",
            "--plastic-layer-count-objective",
            "layer_efficiency",
            "--plastic-layer-count-hold-updates",
            "25",
            "--plastic-layer-count-cost-weight",
            "0.2",
            "--plastic-geometry-learning-rate-multiplier",
            "0.15",
            "--no-plastic-freeze-geometry-during-warmup",
            "--select-depth",
            "-g",
            "PLASTIC_TEST",
            "-n",
            "2",
            "-w",
            "0",
            "-b",
            "1",
            "-A",
            "1",
            "-u",
            "1",
            "-e",
            "1",
            "-l",
            "1",
            "-k",
            "0",
            "-L",
            "4",
            "-H",
            "2",
            "-D",
            "8",
            "-C",
            "8",
            "-P",
            "4",
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
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "plastic depth:" in result.stdout
    dry_run = result.stdout.split("DRY RUN:", 1)[1]
    for expected in (
        "--plastic-enabled",
        "--plastic-do-learn-layer-count",
        "--plastic-initial-layer-count 2",
        "--plastic-max-permitted-layers 4",
        "--plastic-layer-sampling-initialisation random",
        "--plastic-layer-count-objective layer_efficiency",
        "--plastic-layer-count-hold-updates 25",
        "--plastic-layer-count-cost-weight 0.2",
        "--plastic-geometry-learning-rate-multiplier 0.15",
        "--no-plastic-freeze-geometry-during-warmup",
    ):
        assert expected in dry_run


def test_wrapper_help_and_shell_syntax_cover_plastic_depth() -> None:
    subprocess.run(["bash", "-n", "train_OWT.sh"], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", "train_OWT_core.sh"], cwd=ROOT, check=True)
    result = subprocess.run(
        ["bash", "train_OWT.sh", "-h"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "PLASTIC DEPTH:" in result.stdout
    assert "--plastic-layers-to-sample" in result.stdout
    assert "--plastic-do-learn-layer-count" in result.stdout
    assert "--plastic-layer-count-objective" in result.stdout


def test_memory_budget_rejects_cpu_execution() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        config_from_arguments(
            _plastic_cli(
                "--plastic-do-learn-layer-count",
                "--plastic-initial-layer-count",
                "2",
                "--plastic-max-permitted-layers",
                "4",
                "--plastic-layer-count-objective",
                "memory_budget",
                "--plastic-layer-memory-budget-gib",
                "4",
            )
        )


# ^^^ THOG
