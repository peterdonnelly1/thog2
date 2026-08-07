# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from run_thog2_owt_core import build_parser, config_from_arguments
from sheet.checkpoints import _semantic_compact_identity
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
            "--plastic__enabled",
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
            "--plastic__layers_to_sample",
            "6",
            "--plastic__layer_sampling_initialisation",
            "random",
            "--plastic__geometry_learning_rate_multiplier",
            "0.2",
            "--no-plastic__freeze_geometry_during_warmup",
        )
    )
    assert config.plastic__enabled is True
    assert config.n_layer == 6
    assert config.plastic__initial_active_layers == 6
    assert config.plastic__layer_sampling_initialisation == "random"
    assert config.plastic__geometry_learning_rate_multiplier == pytest.approx(0.2)
    assert config.plastic__freeze_geometry_during_warmup is False
    assert "_L_6__" in config.parameter_artifact_fragment()
    assert "_L_dyn__" not in config.parameter_artifact_fragment()
    assert "P__LN_6_LM_6_LI_rndm_LO_loss" in config.artifact_name
    training = config.to_training_config(vocab_size=32, world_size=1, out_dir=Path("out-test"))
    assert training.n_layer == 6
    run_identity = config.compact_identity()["plastic_depth"]
    training_identity = training.compact_identity_metadata()["plastic_depth"]
    assert run_identity["plastic__layers_to_sample"] == 6
    assert run_identity["plastic__initial_active_layers"] == 6
    assert training_identity["plastic__layers_to_sample"] == 6
    assert training_identity["plastic__initial_active_layers"] == 6


def test_plastic_learned_count_cli_resolves_maximum_and_initial_count() -> None:
    config = config_from_arguments(
        _plastic_cli(
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "7",
            "--plastic__layer_count_objective",
            "layer_efficiency",
            "--plastic__layer_count_cost_weight",
            "0.25",
            "--plastic__layer_count_update_brake",
            "50",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "20",
            "--plastic__layer_count_probe_noise_lambda",
            "2.5",
        )
    )
    assert config.n_layer == 7
    assert config.plastic__initial_active_layers == 2
    assert config.plastic__do_learn_layer_count is True
    assert config.plastic__layer_count_objective == "layer_efficiency"
    assert "_L_dyn__" in config.parameter_artifact_fragment()
    assert "_L_7__" not in config.parameter_artifact_fragment()
    assert "P__LN_2_LM_7" in config.artifact_name
    assert "LB_50_LNW_20_LNL_250000" in config.artifact_name


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
                "--plastic__do_learn_layer_count",
                "--plastic__initial_layer_count",
                "2",
                "--plastic__max_permitted_layers",
                "4",
                "--plastic__layer_count_objective",
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
            "--plastic__enabled",
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "4",
            "--plastic__layer_sampling_initialisation",
            "random",
            "--plastic__layer_count_objective",
            "layer_efficiency",
            "--plastic__layer_count_update_brake",
            "25",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "20",
            "--plastic__layer_count_probe_noise_lambda",
            "2.5",
            "--plastic__layer_count_cost_weight",
            "0.2",
            "--plastic__cuda_allocator_reserve_gib",
            "0.75",
            "--plastic__geometry_learning_rate_multiplier",
            "0.15",
            "--no-plastic__freeze_geometry_during_warmup",
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
        "--plastic__enabled",
        "--plastic__do_learn_layer_count",
        "--plastic__initial_layer_count 2",
        "--plastic__max_permitted_layers 4",
        "--plastic__layer_sampling_initialisation random",
        "--plastic__layer_count_objective layer_efficiency",
        "--plastic__layer_count_update_brake 25",
        "--plastic__layer_count_probe__window_size_as_number_of_probes 20",
        "--plastic__layer_count_probe_noise_lambda 2.5",
        "--plastic__layer_count_cost_weight 0.2",
        "--plastic__cuda_allocator_reserve_gib 0.75",
        "--plastic__geometry_learning_rate_multiplier 0.15",
        "--no-plastic__freeze_geometry_during_warmup",
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
    assert "PLASTIC DEPTH COARSE/FINE:" in result.stdout
    assert "--plastic__layers_to_sample" in result.stdout
    assert "--plastic__do_learn_layer_count" in result.stdout
    assert "--plastic__layer_count_objective" in result.stdout
    assert "--plastic__layer_count_update_brake" in result.stdout
    assert "--plastic__layer_count_probe__window_size_as_number_of_probes" in result.stdout
    assert "--plastic__cuda_allocator_reserve_gib" in result.stdout


def test_memory_budget_rejects_cpu_execution() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        config_from_arguments(
            _plastic_cli(
                "--plastic__do_learn_layer_count",
                "--plastic__initial_layer_count",
                "2",
                "--plastic__max_permitted_layers",
                "4",
                "--plastic__layer_count_objective",
                "memory_budget",
                "--plastic__layer_memory_budget_gib",
                "4",
            )
        )


# ^^^ THOG

# vvv THOG CUDA reserve is persistent execution configuration but deliberately not yet an artifact-name component
def test_cuda_allocator_reserve_cli_propagates_without_changing_artifact_identity() -> None:
    default_config = config_from_arguments(
        _plastic_cli(
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "4",
        )
    )
    configured = config_from_arguments(
        _plastic_cli(
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "4",
            "--plastic__cuda_allocator_reserve_gib",
            "0.75",
        )
    )
    assert default_config.plastic__cuda_allocator_reserve_gib == pytest.approx(0.5)
    assert configured.plastic__cuda_allocator_reserve_gib == pytest.approx(0.75)
    assert default_config.artifact_name == configured.artifact_name
    training = configured.to_training_config(
        vocab_size=32,
        world_size=1,
        out_dir=Path("out-test"),
    )
    assert training.plastic__cuda_allocator_reserve_gib == pytest.approx(0.75)
    assert configured.compact_identity()["plastic_depth"]["plastic__cuda_allocator_reserve_gib"] == pytest.approx(0.75)
    assert training.compact_identity_metadata()["plastic_depth"]["plastic__cuda_allocator_reserve_gib"] == pytest.approx(0.75)


def test_cuda_allocator_reserve_rejects_negative_cli_value() -> None:
    with pytest.raises(ValueError, match="cuda_allocator_reserve"):
        config_from_arguments(
            _plastic_cli(
                "--plastic__do_learn_layer_count",
                "--plastic__initial_layer_count",
                "2",
                "--plastic__max_permitted_layers",
                "4",
                "--plastic__cuda_allocator_reserve_gib",
                "-0.1",
            )
        )
# ^^^ THOG


# vvv THOG canonical public-control and dynamic artifact identity coverage

def test_persisted_plastic_identity_uses_only_canonical_public_control_names() -> None:
    config = config_from_arguments(
        _plastic_cli(
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "7",
            "--plastic__layer_sampling_initialisation",
            "random",
            "--plastic__layer_count_objective",
            "relative_training_wall_time",
            "--plastic__layer_count_update_brake",
            "11",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "19",
            "--plastic__layer_count_probe_noise_lambda",
            "1.5",
            "--plastic__layer_count_cost_weight",
            "0.2",
            "--plastic__cuda_allocator_reserve_gib",
            "0.75",
            "--plastic__geometry_learning_rate_multiplier",
            "0.25",
            "--no-plastic__freeze_geometry_during_warmup",
        )
    )
    training = config.to_training_config(vocab_size=32, world_size=1, out_dir=Path("out-test"))
    identities = (
        config.compact_identity()["plastic_depth"],
        training.compact_identity_metadata()["plastic_depth"],
    )
    obsolete_names = {
        "maximum_layers",
        "initial_active_layers",
        "learn_layer_count",
        "sampling_initialisation",
        "count_objective",
        "count_update_brake",
        "probe_noise_window",
        "probe_noise_min_observations",
        "probe_noise_lambda",
        "count_cost_weight",
        "memory_budget_gib",
        "cuda_allocator_reserve_gib",
        "geometry_lr_multiplier",
        "freeze_geometry_during_warmup",
    }
    for identity in identities:
        assert set(identity).isdisjoint(obsolete_names)
        assert identity["version"] == "plastic_depth_v0_3"
        public_names = set(identity) - {"version", "plastic__initial_active_layers"}
        assert public_names
        assert all(name.startswith("plastic__") for name in public_names)
        assert "plastic__sampling_seed" not in identity
        assert identity["plastic__do_learn_layer_count"] is True
        assert identity["plastic__initial_layer_count"] == 2
        assert identity["plastic__max_permitted_layers"] == 7


def test_retired_short_identity_aliases_remain_checkpoint_compatible() -> None:
    config = config_from_arguments(
        _plastic_cli(
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "7",
            "--plastic__layer_count_update_brake",
            "11",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "19",
            "--plastic__layer_count_probe_noise_lambda",
            "1.5",
            "--plastic__layer_count_cost_weight",
            "0.2",
            "--plastic__cuda_allocator_reserve_gib",
            "0.75",
            "--plastic__geometry_learning_rate_multiplier",
            "0.25",
            "--no-plastic__freeze_geometry_during_warmup",
        )
    )
    canonical = config.compact_identity()
    retired = dict(canonical)
    retired["plastic_depth"] = {
        "version": "plastic_depth_v0_3",
        "maximum_layers": 7,
        "initial_active_layers": 2,
        "learn_layer_count": True,
        "sampling_initialisation": "equidistant",
        "count_objective": "lowest_loss",
        "count_update_brake": 11,
        "probe_noise_window": 19,
        "probe_noise_min_observations": 3,
        "probe_noise_lambda": 1.5,
        "count_cost_weight": 0.2,
        "memory_budget_gib": None,
        "cuda_allocator_reserve_gib": 0.75,
        "geometry_lr_multiplier": 0.25,
        "freeze_geometry_during_warmup": False,
    }
    assert _semantic_compact_identity(retired) == _semantic_compact_identity(canonical)


def test_obsolete_hold_control_and_internal_sampling_seed_are_not_public_cli_options() -> None:
    help_text = build_parser().format_help()
    assert "--plastic-layer-count-hold-updates" not in help_text
    assert "--plastic-sampling-seed" not in help_text
    assert "--plastic__sampling_seed" not in help_text
# ^^^ THOG
