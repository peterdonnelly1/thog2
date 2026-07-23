# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from run_thog2_owt import build_parser, config_from_arguments
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.run_config import OwtRunConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PICTON_ORDER_FLAGS = {
    "-P": "--o-depth",
    "-Q": "--o-attn-d-model",
    "-J": "--o-attn-qkv-per-channel",
    "-O": "--o-attn-out-per-channel",
    "-X": "--o-mlp-d-model",
    "-Y": "--o-mlp-hidden",
}
PICTON_CANONICAL_WRAPPERS = (
    REPOSITORY_ROOT / "current_scruffy_train_OWT.sh",
    REPOSITORY_ROOT / "current_dreedle_train_OWT.sh",
)
PICTON_ALL_WRAPPERS = (
    *PICTON_CANONICAL_WRAPPERS,
    REPOSITORY_ROOT / "current_scruffy_train_DENSE_OWT.sh",
    REPOSITORY_ROOT / "current_scruffy_train_SHEET_OWT.sh",
    REPOSITORY_ROOT / "run_scruffy_dense_sheet_l144_smoke_owt.sh",
    REPOSITORY_ROOT / "run_scruffy_sheet_eden_best_long_owt.sh",
)


def test_picton_runner_parser_accepts_all_final_presets_and_six_semantic_orders() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--model-type",
            "sheet",
            "--geometry-preset",
            "full_block",
            "--n-layer",
            "12",
            "--n-head",
            "3",
            "--n-embd",
            "12",
            "--o-depth",
            "2",
            "--o-attn-d-model",
            "7",
            "--o-attn-qkv-per-channel",
            "3",
            "--o-attn-out-per-channel",
            "2",
            "--o-mlp-d-model",
            "5",
            "--o-mlp-hidden",
            "11",
            "--max-iters",
            "20",
            "--warmup-iters",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    config = config_from_arguments(arguments)
    assert config.geometry_preset == GEOMETRY_PRESET_FULL_BLOCK
    assert config.o_depth == 2
    assert config.o_attn_d_model == 7
    assert config.o_attn_qkv_per_channel == 3
    assert config.o_attn_out_per_channel == 2
    assert config.o_mlp_d_model == 5
    assert config.o_mlp_hidden == 11


@pytest.mark.parametrize(
    "retired_arguments",
    (
        ("--geometry-preset", "curve"),
        ("--geometry-preset", "block"),
        ("--depth-order", "2"),
        ("--base-row-order", "7"),
        ("--mlp-channel-order", "11"),
    ),
)
def test_picton_runner_parser_rejects_retired_public_names_and_flags(retired_arguments: tuple[str, str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-type", "sheet", *retired_arguments])


def test_picton_artifact_identity_records_the_complete_order_vector() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        host_label="test",
        experiment_prefix="PICTON",
        run_start_label="260712-120000",
        max_iters=20,
        warmup_iters=1,
        eval_interval=10,
        n_layer=12,
        n_head=3,
        n_embd=12,
        o_depth=2,
        o_attn_d_model=7,
        o_attn_qkv_per_channel=3,
        o_attn_out_per_channel=2,
        o_mlp_d_model=5,
        o_mlp_hidden=11,
        geometry_preset=GEOMETRY_PRESET_FULL_BLOCK,
        device="cpu",
        dtype="float32",
        residual_init_depth_source="dof_implied_depth",
    )
    assert "PICTON" in config.artifact_name
    parameter_fragment = config.parameter_artifact_fragment()
    for fragment in ("P_2", "Q_7", "J_3", "O_2", "X_5", "Y_11"):
        assert fragment in parameter_fragment
    identity = config.compact_identity()
    assert identity["o_depth"] == 2
    assert identity["o_attn_d_model"] == 7
    assert identity["o_attn_qkv_per_channel"] == 3
    assert identity["o_attn_out_per_channel"] == 2
    assert identity["o_mlp_d_model"] == 5
    assert identity["o_mlp_hidden"] == 11


def test_picton_canonical_host_wrappers_share_the_same_getopts_letters_and_long_options() -> None:
    contents = [path.read_text(encoding="utf-8") for path in PICTON_CANONICAL_WRAPPERS]
    getopts_lines = [next(line for line in content.splitlines() if "while getopts" in line) for content in contents]
    assert getopts_lines[0] == getopts_lines[1]
    for content in contents:
        for short_flag, long_flag in PICTON_ORDER_FLAGS.items():
            assert f"{short_flag} " in content
            assert long_flag in content
        for preset in ("dense", "legacy_sheet_col", "depth", "mlp_block", "head_aware_block", "full_block"):
            assert preset in content


def test_picton_all_known_shell_wrappers_are_syntactically_valid() -> None:
    for wrapper in PICTON_ALL_WRAPPERS:
        assert wrapper.is_file(), wrapper
        completed = subprocess.run(
            ["bash", "-n", str(wrapper)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, f"{wrapper.name}: {completed.stderr}"
# ^^^ THOG
