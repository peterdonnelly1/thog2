# vvv THOG
from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import constants
import run_thog2_owt as public_runner
from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
from sheet import plastic_depth_same_batch_visibility_patch as visibility
from sheet import stage6_trainer as stage6
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_training_config


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ANSI_ESCAPE = visibility._ANSI_ESCAPE


@pytest.fixture(autouse=True)
def _isolated_same_batch_environment(monkeypatch):
    monkeypatch.delenv(same_batch._RUNTIME_ENV, raising=False)
    monkeypatch.delenv(same_batch._EXPLICIT_ENV, raising=False)
    visibility._SAMPLED_BY_RUN_ID.clear()
    try:
        yield
    finally:
        os.environ.pop(same_batch._RUNTIME_ENV, None)
        os.environ.pop(same_batch._EXPLICIT_ENV, None)
        visibility._SAMPLED_BY_RUN_ID.clear()


def _trainer(*, window_size: int = 2) -> SharedTrainer:
    same_batch._set_runtime_enabled(True)
    train_tokens, validation_tokens = token_splits(length=2048)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_objective="lowest_loss",
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=window_size,
        plastic__layer_count_probe_noise_lambda=1.0e9,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
        gradient_accumulation_steps=1,
        batch_size=4,
        block_size=8,
        max_updates=8,
        warmup_updates=0,
        checkpoint_segment_size=0,
        device="cpu",
        dtype="float32",
    )
    return SharedTrainer(config, train_tokens, validation_tokens)


def _progress_line_from_audit(row: dict) -> str:
    ordinal = int(row["probe_window_ordinal"])
    window_size = int(row["probe_window_size"])
    payload = {
        "completed_updates": f"{int(row['update_number']):6d}",
        "timestamp": "260809:1300",
        "cumulative_training_seconds": "       10",
        "mean_step_seconds": "  1.0000",
        "tok/s": "       10000",
        "consumed_tokens": "       10000",
        "training_loss": "  7.0000",
        "training_loss_delta": "  -0.001",
        "learning_rate": " 9.000e-04",
        "gradient_norm": "   5.921",
        "current_layer_count": 3,
        "plastic_probe_offsets": (-1, 0, 1),
        "plastic_probe_edge_offsets": (-1, 1),
        "plastic_probe_losses": (7.001, 7.0, 6.999),
        "plastic_score_z": (None, None),
        "plastic_lra_summary": (0, ordinal, 0, ordinal, "R" if ordinal >= window_size else "-"),
        "plastic_lra_left_wins": (),
        "plastic_lra_right_wins": (),
        "plastic_probe_sequence": ordinal,
        "plastic_probe_provenance": tuple(row["probe_window_provenance"]),
        "plastic_probe_decision_ready": ordinal >= window_size,
        "plastic_same_batch_window_id": int(row["probe_window_id"]),
        "plastic_same_batch_window_ordinal": ordinal,
        "plastic_same_batch_window_size": window_size,
        "plastic_same_batch_batch_digest": str(row["probe_batch_digest"]),
    }
    return stage6.format_progress_line("visibility-test", "optimizer_progress", payload)


def _plain(line: str) -> str:
    return _ANSI_ESCAPE.sub("", line)


def _ordinary_progress_line(run_id: str, sample_points: tuple[float, ...], gradient_norm: str = "   5.921") -> str:
    payload = {
        "completed_updates": "     7",
        "timestamp": "260809:1400",
        "cumulative_training_seconds": "       42",
        "mean_step_seconds": "  5.6000",
        "tok/s": "       13000",
        "consumed_tokens": "      516096",
        "training_loss": "  7.5000",
        "training_loss_delta": "  -0.001",
        "learning_rate": " 9.000e-04",
        "gradient_norm": gradient_norm,
        "current_layer_count": len(sample_points),
        "depth_sample_points": sample_points,
    }
    return stage6.format_progress_line(run_id, "optimizer_progress", payload)


def test_same_batch_visible_probe_number_resets_with_fresh_batch() -> None:
    trainer = _trainer(window_size=2)
    try:
        trainer.train_one_update()
        first_audit = trainer.plastic_depth_count_audit[-1]
        first_line = _progress_line_from_audit(first_audit)

        trainer.train_one_update()
        second_audit = trainer.plastic_depth_count_audit[-1]
        second_line = _progress_line_from_audit(second_audit)

        trainer.train_one_update()
        third_audit = trainer.plastic_depth_count_audit[-1]
        third_line = _progress_line_from_audit(third_audit)

        first_plain = _plain(first_line)
        second_plain = _plain(second_line)
        third_plain = _plain(third_line)
        assert "P1  probe_Δloss" in first_plain
        assert "P2  probe_Δloss" in second_plain
        assert "P1  probe_Δloss" in third_plain
        assert first_plain.expandtabs(8).index("P1  probe_Δloss") == 339
        assert second_plain.expandtabs(8).index("P2  probe_Δloss") == 339
        assert third_plain.expandtabs(8).index("P1  probe_Δloss") == 339
        assert "▼|▲|? =" not in first_plain
        assert "▼|▲|? =" in second_plain
        assert "(P1,2)" in second_plain
        assert "▼|▲|? =" not in third_plain
        assert "same_batch W" not in first_plain
        assert "same_batch W" not in second_plain
        assert "same_batch W" not in third_plain
        assert f"=>{constants.BOLD}{constants.YELLOW}{constants.UP_ARROW}{constants.R}" in second_line

        audits = trainer.plastic_depth_count_audit[-3:]
        assert [row["probe_window_id"] for row in audits] == [1, 1, 2]
        assert [tuple(row["probe_window_provenance"]) for row in audits] == [
            (1,),
            (1, 2),
            (1,),
        ]
        assert [row["probe_global_sequence"] for row in audits] == [1, 2, 3]
        assert [tuple(row["probe_global_provenance"]) for row in audits] == [
            (1,),
            (1, 2),
            (3,),
        ]
    finally:
        trainer.close()


def test_same_batch_forensic_marker_requires_debug_above_nine(monkeypatch) -> None:
    trainer = _trainer(window_size=2)
    try:
        trainer.train_one_update()
        row = trainer.plastic_depth_count_audit[-1]
        monkeypatch.setattr(constants, "DEBUG", 10)
        line = _plain(_progress_line_from_audit(row))
        assert "same_batch W1:1/2 B=" in line
    finally:
        trainer.close()


def test_sample_change_is_yellow_for_exactly_one_progress_row() -> None:
    first = _ordinary_progress_line("sample-change", (1.0, 19.0, 20.0))
    second = _ordinary_progress_line("sample-change", (1.0, 18.9, 20.0))
    third = _ordinary_progress_line("sample-change", (1.0, 18.9, 20.0))

    assert f"{constants.YELLOW}18.9{constants.R}" not in first
    assert f"{constants.YELLOW}18.9{constants.R}" in second
    assert f"{constants.YELLOW}18.9{constants.R}" not in third


def test_gradient_norm_is_fixed_width() -> None:
    small = _plain(_ordinary_progress_line("grad-small", (1.0, 2.0), gradient_norm="   5.921"))
    large = _plain(_ordinary_progress_line("grad-large", (1.0, 2.0), gradient_norm="  10.746"))
    assert "g nrm=  5.921" in small
    assert "g nrm= 10.746" in large


def test_plastic_startup_section_prints_resolved_same_batch_mode() -> None:
    same_batch._set_runtime_enabled(True)
    visibility._STARTUP_VISIBILITY_INSTALLED = False
    visibility._install_startup_visibility()
    stream = io.StringIO()
    with redirect_stdout(stream):
        public_runner._print_plastic_option(
            "plastic__layer_count_probe__window_size_as_number_of_probes:",
            "4",
        )
    rendered = stream.getvalue()
    assert "plastic__layer_count_probe__window_size_as_number_of_probes:" in rendered
    assert "plastic__layer_count__same_batch_all_probes:" in rendered
    assert "true" in rendered


def test_startup_visibility_resolves_direct_main_module() -> None:
    fake_main = SimpleNamespace(_print_plastic_option=lambda _label, _value: None)
    resolved = visibility._startup_runner_module({"__main__": fake_main})
    assert resolved is fake_main


def test_plastic_header_declares_every_parser_plastic_control() -> None:
    parser = public_runner._core.build_parser()
    public_controls = {
        action.dest
        for action in parser._actions
        if str(getattr(action, "dest", "")).startswith("plastic__")
    }
    declared_controls = {
        label.rstrip(":")
        for label in public_runner._PLASTIC_STARTUP_LABELS
        if label.startswith("plastic__")
    }
    declared_controls.add("plastic__layer_count__same_batch_all_probes")
    assert public_controls <= declared_controls, sorted(public_controls - declared_controls)


def test_public_wrapper_compact_header_prints_same_batch_true() -> None:
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = sys.executable
    completed = subprocess.run(
        [
            "bash",
            "train_OWT.sh",
            "-x",
            "true",
            "-I",
            "none",
            "-g",
            "SAME_BATCH_VISIBILITY",
            "-n",
            "2",
            "-w",
            "0",
            "-b",
            "1",
            "-A",
            "1",
            "-L",
            "4",
            "-H",
            "2",
            "-D",
            "8",
            "-C",
            "8",
            "-P",
            "2",
            "-S",
            "1",
            "--plastic__enabled",
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "2",
            "--plastic__max_permitted_layers",
            "4",
            "--plastic__layer_count__same_batch_all_probes",
            "--plastic__layer_count_probe__number_of_sampled_valid_tokens",
            "8",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "2",
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    compact_line = next(
        line for line in completed.stdout.splitlines()
        if line.lstrip().startswith("plastic fine:")
    )
    assert "same_batch=true" in compact_line
# ^^^ THOG
