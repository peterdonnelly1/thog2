# vvv THOG
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import run_thog2_owt_core as core
from sheet import help_registry_descriptor_patch as descriptor_help
from sheet import plastic_depth_warmup_guard_patch as guard


ROOT = Path(__file__).resolve().parents[1]
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _visible(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).expandtabs(8)


def test_t_row_compaction_uses_short_labels_and_single_space_vectors(monkeypatch) -> None:
    monkeypatch.setattr(
        guard,
        "_ORIGINAL_FORMAT_PROGRESS_LINE",
        lambda run_id, event, payload: (
            "T       1  260806:2138  00:00:08       8s  Δstep=  7.9384s  "
            "tok/s=  9288  tokens=     73,728  training loss  =  11.1516  "
            "Δloss=     n/a  lr= 8.911e-06  gradient norm=  11.066  layers = 32  "
            "probe_losses [L-4, L-3, L-2] = [ 10.999,  11.042,  11.077]  "
            "score_z [L-4, L-3, L-2] = [    -3.05,     -3.08,     -1.63]  "
            "sampled = [  1.0,   2.0,  10.0]"
        ),
    )

    rendered = guard._format_progress_line_with_compact_console_fields(
        "run",
        "optimizer_progress",
        {},
    )

    assert rendered.startswith("T     1")
    assert "loss  =  11.1516" in rendered
    assert "training loss" not in rendered
    assert "Δ=     n/a" in rendered
    assert "Δloss" not in rendered
    assert "grad norm=  11.066" in rendered
    assert "gradient norm" not in rendered
    assert "probe_losses [L-4, L-3, L-2] = [10.999, 11.042, 11.077]" in rendered
    assert "score_z [L-4, L-3, L-2] = [-3.05, -3.08, -1.63]" in rendered
    assert rendered.endswith("sampled = [1.0, 2.0, 10.0]")


def test_v_row_uses_the_same_prefix_and_loss_column_as_t(monkeypatch) -> None:
    rows = {
        "optimizer_progress": (
            "T      10  260806:2139  00:01:17      77s  Δstep=  7.6203s  "
            "tok/s=  9635  tokens=    737,280  training loss  =   9.2077  "
            "Δloss=  -1.944  lr= 8.911e-05  gradient norm=   2.546"
        ),
        "evaluation_completed": (
            "\x1b[33mV      10  260806:2139  00:01:17      77s  Δstep=  7.6203s  "
            "tok/s=  9635  tokens=    737,280  training loss  =   9.2077  "
            "\x1b[1;93mvalidation loss=   9.1000\x1b[33m\x1b[0m"
        ),
    }
    monkeypatch.setattr(
        guard,
        "_ORIGINAL_FORMAT_PROGRESS_LINE",
        lambda run_id, event, payload: rows[event],
    )

    training = guard._format_progress_line_with_compact_console_fields(
        "run",
        "optimizer_progress",
        {},
    )
    validation = guard._format_progress_line_with_compact_console_fields(
        "run",
        "evaluation_completed",
        {},
    )
    visible_training = _visible(training)
    visible_validation = _visible(validation)

    assert visible_training.startswith("T    10")
    assert visible_validation.startswith("V    10")
    assert visible_training.index("loss") == visible_validation.index("loss")
    assert "training loss" not in visible_validation


def test_descriptor_registry_uses_actual_keys_and_explicit_missing_marker() -> None:
    registry = descriptor_help.format_descriptor_registry()
    assert "getopt / artifact descriptor registry" in registry
    assert "LCS" in registry and "--plastic__phase_1_n_steps N" in registry
    assert "DLB" in registry and "--depth-compress-layer-norm-and-bias" in registry
    assert "HFC" in registry and "--hyperblock-common-family-order N" in registry
    assert "—" in registry and "--plastic__cuda_allocator_reserve_gib VALUE" in registry


def test_registered_parser_help_appends_descriptor_registry_once() -> None:
    rendered = core.build_parser().format_help()
    assert rendered.count("getopt / artifact descriptor registry") == 1
    assert "LPI" in rendered
    assert "--plastic__layer_count_probe__probe_every_n_steps N" in rendered


def test_wrapper_registry_emits_descriptor_table_once() -> None:
    completed = subprocess.run(
        ("bash", "./train_OWT.sh", "--print-geometry-registry"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.count("getopt / artifact descriptor registry") == 1
    assert "abbrev" in completed.stdout
    assert "parameter" in completed.stdout
    assert "description" in completed.stdout
# ^^^ THOG
