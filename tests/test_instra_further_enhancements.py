# vvv THOG
from __future__ import annotations

import sqlite3
from pathlib import Path

import run_thog2_owt_core
from sheet import dense_snapshot, local_dashboard_notes_patch, wandb_telemetry


def test_runtime_overview_metadata_captures_wrapper_and_environment(monkeypatch):
    runstring = "./train_OWT_core.sh --max-updates 50"
    monkeypatch.setenv("THOG2_RUNSTRING", runstring)

    metadata = run_thog2_owt_core.runtime_overview_metadata(argv=["--ignored"])

    assert metadata["command"] == runstring
    assert metadata["hostname"]
    assert metadata["os"]
    assert metadata["python_version"]
    assert metadata["python_executable"]
    assert metadata["git_repository"]


def test_runtime_overview_metadata_has_a_shell_safe_python_fallback(monkeypatch):
    monkeypatch.delenv("THOG2_RUNSTRING", raising=False)

    metadata = run_thog2_owt_core.runtime_overview_metadata(
        argv=["--label", "two words"],
        module_name="run_thog2_owt",
    )

    assert "-m run_thog2_owt" in metadata["command"]
    assert "'two words'" in metadata["command"]


def test_run_notes_are_durable_and_normalised(tmp_path):
    database_path = tmp_path / "charts.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    written = local_dashboard_notes_patch._write_run_notes(
        database_path,
        "first line\r\nsecond line",
    )

    with sqlite3.connect(database_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    assert written == "first line\nsecond line"
    assert metadata["user_notes"] == written
    assert metadata["user_notes_updated_at"]


def test_dense_snapshot_completion_path_is_repository_relative(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("THOG2_WRAPPER_OWNS_SNAPSHOT_FOOTER", raising=False)
    absolute = tmp_path / "dense_baseline_snapshots" / "example.dense_snapshot.pt"

    dense_snapshot.print_dense_snapshot_completion({"snapshot_path": str(absolute)})

    assert capsys.readouterr().out == (
        "  snapshot path:               "
        "dense_baseline_snapshots/example.dense_snapshot.pt\n"
    )


def test_verbose_wandb_console_requires_debug_above_99(monkeypatch):
    monkeypatch.setattr(wandb_telemetry._constants, "DEBUG", 99)
    assert wandb_telemetry.verbose_wandb_console_enabled() is False
    monkeypatch.setattr(wandb_telemetry._constants, "DEBUG", 100)
    assert wandb_telemetry.verbose_wandb_console_enabled() is True


def test_further_enhancement_assets_and_api_are_registered():
    repository_root = Path(__file__).parents[1]
    dashboard = (repository_root / "run_thog2_dashboard.py").read_text()
    lifecycle = (repository_root / "run_thog2_lifecycle.py").read_text()
    shell = (repository_root / "train_OWT_core.sh").read_text()

    assert "local_dashboard_notes_patch.install(_dashboard)" in dashboard
    assert '"dashboard_instra_further_enhancements_patch.js"' in dashboard
    assert "runtime_overview_metadata(argv=actual_argv)" in lifecycle
    assert "module_name=__name__" not in lifecycle
    assert 'printf -v THOG2_RUNSTRING \'%q \' "$0" "$@"' in shell
    assert "THOG2_WRAPPER_OWNS_SNAPSHOT_FOOTER=true" in shell
    assert "snapshot_done=" in shell
# ^^^ THOG
