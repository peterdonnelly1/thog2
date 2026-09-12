# vvv THOG
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from sheet.premat_processing import (
    normalize_nsys_sqlite,
    processing_requested_from_argv,
    rewrite_processing_cli_for_core,
    validate_processing_configuration,
)


def test_processing_cli_surface_exact_names() -> None:
    enabled, frequency = processing_requested_from_argv([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz", "12345",
    ])
    assert enabled is True
    assert frequency == 12345


def test_processing_public_cli_rewrites_to_hidden_core_aliases() -> None:
    assert rewrite_processing_cli_for_core([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz=12345",
        "--model-type", "sheet",
    ]) == [
        "--processing_logging_internal", "enabled",
        "--processing_logging_capture_frequency_hz_internal=12345",
        "--model-type", "sheet",
    ]


def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:
    validate_processing_configuration("enabled", 10000, "cuda")
    validate_processing_configuration("disabled", 10, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        validate_processing_configuration("enabled", 10000, "cpu")
    with pytest.raises(ValueError, match="capture_frequency_hz"):
        validate_processing_configuration("enabled", 9, "cuda")


def _synthetic_nsys_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER, correlationId INTEGER, shortName INTEGER);
            CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL);
            """
        )
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "main_kernel"), (2, "premat_kernel")])
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (1000, 10000, 7, "THOG2_PREMAT_PROCESSING_CAPTURE"),
                (1800, 2400, 7, "THOG2_PROCESSING|owner=MAIN|operation=consume|family=QKV|layer=0"),
                (2000, 2600, 7, "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=1"),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            [(1900, 1950, 7, 11), (2100, 2150, 7, 12)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            [(2500, 4500, 3, 11, 1), (3500, 4300, 9, 12, 2)],
        )
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
            [
                (1, "SMs Active %"),
                (2, "SM Issue %"),
                (3, "Tensor Active %"),
                (4, "Active SM Unused Warp Slots %"),
            ],
        )
        for timestamp, values in ((3000, (80, 70, 90, 20)), (4000, (90, 75, 95, 15))):
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?, ?)",
                [(timestamp, metric_id, value) for metric_id, value in enumerate(values, start=1)],
            )
        connection.commit()


def test_processing_normalizer_emits_graph_and_download_data(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    output = tmp_path / "processing"
    _synthetic_nsys_database(database)
    payload = normalize_nsys_sqlite(
        database,
        output,
        capture_frequency_hz=10000,
        handoff={"run_name": "fixture"},
        capture_metadata={"optimizer_update": 10, "micro_step": 1},
    )
    assert payload["metadata"]["capture_frequency_hz"] == 10000
    assert payload["samples"][0]["sm_active_pct"] == 80.0
    assert {row["owner"] for row in payload["intervals"]} == {"MAIN", "PREMAT"}
    assert len(payload["summary"]) == 1
    summary = payload["summary"][0]
    assert summary["family"] == "QKV"
    assert summary["duration_ms"] == pytest.approx(0.002)
    assert summary["premat_overlap_ms"] == pytest.approx(0.0008)
    assert summary["premat_overlap_pct"] == pytest.approx(40.0)
    expected = {
        "processing_samples.csv",
        "processing_intervals.csv",
        "processing_summary.csv",
        "processing_metadata.json",
        "processing_data.json",
        "processing_bundle.zip",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    with (output / "processing_summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["family"] == "QKV"
# ^^^ THOG
