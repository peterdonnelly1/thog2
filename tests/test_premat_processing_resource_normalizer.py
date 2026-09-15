# vvv THOG
import csv
import json
import sqlite3

from sheet.premat_processing import PROCESSING_CAPTURE_RANGE, normalize_nsys_sqlite


def _build_trace(path):
    start = 1_000_000
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, streamId INTEGER, contextId INTEGER, correlationId INTEGER, shortName TEXT);
            CREATE TABLE TARGET_INFO_GPU_METRICS(metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS(timestamp INTEGER, metricId INTEGER, value REAL);
            """
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)",
            [
                (start, start + 100_000, 1, PROCESSING_CAPTURE_RANGE),
                (start + 5_000, start + 35_000, 1, "THOG2_PROCESSING|owner=MAIN|operation=consume|family=DOWN|layer=0"),
                (start + 45_000, start + 70_000, 1, "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=1"),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
            [
                (start + 10_000, start + 11_000, 1, 101),
                (start + 50_000, start + 51_000, 1, 102),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
            [
                (start + 12_000, start + 20_000, 7, 1, 101, "main_explicit"),
                (start + 22_000, start + 26_000, 7, 1, 999, "main_stream_inferred"),
                (start + 52_000, start + 60_000, 8, 1, 102, "premat_explicit"),
                (start + 78_000, start + 82_000, 9, 1, 998, "unknown_kernel"),
            ],
        )
        metrics = [
            (1, "SMs Active"),
            (2, "SM Issue"),
            (3, "Tensor Active"),
            (4, "Active SM Unused Warp Slots"),
            (5, "DRAM Read"),
        ]
        connection.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES(?,?)", metrics)
        for offset_us, base in [(10, 10), (30, 30), (50, 50), (70, 70), (90, 90)]:
            timestamp = start + offset_us * 1000
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES(?,?,?)",
                [(timestamp, metric_id, base + metric_id) for metric_id, _name in metrics],
            )
        connection.commit()


def test_normalizer_emits_exact_owners_stream_resource_csv_and_idle_intervals(tmp_path):
    sqlite_path = tmp_path / "trace.sqlite"
    output = tmp_path / "processing"
    _build_trace(sqlite_path)
    payload = normalize_nsys_sqlite(
        sqlite_path,
        output,
        capture_frequency_hz=25_000,
        handoff={"run_name": "TEST_RUN"},
        capture_metadata={"optimizer_update": 50, "micro_step": 1},
    )

    assert payload["metadata"]["schema_version"] == 2
    assert payload["metadata"]["metric_catalog"]["dram_read_pct"]["available"] is True
    assert payload["metadata"]["metric_catalog"]["dram_write_pct"]["available"] is False

    owners = [(row["kernel_name"], row["owner"], row["owner_source"]) for row in payload["intervals"]]
    assert owners == [
        ("main_explicit", "MAIN", "NVTX"),
        ("main_stream_inferred", "MAIN", "STREAM_INFERRED"),
        ("premat_explicit", "PREMAT", "NVTX"),
        ("unknown_kernel", "UNKNOWN", "UNKNOWN"),
    ]
    assert not any(row["kernel_name"] == "unknown_kernel" and row["owner"] == "MAIN" for row in payload["intervals"])

    states = {row["attribution_state"] for row in payload["stream_resources"]}
    assert "MAIN_ONLY" in states
    assert "PREMAT_ONLY" in states
    assert "OTHER_OR_UNKNOWN" in states

    files = payload["metadata"]["files"]
    stream_resource_path = output / files["stream_resources"]
    assert stream_resource_path.exists()
    with stream_resource_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows
    assert "main_premat_simultaneous_pct" in rows[0]
    assert "dram_read_pct" in rows[0]

    bundle_metadata = json.loads((output / files["metadata"]).read_text())
    assert bundle_metadata["files"]["stream_resources"].endswith("processing_stream_resources.csv")
    assert payload["main_idle_intervals"]
# ^^^ THOG
