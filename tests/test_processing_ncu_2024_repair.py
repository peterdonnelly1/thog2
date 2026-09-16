# vvv THOG
from __future__ import annotations

import csv
from pathlib import Path

from sheet import processing_ncu_2024_instra_patch as repair
from sheet import processing_ncu_compatibility as compatibility


def _write_wide_ncu(path: Path) -> None:
    headers = [
        "ID", "Process ID", "Kernel Name", "Context", "Stream", "Block Size", "CC",
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
        "gpu__time_duration.sum",
        "launch__registers_per_thread_allocated",
        "launch__shared_mem_per_block_allocated",
        "launch__stream_id",
    ]
    units = ["", "", "", "", "", "", "", "", "ns", "register/thread", "byte/block", ""]
    rows = [
        [
            "0", "111", "main_gemm", "1", "7", "(128, 1, 1)", "8.9",
            "THOG2_PROCESSING|owner=MAIN|operation=consume|family=DOWN|layer=8",
            "5,472", "32", "16,384", "7",
        ],
        [
            "1", "111", "premat_gemm", "1", "13", "(128, 1, 1)", "8.9",
            "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=8",
            "8,192", "40", "8,192", "13",
        ],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(units)
        writer.writerows(rows)


def test_ncu_2024_wide_export_normalizes_resources_and_semantics(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    semantic_path = tmp_path / "semantic.csv"
    _write_wide_ncu(raw_path)
    _write_wide_ncu(semantic_path)

    rows = compatibility.normalize_semantic_ncu_exports(raw_path, semantic_path)

    assert {row["role"] for row in rows} == {"MAIN", "PREMAT"}
    assert {row["family"] for row in rows} == {"DOWN"}
    assert {row["layer"] for row in rows} == {8}

    main = next(row for row in rows if row["role"] == "MAIN")
    premat = next(row for row in rows if row["role"] == "PREMAT")
    assert main["duration_ns"] == 5472
    assert main["threads_per_block"] == 128
    assert main["registers_per_thread"] == 32
    assert main["registers_per_block"] == 4096
    assert main["dynamic_shared_mem_bytes"] == 16384
    assert main["stream_id"] == "7"
    assert premat["duration_ns"] == 8192
    assert premat["dynamic_shared_mem_bytes"] == 8192
    assert premat["stream_id"] == "13"

    representative = compatibility.select_representative_kernel_resources(rows)
    result = compatibility.build_compatibility_rows(representative)
    assert result
    assert {row["compatibility_class"] for row in result} <= {"GREEN", "ORANGE", "YELLOW", "RED"}


def test_postrun_progress_is_staged_and_flushed(capsys) -> None:
    repair._postrun_progress(3, 7, "semantic NCU CSV export complete")
    output = capsys.readouterr().out
    assert "THOG2 INSTRA post-run [3/7" in output
    assert "43%" in output
    assert "semantic NCU CSV export complete" in output
# ^^^ THOG
