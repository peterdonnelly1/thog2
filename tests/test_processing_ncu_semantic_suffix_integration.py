# vvv THOG
import csv
from pathlib import Path

from sheet import processing_ncu_compatibility as compatibility


def _write_wide(path: Path, *, semantic: bool) -> None:
    headers = [
        "ID", "Process ID", "Kernel Name", "Context", "Stream", "Block Size", "CC",
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
        "gpu__time_duration.sum", "launch__registers_per_thread_allocated",
        "launch__shared_mem_per_block_allocated", "launch__stream_id",
    ]
    units = ["", "", "", "", "", "", "", "", "ns", "register/thread", "byte/block", ""]
    premat_label = "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=8"
    main_label = "THOG2_PROCESSING|owner=MAIN|operation=consume|family=DOWN|layer=8"
    rows = [
        ["0", "1614690", f"{premat_label}/void premat_kernel" if semantic else "premat_kernel", "1", "13", "(128, 1, 1)", "8.9", f"{premat_label}:none:none", "5376", "32", "2048", "13"],
        ["3", "1614690", f"{main_label}/void main_kernel" if semantic else "main_kernel", "1", "7", "(128, 1, 1)", "8.9", f"{main_label}:none:none", "8192", "40", "4096", "7"],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(units)
        writer.writerows(rows)


def test_wide_ncu_suffix_semantics_join_main_and_premat(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    semantic = tmp_path / "semantic.csv"
    _write_wide(raw, semantic=False)
    _write_wide(semantic, semantic=True)

    rows = compatibility.normalize_semantic_ncu_exports(raw, semantic)

    assert {row["role"] for row in rows} == {"MAIN", "PREMAT"}
    assert {row["operation"] for row in rows} == {"consume", "materialize"}
    assert {row["family"] for row in rows} == {"DOWN"}
    assert {row["layer"] for row in rows} == {8}
# ^^^ THOG
