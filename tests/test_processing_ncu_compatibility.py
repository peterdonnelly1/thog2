# vvv THOG
import csv

from sheet.processing_ncu_compatibility import (
    build_compatibility_rows,
    compatibility_row,
    normalize_raw_ncu_csv,
    parse_processing_nvtx_label,
    select_representative_kernel_resources,
)


def _write_wide_ncu_csv(path, *, threads, registers_per_thread, registers_per_block, shared_bytes):
    fields = [
        "ID", "Kernel Name", "Context", "Stream", "Block Size", "CC",
        "launch__registers_per_thread",
        "launch__registers_per_thread_allocated",
        "launch__shared_mem_per_block_allocated",
        "gpu__time_duration.sum",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "launch__registers_per_thread": "register/thread",
            "launch__registers_per_thread_allocated": "register/thread",
            "launch__shared_mem_per_block_allocated": "byte/block",
            "gpu__time_duration.sum": "ns",
        })
        writer.writerow({
            "ID": "4",
            "Kernel Name": "test_kernel",
            "Context": "1",
            "Stream": "7",
            "Block Size": f"({threads}, 1, 1)",
            "CC": "8.9",
            "launch__registers_per_thread": str(registers_per_thread),
            "launch__registers_per_thread_allocated": f"{registers_per_block:,}",
            "launch__shared_mem_per_block_allocated": f"{shared_bytes:,}",
            "gpu__time_duration.sum": "1000",
        })


def test_wide_ncu_register_allocation_is_per_block_not_per_thread(tmp_path):
    source = tmp_path / "main.csv"
    _write_wide_ncu_csv(
        source,
        threads=256,
        registers_per_thread=224,
        registers_per_block=57344,
        shared_bytes=74752,
    )

    rows = normalize_raw_ncu_csv(
        source,
        role="MAIN",
        family="DOWN",
        operation="consume",
        layer=8,
    )

    assert len(rows) == 1
    assert rows[0]["registers_per_thread"] == 224
    assert rows[0]["registers_per_block"] == 57344
    assert rows[0]["theoretical_blocks_per_sm"] == 1


def test_ncu_nvtx_renamed_kernel_suffix_is_not_part_of_layer():
    assert parse_processing_nvtx_label(
        "THOG2_PROCESSING|owner=PREMAT|operation=materialize|"
        "family=DOWN|layer=8/void at::vectorized_elementwise_kernel(...)"
    ) == {
        "role": "PREMAT",
        "operation": "materialize",
        "family": "DOWN",
        "layer": 8,
    }


def test_corrected_down_pair_is_structurally_co_resident(tmp_path):
    main_path = tmp_path / "main.csv"
    premat_path = tmp_path / "premat.csv"
    _write_wide_ncu_csv(
        main_path,
        threads=256,
        registers_per_thread=224,
        registers_per_block=57344,
        shared_bytes=74752,
    )
    _write_wide_ncu_csv(
        premat_path,
        threads=128,
        registers_per_thread=30,
        registers_per_block=4096,
        shared_bytes=1024,
    )
    main = normalize_raw_ncu_csv(
        main_path,
        role="MAIN",
        family="DOWN",
        operation="consume",
        layer=8,
    )[0]
    premat = normalize_raw_ncu_csv(
        premat_path,
        role="PREMAT",
        family="DOWN",
        operation="materialize",
        layer=8,
    )[0]

    result = compatibility_row(main, premat)

    assert result["pair_can_co_reside"] is True
    assert result["main_registers_per_block"] == 57344
    assert result["premat_registers_per_block"] == 4096
    assert result["premat_blocks_with_one_main_block"] == 2
    assert result["compatibility_class"] == "GREEN"


def _resource_row(*, role, launch_id, duration, registers, kernel):
    return {
        "role": role,
        "operation": "consume" if role == "MAIN" else "materialize",
        "family": "DOWN",
        "layer": 8,
        "launch_id": str(launch_id),
        "cuda_kernel_name": kernel,
        "duration_ns": duration,
        "threads_per_block": 128,
        "warps_per_block": 4,
        "registers_per_thread": registers / 128,
        "registers_per_block": registers,
        "static_shared_mem_bytes": 0,
        "dynamic_shared_mem_bytes": 0,
        "theoretical_blocks_per_sm": 2 if role == "MAIN" else 8,
        "sm_register_capacity": 65536,
        "sm_shared_mem_bytes": 102400,
        "sm_max_warps": 48,
        "sm_max_threads": 1536,
        "sm_max_blocks": 24,
    }


def test_all_distinct_premat_stages_survive_and_worst_stage_controls_result():
    rows = [
        _resource_row(role="MAIN", launch_id=1, duration=500, registers=30720, kernel="main_gemm"),
        _resource_row(role="PREMAT", launch_id=2, duration=20, registers=1024, kernel="setup"),
        _resource_row(role="PREMAT", launch_id=3, duration=400, registers=4096, kernel="vectorized"),
        _resource_row(role="PREMAT", launch_id=4, duration=180, registers=5120, kernel="gemvx"),
    ]

    selected = select_representative_kernel_resources(rows)
    premat = [row for row in selected if row["role"] == "PREMAT"]
    assert [row["cuda_kernel_name"] for row in premat] == ["setup", "vectorized", "gemvx"]
    assert [row["premat_stage_index"] for row in premat] == [1, 2, 3]
    assert all(row["premat_stage_count"] == 3 for row in premat)

    compatibility = build_compatibility_rows(selected)
    by_kernel = {row["premat_cuda_kernel_name"]: row for row in compatibility}
    assert by_kernel["vectorized"]["compatibility_class"] == "GREEN"
    assert by_kernel["gemvx"]["compatibility_class"] == "YELLOW"
    assert by_kernel["gemvx"]["premat_stage_is_most_constrained"] is True
    assert by_kernel["vectorized"]["premat_stage_is_most_constrained"] is False
# ^^^ THOG
