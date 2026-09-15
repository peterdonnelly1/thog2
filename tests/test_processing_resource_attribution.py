# vvv THOG
from sheet.processing_resource_attribution import (
    build_stream_resource_rows,
    infer_kernel_owners,
    main_idle_intervals,
    metric_catalog,
    sample_bins,
)


def _kernel(start, end, owner, *, stream=7, context_id=1, explicit=True, operation="consume"):
    return {
        "start_us": start,
        "end_us": end,
        "duration_us": end - start,
        "stream": stream,
        "context_id": context_id,
        "owner": owner,
        "_owner_explicit": explicit,
        "operation": operation,
        "layer": 0,
        "family": "DOWN",
    }


def test_unlabelled_kernel_is_inferred_only_from_unambiguous_context_stream():
    rows = infer_kernel_owners([
        _kernel(0, 10, "MAIN", stream=7, context_id=1),
        _kernel(10, 20, "", stream=7, context_id=1, explicit=False),
        _kernel(20, 30, "", stream=7, context_id=2, explicit=False),
    ])
    assert rows[0]["owner"] == "MAIN"
    assert rows[0]["owner_source"] == "NVTX"
    assert rows[1]["owner"] == "MAIN"
    assert rows[1]["owner_source"] == "STREAM_INFERRED"
    assert rows[2]["owner"] == "UNKNOWN"
    assert rows[2]["owner_source"] == "UNKNOWN"


def test_stream_inference_refuses_a_stream_with_conflicting_explicit_owners():
    rows = infer_kernel_owners([
        _kernel(0, 10, "MAIN", stream=9, context_id=1),
        _kernel(10, 20, "PREMAT", stream=9, context_id=1, operation="materialize"),
        _kernel(20, 30, "", stream=9, context_id=1, explicit=False),
    ])
    assert rows[2]["owner"] == "UNKNOWN"
    assert rows[2]["owner_source"] == "UNKNOWN"


def test_sample_bins_use_midpoints_and_capture_edges():
    rows = sample_bins([{"time_us": 10}, {"time_us": 30}, {"time_us": 50}], 60)
    assert [(row["sample_start_us"], row["sample_end_us"]) for row in rows] == [
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
    ]


def test_resource_attribution_distinguishes_exact_overlap_sequential_other_and_idle():
    samples = [
        {"time_us": 10, "sm_active_pct": 20},
        {"time_us": 30, "sm_active_pct": 30},
        {"time_us": 50, "sm_active_pct": 90},
        {"time_us": 70, "sm_active_pct": 60},
        {"time_us": 90, "sm_active_pct": 10},
    ]
    intervals = [
        {**_kernel(0, 38, "MAIN"), "owner_source": "NVTX"},
        {**_kernel(42, 58, "MAIN"), "owner_source": "NVTX"},
        {**_kernel(45, 55, "PREMAT", stream=8, operation="materialize"), "owner_source": "NVTX"},
        {**_kernel(61, 68, "MAIN"), "owner_source": "NVTX"},
        {**_kernel(72, 78, "PREMAT", stream=8, operation="materialize"), "owner_source": "NVTX"},
        {**_kernel(85, 90, "UNKNOWN", stream=11, explicit=False), "owner_source": "UNKNOWN"},
    ]
    for row in intervals:
        row.pop("_owner_explicit", None)
    resources = build_stream_resource_rows(samples, intervals, 100)
    assert [row["attribution_state"] for row in resources] == [
        "MAIN_ONLY",
        "MAIN_ONLY",
        "MAIN_PREMAT_OVERLAP",
        "MIXED_SEQUENTIAL",
        "OTHER_OR_UNKNOWN",
    ]
    assert resources[2]["main_premat_simultaneous_pct"] == 50.0
    assert resources[3]["main_premat_simultaneous_pct"] == 0.0
    assert resources[2]["main_layers"] == "1"
    assert resources[2]["premat_families"] == "DOWN"


def test_idle_sample_is_not_assigned_to_main():
    resources = build_stream_resource_rows([{"time_us": 50, "sm_active_pct": 0}], [], 100)
    assert resources[0]["attribution_state"] == "IDLE"
    assert resources[0]["main_coverage_pct"] == 0.0


def test_main_idle_intervals_are_exact_complement_of_main_kernel_union():
    intervals = [
        {"start_us": 10, "end_us": 20, "owner": "MAIN"},
        {"start_us": 15, "end_us": 30, "owner": "MAIN"},
        {"start_us": 40, "end_us": 50, "owner": "PREMAT"},
        {"start_us": 70, "end_us": 80, "owner": "MAIN"},
    ]
    assert main_idle_intervals(intervals, 100) == [
        {"start_us": 0.0, "end_us": 10.0, "duration_us": 10.0},
        {"start_us": 30.0, "end_us": 70.0, "duration_us": 40.0},
        {"start_us": 80.0, "end_us": 100.0, "duration_us": 20.0},
    ]


def test_metric_catalog_marks_optional_unavailable_metrics_without_inventing_values():
    catalog = metric_catalog({"sm_active_pct": "SMs Active", "dram_read_pct": "DRAM Read"})
    assert catalog["sm_active_pct"]["available"] is True
    assert catalog["dram_read_pct"]["available"] is True
    assert catalog["dram_write_pct"]["available"] is False
    assert catalog["l2_active_pct"]["available"] is False
# ^^^ THOG
