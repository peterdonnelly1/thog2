# vvv THOG
from __future__ import annotations

import csv
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from run_thog2_local_dashboard import RunDashboardState
from sheet.local_chart_store import LocalChartReader, LocalChartStore
from sheet.premat_processing import (
    _mean_metric,
    _nsys_profile_command,
    normalize_nsys_sqlite,
    processing_invocation_is_non_training,
    processing_requested_from_argv,
    register_processing_handoff,
    should_capture_processing_forward,
    rewrite_processing_cli_for_core,
    validate_processing_configuration,
)
from sheet.processing_update_timing_patch import (
    _apply_official_elapsed,
    _gap_phase,
    _premat_is_enabled,
)


def test_processing_cli_surface_exact_names() -> None:
    enabled, frequency, capture_update = processing_requested_from_argv([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz", "12345",
        "--premat_processing_logging_capture_update", "5",
    ])
    assert enabled is True
    assert frequency == 12345
    assert capture_update == 5


def test_processing_public_cli_rewrites_to_hidden_core_aliases() -> None:
    assert rewrite_processing_cli_for_core([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz=12345",
        "--premat_processing_logging_capture_update", "5",
        "--model-type", "sheet",
    ]) == [
        "--processing_logging_internal", "enabled",
        "--processing_logging_capture_frequency_hz_internal=12345",
        "--processing_logging_capture_update_internal", "5",
        "--model-type", "sheet",
    ]


def test_processing_metadata_probe_does_not_require_nsys() -> None:
    assert processing_invocation_is_non_training([
        "--premat_processing_logging", "enabled",
        "--print-resolved-json",
    ])
    assert processing_invocation_is_non_training(["--dry-run"])
    assert processing_invocation_is_non_training(["--print-artifact-name=true"])
    assert processing_invocation_is_non_training(["--help"])
    assert not processing_invocation_is_non_training([
        "--premat_processing_logging", "enabled",
        "--model-type", "sheet",
    ])


def test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:
    command = _nsys_profile_command(
        "/usr/bin/nsys",
        report_base=tmp_path / "trace",
        frequency=10000,
        entrypoint=tmp_path / "runner.py",
        arguments=["--processing_logging_internal", "enabled"],
    )
    assert "--inherit-environment=true" in command
    assert "--show-output=true" in command
    assert "--wait=primary" in command
    assert "--sample=none" in command
    assert "--cpuctxsw=none" in command
    assert "--capture-range-end=stop" in command
    assert "--gpu-metrics-frequency=10000" in command


def test_processing_sparse_metric_samples_ignore_empty_cells() -> None:
    samples = [
        {"time_us": 1.0, "sm_active_pct": 80.0, "tensor_active_pct": ""},
        {"time_us": 2.0, "sm_active_pct": "", "tensor_active_pct": 90.0},
        {"time_us": 3.0, "sm_active_pct": None, "tensor_active_pct": "not-a-number"},
    ]
    assert _mean_metric(samples, "sm_active_pct", 0.0, 4.0) == 80.0
    assert _mean_metric(samples, "tensor_active_pct", 0.0, 4.0) == 90.0
    assert _mean_metric(samples, "sm_issue_pct", 0.0, 4.0) is None


def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:
    validate_processing_configuration("enabled", 10000, "cuda")
    validate_processing_configuration("disabled", 10, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        validate_processing_configuration("enabled", 10000, "cpu")
    with pytest.raises(ValueError, match="capture_frequency_hz"):
        validate_processing_configuration("enabled", 9, "cuda")
    with pytest.raises(ValueError, match="capture_update"):
        validate_processing_configuration("enabled", 10000, "cuda", 0)
    with pytest.raises(ValueError, match="must not exceed"):
        validate_processing_configuration("enabled", 10000, "cuda", 6, 5)




def test_processing_capture_update_is_exact_and_log_interval_independent(monkeypatch) -> None:
    import sheet.premat_processing as processing
    monkeypatch.setattr(processing, "_capture_done", False)
    assert not should_capture_processing_forward(
        enabled=True, completed_updates=3, max_updates=5, log_interval=1, capture_update=5, micro_step=0
    )
    assert should_capture_processing_forward(
        enabled=True, completed_updates=4, max_updates=5, log_interval=99, capture_update=5, micro_step=0
    )
    assert not should_capture_processing_forward(
        enabled=True, completed_updates=4, max_updates=5, log_interval=1, capture_update=5, micro_step=1
    )

def test_processing_handoff_records_target_matrix(tmp_path: Path, monkeypatch) -> None:
    handoff_path = tmp_path / "handoff.json"
    monkeypatch.setenv("THOG2_PREMAT_PROCESSING_HANDOFF", str(handoff_path))
    register_processing_handoff(
        tmp_path / "run",
        run_name="fixture",
        config={"premat": "enabled", "premat_target_layer": 1, "premat_target_matrix": 2, "premat_attention_mode": "fused"},
    )
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert payload["config"]["premat_target_layer"] == 1
    assert payload["config"]["premat_target_matrix"] == 2


def test_processing_charts_use_standard_instra_panel_contract() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")
    for chart_name in ("processing_timeline", "processing_throughput", "processing_contention"):
        assert f'data-chart="{chart_name}"' in html
        assert f'data-maximize="{chart_name}"' in html
        assert f'id="{chart_name}_plot"' in html
    assert html.count('class="plot-mount processing-plot') >= 3
    assert html.count('class="panel-resizer panel-resizer-corner"') >= 3
    assert ".processing-card.chart-card" in css
    assert ".processing-grid.chart-grid" in css




def test_processing_visibility_is_owned_by_processing_view() -> None:
    premat_js = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")
    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert ":not(#processing_chart_group)" in premat_js
    assert "processing_apply_detail_tab" in premat_js
    assert "trace_available" in processing_js
    assert "Nsight charts appear after run completion" in processing_js
    assert 'group_count.textContent = trace_available ? "4" : "1"' in processing_js
    assert "Plotly.newPlot" in processing_js
    assert 'dataset.plotReady = "true"' in processing_js

# vvv THOG Processing presentation keeps diagnostic evidence first and throughput last, with explicit pane-filling maximize geometry
def test_processing_chart_order_and_maximize_geometry() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")
    assert html.index('data-chart="processing_timeline"') < html.index('data-chart="processing_contention"') < html.index('data-chart="processing_throughput"')
    assert ".processing-group.maximized" in css
    assert "height: 100% !important" in css
    assert "height: auto !important" not in css
    assert "align-content: stretch" in css
# ^^^ THOG


def test_processing_throughput_round_trips_through_local_store(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(database, run_name="fixture", config={})
    store.append_processing_throughput(1, 12345.5)
    store.append_processing_throughput(2, 13001.25)
    store.close()
    assert LocalChartReader(database).processing_throughput() == (
        {"optimizer_update": 1, "tokens_per_second": 12345.5},
        {"optimizer_update": 2, "tokens_per_second": 13001.25},
    )


def test_processing_group_is_available_before_first_live_throughput_sample(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(
        database,
        run_name="fixture",
        config={"premat_processing_logging": "enabled"},
    )
    store.close()

    payload = RunDashboardState(database).processing()

    assert payload["available"] is True
    assert payload["trace_available"] is False
    assert payload["data"]["throughput"] == []


def test_processing_group_remains_hidden_for_runs_without_processing_data(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(
        database,
        run_name="fixture",
        config={"premat_processing_logging": "disabled"},
    )
    store.close()

    payload = RunDashboardState(database).processing()

    assert payload["available"] is False
    assert payload["trace_available"] is False
    assert payload["data"] is None


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
                (4, "Unallocated Warps in Active SMs [Throughput %]"),
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
    assert payload["samples"][0]["active_sm_unused_warp_slots_pct"] == 20.0
    assert "active_sm_unused_warp_slots_pct" in payload["metadata"]["metric_mapping"]
    assert "Unallocated Warps in Active SMs [Throughput %]" in payload["metadata"]["available_gpu_metric_names"]
    assert {row["owner"] for row in payload["intervals"]} == {"MAIN", "PREMAT"}
    assert len(payload["summary"]) == 1
    summary = payload["summary"][0]
    assert summary["family"] == "QKV"
    assert summary["duration_ms"] == pytest.approx(0.002)
    assert summary["premat_overlap_ms"] == pytest.approx(0.0008)
    assert summary["premat_overlap_pct"] == pytest.approx(40.0)
    expected = {
        "fixture_processing_samples.csv",
        "fixture_processing_intervals.csv",
        "fixture_processing_summary.csv",
        "fixture_processing_metadata.json",
        "processing_data.json",
        "fixture_processing_bundle.zip",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert payload["metadata"]["files"] == {
        "samples": "fixture_processing_samples.csv",
        "intervals": "fixture_processing_intervals.csv",
        "summary": "fixture_processing_summary.csv",
        "metadata": "fixture_processing_metadata.json",
        "bundle": "fixture_processing_bundle.zip",
        "raw_trace": "fixture_processing_trace.nsys-rep",
    }
    with (output / "fixture_processing_summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["family"] == "QKV"
    with zipfile.ZipFile(output / "fixture_processing_bundle.zip") as archive:
        members = set(archive.namelist())
    assert "fixture_processing_samples.csv" in members
    assert "fixture_processing_intervals.csv" in members
    assert "fixture_processing_summary.csv" in members
    assert "fixture_processing_metadata.json" in members
    assert "processing_data.json" in members
# ^^^ THOG


# vvv THOG PREMAT in a gap between Main kernels is not simultaneous Main/PREMAT execution
def test_processing_overlap_excludes_internal_main_kernel_gaps(tmp_path: Path) -> None:
    database = tmp_path / "gap.sqlite"
    output = tmp_path / "processing_gap"
    with sqlite3.connect(database) as connection:
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
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "main_a"), (2, "main_b"), (3, "premat_gap")])
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (1000, 10000, 7, "THOG2_PREMAT_PROCESSING_CAPTURE"),
                (1500, 7500, 7, "THOG2_PROCESSING|owner=MAIN|operation=consume|family=QKV|layer=0"),
                (3500, 5500, 7, "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=O|layer=1"),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            [(1600, 1650, 7, 11), (5600, 5650, 7, 12), (3600, 3650, 7, 13)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            [
                (2000, 3000, 3, 11, 1),
                (6000, 7000, 3, 12, 2),
                (4000, 5000, 9, 13, 3),
            ],
        )
        connection.commit()
    payload = normalize_nsys_sqlite(database, output, capture_frequency_hz=10000)
    summary = payload["summary"][0]
    assert summary["duration_ms"] == pytest.approx(0.002)
    assert summary["premat_overlap_ms"] == pytest.approx(0.0)
    assert summary["premat_overlap_pct"] == pytest.approx(0.0)
# ^^^ THOG


# vvv THOG regress parent post-processing handoff, four-family scoreboard, and selector-aware Premat rendering
def test_processing_parent_retains_normalizer_payload_for_artifact_copy() -> None:
    source = Path("sheet/premat_processing.py").read_text(encoding="utf-8")
    assert "processing_data = normalize_nsys_sqlite(" in source
    assert 'processing_files = processing_data["metadata"]["files"]' in source


def test_processing_matrix_summary_reserves_all_four_families(tmp_path: Path) -> None:
    database = tmp_path / "matrix_summary.sqlite"
    output = tmp_path / "matrix_summary"
    _synthetic_nsys_database(database)
    payload = normalize_nsys_sqlite(database, output, capture_frequency_hz=10000, handoff={"run_name": "fixture"})
    assert list(payload["matrix_summary"]) == ["QKV", "O", "UP", "DOWN"]
    assert payload["matrix_summary"]["QKV"] is None
    assert payload["matrix_summary"]["O"] is None
    assert payload["matrix_summary"]["UP"] is None
    down = payload["matrix_summary"]["DOWN"]
    assert down is not None
    assert down["premat_materialisations"] == 1
    assert down["premat_gpu_ms_total"] == pytest.approx(0.0008)
    assert down["main_consume_overlap_ms"] == pytest.approx(0.0008)


def test_processing_dashboard_has_four_matrix_scoreboard() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert 'id="processing_matrix_summary_body"' in html
    assert '<th>QKV</th><th>O</th><th>UP</th><th>DOWN</th>' in html
    assert '"PREMAT GPU work"' in js
    assert '"Mean ready-before-consumption lead"' in js


def test_premat_matrix_selector_renders_non_targets_as_main_without_counting_misses() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")
    css = Path("sheet/local_dashboard_assets/dashboard_premat.css").read_text(encoding="utf-8")
    assert "function premat_target_family" in js
    assert 'record.outcome = targeted ? "COMPLETE MISS" : "NOT TARGETED"' in js
    assert 'record.path = targeted ? "main" : "not-targeted-main"' in js
    assert "if (record.targeted === false) continue;" not in js
    assert 'premat-not-targeted' in js
    assert '.premat-stage.premat-not-targeted' in css
# ^^^ THOG


# vvv THOG existing Processing captures must populate the new matrix scoreboard without regeneration
def test_processing_matrix_summary_has_legacy_interval_fallback() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert "function processing_matrix_summary_from_intervals" in js
    assert "payload.matrix_summary || processing_matrix_summary_from_intervals(payload.intervals || [])" in js
# ^^^ THOG


# vvv THOG Processing maximize/restore must resize Plotly from observed final card geometry, not one timing-sensitive callback
def test_processing_charts_observe_card_geometry_for_plotly_resize() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert "const processing_resize_observers = []" in js
    assert "new ResizeObserver(() => processing_resize_ready_card(card))" in js
    assert "observer.observe(card)" in js
    assert "Plotly.Plots.resize(mount)" in js
    assert "processing_install_resize_observers();" in js
# ^^^ THOG


# vvv THOG Processing group/navigation and selector-aware presentation regressions
def test_processing_group_is_collapsible_and_summary_is_visible_card() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")
    assert 'id="processing_group_toggle"' in html
    assert 'aria-controls="processing_grid"' in html
    assert 'id="processing_grid"' in html
    assert 'id="processing_matrix_summary_card"' in html
    assert html.index('id="processing_contention_plot"') < html.index('id="processing_matrix_summary_card"')
    assert ".processing-group.collapsed" in css
    assert ".processing-summary-card" in css


def test_processing_overlap_graph_uses_direct_two_denominator_profile() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert 'name: "PREMAT work concurrent with Main"' in js
    assert 'name: "Main busy time concurrent with PREMAT"' in js
    assert 'title: {text: "temporal overlap (%)", standoff: 12}' in js
    assert "range: [0, 100]" in js
    assert 'ticksuffix: "%"' in js
    assert js.count('textposition: "outside"') >= 2
    assert 'type: "scatter"' not in js[js.index("async function processing_render_contention"):js.index("function processing_format")]


def test_processing_gpu_charts_show_selected_run_artifact() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert 'id="processing_timeline_artifact"' in html
    assert 'id="processing_contention_artifact"' in html
    assert 'run.artifact_name || run.run_name || processing_current_run()' in js
    assert 'element.textContent = `Run artifact: ${artifact}`' in js
    assert "processing_set_chart_artifacts(payload);" in js


def test_premat_selector_exclusions_keep_main_activity_without_becoming_misses() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")
    assert '"not-targeted-main"' in js
    assert '"MAIN STREAM MATERIALISING · NOT TARGETED"' in js
    assert '"MAIN STREAM CONSUMING · NOT TARGETED"' in js
    assert 'record.outcome = targeted ? "COMPLETE MISS" : "NOT TARGETED"' in js
    assert "if (record.targeted === false) continue" not in js
# ^^^ THOG


# vvv THOG Training throughput overlays visible Workspace runs while capture-specific Processing charts remain selected-run diagnostics
def test_processing_throughput_workspace_multirun_contract() -> None:
    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    server = Path("run_thog2_local_dashboard.py").read_text(encoding="utf-8")
    assert "/api/processing-throughput" in server
    assert "def processing_throughput(self)" in server
    assert "processing_throughput_workspace_runs" in processing_js
    assert "app.workspace_mode === true" in processing_js
    assert "is_visible(run_identifier(run))" in processing_js
    assert "colour_for_run(entry.run_id)" in processing_js
    assert "/api/processing-throughput?run=" in processing_js
    assert processing_js.count("/api/processing-throughput?run=") == 1
    assert "processing_render_timeline(payload)" in processing_js
    assert "processing_render_contention(payload)" in processing_js
    assert "processing_render_summary(payload)" in processing_js
# ^^^ THOG


# vvv THOG Processing participates in the ordinary group stack and throughput follows ordinary INSTRA line presentation
def test_processing_group_sits_after_val_and_throughput_has_no_curve_markers() -> None:
    group_js = Path("sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js").read_text(encoding="utf-8")
    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    assert 'const processing_anchor = group_section("val") || group_section("train");' in group_js
    assert 'processing_anchor.after(processing_group)' in group_js
    assert 'entry.rows.length === 1 ? "markers" : "lines"' in processing_js
    assert 'entry.rows.length === 1 ? "markers" : "lines+markers"' not in processing_js
# ^^^ THOG


# vvv THOG complete-update accounting and paired Processing comparison regressions
def test_processing_update_timing_parses_string_premat_mode_exactly() -> None:
    assert _premat_is_enabled("enabled") is True
    assert _premat_is_enabled("disabled") is False
    assert _premat_is_enabled(False) is False


def test_processing_update_timing_gap_classification_covers_complete_update() -> None:
    forward = {"phase": "forward", "micro_step": 1}
    backward = {"phase": "backward", "micro_step": 1}
    optimizer = {"phase": "optimizer", "micro_step": None}
    assert _gap_phase(None, forward) == "setup"
    assert _gap_phase(forward, backward) == "forward"
    assert _gap_phase(backward, optimizer) == "post_backward"
    assert _gap_phase(optimizer, None) == "update_cleanup"


def test_processing_update_timing_reconciles_to_official_synchronized_elapsed() -> None:
    payload = {
        "captured_update_host_ms": 99.5,
        "timeline": [],
        "phase_totals_host_ms": {
            "setup": 4.0,
            "forward": 20.0,
            "backward": 50.0,
            "post_backward": 5.0,
            "optimizer": 10.0,
            "update_cleanup": 2.0,
            "gpu_completion_drain": 8.5,
            "unexplained_residual": 0.0,
        },
    }
    result = _apply_official_elapsed(payload, 100.0)
    assert result["host_update_ms"] == 100.0
    assert result["official_update_ms"] == 100.0
    assert result["unexplained_residual_ms"] == 0.5
    assert result["host_partition_sum_ms"] == 100.0
    assert result["host_partition_residual_ms"] == 0.0
    assert result["timeline"][-1]["phase"] == "unexplained_residual"


def test_processing_update_timing_uses_device_drain_and_preserves_trainer_identity() -> None:
    source = Path("sheet/processing_update_timing_patch.py").read_text(encoding="utf-8")
    assert "torch.cuda.synchronize(self.device)" in source
    assert "cuda_update_end.synchronize()" not in source
    assert "@wraps(original_train_one_update)" in source
    assert "result, elapsed = original_timed(function)" in source
    assert source.index("result, elapsed = original_timed(function)") < source.index("_publish_payload(", source.index("def timed_complete_update"))


def test_processing_update_timing_ui_uses_g_labels_pair_warning_and_fresh_workspace_data() -> None:
    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")
    assert "processing_update_timing_experiment_label" in js
    assert "capture?.run_label" in js
    assert "processing_update_timing_pair_assessment" in js
    assert "NOMAT followed by PREMAT" in js
    assert "match_signature" in js
    assert "timing_render_generation" in js
    assert "cached.revision === revision" in js
    assert "processing_update_timing_clear_plots" in js
    assert 'type: "bar"' in js
    assert "gpu_completion_drain" in js
    assert "unexplained_residual" in js
    assert ".processing-pair-warning" in css
    assert ".processing-update-timing-timeline-card.maximized" in css
# ^^^ THOG
