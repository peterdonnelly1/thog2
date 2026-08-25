# vvv THOG
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from sheet.local_chart_store import LocalChartStore


_WEIGHT_CHARTS = (
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _legacy_probe(step: int) -> dict[str, object]:
    active = 16
    return {
        "optimizer_update": step,
        "probe_id": f"legacy-{step}",
        "active_layers": active,
        "selected_layers": active,
        "shrink": tuple(
            (distance, -0.0025 * distance - step * 1e-6, active - distance, -distance)
            for distance in range(1, 6)
        ),
        "growth": tuple(
            (distance, 0.0020 * distance + step * 1e-6, active + distance, distance)
            for distance in range(1, 6)
        ),
        # Deliberately no current_loss: this is the historical-store compatibility case.
    }


def _weight_snapshot(step: int) -> dict[str, object]:
    families: dict[str, object] = {}
    for chart_index, chart_name in enumerate(_WEIGHT_CHARTS):
        values = tuple(
            0.01 * (chart_index + 1) + 0.001 * layer + step * 1e-7
            for layer in range(1, 17)
        )
        families[chart_name] = {
            "semantic_family": chart_name,
            "depth_coordinates": tuple(float(layer) for layer in range(1, 17)),
            "executed_layer_coordinates": tuple(float(layer) for layer in range(1, 17)),
            "curves": (
                {
                    "scalar_id": "r0_c0",
                    "output_row": 0,
                    "row_index": 0,
                    "values": values,
                    "executed_values": values,
                },
            ),
        }
    return {
        "optimizer_update": step,
        "attention_head": 0,
        "families": families,
    }


def _make_legacy_store(root: Path) -> tuple[Path, str]:
    run_id = "firefox_acceptance"
    path = root / "firefox_acceptance_artifact" / run_id / "charts.sqlite3"
    store = LocalChartStore(
        path,
        run_name="firefox_acceptance_artifact",
        run_id=run_id,
        config={
            "host_label": "scruffy",
            "model_type": "sheet",
            "instrumentation__depth_weight_curves__history_length": 100,
            "instrumentation__delta_loss_v_layer_heatmap": True,
            "instrumentation__delta_loss_v_layer_heatmap__destination": "local",
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
        },
    )
    store.append_heatmap_records(_legacy_probe(step) for step in range(912, 1004))
    for step in range(999, 1004):
        store.append_depth_weight_snapshot(_weight_snapshot(step), history_length=100)
    store.close()
    return path, run_id


def _make_selected_thog_store(root: Path) -> tuple[Path, str]:
    run_id = "firefox_selected_thog"
    path = root / "firefox_selected_thog_artifact" / run_id / "charts.sqlite3"
    store = LocalChartStore(
        path,
        run_name="firefox_selected_thog_artifact",
        run_id=run_id,
        config={
            "host_label": "scruffy",
            "model_type": "sheet",
            "instrumentation__depth_weight_curves__history_length": 100,
            "instrumentation__delta_loss_v_layer_heatmap": True,
            "instrumentation__delta_loss_v_layer_heatmap__destination": "local",
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
        },
    )
    store.append_heatmap_records(_legacy_probe(step) for step in range(120, 128))
    for step in (126, 127):
        families: dict[str, object] = {}
        for chart_index, chart_name in enumerate(_WEIGHT_CHARTS):
            coordinates = tuple(float(layer) for layer in range(1, 17))
            reverse = chart_name in {"attn_out_head_N", "mlp_down"}
            random_model, random_intermediate = 2, 3
            selected_model, selected_intermediate = 12, 14
            random_row, random_column = (
                (random_model, random_intermediate)
                if reverse
                else (random_intermediate, random_model)
            )
            selected_row, selected_column = (
                (selected_model, selected_intermediate)
                if reverse
                else (selected_intermediate, selected_model)
            )
            curves = []
            for curve_index, (row, column, model_feature, intermediate_feature, kind) in enumerate((
                (random_row, random_column, random_model, random_intermediate, "random"),
                (selected_row, selected_column, selected_model, selected_intermediate, "user"),
            )):
                values = tuple(
                    0.01 * (chart_index + 1)
                    + 0.002 * layer
                    + 0.00004 * curve_index * layer * layer
                    + step * 1e-7
                    for layer in range(1, 17)
                )
                curves.append({
                    "scalar_id": f"r{row}_c{column}",
                    "output_row": row,
                    "row_index": column,
                    "values": values,
                    "executed_values": values,
                    "model_feature": model_feature,
                    "intermediate_feature": intermediate_feature,
                    "selection_kind": kind,
                })
            families[chart_name] = {
                "semantic_family": chart_name,
                "depth_coordinates": coordinates,
                "executed_layer_coordinates": coordinates,
                "curves": tuple(curves),
            }
        store.append_depth_weight_snapshot(
            {
                "optimizer_update": step,
                "attention_head": 0,
                "families": families,
                "weight_selection": {
                    "protocol": "matched_six_v1",
                    "user_selected": True,
                    "model_feature": 12,
                    "intermediate_feature": 14,
                    "feature_count": 16,
                    "applied": True,
                },
            },
            history_length=100,
        )
    store.close()
    (root / ".instra_weight_selection.json").write_text(
        json.dumps({
            "protocol": "matched_six_v1",
            "user_selected": True,
            "model_feature": 12,
            "intermediate_feature": 14,
        }),
        encoding="utf-8",
    )
    return path, run_id


def _wait_for_server(url: str, process: subprocess.Popen[str], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                f"INSTRA server exited early ({process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with urlopen(f"{url}/api/runs", timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("INSTRA server did not become ready")


def _firefox_driver():
    selenium = pytest.importorskip("selenium")
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    options = Options()
    options.add_argument("-headless")
    options.set_preference("browser.cache.disk.enable", False)
    options.set_preference("browser.cache.memory.enable", False)
    options.set_preference("browser.cache.offline.enable", False)
    try:
        return webdriver.Firefox(options=options)
    except Exception as error:  # pragma: no cover - CI environment diagnostic
        raise AssertionError(f"headless Firefox could not start: {error}") from error


def _wait(driver, predicate, *, timeout: float = 15.0, message: str) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as error:  # DOM can be replaced between polls.
            last_error = error
        time.sleep(0.05)
    suffix = f"; last error: {last_error}" if last_error is not None else ""
    raise AssertionError(f"{message}{suffix}")


def _open_group(driver, group_id: str, toggle_id: str) -> None:
    driver.execute_script(
        """
        const group = document.getElementById(arguments[0]);
        const toggle = document.getElementById(arguments[1]);
        if (!group || !toggle) throw new Error(`missing ${arguments[0]} / ${arguments[1]}`);
        if (group.classList.contains('collapsed')) toggle.click();
        """,
        group_id,
        toggle_id,
    )


def _checkbox_enabled(driver, element_id: str) -> bool:
    return bool(
        driver.execute_script(
            "const el=document.getElementById(arguments[0]); return !!el && !el.disabled;",
            element_id,
        )
    )


def _checkbox_checked(driver, element_id: str) -> bool:
    return bool(
        driver.execute_script(
            "const el=document.getElementById(arguments[0]); return !!el && !!el.checked;",
            element_id,
        )
    )


def _open_settings_and_wait(driver, button_selector: str) -> None:
    driver.execute_script(
        "const el=document.querySelector(arguments[0]); if(!el) throw new Error(arguments[0]); el.click();",
        button_selector,
    )
    _wait(
        driver,
        lambda: bool(driver.execute_script(
            "const el=document.getElementById('chart_settings_overlay'); return !!el && !el.hidden;"
        )),
        message=f"settings overlay did not open from {button_selector}",
    )


def _save_settings_and_wait(driver) -> None:
    driver.execute_script("document.getElementById('save_chart_settings').click();")
    _wait(
        driver,
        lambda: bool(driver.execute_script(
            "const el=document.getElementById('chart_settings_overlay'); return !!el && el.hidden;"
        )),
        message="settings overlay did not close after Apply",
    )


def test_real_firefox_heatmap_and_weight_group_chart_overrides(tmp_path: Path) -> None:
    _database, run_id = _make_legacy_store(tmp_path)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "run_thog2_dashboard.py",
            "--root",
            str(tmp_path),
            "--run",
            run_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    driver = None
    try:
        _wait_for_server(base_url, process)
        driver = _firefox_driver()
        driver.set_page_load_timeout(30)
        driver.get(f"{base_url}/runs/{run_id}")

        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.querySelector('tr.current-run') !== null;"
            )),
            timeout=20,
            message="fixture run was not selected",
        )

        _open_group(driver, "heatmap_chart_group", "heatmap_group_toggle")
        heatmap_started = time.monotonic()
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "const el=document.getElementById('heatmap_plot'); return !!el && el.dataset.plotReady === 'true';"
            )),
            timeout=15,
            message="legacy heatmap never reached Plotly plotReady",
        )
        heatmap_seconds = time.monotonic() - heatmap_started
        assert heatmap_seconds < 15.0
        assert driver.execute_script(
            "const el=document.getElementById('heatmap_placeholder'); return !!el && el.hidden;"
        )
        finite_cells = driver.execute_script(
            """
            const mount = document.getElementById('heatmap_plot');
            const trace = (mount?.data || []).find(item => item.type === 'heatmap');
            if (!trace || !Array.isArray(trace.z)) return 0;
            return trace.z.flat().filter(value => Number.isFinite(Number(value)) && Number(value) !== 0).length;
            """
        )
        assert int(finite_cells or 0) > 0, "legacy heatmap rendered without coloured finite cells"
        assert driver.execute_script(
            "return document.getElementById('heatmap_delta_loss_mode')?.textContent?.trim();"
        ) == "|abs|"

        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.getElementById('weights_group_settings_button') !== null;"
            )),
            message="Weights group settings button did not appear",
        )

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_enabled(driver, "chart_current_weights_only"), (
            "group-level Current weights only is disabled"
        )
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), (
            "group-level Join with line segments is disabled"
        )
        before = _checkbox_checked(driver, "chart_current_weights_only")
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        assert _checkbox_checked(driver, "chart_current_weights_only") is (not before)
        group_expected = not before
        _save_settings_and_wait(driver)

        _open_settings_and_wait(
            driver,
            '.chart-card[data-chart="attn_q_head_N"] .chart-settings-button',
        )
        assert _checkbox_enabled(driver, "chart_current_weights_only"), (
            "individual-chart Current weights only is disabled"
        )
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), (
            "individual-chart Join with line segments is disabled"
        )
        assert _checkbox_checked(driver, "chart_current_weights_only") is group_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is True
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        chart_expected = not group_expected
        assert _checkbox_checked(driver, "chart_current_weights_only") is chart_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is False
        _save_settings_and_wait(driver)

        _open_settings_and_wait(
            driver,
            '.chart-card[data-chart="attn_q_head_N"] .chart-settings-button',
        )
        assert _checkbox_checked(driver, "chart_current_weights_only") is chart_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is False
        driver.execute_script("document.getElementById('cancel_chart_settings').click();")

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_enabled(driver, "chart_current_weights_only")
        assert _checkbox_checked(driver, "chart_current_weights_only") is group_expected
    finally:
        if driver is not None:
            driver.quit()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
# ^^^ THOG


# vvv THOG the modern THOG path keeps heatmaps discoverable and renders only the group-selected integer-segment coupling without duplicate Plotly chrome
def test_real_firefox_selected_thog_render_contract(tmp_path: Path) -> None:
    _database, run_id = _make_selected_thog_store(tmp_path)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "run_thog2_dashboard.py",
            "--root",
            str(tmp_path),
            "--run",
            run_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    driver = None
    try:
        _wait_for_server(base_url, process)
        driver = _firefox_driver()
        driver.set_page_load_timeout(30)
        driver.get(f"{base_url}/runs/{run_id}")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.querySelector('tr.current-run') !== null;"
            )),
            timeout=20,
            message="selected THOG fixture run was not selected",
        )

        assert driver.execute_script(
            "return document.getElementById('heatmap_chart_group')?.hidden === false;"
        ), "THOG heatmap group was hidden"
        _open_group(driver, "heatmap_chart_group", "heatmap_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.getElementById('heatmap_plot')?.dataset.plotReady === 'true';"
            )),
            timeout=20,
            message="THOG heatmap did not render",
        )

        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.getElementById('weights_group_settings_button') !== null;"
            )),
            message="Weights group settings button did not appear",
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return /12\\s*→\\s*14/.test(document.getElementById('weight_index_group_summary')?.textContent || '');"
            )),
            message="group-level selected coupling did not load",
        )

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        for element_id in ("chart_current_weights_only", "chart_join_with_line_segments"):
            if not _checkbox_checked(driver, element_id):
                driver.execute_script(
                    "document.getElementById(arguments[0]).click();",
                    element_id,
                )
        render_started = time.monotonic()
        _save_settings_and_wait(driver)

        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const mount = document.getElementById('attn_q_head_N_plot');
                const traces = (mount?.data || []).filter(
                  trace => trace?.meta?.instra_top_axis_anchor !== true
                );
                const selected = traces.find(trace => (
                  trace?.meta?.instra_weight_model_feature === 12
                  && trace?.meta?.instra_weight_intermediate_feature === 14
                  && trace?.meta?.instra_thog_weight === true
                  && String(trace?.mode || '').includes('lines')
                ));
                return (
                  traces.length === 1
                  && !!selected
                  && selected?.line?.shape === 'linear'
                  && traces.every(trace => trace?.meta?.instra_thog_executed_overlay !== true)
                );
                """
            )),
            timeout=20,
            message="selected THOG coupling did not reach the post-save line-segment render contract",
        )
        render_seconds = time.monotonic() - render_started
        assert render_seconds < 10.0, f"post-save Weights render took {render_seconds:.2f}s"

        render_contract = driver.execute_script(
            """
            const mount = document.getElementById('attn_q_head_N_plot');
            const traces = (mount?.data || []).filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
            const selected = traces.find(trace => (
              trace?.meta?.instra_weight_model_feature === 12
              && trace?.meta?.instra_weight_intermediate_feature === 14
              && String(trace?.mode || '').includes('lines')
            ));
            return {
              trace_count: traces.length,
              selected_x: selected?.x || [],
              selected_shape: selected?.line?.shape || null,
              selected_mode: selected?.mode || null,
              wrong_coupling_count: traces.filter(trace => (
                trace?.meta?.instra_weight_selection_protocol === 'matched_six_v1'
                && (
                  trace?.meta?.instra_weight_model_feature !== 12
                  || trace?.meta?.instra_weight_intermediate_feature !== 14
                )
              )).length,
              executed_overlay_count: traces.filter(
                trace => trace?.meta?.instra_thog_executed_overlay === true
              ).length,
              title: mount?.layout?.title?.text ?? mount?.layout?.title ?? null,
              showlegend: mount?.layout?.showlegend ?? null,
              legend: mount?.layout?.legend ?? null,
              top_title: mount?.layout?.xaxis2?.title?.text ?? mount?.layout?.xaxis2?.title ?? null,
            };
            """
        )
        assert render_contract["trace_count"] == 1, render_contract
        assert render_contract["selected_x"] == list(range(1, 17)), render_contract
        assert render_contract["selected_shape"] == "linear", render_contract
        assert "lines" in render_contract["selected_mode"], render_contract
        assert render_contract["wrong_coupling_count"] == 0, render_contract
        assert render_contract["executed_overlay_count"] == 0, render_contract
        assert render_contract["title"] is None, render_contract
        assert render_contract["showlegend"] is False, render_contract
        assert render_contract["legend"] is None, render_contract
        assert render_contract["top_title"] is None, render_contract
    finally:
        if driver is not None:
            driver.quit()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
# ^^^ THOG
