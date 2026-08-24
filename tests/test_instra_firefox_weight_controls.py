# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from tests.test_instra_firefox_acceptance import (
    _checkbox_checked,
    _checkbox_enabled,
    _firefox_driver,
    _free_port,
    _make_legacy_store,
    _make_selected_thog_store,
    _open_group,
    _open_settings_and_wait,
    _save_settings_and_wait,
    _wait,
    _wait_for_server,
)


def _start_dashboard(tmp_path: Path, run_id: str) -> tuple[subprocess.Popen[str], str]:
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
    _wait_for_server(base_url, process)
    return process, base_url


def _stop_dashboard(process: subprocess.Popen[str], driver) -> None:
    if driver is not None:
        driver.quit()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_real_firefox_weight_group_and_chart_overrides_are_clickable(tmp_path: Path) -> None:
    _database, run_id = _make_legacy_store(tmp_path)
    process, base_url = _start_dashboard(tmp_path, run_id)
    driver = None
    try:
        driver = _firefox_driver()
        driver.set_page_load_timeout(30)
        driver.get(f"{base_url}/runs/{run_id}")
        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.querySelector('tr.current-run') !== null;")),
            timeout=20,
            message="fixture run was not selected",
        )
        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.getElementById('weights_group_settings_button') !== null;")),
            message="Weights group settings button did not appear",
        )

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        diagnostic = driver.execute_script(
            """
            const current = document.getElementById('chart_current_weights_only');
            return {
              disabled: current?.disabled,
              axis_chart_name: window.app?.axis_chart_name ?? null,
              overlay_hidden: document.getElementById('chart_settings_overlay')?.hidden,
              title: document.getElementById('chart_settings_title')?.textContent,
              stability_loaded: !!window.__instra_weight_stability_final,
              step_filter_active: window.__instra_weight_step_filter?.active?.() ?? null,
            };
            """
        )
        assert _checkbox_enabled(driver, "chart_current_weights_only"), f"group Current-only disabled: {diagnostic}"
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), f"group Join disabled: {diagnostic}"
        before = _checkbox_checked(driver, "chart_current_weights_only")
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        group_expected = not before
        assert _checkbox_checked(driver, "chart_current_weights_only") is group_expected
        _save_settings_and_wait(driver)

        _open_settings_and_wait(driver, '.chart-card[data-chart="attn_q_head_N"] .chart-settings-button')
        diagnostic = driver.execute_script(
            """
            const current = document.getElementById('chart_current_weights_only');
            return {
              disabled: current?.disabled,
              axis_chart_name: window.app?.axis_chart_name ?? null,
              overlay_hidden: document.getElementById('chart_settings_overlay')?.hidden,
              title: document.getElementById('chart_settings_title')?.textContent,
              stability_loaded: !!window.__instra_weight_stability_final,
              step_filter_active: window.__instra_weight_step_filter?.active?.() ?? null,
            };
            """
        )
        assert _checkbox_enabled(driver, "chart_current_weights_only"), f"individual Current-only disabled: {diagnostic}"
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), f"individual Join disabled: {diagnostic}"
        assert _checkbox_checked(driver, "chart_current_weights_only") is group_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is True
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        chart_expected = not group_expected
        assert _checkbox_checked(driver, "chart_current_weights_only") is chart_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is False
        _save_settings_and_wait(driver)

        _open_settings_and_wait(driver, '.chart-card[data-chart="attn_q_head_N"] .chart-settings-button')
        assert _checkbox_checked(driver, "chart_current_weights_only") is chart_expected
        assert _checkbox_checked(driver, "chart_inherit_weights_group") is False
        driver.execute_script("document.getElementById('cancel_chart_settings').click();")

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_enabled(driver, "chart_current_weights_only")
        assert _checkbox_checked(driver, "chart_current_weights_only") is group_expected
    finally:
        _stop_dashboard(process, driver)


def test_real_firefox_explicit_weight_range_overrides_current_only_and_has_step_hover(tmp_path: Path) -> None:
    _database, run_id = _make_legacy_store(tmp_path)
    process, base_url = _start_dashboard(tmp_path, run_id)
    driver = None
    try:
        driver = _firefox_driver()
        driver.set_page_load_timeout(30)
        driver.get(f"{base_url}/runs/{run_id}")
        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.querySelector('tr.current-run') !== null;")),
            timeout=20,
            message="range fixture run was not selected",
        )
        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return !!window.__instra_weight_range_interaction_final && document.getElementById('attn_q_head_N_plot')?.dataset.plotReady === 'true';"
            )),
            timeout=20,
            message="final weight-range owner did not install",
        )

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        if not _checkbox_checked(driver, "chart_current_weights_only"):
            driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        _save_settings_and_wait(driver)
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                const steps = new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite));
                return steps.size === 1;
                """
            )),
            timeout=20,
            message="Current weights only did not collapse the no-range view to one step",
        )

        driver.execute_script("window.__instra_weight_stability_final.set_range(999, 1003);")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                const steps = new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite));
                return steps.size === 5;
                """
            )),
            timeout=20,
            message="explicit 999–1003 range did not render all five retained steps",
        )

        contract = driver.execute_script(
            """
            const mount = document.getElementById('attn_q_head_N_plot');
            const traces = (mount?.data || []).filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
            const steps = [...new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite))].sort((a,b) => a-b);
            const hover_ok = traces.every(trace => {
              const step = trace_optimizer_update(trace);
              return !Number.isFinite(step) || String(trace.hovertemplate || '').includes(`step ${step}`);
            });
            const input = document.getElementById('weight_coupling_input');
            const output = document.getElementById('weight_coupling_output');
            return {
              steps,
              hover_ok,
              current_only: window.__instra_weight_stability_final.effective('attn_q_head_N').current_weights_only,
              selected_range: window.__instra_weight_stability_final.selected_range(),
              input_width: input?.getBoundingClientRect().width || 0,
              output_width: output?.getBoundingClientRect().width || 0,
            };
            """
        )
        assert contract["steps"] == [999, 1000, 1001, 1002, 1003], contract
        assert contract["hover_ok"] is True, contract
        assert contract["current_only"] is True, contract
        assert contract["selected_range"] == {"minimum": 999, "maximum": 1003}, contract
        assert contract["input_width"] >= 60, contract
        assert contract["output_width"] >= 60, contract
    finally:
        _stop_dashboard(process, driver)


def test_real_firefox_rnd_changes_weight_coupling(tmp_path: Path) -> None:
    _database, run_id = _make_selected_thog_store(tmp_path)
    process, base_url = _start_dashboard(tmp_path, run_id)
    driver = None
    try:
        driver = _firefox_driver()
        driver.set_page_load_timeout(30)
        driver.get(f"{base_url}/runs/{run_id}")
        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.querySelector('tr.current-run') !== null;")),
            timeout=20,
            message="RND fixture run was not selected",
        )
        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const button = document.getElementById('weight_random_jump');
                const input = document.getElementById('weight_coupling_input');
                const output = document.getElementById('weight_coupling_output');
                return !!window.__instra_weight_range_interaction_final && !!button && !button.disabled && !!input && !input.disabled && !!output && !output.disabled;
                """
            )),
            timeout=20,
            message="RND controls did not become ready",
        )
        before = driver.execute_script("return window.__instra_matched_weight_selection.selection();")
        driver.execute_script("document.getElementById('weight_random_jump').click();")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const before = arguments[0];
                const current = window.__instra_matched_weight_selection.selection();
                return current.user_selected === true
                  && current.model_feature !== before.model_feature
                  && current.intermediate_feature !== before.intermediate_feature;
                """,
                before,
            )),
            timeout=10,
            message="RND did not select a different valid coupling",
        )
        after = driver.execute_script("return window.__instra_matched_weight_selection.selection();")
        assert after["model_feature"] != before["model_feature"], (before, after)
        assert after["intermediate_feature"] != before["intermediate_feature"], (before, after)
        assert 0 <= after["model_feature"] < 16, after
        assert 0 <= after["intermediate_feature"] < 16, after
    finally:
        _stop_dashboard(process, driver)
# ^^^ THOG
