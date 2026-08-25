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
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                return document.getElementById('weight_step_from')?.value === '999'
                  && document.getElementById('weight_step_to')?.value === '1003';
                """
            )),
            timeout=20,
            message="initial retained range controls were not populated",
        )

        initial = driver.execute_script(
            """
            const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
              .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
            const steps = [...new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite))].sort((a,b) => a-b);
            return {
              steps,
              from: document.getElementById('weight_step_from')?.value,
              to: document.getElementById('weight_step_to')?.value,
              colours: [...new Set(traces.map(trace => trace?.line?.color).filter(Boolean))],
            };
            """
        )
        assert initial["steps"] == [999, 1000, 1001, 1002, 1003], initial
        assert initial["from"] == "999", initial
        assert initial["to"] == "1003", initial
        assert len(initial["colours"]) == 5, initial

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

        driver.execute_script("document.getElementById('weight_step_latest').click();")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                const steps = new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite));
                return steps.size === 1
                  && document.getElementById('weight_step_from')?.value === '1003'
                  && document.getElementById('weight_step_to')?.value === '1003';
                """
            )),
            timeout=20,
            message="Latest step did not select and populate step 1003",
        )
        latest_colour = driver.execute_script(
            """
            const trace = (document.getElementById('attn_q_head_N_plot')?.data || [])
              .find(value => value?.meta?.instra_top_axis_anchor !== true);
            return {trace: trace?.line?.color || null, run: colour_for_run(window.app.current_run_id)};
            """
        )
        assert latest_colour["trace"] == latest_colour["run"], latest_colour

        driver.execute_script("document.getElementById('weight_step_whole_range').click();")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                const steps = new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite));
                return steps.size === 5
                  && document.getElementById('weight_step_from')?.value === '999'
                  && document.getElementById('weight_step_to')?.value === '1003';
                """
            )),
            timeout=20,
            message="Whole range did not restore all retained curves and bounds",
        )

        invalid = driver.execute_script(
            """
            const to = document.getElementById('weight_step_to');
            to.value = '2000';
            to.dispatchEvent(new Event('input', {bubbles: true}));
            document.getElementById('weight_step_apply').click();
            const error = document.getElementById('weight_step_range_error');
            return {to: to.value, text: error?.textContent || '', hidden: error?.hidden};
            """
        )
        assert invalid["to"] == "1003", invalid
        assert invalid["hidden"] is False, invalid
        assert "'to' value cannot be greater than 1003" in invalid["text"], invalid
        driver.execute_script("document.getElementById('weight_step_apply').click();")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return document.getElementById('weight_step_range_error')?.hidden === true;"
            )),
            message="corrected range could not be accepted with a second Show",
        )
    finally:
        _stop_dashboard(process, driver)


def test_real_firefox_weight_settings_previews_show_current_curves(tmp_path: Path) -> None:
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
            message="preview fixture run was not selected",
        )
        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return !!window.__instra_weight_range_interaction_final && document.getElementById('weights_group_settings_button') !== null;"
            )),
            timeout=20,
            message="weight preview controls did not become ready",
        )

        def preview_step_count() -> int:
            return int(driver.execute_script(
                """
                const traces = (document.getElementById('chart_settings_preview')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                return new Set(traces.map(trace => trace_optimizer_update(trace)).filter(Number.isFinite)).size;
                """
            ))

        for selector in (
            "#weights_group_settings_button",
            '.chart-card[data-chart="attn_q_head_N"] .chart-settings-button',
        ):
            _open_settings_and_wait(driver, selector)
            initial_current_only = _checkbox_checked(driver, "chart_current_weights_only")
            _wait(
                driver,
                lambda: preview_step_count() == (1 if initial_current_only else 2),
                timeout=20,
                message=f"initial preview did not match Current-only for {selector}",
            )
            if initial_current_only:
                driver.execute_script("document.getElementById('chart_current_weights_only').click();")
                _wait(
                    driver,
                    lambda: preview_step_count() == 2,
                    message=f"history preview did not recover for {selector}",
                )
            driver.execute_script("document.getElementById('chart_current_weights_only').click();")
            _wait(
                driver,
                lambda: preview_step_count() == 1,
                timeout=20,
                message=f"Current-only preview was blank for {selector}",
            )
            driver.execute_script("document.getElementById('chart_current_weights_only').click();")
            _wait(
                driver,
                lambda: preview_step_count() == 2,
                message=f"history preview did not return for {selector}",
            )
            driver.execute_script("document.getElementById('cancel_chart_settings').click();")
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
        before = driver.execute_script(
            """
            const mount = document.getElementById('attn_q_head_N_plot');
            const traces = (mount?.data || []).filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
            return {
              viewer: window.__instra_weight_viewer_selection.selection(),
              capture: window.__instra_matched_weight_selection.selection(),
              values: traces.map(trace => trace.y),
            };
            """
        )
        driver.execute_script("document.getElementById('weight_random_jump').click();")
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const before = arguments[0];
                const current = window.__instra_weight_viewer_selection.selection();
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                return current.user_selected === true
                  && current.model_feature !== before.viewer.model_feature
                  && current.intermediate_feature !== before.viewer.intermediate_feature
                  && JSON.stringify(traces.map(trace => trace.y)) !== JSON.stringify(before.values);
                """,
                before,
            )),
            timeout=10,
            message="RND changed neither the retained coupling nor the rendered curves",
        )
        after = driver.execute_script(
            """
            return {
              viewer: window.__instra_weight_viewer_selection.selection(),
              capture: window.__instra_matched_weight_selection.selection(),
            };
            """
        )
        assert after["viewer"]["model_feature"] != before["viewer"]["model_feature"], (before, after)
        assert after["viewer"]["intermediate_feature"] != before["viewer"]["intermediate_feature"], (before, after)
        assert after["capture"] == before["capture"], (before, after)

        driver.execute_script(
            """
            const input = document.getElementById('weight_coupling_input');
            const output = document.getElementById('weight_coupling_output');
            input.value = '12';
            output.value = '14';
            output.dispatchEvent(new Event('change', {bubbles: true}));
            """
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const pair = window.__instra_weight_viewer_selection.pair();
                return pair?.model_feature === 12 && pair?.intermediate_feature === 14;
                """
            )),
            message="manual recorded coupling did not redraw immediately",
        )
        rejected = driver.execute_script(
            """
            const input = document.getElementById('weight_coupling_input');
            const output = document.getElementById('weight_coupling_output');
            input.value = '7';
            output.value = '8';
            output.dispatchEvent(new Event('change', {bubbles: true}));
            return {
              pair: window.__instra_weight_viewer_selection.pair(),
              input: input.value,
              output: output.value,
              error: document.getElementById('weight_coupling_view_error')?.textContent || '',
            };
            """
        )
        assert rejected["pair"] == {"model_feature": 12, "intermediate_feature": 14}, rejected
        assert rejected["input"] == "12" and rejected["output"] == "14", rejected
        assert "was not recorded" in rejected["error"], rejected
    finally:
        _stop_dashboard(process, driver)
# ^^^ THOG
