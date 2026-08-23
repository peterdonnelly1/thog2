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
    _open_group,
    _open_settings_and_wait,
    _save_settings_and_wait,
    _wait,
    _wait_for_server,
)


def test_real_firefox_global_weight_controls_are_clickable(tmp_path: Path) -> None:
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
              repair_loaded: !!window.__instra_regression_repair,
              step_filter_active: window.__instra_weight_step_filter?.active?.() ?? null,
            };
            """
        )
        assert _checkbox_enabled(driver, "chart_current_weights_only"), f"group Current-only disabled: {diagnostic}"
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), f"group Join disabled: {diagnostic}"
        before = _checkbox_checked(driver, "chart_current_weights_only")
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        expected = not before
        assert _checkbox_checked(driver, "chart_current_weights_only") is expected
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
              repair_loaded: !!window.__instra_regression_repair,
              step_filter_active: window.__instra_weight_step_filter?.active?.() ?? null,
            };
            """
        )
        assert _checkbox_enabled(driver, "chart_current_weights_only"), f"individual Current-only disabled: {diagnostic}"
        assert _checkbox_enabled(driver, "chart_join_with_line_segments"), f"individual Join disabled: {diagnostic}"
        assert _checkbox_checked(driver, "chart_current_weights_only") is expected
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        expected = not expected
        _save_settings_and_wait(driver)

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_enabled(driver, "chart_current_weights_only")
        assert _checkbox_checked(driver, "chart_current_weights_only") is expected
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
