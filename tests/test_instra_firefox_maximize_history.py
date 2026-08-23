# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from tests.test_instra_firefox_acceptance import (
    _checkbox_checked,
    _firefox_driver,
    _free_port,
    _make_legacy_store,
    _open_group,
    _open_settings_and_wait,
    _save_settings_and_wait,
    _wait,
    _wait_for_server,
)


def _q_trace_count(driver) -> int:
    return int(driver.execute_script(
        """
        const mount = document.getElementById('attn_q_head_N_plot');
        return (mount?.data || []).filter(trace => {
          if (trace?.meta?.instra_top_axis_anchor === true) return false;
          const mode = String(trace?.mode || '');
          return mode.includes('lines') && Array.isArray(trace?.x) && trace.x.length > 0;
        }).length;
        """
    ))


def test_real_firefox_maximize_current_only_history_round_trip(tmp_path: Path) -> None:
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
            lambda: bool(driver.execute_script(
                "const el=document.getElementById('attn_q_head_N_plot'); return !!el && el.dataset.plotReady === 'true';"
            )),
            timeout=20,
            message="Q history chart did not render",
        )
        _wait(
            driver,
            lambda: _q_trace_count(driver) > 1,
            message="Q history did not contain multiple recorded snapshots",
        )
        history_count = _q_trace_count(driver)

        driver.execute_script(
            """
            const button = document.querySelector('.chart-card[data-chart="attn_q_head_N"] .maximize-button');
            if (!button) throw new Error('Q maximize button missing');
            button.click();
            """
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const card = document.querySelector('.chart-card[data-chart="attn_q_head_N"]');
                return !!card?.classList.contains('maximized')
                  && document.getElementById('charts_scroll')?.classList.contains('maximized-mode');
                """
            )),
            message="Q chart did not enter maximized mode",
        )

        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.getElementById('weights_group_settings_button') !== null;")),
            message="Weights group settings button did not appear",
        )
        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_checked(driver, "chart_current_weights_only") is False
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        assert _checkbox_checked(driver, "chart_current_weights_only") is True
        _save_settings_and_wait(driver)
        _wait(
            driver,
            lambda: 0 < _q_trace_count(driver) < history_count,
            timeout=20,
            message="maximized Q chart did not collapse to current-only data",
        )
        current_count = _q_trace_count(driver)

        driver.execute_script(
            """
            const button = document.querySelector('.chart-card[data-chart="attn_q_head_N"] .maximize-button');
            if (!button) throw new Error('Q restore button missing');
            button.click();
            """
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "return !document.getElementById('charts_scroll')?.classList.contains('maximized-mode');"
            )),
            message="Q chart did not restore from maximized mode",
        )
        assert _q_trace_count(driver) == current_count

        _open_settings_and_wait(driver, "#weights_group_settings_button")
        assert _checkbox_checked(driver, "chart_current_weights_only") is True
        driver.execute_script("document.getElementById('chart_current_weights_only').click();")
        assert _checkbox_checked(driver, "chart_current_weights_only") is False
        _save_settings_and_wait(driver)
        _wait(
            driver,
            lambda: _q_trace_count(driver) == history_count,
            timeout=20,
            message="turning current-only off did not restore the original Q history",
        )
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
