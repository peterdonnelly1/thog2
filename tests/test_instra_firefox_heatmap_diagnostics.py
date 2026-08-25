# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from tests.test_instra_firefox_acceptance import (
    _firefox_driver,
    _free_port,
    _make_legacy_store,
    _open_group,
    _wait,
    _wait_for_server,
)


def test_real_firefox_heatmap_pipeline_diagnostics(tmp_path: Path) -> None:
    _database, run_id = _make_legacy_store(tmp_path)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, "run_thog2_dashboard.py", "--root", str(tmp_path), "--run", run_id,
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    driver = None
    try:
        _wait_for_server(base_url, process)
        driver = _firefox_driver()
        driver.get(f"{base_url}/runs/{run_id}")
        _wait(driver, lambda: bool(driver.execute_script("return document.querySelector('tr.current-run') !== null;")),
              timeout=20, message="fixture run was not selected")
        _open_group(driver, "heatmap_chart_group", "heatmap_group_toggle")
        _wait(driver, lambda: bool(driver.execute_script(
            "const el=document.getElementById('heatmap_plot'); return !!el && el.dataset.plotReady === 'true';")),
            timeout=15, message="heatmap never reached plotReady")
        _wait(driver, lambda: bool(driver.execute_script(
            "return !!window.__instra_dashboard_consistency_final;")),
            timeout=15, message="final heatmap consistency guard did not install")
        diagnostic = driver.execute_script(
            """
            const mount = document.getElementById('heatmap_plot');
            const trace = (mount?.data || []).find(item => item.type === 'heatmap');
            const rawFigure = window.app?.figures?.heatmap;
            const rawTrace = (rawFigure?.data || []).find(item => item.type === 'heatmap');
            const preparedCells = Array.isArray(trace?.z) ? trace.z.flat() : [];
            const rawCells = Array.isArray(rawTrace?.z) ? rawTrace.z.flat() : [];
            const customCells = Array.isArray(trace?.customdata) ? trace.customdata.flat().filter(Array.isArray) : [];
            return {
              repair_loaded: !!window.__instra_regression_repair,
              consistency_loaded: !!window.__instra_dashboard_consistency_final,
              prepared_nonzero: preparedCells.filter(v => Number.isFinite(Number(v)) && Number(v) !== 0).length,
              prepared_null: preparedCells.filter(v => v === null || v === undefined).length,
              raw_nonzero: rawCells.filter(v => Number.isFinite(Number(v)) && Number(v) !== 0).length,
              raw_null: rawCells.filter(v => v === null || v === undefined).length,
              custom_count: customCells.length,
              custom_raw_nonzero: customCells.filter(c => Number.isFinite(Number(c?.[3])) && Number(c?.[3]) !== 0).length,
              custom_sample: customCells.slice(0, 4),
              current_losses_sample: (trace?.layout?.meta?.thog2_current_losses || rawFigure?.layout?.meta?.thog2_current_losses || []).slice(0, 4),
              fallback_meta: mount?.layout?.meta?.thog2_legacy_absolute_fallback ?? null,
              mode_text: document.getElementById('heatmap_delta_loss_mode')?.textContent?.trim() ?? null,
              mode_disabled: document.getElementById('heatmap_delta_loss_mode')?.disabled ?? null,
              colour_key_x: Number(trace?.colorbar?.x ?? 0),
              colour_key_xpad: Number(trace?.colorbar?.xpad ?? 0),
              right_margin: Number(mount?.layout?.margin?.r ?? 0),
            };
            """
        )
        assert diagnostic["prepared_nonzero"] > 0, f"heatmap pipeline diagnostic: {diagnostic}"
        assert diagnostic["consistency_loaded"] is True, diagnostic
        assert diagnostic["colour_key_x"] > 1.05, diagnostic
        assert diagnostic["colour_key_xpad"] >= 12, diagnostic
        assert diagnostic["right_margin"] >= 270, diagnostic
    finally:
        if driver is not None:
            driver.quit()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)
# ^^^ THOG
