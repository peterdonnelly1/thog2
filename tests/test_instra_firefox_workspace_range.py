# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from sheet.local_chart_store import LocalChartStore
from tests.test_instra_firefox_acceptance import (
    _firefox_driver,
    _free_port,
    _open_group,
    _wait,
    _wait_for_server,
    _weight_snapshot,
)


def _make_weight_store(
    root: Path,
    *,
    artifact_name: str,
    run_id: str,
    first_step: int,
    last_step: int,
    model_type: str = "sheet",
) -> None:
    path = root / artifact_name / run_id / "charts.sqlite3"
    store = LocalChartStore(
        path,
        run_name=artifact_name,
        run_id=run_id,
        config={
            "host_label": "scruffy",
            "model_type": model_type,
            "geometry_preset": "depth" if model_type == "sheet" else None,
            "instrumentation__depth_weight_curves__history_length": 100,
        },
    )
    for step in range(first_step, last_step + 1):
        store.append_depth_weight_snapshot(_weight_snapshot(step), history_length=100)
    store.close()


def _eye_click(driver, run_id: str) -> None:
    driver.execute_script(
        """
        const wanted = String(arguments[0]);
        const row = [...document.querySelectorAll('tr[data-run-id]')].find(candidate => {
          const value = String(candidate.dataset.runId || '');
          return value === wanted || value === `legacy:${wanted}`;
        });
        const eye = row?.querySelector('.eye-button');
        if (!eye) throw new Error(`missing eye for ${wanted}`);
        eye.click();
        """,
        run_id,
    )


def _set_step_window(driver, minimum: int, maximum: int) -> None:
    driver.execute_script(
        """
        const from = document.getElementById('weight_step_from');
        const to = document.getElementById('weight_step_to');
        const apply = document.getElementById('weight_step_apply');
        if (!from || !to || !apply) throw new Error('weight step controls missing');
        from.value = String(arguments[0]);
        to.value = String(arguments[1]);
        from.dispatchEvent(new Event('input', {bubbles: true}));
        to.dispatchEvent(new Event('input', {bubbles: true}));
        apply.click();
        """,
        minimum,
        maximum,
    )


def test_real_firefox_workspace_intersection_and_step_windows(tmp_path: Path) -> None:
    _make_weight_store(
        tmp_path,
        artifact_name="260824-0901_workspace_a",
        run_id="workspace_a",
        first_step=1000,
        last_step=1004,
    )
    _make_weight_store(
        tmp_path,
        artifact_name="260824-0911_workspace_b",
        run_id="workspace_b",
        first_step=1002,
        last_step=1006,
    )
    _make_weight_store(
        tmp_path,
        artifact_name="260824-0921_workspace_c",
        run_id="workspace_c",
        first_step=1010,
        last_step=1014,
        model_type="dense",
    )

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
        driver.get(f"{base_url}/runs/workspace_a")
        _wait(
            driver,
            lambda: int(driver.execute_script("return document.querySelectorAll('tr[data-run-id]').length;")) >= 3,
            timeout=20,
            message="three Workspace fixture runs did not appear",
        )
        preset_contract_script = """
        const row = wanted => [...document.querySelectorAll('tr[data-run-id]')].find(candidate => {
          const value = String(candidate.dataset.runId || '');
          return value === wanted || value === `legacy:${wanted}`;
        });
        const depth = row('workspace_a')?.querySelector('[data-instra-run-shape-cell="preset"]');
        const dense = row('workspace_c')?.querySelector('[data-instra-run-shape-cell="preset"]');
        return {
          depth_text: depth?.textContent || '',
          depth_bold: !!depth?.querySelector('strong'),
          dense_text: dense?.textContent || '',
          dense_bold: dense?.querySelector('strong')?.textContent || '',
        };
        """
        _wait(
            driver,
            lambda: driver.execute_script(preset_contract_script).get("dense_bold") == "dense",
            message="delayed preset column did not render DENSE in bold",
        )
        preset_contract = driver.execute_script(preset_contract_script)
        assert preset_contract == {
            "depth_text": "depth",
            "depth_bold": False,
            "dense_text": "dense",
            "dense_bold": "dense",
        }
        capture_columns = driver.execute_script(
            """
            const headers = [...document.querySelectorAll('.runs-table thead th')]
              .map(cell => String(cell.textContent || '').trim());
            const row = [...document.querySelectorAll('tr[data-run-id]')].find(candidate =>
              ['workspace_a', 'legacy:workspace_a'].includes(String(candidate.dataset.runId || ''))
            );
            const value = label => row?.children[headers.indexOf(label)]?.textContent?.trim() || '';
            return {
              sequence: headers.filter(label => ['P_s', 'P_e', 'C_s', 'C_e'].includes(label)),
              curve_start: value('C_s'),
              curve_end: value('C_e'),
            };
            """
        )
        assert capture_columns == {
            "sequence": ["P_s", "P_e", "C_s", "C_e"],
            "curve_start": "1,000",
            "curve_end": "1,004",
        }

        driver.execute_script(
            "const el=document.getElementById('workspace_nav'); if(!el) throw new Error('workspace_nav'); el.click();"
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                """
                const nav = document.getElementById('workspace_nav');
                return document.body.classList.contains('instra-workspace-mode')
                  && !!nav?.classList.contains('selected');
                """
            )),
            message="Workspace mode did not activate",
        )
        _open_group(driver, "coefficients_chart_group", "coefficients_group_toggle")
        _wait(
            driver,
            lambda: bool(driver.execute_script("return document.getElementById('weight_step_availability') !== null;")),
            message="Workspace weight-step controls did not appear",
        )

        assert driver.execute_script(
            "return document.getElementById('weight_step_availability')?.textContent || '';"
        ) == ""
        driver.execute_script("document.getElementById('weight_step_overlapping_range').click();")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const button = document.getElementById('weight_step_overlapping_range');
                const error = document.getElementById('weight_step_range_error');
                return !!button && !button.hidden && !!error && !error.hidden
                  && error.textContent === 'No overlapping retained weight steps.';
                """
            ) is True,
            message="no-overlap Workspace action did not show the red inline error",
        )

        _eye_click(driver, "workspace_c")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const range = window.__instra_weight_stability_final?.available_range?.();
                return range?.minimum === 1002 && range?.maximum === 1004;
                """
            ) is True,
            message="eye-ablation did not restore the A/B retained-step intersection",
        )
        driver.execute_script("document.getElementById('weight_step_overlapping_range').click();")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const range = window.__instra_weight_step_filter?.request_range?.();
                const error = document.getElementById('weight_step_range_error');
                return range?.minimum === 1002 && range?.maximum === 1004
                  && document.getElementById('weight_step_from')?.value === '1002'
                  && document.getElementById('weight_step_to')?.value === '1004'
                  && !!error && error.hidden;
                """
            ) is True,
            message="Workspace overlap action did not select the retained intersection",
        )
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => (
                    trace?.meta?.instra_top_axis_anchor !== true
                    && trace?.meta?.instra_thog_executed_overlay !== true
                  ));
                const byRun = new Map();
                for (const trace of traces) {
                  const id = String(trace?.meta?.instra_workspace_run_id || '');
                  const step = Number(trace?.meta?.instra_workspace_optimizer_update);
                  if (!id || !Number.isFinite(step)) continue;
                  if (!byRun.has(id)) byRun.set(id, new Set());
                  byRun.get(id).add(step);
                }
                return byRun.size === 2 && [...byRun.values()].every(steps => (
                  steps.size === 3 && [1002, 1003, 1004].every(step => steps.has(step))
                ));
                """
            ) is True,
            timeout=20,
            message="Workspace overlap range did not finish redrawing both runs",
        )

        driver.execute_script("document.getElementById('weight_step_gradient').click();")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const button = document.getElementById('weight_step_gradient');
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => (
                    trace?.meta?.instra_top_axis_anchor !== true
                    && trace?.meta?.instra_thog_executed_overlay !== true
                  ));
                const byRun = new Map();
                for (const trace of traces) {
                  const id = String(trace?.meta?.instra_workspace_run_id || '');
                  const step = Number(trace?.meta?.instra_workspace_optimizer_update);
                  if (!id || !Number.isFinite(step)) continue;
                  if (!byRun.has(id)) byRun.set(id, []);
                  byRun.get(id).push({step, colour: String(trace?.line?.color || '')});
                }
                const rgb = colour => {
                  const probe = document.createElement('span');
                  probe.style.color = colour;
                  probe.hidden = true;
                  document.body.appendChild(probe);
                  const values = (getComputedStyle(probe).color.match(/[0-9.]+/g) || [])
                    .slice(0, 3).map(Number);
                  probe.remove();
                  return values;
                };
                const sameColour = (left, right) => {
                  const a = rgb(left); const b = rgb(right);
                  return a.length === 3 && b.length === 3 && a.every((value, index) => value === b[index]);
                };
                const lum = colour => { const [r,g,b] = rgb(colour); return .2126*r + .7152*g + .0722*b; };
                return button?.getAttribute('aria-pressed') === 'true'
                  && [...byRun].every(([id, values]) => {
                    values.sort((a, b) => a.step - b.step);
                    if (values.length !== 3) return false;
                    const base = String(colour_for_run(id));
                    return sameColour(values[1].colour, base)
                      && lum(values[0].colour) > lum(values[1].colour)
                      && lum(values[2].colour) < lum(values[1].colour);
                  });
                """
            ) is True,
            timeout=20,
            message="Workspace gradient did not use light/run-colour/dark step ordering",
        )

        _set_step_window(driver, 1002, 1003)
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const range = window.__instra_weight_step_filter?.request_range?.();
                return range?.minimum === 1002 && range?.maximum === 1003;
                """
            ) is True,
            message="explicit Workspace step window did not become active",
        )
        _wait(
            driver,
            lambda: bool(driver.execute_script(
                "const el=document.getElementById('attn_q_head_N_plot'); return !!el && el.dataset.plotReady === 'true';"
            )),
            timeout=20,
            message="Workspace Q chart did not render selected step window",
        )
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const mount = document.getElementById('attn_q_head_N_plot');
                const steps = (mount?.data || [])
                  .map(trace => Number(trace?.meta?.instra_workspace_optimizer_update))
                  .filter(Number.isFinite);
                return steps.length > 0 && steps.every(step => step >= 1002 && step <= 1003);
                """
            ) is True,
            message="Workspace rendered traces outside explicit 1002–1003 window",
        )
        compact_hover = driver.execute_script(
            """
            const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
              .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
            return traces.map(trace => ({
              artifact: trace?.meta?.instra_workspace_artifact_name,
              rows: String(trace?.hovertemplate || '').split('<extra', 1)[0].split('<br>').slice(0, 2),
              step: Number(trace?.meta?.instra_workspace_optimizer_update),
            }));
            """
        )
        assert compact_hover
        assert all(
            entry["rows"] == [f"<b>{entry['artifact'][:11]}</b>", f"step {entry['step']}"]
            for entry in compact_hover
        ), compact_hover

        driver.execute_script(
            "document.querySelector('.chart-card[data-chart=\"attn_q_head_N\"] .maximize-button').click();"
        )
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const traces = (document.getElementById('attn_q_head_N_plot')?.data || [])
                  .filter(trace => trace?.meta?.instra_top_axis_anchor !== true);
                const card = document.querySelector('.chart-card[data-chart="attn_q_head_N"]');
                const actions = card?.querySelector('.chart-card-actions');
                const groupHeader = document.querySelector('#coefficients_chart_group > .chart-group-header');
                return !!card?.classList.contains('maximized') && traces.length > 0
                  && getComputedStyle(groupHeader).display !== 'none'
                  && !!actions?.classList.contains('thog2-maximized-weight-controls')
                  && getComputedStyle(actions).visibility === 'visible'
                  && !!actions.querySelector('.explicit-trajectory-modes')
                  && !!actions.querySelector('.chart-settings-button')
                  && traces.every(trace => {
                    const rows = String(trace?.hovertemplate || '').split('<extra', 1)[0].split('<br>');
                    return rows[0] === `<b>${trace.meta.instra_workspace_artifact_name}</b>`
                      && rows[1] === `step ${trace.meta.instra_workspace_optimizer_update}`;
                  });
                """
            ) is True,
            message="maximized Workspace hover did not retain full artifact with step second",
        )
        driver.execute_script(
            "document.querySelector('.chart-card[data-chart=\"attn_q_head_N\"] .maximize-button').click();"
        )
        _wait(
            driver,
            lambda: driver.execute_script(
                "return !document.getElementById('charts_scroll')?.classList.contains('maximized-mode');"
            ) is True,
            message="Workspace Q chart did not restore after hover test",
        )

        _set_step_window(driver, 1010, 1011)
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const error = document.getElementById('weight_step_range_error');
                return document.getElementById('weight_step_from')?.value === '1004'
                  && document.getElementById('weight_step_to')?.value === '1004'
                  && !!error && !error.hidden
                  && error.textContent.includes('cannot be greater than 1004');
                """
            ) is True,
            message="invalid Workspace bounds were not corrected inline",
        )

        driver.execute_script("document.getElementById('weight_step_whole_range').click();")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const range = window.__instra_weight_step_filter?.request_range?.();
                return window.__instra_weight_step_filter?.active?.() === true
                  && range?.minimum === 1002 && range?.maximum === 1004
                  && document.getElementById('weight_step_from')?.value === '1002'
                  && document.getElementById('weight_step_to')?.value === '1004';
                """
            ) is True,
            message="whole range did not restore the full Workspace intersection",
        )
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const placeholder = document.getElementById('attn_q_head_N_placeholder');
                return !!placeholder && placeholder.hidden;
                """
            ) is True,
            timeout=20,
            message="Q chart did not leave historical-range placeholder after whole range",
        )
        driver.execute_script("document.getElementById('runs_nav').click();")
        _wait(
            driver,
            lambda: driver.execute_script(
                """
                const error = document.getElementById('weight_step_range_error');
                return document.body.classList.contains('instra-workspace-mode') === false
                  && (!error || error.hidden);
                """
            ) is True,
            message="Workspace range error leaked back into Runs",
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
