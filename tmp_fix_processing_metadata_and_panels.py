# vvv THOG temporary exact-source transformer for Processing metadata and INSTRA-standard panels
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def thog_line(code: str, explanation: str) -> str:
    marker = "# <<< THOG " + explanation
    if len(code) >= 155:
        raise RuntimeError(f"THOG line too long for column-156 marker: {code}")
    return code + (" " * (155 - len(code))) + marker + "\n"


# Processing bundle self-identification: retain the new fixed matrix selector in the handoff metadata.
replace_once(
    "sheet/premat_processing.py",
    '            "premat": config.get("premat"),\n            "premat_target_layer": config.get("premat_target_layer"),\n            "premat_attention_mode": config.get("premat_attention_mode"),\n',
    '            "premat": config.get("premat"),\n            "premat_target_layer": config.get("premat_target_layer"),\n'
    + thog_line('            "premat_target_matrix": config.get("premat_target_matrix"),', "record fixed PREMAT matrix selector in Processing bundle metadata")
    + '            "premat_attention_mode": config.get("premat_attention_mode"),\n',
)

# Make the two Processing plots normal INSTRA chart cards: same resize handles, persisted panel size,
# reset-panels behaviour, responsive Plotly resize path and maximize control.
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '''          <div class="processing-grid">\n            <article class="processing-card"><h2>GPU processing timeline</h2><div class="processing-plot" id="processing_timeline_plot"></div></article>\n            <article class="processing-card"><h2>Main duration vs PREMAT overlap</h2><div class="processing-plot processing-contention-plot" id="processing_contention_plot"></div><div class="processing-summary-wrap"><table class="processing-summary"><thead><tr><th>Matrix</th><th>N</th><th>Main mean</th><th>PREMAT overlap</th><th>SM active</th><th>SM issue</th><th>Tensor active</th><th>Unused warp slots</th></tr></thead><tbody id="processing_summary_body"></tbody></table></div></article>\n          </div>\n''',
    '''          <!-- vvv THOG Processing plots use the established INSTRA chart-card contract so resize/maximize/reset/persistence work identically -->\n          <div class="processing-grid chart-grid">\n            <article class="processing-card chart-card processing-timeline-card" data-chart="processing_timeline">\n              <header class="chart-card-header"><div class="chart-heading-copy"><h2>GPU processing timeline</h2></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_timeline" type="button" aria-label="Maximize GPU processing timeline" title="Maximize chart"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="pointer-events:none;vertical-align:middle"><rect x="4" y="4" width="16" height="16" rx="1"/></svg></button></div></header>\n              <div class="processing-plot" id="processing_timeline_plot"></div>\n              <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>\n              <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>\n              <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>\n            </article>\n            <article class="processing-card chart-card processing-contention-card" data-chart="processing_contention">\n              <header class="chart-card-header"><div class="chart-heading-copy"><h2>Main duration vs PREMAT overlap</h2></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_contention" type="button" aria-label="Maximize Main duration vs PREMAT overlap" title="Maximize chart"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="pointer-events:none;vertical-align:middle"><rect x="4" y="4" width="16" height="16" rx="1"/></svg></button></div></header>\n              <div class="processing-plot processing-contention-plot" id="processing_contention_plot"></div>\n              <div class="processing-summary-wrap"><table class="processing-summary"><thead><tr><th>Matrix</th><th>N</th><th>Main mean</th><th>PREMAT overlap</th><th>SM active</th><th>SM issue</th><th>Tensor active</th><th>Unused warp slots</th></tr></thead><tbody id="processing_summary_body"></tbody></table></div>\n              <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>\n              <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>\n              <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>\n            </article>\n          </div>\n          <!-- ^^^ THOG -->\n''',
)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.css",
    '''.processing-grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(360px, 1fr); gap: 12px; padding: 12px; }\n.processing-card { min-width: 0; border: 1px solid rgba(127,127,127,.18); border-radius: 8px; padding: 10px; }\n.processing-card h2 { margin: 0 0 8px; font-size: 14px; }\n.processing-plot { width: 100%; height: 420px; }\n.processing-contention-plot { height: 300px; }\n.processing-summary-wrap { overflow-x: auto; margin-top: 10px; }\n''',
    '''/* vvv THOG reuse the established INSTRA chart-card geometry rather than a fixed Processing-only grid */\n.processing-grid.chart-grid { min-height: 0; display: flex; flex-wrap: wrap; align-content: flex-start; gap: 12px; padding: 12px; }\n.processing-card.chart-card { min-width: 320px; height: 420px; min-height: 300px; display: flex; flex-direction: column; padding: 0; overflow: hidden; }\n.processing-grid .processing-timeline-card { flex: 1.6 1 620px; }\n.processing-grid .processing-contention-card { flex: 1 1 420px; }\n.processing-grid.chart-grid.is-maximized .processing-card.chart-card.maximized { display: flex; }\n.processing-plot { width: 100%; min-width: 0; min-height: 140px; height: auto; flex: 1 1 auto; overflow: hidden; }\n.processing-contention-plot { min-height: 140px; }\n.processing-summary-wrap { flex: 0 1 122px; max-height: 122px; min-height: 74px; overflow: auto; margin: 0 10px 10px; }\n/* ^^^ THOG */\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.css",
    '@media (max-width: 1180px) { .processing-grid { grid-template-columns: 1fr; } }\n',
    '@media (max-width: 1180px) { .processing-grid .processing-card.chart-card { flex-basis: 100%; } }\n',
)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''function processing_render(payload) {\n  const group = by_id("processing_chart_group");\n  group.hidden = false;\n''',
    '''function processing_render(payload) {\n  const group = by_id("processing_chart_group");\n  group.hidden = false;\n  // vvv THOG static Processing cards participate in the same saved-size contract as ordinary INSTRA charts\n  if (typeof apply_saved_panel_sizes === "function") apply_saved_panel_sizes();\n  // ^^^ THOG\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''  processing_render_timeline(payload);\n  processing_render_contention(payload);\n  processing_render_summary(payload);\n}\n''',
    '''  processing_render_timeline(payload);\n  processing_render_contention(payload);\n  processing_render_summary(payload);\n  // vvv THOG Plotly must re-measure after the hidden Processing group becomes visible and after any restored panel geometry is applied\n  requestAnimationFrame(() => {\n    for (const chart_name of ["processing_timeline", "processing_contention"]) {\n      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);\n      if (card && typeof resize_plot_in_card === "function") resize_plot_in_card(card);\n    }\n  });\n  // ^^^ THOG\n}\n''',
)

# Regression coverage for the bundle identity and the dashboard contract.
replace_once(
    "tests/test_premat_processing.py",
    'import csv\nimport sqlite3\n',
    'import csv\nimport json\nimport sqlite3\n',
)
replace_once(
    "tests/test_premat_processing.py",
    '    processing_requested_from_argv,\n    rewrite_processing_cli_for_core,\n',
    '    processing_requested_from_argv,\n    register_processing_handoff,\n    rewrite_processing_cli_for_core,\n',
)
replace_once(
    "tests/test_premat_processing.py",
    '''def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n    validate_processing_configuration("enabled", 10000, "cuda")\n    validate_processing_configuration("disabled", 10, "cpu")\n    with pytest.raises(ValueError, match="CUDA"):\n        validate_processing_configuration("enabled", 10000, "cpu")\n    with pytest.raises(ValueError, match="capture_frequency_hz"):\n        validate_processing_configuration("enabled", 9, "cuda")\n\n\n''',
    '''def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n    validate_processing_configuration("enabled", 10000, "cuda")\n    validate_processing_configuration("disabled", 10, "cpu")\n    with pytest.raises(ValueError, match="CUDA"):\n        validate_processing_configuration("enabled", 10000, "cpu")\n    with pytest.raises(ValueError, match="capture_frequency_hz"):\n        validate_processing_configuration("enabled", 9, "cuda")\n\n\ndef test_processing_handoff_records_target_matrix(tmp_path: Path, monkeypatch) -> None:\n    handoff_path = tmp_path / "handoff.json"\n    monkeypatch.setenv("THOG2_PREMAT_PROCESSING_HANDOFF", str(handoff_path))\n    register_processing_handoff(\n        tmp_path / "run",\n        run_name="fixture",\n        config={"premat": "enabled", "premat_target_layer": 1, "premat_target_matrix": 2, "premat_attention_mode": "fused"},\n    )\n    payload = json.loads(handoff_path.read_text(encoding="utf-8"))\n    assert payload["config"]["premat_target_layer"] == 1\n    assert payload["config"]["premat_target_matrix"] == 2\n\n\ndef test_processing_charts_use_standard_instra_panel_contract() -> None:\n    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")\n    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")\n    for chart_name in ("processing_timeline", "processing_contention"):\n        assert f'data-chart="{chart_name}"' in html\n        assert f'data-maximize="{chart_name}"' in html\n    assert html.count('class="panel-resizer panel-resizer-corner"') >= 3\n    assert ".processing-card.chart-card" in css\n    assert ".processing-grid.chart-grid" in css\n\n\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text(encoding="utf-8")
entry = "\n- Processing bundles now retain `premat_target_matrix` in `processing_metadata.json` run configuration. The two Processing Plotly panels now use the ordinary INSTRA chart-card contract, including width/height/corner resizing, persisted panel dimensions, Reset panels, responsive Plotly resizing, and maximize controls.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry, encoding="utf-8")

print("Processing metadata and INSTRA-standard panel update applied")
# ^^^ THOG
