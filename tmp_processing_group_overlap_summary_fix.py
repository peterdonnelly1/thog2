# vvv THOG temporary exact-source transformer for Processing group/navigation, overlap graphic, summary placement, and selector-aware Main display
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one match in {path}, found {text.count(old)}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve selector-excluded matrices in the recapitulation as real Main work, but never count them as PREMAT misses.
premat_path = "sheet/local_dashboard_assets/dashboard_premat.js"
replace_once(
    premat_path,
    '''      record.targeted = target_family === null || family === target_family;                                                                          // <<< THOG distinguish PREMAT eligibility from ordinary Main materialisation\n      if (!record.targeted) {\n        record.path = "not-targeted";\n        record.outcome = "NOT TARGETED";\n        record.trace = ["NOT TARGETED"];\n      }\n      records.set(record.key, record);''',
    '''      record.targeted = target_family === null || family === target_family;                                                                          // <<< THOG distinguish PREMAT eligibility from ordinary Main materialisation\n      records.set(record.key, record);''',
)
replace_once(
    premat_path,
    '''    if (!record) continue;\n    if (record.targeted === false) continue;                                                                                                          // <<< THOG ordinary Main events for selector-excluded matrices are not PREMAT failures\n    premat_update_from_event(record, event);''',
    '''    if (!record) continue;\n    premat_update_from_event(record, event);''',
)
replace_once(
    premat_path,
    '''      if (main_owned) {\n        record.path = "main";\n        record.outcome = "COMPLETE MISS";\n        premat_append_trace(record, "PREMAT NOT STARTED - MAIN STREAM MATERIALISING");\n        add_frame({key, state: "main-materialising", outcome: record.outcome}, event, "PREMAT NOT STARTED - MAIN STREAM MATERIALISING");\n      } else {''',
    '''      if (main_owned) {\n        const targeted = record.targeted !== false;                                                                                                     // <<< THOG selector-excluded matrices still show their real Main materialisation path\n        record.path = targeted ? "main" : "not-targeted-main";                                                                                         // <<< THOG keep PREMAT miss semantics separate from ordinary non-targeted Main work\n        record.outcome = targeted ? "COMPLETE MISS" : "NOT TARGETED";                                                                                 // <<< THOG non-targeted Main work is excluded from PREMAT F/P/M totals\n        const main_state = targeted ? "PREMAT NOT STARTED - MAIN STREAM MATERIALISING" : "MAIN STREAM MATERIALISING · NOT TARGETED";                // <<< THOG make selector semantics explicit while preserving physical Main activity\n        premat_append_trace(record, main_state);\n        add_frame({key, state: "main-materialising", outcome: record.outcome}, event, main_state);\n      } else {''',
)
replace_once(
    premat_path,
    '''      if (main_owned) {\n        record.path = "main";\n        record.outcome = "COMPLETE MISS";\n        premat_append_trace(record, "MAIN STREAM CONSUMING");\n        add_frame({key, state: "main-consuming", outcome: record.outcome}, event, "MAIN STREAM CONSUMING");\n      } else if (waited || event.critical_path_miss) {''',
    '''      if (main_owned) {\n        const targeted = record.targeted !== false;                                                                                                     // <<< THOG selector-excluded matrices still show their real Main consumption path\n        record.path = targeted ? "main" : "not-targeted-main";                                                                                         // <<< THOG preserve NOT TARGETED classification through consumption\n        record.outcome = targeted ? "COMPLETE MISS" : "NOT TARGETED";                                                                                 // <<< THOG only targeted ordinary Main fallback is a PREMAT miss\n        const main_state = targeted ? "MAIN STREAM CONSUMING" : "MAIN STREAM CONSUMING · NOT TARGETED";                                             // <<< THOG make ordinary Main work explicit in playback\n        premat_append_trace(record, main_state);\n        add_frame({key, state: "main-consuming", outcome: record.outcome}, event, main_state);\n      } else if (waited || event.critical_path_miss) {''',
)
replace_once(
    premat_path,
    '''      if (record.path === "none") {\n        record.path = "main";\n        record.outcome = "COMPLETE MISS";\n      }''',
    '''      if (record.path === "none") {\n        record.path = record.targeted === false ? "not-targeted-main" : "main";                                                                         // <<< THOG preserve selector semantics if only the terminal Main event was retained\n        record.outcome = record.targeted === false ? "NOT TARGETED" : "COMPLETE MISS";                                                                 // <<< THOG never fabricate a miss for a selector-excluded matrix\n      }''',
)
replace_once(
    premat_path,
    '''      const target_text = `l+${record.target_offset} · ${record.target_order} #${record.target_order_position + 1}`;''',
    '''      const target_text = record.targeted === false\n        ? "NOT TARGETED · ordinary Main path"                                                                                                           // <<< THOG inspector distinguishes selector exclusion from PREMAT failure\n        : `l+${record.target_offset} · ${record.target_order} #${record.target_order_position + 1}`;''',
)

# Give Processing the standard collapsible group header, make the matrix summary visible as its own card, and relabel the overlap diagnostic.
index_path = "sheet/local_dashboard_assets/index.html"
replace_once(
    index_path,
    '''          <header class="chart-group-header processing-header">\n            <div class="processing-header-copy"><strong>Processing - Step <span id="processing_step">—</span></strong><span class="processing-status" id="processing_status">Waiting for capture</span></div>\n            <div class="processing-downloads">''',
    '''          <header class="chart-group-header processing-header">\n            <button class="chart-group-toggle" id="processing_group_toggle" type="button" aria-expanded="true" aria-controls="processing_grid">\n              <span class="group-caret" aria-hidden="true">⌄</span><strong>processing</strong><span class="group-count">4</span>\n            </button>\n            <div class="processing-header-copy"><span>Step <strong id="processing_step">—</strong></span><span class="processing-status" id="processing_status">Waiting for capture</span></div>\n            <div class="processing-downloads">''',
)
replace_once(index_path, '<div class="processing-grid chart-grid">', '<div class="processing-grid chart-grid" id="processing_grid">')
replace_once(
    index_path,
    '''<header class="chart-card-header"><div class="chart-heading-copy"><h2>Main duration vs PREMAT overlap</h2></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_contention" type="button" aria-label="Maximize Main duration vs PREMAT overlap" title="Maximize chart">''',
    '''<header class="chart-card-header"><div class="chart-heading-copy"><h2>PREMAT/Main temporal overlap</h2><p>Temporal coincidence only; not proof of resource contention.</p></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_contention" type="button" aria-label="Maximize PREMAT/Main temporal overlap" title="Maximize chart">''',
)
replace_once(
    index_path,
    '''              <!-- vvv THOG fixed four-family Processing scoreboard supports matrix-isolation comparisons as QKV/O/UP/DOWN diagnostics are accumulated -->\n              <div class="processing-summary-wrap"><table class="processing-summary processing-matrix-summary"><thead><tr><th>Metric</th><th>QKV</th><th>O</th><th>UP</th><th>DOWN</th></tr></thead><tbody id="processing_matrix_summary_body"></tbody></table></div>\n              <!-- ^^^ THOG -->\n''',
    '',
)
anchor = '''            </article>\n            <!-- ^^^ THOG -->\n          </div>\n          <!-- ^^^ THOG -->'''
summary_card = '''            </article>\n            <!-- ^^^ THOG -->\n            <!-- vvv THOG visible four-family Processing scoreboard is a first-class INSTRA card rather than being hidden under an absolutely positioned Plotly shell -->\n            <article class="processing-card chart-card processing-summary-card" id="processing_matrix_summary_card" data-chart="processing_matrix_summary" hidden>\n              <header class="chart-card-header"><div class="chart-heading-copy"><h2>Matrix summary</h2><p>QKV / O / UP / DOWN; untargeted families remain blank.</p></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_matrix_summary" type="button" aria-label="Maximize matrix summary" title="Maximize chart"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="pointer-events:none;vertical-align:middle"><rect x="4" y="4" width="16" height="16" rx="1"/></svg></button></div></header>\n              <div class="processing-summary-wrap"><table class="processing-summary processing-matrix-summary"><thead><tr><th>Metric</th><th>QKV</th><th>O</th><th>UP</th><th>DOWN</th></tr></thead><tbody id="processing_matrix_summary_body"></tbody></table></div>\n              <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>\n              <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>\n              <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>\n            </article>\n            <!-- ^^^ THOG -->\n          </div>\n          <!-- ^^^ THOG -->'''
replace_once(index_path, anchor, summary_card)
replace_once(
    index_path,
    '''              <div class="premat-key-flow"><strong>MISS</strong><span class="premat-key-flow-step"><span>PREMAT NOT STARTED - MAIN STREAM MATERIALISING</span><span class="premat-key-swatch premat-state-main-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>MAIN STREAM CONSUMING</span><span class="premat-key-swatch premat-state-main-consuming"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>COMPLETE MISS</span><span class="premat-key-swatch premat-state-consumed-main"></span></span></div>''',
    '''              <div class="premat-key-flow"><strong>MISS</strong><span class="premat-key-flow-step"><span>PREMAT NOT STARTED - MAIN STREAM MATERIALISING</span><span class="premat-key-swatch premat-state-main-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>MAIN STREAM CONSUMING</span><span class="premat-key-swatch premat-state-main-consuming"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>COMPLETE MISS</span><span class="premat-key-swatch premat-state-consumed-main"></span></span></div>\n              <div class="premat-key-flow"><strong>NOT TARGETED</strong><span class="premat-key-flow-step"><span>MAIN STREAM MATERIALISING</span><span class="premat-key-swatch premat-state-main-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>MAIN STREAM CONSUMING</span><span class="premat-key-swatch premat-state-main-consuming"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>ORDINARY MAIN PATH</span><span class="premat-key-swatch premat-state-consumed-main"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>excluded from F/P/M totals</span><span class="premat-key-swatch premat-neutral"></span></span></div>''',
)

# Replace the degenerate duration-v-overlap scatter with a direct two-denominator overlap profile.
processing_js = "sheet/local_dashboard_assets/dashboard_processing.js"
old_render = '''async function processing_render_contention(payload) {\n  const families = [...new Set((payload.summary || []).map(row => String(row.family || "?")))];\n  const traces = families.map(family => {\n    const rows = payload.summary.filter(row => String(row.family || "?") === family);\n    return {\n      type: "scatter",\n      mode: "markers",\n      name: family,\n      x: rows.map(row => Number(row.premat_overlap_pct)),\n      y: rows.map(row => Number(row.duration_ms)),\n      text: rows.map(row => `L${Number(row.layer) + 1}`),\n      hovertemplate: `${family} %{text}<br>PREMAT overlap of Main GPU kernels %{x:.1f}%<br>Main GPU kernel duration %{y:.4f} ms<extra></extra>`,\n    };\n  });\n  await processing_plot("processing_contention_plot", traces, {\n    margin: {l: 70, r: 24, t: 12, b: 58},\n    xaxis: {title: "Main GPU kernel time overlapped by PREMAT (%)", range: [0, 100]},\n    yaxis: {title: "Main GPU kernel duration (ms)"},\n    legend: {orientation: "h", y: 1.08},\n  }, plot_config);\n}\n'''
new_render = '''async function processing_render_contention(payload) {\n  const family_order = ["QKV", "O", "UP", "DOWN"];\n  const summary = processing_resolved_matrix_summary(payload);                                                                                             // <<< THOG drive overlap graphic from true kernel-union summary rather than semantic-span scatter points\n  const families = family_order.filter(family => summary[family]);\n  const premat_pct = families.map(family => Number(summary[family].premat_concurrent_with_main_pct));\n  const main_pct = families.map(family => Number(summary[family].main_busy_concurrent_with_premat_pct));\n  const maximum = Math.max(0, ...premat_pct.filter(Number.isFinite), ...main_pct.filter(Number.isFinite));\n  const axis_maximum = Math.min(100, Math.max(5, maximum * 1.18));                                                                                         // <<< THOG keep isolated-matrix low-overlap runs readable instead of wasting a fixed 0..100 axis\n  const traces = families.length ? [\n    {\n      type: "bar", orientation: "h", name: "PREMAT work concurrent with Main",\n      y: families, x: premat_pct,\n      hovertemplate: "%{y}<br>%{x:.2f}% of PREMAT GPU work coincides with any Main kernel<extra></extra>",\n    },\n    {\n      type: "bar", orientation: "h", name: "Main busy time concurrent with PREMAT",\n      y: families, x: main_pct,\n      hovertemplate: "%{y}<br>%{x:.2f}% of Main GPU busy time coincides with PREMAT<extra></extra>",\n    },\n  ] : [];\n  await processing_plot("processing_contention_plot", traces, {\n    margin: {l: 70, r: 24, t: 18, b: 58},\n    barmode: "group",\n    xaxis: {title: "temporal overlap (%)", range: [0, axis_maximum], rangemode: "tozero"},\n    yaxis: {categoryorder: "array", categoryarray: [...family_order].reverse()},\n    legend: {orientation: "h", y: 1.12},\n    annotations: families.length ? [] : [{text: "No PREMAT materialisation intervals in this capture", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}],\n  }, plot_config);\n}\n'''
replace_once(processing_js, old_render, new_render)
replace_once(
    processing_js,
    '''function processing_render_summary(payload) {\n  const body = by_id("processing_matrix_summary_body");\n  if (!body) return;\n  const families = ["QKV", "O", "UP", "DOWN"];\n  const summary = payload.matrix_summary || processing_matrix_summary_from_intervals(payload.intervals || []);                                           // <<< THOG backfill scoreboard for already-captured Processing bundles''',
    '''function processing_resolved_matrix_summary(payload) {\n  return payload.matrix_summary || processing_matrix_summary_from_intervals(payload.intervals || []);                                                   // <<< THOG one canonical current-or-legacy summary feeds both the overlap graphic and table\n}\n\nfunction processing_render_summary(payload) {\n  const body = by_id("processing_matrix_summary_body");\n  if (!body) return;\n  const families = ["QKV", "O", "UP", "DOWN"];\n  const summary = processing_resolved_matrix_summary(payload);                                                                                          // <<< THOG backfill scoreboard for already-captured Processing bundles''',
)
replace_once(
    processing_js,
    '''  const timeline = by_id("processing_timeline_card");\n  const contention = by_id("processing_contention_card");\n  if (timeline) timeline.hidden = !processing_view.trace_available;\n  if (contention) contention.hidden = !processing_view.trace_available;''',
    '''  const timeline = by_id("processing_timeline_card");\n  const contention = by_id("processing_contention_card");\n  const matrix_summary = by_id("processing_matrix_summary_card");                                                                                      // <<< THOG summary is a separate trace-backed Processing card\n  if (timeline) timeline.hidden = !processing_view.trace_available;\n  if (contention) contention.hidden = !processing_view.trace_available;\n  if (matrix_summary) matrix_summary.hidden = !processing_view.trace_available;''',
)

# Make the new table card obvious and let the standard collapsed group remove Processing from the way.
css_path = "sheet/local_dashboard_assets/dashboard_processing.css"
replace_once(
    css_path,
    '''.processing-group { padding-bottom: 18px; }\n.processing-header { align-items: center; gap: 14px; }\n.processing-header-copy { display: flex; flex-direction: column; gap: 3px; }''',
    '''.processing-group { padding-bottom: 18px; }\n.processing-group.collapsed { padding-bottom: 0; }                                                                                                        /* <<< THOG standard collapsed group gives immediate access to following train/val/system/depth groups */\n.processing-header { align-items: center; gap: 14px; }\n.processing-header-copy { display: flex; align-items: center; gap: 10px; color: #4f5967; font-size: 11px; }''',
)
replace_once(
    css_path,
    '''.processing-grid .processing-throughput-card { flex: 1 1 420px; }                                           /* THOG net-throughput scoreboard */''',
    '''.processing-grid .processing-throughput-card { flex: 1 1 420px; }                                           /* THOG net-throughput scoreboard */\n.processing-grid .processing-summary-card { flex: 1 1 100%; height: 300px; min-height: 240px; }                                                /* <<< THOG visible full-width four-family scoreboard */''',
)
replace_once(
    css_path,
    '''.processing-summary-wrap { flex: 0 1 210px; max-height: 210px; min-height: 160px; overflow: auto; margin: 0 10px 10px; } /* <<< THOG reserve enough room for the eight-row four-family Processing scoreboard */''',
    '''.processing-summary-wrap { flex: 1 1 auto; min-height: 0; overflow: auto; margin: 0; padding: 8px 12px 12px; }                                   /* <<< THOG table owns its dedicated card instead of sitting underneath an absolute plot shell */''',
)

# Static/regression tests for the exact UI semantics repaired here.
test_path = "tests/test_premat_processing.py"
test_text = Path(test_path).read_text(encoding="utf-8")
append = '''\n\n# vvv THOG Processing group/navigation and selector-aware presentation regressions\ndef test_processing_group_is_collapsible_and_summary_is_visible_card() -> None:\n    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")\n    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")\n    assert 'id="processing_group_toggle"' in html\n    assert 'aria-controls="processing_grid"' in html\n    assert 'id="processing_grid"' in html\n    assert 'id="processing_matrix_summary_card"' in html\n    assert html.index('id="processing_contention_plot"') < html.index('id="processing_matrix_summary_card"')\n    assert ".processing-group.collapsed" in css\n    assert ".processing-summary-card" in css\n\n\ndef test_processing_overlap_graph_uses_direct_two_denominator_profile() -> None:\n    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")\n    assert 'name: "PREMAT work concurrent with Main"' in js\n    assert 'name: "Main busy time concurrent with PREMAT"' in js\n    assert 'xaxis: {title: "temporal overlap (%)"' in js\n    assert 'type: "scatter"' not in js[js.index("async function processing_render_contention"):js.index("function processing_format")]\n\n\ndef test_premat_selector_exclusions_keep_main_activity_without_becoming_misses() -> None:\n    js = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")\n    assert '"not-targeted-main"' in js\n    assert '"MAIN STREAM MATERIALISING · NOT TARGETED"' in js\n    assert '"MAIN STREAM CONSUMING · NOT TARGETED"' in js\n    assert 'record.outcome = targeted ? "COMPLETE MISS" : "NOT TARGETED"' in js\n    assert "if (record.targeted === false) continue" not in js\n# ^^^ THOG\n'''
if "test_processing_group_is_collapsible_and_summary_is_visible_card" in test_text:
    raise RuntimeError("tests already transformed")
Path(test_path).write_text(test_text + append, encoding="utf-8")

log_path = Path("THOG2_DYNAMIC_PREMATERIALISATION_LOG.md")
log = log_path.read_text(encoding="utf-8")
log += '''\n\n## 2026-09-13 Processing UI follow-up\n- Corrected selector semantics in Premat Recapitulation: selector-excluded matrices continue to show their genuine ordinary Main materialisation/consumption activity, but are labelled NOT TARGETED and excluded from PREMAT full/partial/miss totals.\n- Replaced the degenerate Main-duration-v-overlap scatter with a direct horizontal temporal-overlap profile: PREMAT-work overlap and Main-busy overlap use their own denominators.\n- Moved the four-family QKV/O/UP/DOWN summary into its own visible full-width Processing card; the previous placement was underneath the generic absolutely positioned Plotly shell.\n- Made Processing a standard collapsible INSTRA chart group so it no longer traps navigation above later chart groups.\n'''
log_path.write_text(log, encoding="utf-8")
# ^^^ THOG temporary transformer
