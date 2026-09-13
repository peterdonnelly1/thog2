from __future__ import annotations

from pathlib import Path


GROUP_JS = Path("sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js")
PROCESSING_JS = Path("sheet/local_dashboard_assets/dashboard_processing.js")
TESTS = Path("tests/test_premat_processing.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


group_js = GROUP_JS.read_text(encoding="utf-8")
old_order = '''      for (const summary of sorted_group_summaries(summaries)) {\n        const section = update_group_section(summary);\n        parent.insertBefore(section, depth_group || null);\n      }\n    };\n'''
new_order = '''      for (const summary of sorted_group_summaries(summaries)) {\n        const section = update_group_section(summary);\n        parent.insertBefore(section, depth_group || null);\n      }\n      // vvv THOG Processing is an ordinary chart group in the stack: after Val (or Train when Val is unavailable), before Memory/System\n      const processing_group = by_id("processing_chart_group");\n      const processing_anchor = group_section("val") || group_section("train");\n      if (processing_group && processing_anchor) processing_anchor.after(processing_group);\n      // ^^^ THOG\n    };\n'''
if "processing_anchor = group_section(\"val\")" not in group_js:
    group_js = replace_once(group_js, old_order, new_order, label="metric group ordering")
GROUP_JS.write_text(group_js, encoding="utf-8")


processing_js = PROCESSING_JS.read_text(encoding="utf-8")
old_mode = '      mode: entry.rows.length === 1 ? "markers" : "lines+markers",\n'
new_mode = '      mode: entry.rows.length === 1 ? "markers" : "lines",                                                                                     // <<< THOG match ordinary INSTRA curves: no per-point markers on multi-point throughput lines\n'
if 'entry.rows.length === 1 ? "markers" : "lines",' not in processing_js:
    processing_js = replace_once(processing_js, old_mode, new_mode, label="throughput curve mode")
PROCESSING_JS.write_text(processing_js, encoding="utf-8")


tests = TESTS.read_text(encoding="utf-8")
marker = "def test_processing_group_sits_after_val_and_throughput_has_no_curve_markers()"
if marker not in tests:
    tests += '''\n\n# vvv THOG Processing participates in the ordinary group stack and throughput follows ordinary INSTRA line presentation\ndef test_processing_group_sits_after_val_and_throughput_has_no_curve_markers() -> None:\n    group_js = Path("sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js").read_text(encoding="utf-8")\n    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")\n    assert 'const processing_anchor = group_section("val") || group_section("train");' in group_js\n    assert 'processing_anchor.after(processing_group)' in group_js\n    assert 'entry.rows.length === 1 ? "markers" : "lines"' in processing_js\n    assert 'entry.rows.length === 1 ? "markers" : "lines+markers"' not in processing_js\n# ^^^ THOG\n'''
TESTS.write_text(tests, encoding="utf-8")
