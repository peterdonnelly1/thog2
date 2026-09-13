# vvv THOG temporary exact-source transformer for Processing maximize/order repair
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


html_path = ROOT / "sheet/local_dashboard_assets/index.html"
html = html_path.read_text(encoding="utf-8")
throughput_start = html.index('            <!-- vvv THOG duplicate the run-level net-throughput scoreboard beside the Nsight diagnostics for convenience -->')
throughput_article_start = html.index('            <article class="processing-card chart-card processing-throughput-card"', throughput_start)
throughput_end = html.index('            <!-- ^^^ THOG -->', throughput_article_start) + len('            <!-- ^^^ THOG -->\n')
throughput_block = html[throughput_start:throughput_end]
html = html[:throughput_start] + html[throughput_end:]
contention_start = html.index('            <article class="processing-card chart-card processing-contention-card"')
contention_end = html.index('            </article>', contention_start) + len('            </article>\n')
html = html[:contention_end] + throughput_block + html[contention_end:]
html_path.write_text(html, encoding="utf-8")

css_path = ROOT / "sheet/local_dashboard_assets/dashboard_processing.css"
css = css_path.read_text(encoding="utf-8")
needle = '.processing-grid.chart-grid.is-maximized .processing-card.chart-card.maximized { display: flex; }\n'
replacement = '''.processing-grid.chart-grid.is-maximized .processing-card.chart-card.maximized { display: flex; }\n/* vvv THOG Processing maximized cards consume the remaining INSTRA chart pane instead of retaining their normal 420px geometry */\n.processing-group.maximized { height: 100% !important; min-height: 0 !important; padding-bottom: 0; overflow: hidden; }\n.processing-group.maximized > .processing-grid.chart-grid.is-maximized { flex: 1 1 auto; height: auto; min-height: 0; align-content: stretch; }\n.processing-group.maximized > .processing-grid.chart-grid.is-maximized > .processing-card.chart-card.maximized { flex: 1 1 auto !important; width: 100% !important; height: auto !important; min-height: 0 !important; max-height: none !important; }\n/* ^^^ THOG */\n'''
if needle not in css:
    raise RuntimeError("dashboard_processing.css: maximized Processing rule not found")
css_path.write_text(css.replace(needle, replacement, 1), encoding="utf-8")

test_path = ROOT / "tests/test_premat_processing.py"
tests = test_path.read_text(encoding="utf-8")
anchor = '''def test_processing_throughput_round_trips_through_local_store(tmp_path: Path) -> None:\n'''
addition = '''# vvv THOG Processing presentation keeps diagnostic evidence first and throughput last, with explicit pane-filling maximize geometry\ndef test_processing_chart_order_and_maximize_geometry() -> None:\n    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")\n    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")\n    assert html.index('data-chart="processing_timeline"') < html.index('data-chart="processing_contention"') < html.index('data-chart="processing_throughput"')\n    assert ".processing-group.maximized" in css\n    assert "height: auto !important" in css\n    assert "align-content: stretch" in css\n# ^^^ THOG\n\n\n'''
if anchor not in tests:
    raise RuntimeError("tests/test_premat_processing.py: insertion anchor not found")
test_path.write_text(tests.replace(anchor, addition + anchor, 1), encoding="utf-8")

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text(encoding="utf-8")
log += '- Processing INSTRA presentation now places Training throughput after the two Nsight diagnostics. Processing maximize explicitly stretches the selected card through the remaining chart-pane height, avoiding retention of the normal 420px Processing card geometry. The Nsight wrapper already injects `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` automatically whenever Processing profiling is launched, so shell wrappers need no manual prefix.\n'
log_path.write_text(log, encoding="utf-8")
# ^^^ THOG
