# vvv THOG temporary transformer: make Processing Plotly mounts follow actual card geometry through maximize/restore
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


needle = '''window.processing_apply_detail_tab = charts_selected => {\n  processing_view.charts_tab_visible = Boolean(charts_selected);\n  processing_sync_visibility();\n};\n// ^^^ THOG\n'''
replacement = '''window.processing_apply_detail_tab = charts_selected => {\n  processing_view.charts_tab_visible = Boolean(charts_selected);\n  processing_sync_visibility();\n};\n\n// vvv THOG Processing Plotly mounts follow actual card geometry; this closes the maximize/restore race left by one-shot layout resizing\nconst processing_resize_observers = [];\n\nfunction processing_resize_ready_card(card) {\n  if (!card || card.offsetParent === null) return;\n  const mount = card.querySelector(".plot-mount");\n  if (!mount || mount.dataset.plotReady !== "true") return;\n  requestAnimationFrame(() => {\n    if (card.offsetParent !== null && mount.dataset.plotReady === "true") Plotly.Plots.resize(mount);\n  });\n}\n\nfunction processing_install_resize_observers() {\n  if (typeof ResizeObserver !== "function" || processing_resize_observers.length) return;\n  for (const chart_name of ["processing_timeline", "processing_contention", "processing_throughput"]) {\n    const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);\n    if (!card) continue;\n    const observer = new ResizeObserver(() => processing_resize_ready_card(card));\n    observer.observe(card);\n    processing_resize_observers.push(observer);\n  }\n}\n// ^^^ THOG\n// ^^^ THOG\n'''
replace_once("sheet/local_dashboard_assets/dashboard_processing.js", needle, replacement)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''processing_view.timer = window.setInterval(processing_refresh, 1500);\nwindow.addEventListener("load", processing_refresh);\n''',
    '''processing_view.timer = window.setInterval(processing_refresh, 1500);\nwindow.addEventListener("load", () => {\n  processing_install_resize_observers();                                                                                                                  // <<< THOG observe Processing card geometry before first completed-trace render\n  processing_refresh();\n});\n''',
)

with (ROOT / "tests/test_premat_processing.py").open("a", encoding="utf-8") as test:
    test.write('''\n\n# vvv THOG Processing maximize/restore must resize Plotly from observed final card geometry, not one timing-sensitive callback\ndef test_processing_charts_observe_card_geometry_for_plotly_resize() -> None:\n    js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")\n    assert "const processing_resize_observers = []" in js\n    assert "new ResizeObserver(() => processing_resize_ready_card(card))" in js\n    assert "observer.observe(card)" in js\n    assert "Plotly.Plots.resize(mount)" in js\n    assert "processing_install_resize_observers();" in js\n# ^^^ THOG\n''')

with (ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md").open("a", encoding="utf-8") as log:
    log.write("- Processing charts now observe their actual card geometry with `ResizeObserver` and resize the ready Plotly mount after maximize/restore/show transitions. This supplements the generic INSTRA double-rAF resize and removes the remaining timing dependency that could leave a correctly maximized Processing card visually empty.\\n")
# ^^^ THOG
