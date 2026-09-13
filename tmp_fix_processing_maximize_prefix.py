# vvv THOG temporary exact-source transformer for Processing maximize and artifact-prefixed filenames
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:180]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.css",
    ".processing-group.maximized > .processing-grid.chart-grid.is-maximized > .processing-card.chart-card.maximized { flex: 1 1 auto !important; width: 100% !important; height: auto !important; min-height: 0 !important; max-height: none !important; }\n",
    ".processing-group.maximized > .processing-grid.chart-grid.is-maximized > .processing-card.chart-card.maximized { flex: 1 1 auto !important; width: 100% !important; height: 100% !important; min-height: 0 !important; max-height: none !important; } /* <<< THOG preserve standard INSTRA full-pane maximize height instead of reverting to ordinary Processing card height */\n",
)

old_outputs = '''    _write_csv(output_directory / "processing_samples.csv", sample_rows, sample_fields)\n    _write_csv(output_directory / "processing_intervals.csv", interval_rows, interval_fields)\n    _write_csv(output_directory / "processing_summary.csv", summary_rows, summary_fields)\n\n    metadata = {\n'''
new_outputs = '''    # vvv THOG user-facing Processing artifacts carry the canonical run artifact as a filename prefix; processing_data.json remains a private fixed INSTRA lookup\n    run_artifact = str((handoff or {}).get("run_name", "")).strip()\n    if "/" in run_artifact or "\\\\" in run_artifact:\n        run_artifact = Path(run_artifact).name\n    prefix = f"{run_artifact}_" if run_artifact else ""\n    processing_files = {\n        "samples": f"{prefix}processing_samples.csv",\n        "intervals": f"{prefix}processing_intervals.csv",\n        "summary": f"{prefix}processing_summary.csv",\n        "metadata": f"{prefix}processing_metadata.json",\n        "bundle": f"{prefix}processing_bundle.zip",\n        "raw_trace": f"{prefix}processing_trace.nsys-rep",\n    }\n    _write_csv(output_directory / processing_files["samples"], sample_rows, sample_fields)\n    _write_csv(output_directory / processing_files["intervals"], interval_rows, interval_fields)\n    _write_csv(output_directory / processing_files["summary"], summary_rows, summary_fields)\n    # ^^^ THOG\n\n    metadata = {\n'''
replace_once("sheet/premat_processing.py", old_outputs, new_outputs)

old_files = '''        "files": {\n            "samples": "processing_samples.csv",\n            "intervals": "processing_intervals.csv",\n            "summary": "processing_summary.csv",\n            "metadata": "processing_metadata.json",\n            "bundle": "processing_bundle.zip",\n            "raw_trace": "processing_trace.nsys-rep",\n        },\n    }\n    (output_directory / "processing_metadata.json").write_text(\n        json.dumps(metadata, indent=2, sort_keys=True)\n    )\n'''
new_files = '''        "files": dict(processing_files),                                                                                                                        # <<< THOG expose artifact-prefixed downloadable Processing filenames to INSTRA\n    }\n    (output_directory / processing_files["metadata"]).write_text(\n        json.dumps(metadata, indent=2, sort_keys=True)\n    )\n'''
replace_once("sheet/premat_processing.py", old_files, new_files)

old_bundle = '''    with zipfile.ZipFile(output_directory / "processing_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:\n        for name in (\n            "processing_samples.csv",\n            "processing_intervals.csv",\n            "processing_summary.csv",\n            "processing_metadata.json",\n            "processing_data.json",\n        ):\n            archive.write(output_directory / name, arcname=name)\n'''
new_bundle = '''    with zipfile.ZipFile(output_directory / processing_files["bundle"], "w", zipfile.ZIP_DEFLATED) as archive:\n        for name in (\n            processing_files["samples"],\n            processing_files["intervals"],\n            processing_files["summary"],\n            processing_files["metadata"],\n            "processing_data.json",\n        ):\n            archive.write(output_directory / name, arcname=name)\n'''
replace_once("sheet/premat_processing.py", old_bundle, new_bundle)

old_copy = '''    shutil.copy2(report_path, processing_directory / "processing_trace.nsys-rep")\n    print(\n        "THOG2 PREMAT processing data: "\n        f"{processing_directory / 'processing_bundle.zip'}",\n        flush=True,\n    )\n'''
new_copy = '''    processing_files = processing_data["metadata"]["files"]                                                                                                   # <<< THOG consume the normalizer's canonical artifact-prefixed filenames\n    shutil.copy2(report_path, processing_directory / processing_files["raw_trace"])\n    print(\n        "THOG2 PREMAT processing data: "\n        f"{processing_directory / processing_files['bundle']}",\n        flush=True,\n    )\n'''
replace_once("sheet/premat_processing.py", old_copy, new_copy)

replace_once(
    "tests/test_premat_processing.py",
    '    assert "height: auto !important" in css\n',
    '    assert "height: 100% !important" in css\n    assert "height: auto !important" not in css\n',
)

old_expected = '''    expected = {\n        "processing_samples.csv",\n        "processing_intervals.csv",\n        "processing_summary.csv",\n        "processing_metadata.json",\n        "processing_data.json",\n        "processing_bundle.zip",\n    }\n    assert expected.issubset({path.name for path in output.iterdir()})\n    with (output / "processing_summary.csv").open() as source:\n        rows = list(csv.DictReader(source))\n    assert rows[0]["family"] == "QKV"\n'''
new_expected = '''    expected = {\n        "fixture_processing_samples.csv",\n        "fixture_processing_intervals.csv",\n        "fixture_processing_summary.csv",\n        "fixture_processing_metadata.json",\n        "processing_data.json",\n        "fixture_processing_bundle.zip",\n    }\n    assert expected.issubset({path.name for path in output.iterdir()})\n    assert payload["metadata"]["files"] == {\n        "samples": "fixture_processing_samples.csv",\n        "intervals": "fixture_processing_intervals.csv",\n        "summary": "fixture_processing_summary.csv",\n        "metadata": "fixture_processing_metadata.json",\n        "bundle": "fixture_processing_bundle.zip",\n        "raw_trace": "fixture_processing_trace.nsys-rep",\n    }\n    with (output / "fixture_processing_summary.csv").open() as source:\n        rows = list(csv.DictReader(source))\n    assert rows[0]["family"] == "QKV"\n    with zipfile.ZipFile(output / "fixture_processing_bundle.zip") as archive:\n        members = set(archive.namelist())\n    assert "fixture_processing_samples.csv" in members\n    assert "fixture_processing_intervals.csv" in members\n    assert "fixture_processing_summary.csv" in members\n    assert "fixture_processing_metadata.json" in members\n    assert "processing_data.json" in members\n'''
replace_once("tests/test_premat_processing.py", old_expected, new_expected)

replace_once(
    "tests/test_premat_processing.py",
    "import sqlite3\nfrom pathlib import Path\n",
    "import sqlite3\nimport zipfile\nfrom pathlib import Path\n",
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
with log_path.open("a", encoding="utf-8") as log:
    log.write('- Corrected Processing maximize after the previous Processing-specific CSS accidentally overrode the standard INSTRA `height: 100% !important` with `height: auto !important`; the selected Processing card now consumes the full chart grid while siblings are hidden. The six user-facing Processing artifacts now use the canonical run artifact as a filename prefix: bundle, samples, intervals, summary, metadata and raw Nsight trace. The private `processing_data.json` remains fixed-name for INSTRA lookup.\\n')
# retry trigger after transient GitHub commit_refs rejection
# ^^^ THOG
