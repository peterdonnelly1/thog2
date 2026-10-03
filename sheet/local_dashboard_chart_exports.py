# vvv THOG produce genuine BIFF8 Excel workbooks containing every retained point in each exported visible curve
"""Structured chart exports, with a bundled BSD licensed BIFF8 writer."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlparse


def _writer():
    try:
        import xlwt
    except ImportError:
        sys.path.insert(0, str(Path(__file__).with_name("vendor") / "xlwt-1.3.0.zip"))
        import xlwt
    return xlwt


def validate_export(payload):
    if not isinstance(payload, dict) or payload.get("schema") != "instra.visible_curves.v1":
        raise ValueError("Invalid chart export schema")
    if payload.get("metric") not in {"training_loss", "tokens_throughput"}:
        raise ValueError("Unknown chart metric")
    series = payload.get("series")
    if not isinstance(series, list) or not 1 <= len(series) <= 1000:
        raise ValueError("Export must contain 1 to 1000 visible curves")
    points = 0
    for curve in series:
        if not isinstance(curve, dict):
            raise ValueError("Invalid curve")
        x, y = curve.get("x"), curve.get("y")
        if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
            raise ValueError("Curve x and y lengths must match")
        if not isinstance(curve.get("x_variants", {}), dict) or len(curve.get("x_variants", {})) > 16:
            raise ValueError("Invalid curve axis variants")
        points += len(y)
    if points > 2_000_000:
        raise ValueError("Excel export exceeds two million points; use JSON")
    return payload


def workbook_bytes(payload):
    validate_export(payload)
    xlwt = _writer()
    workbook = xlwt.Workbook(encoding="utf-8")
    bold = xlwt.easyxf("font: bold on")
    metadata = workbook.add_sheet("Export")
    for row, key in enumerate(("schema", "metric", "exported_at", "selected_run_id", "selected_run_name",
                               "x_axis", "y_axis", "data_scope")):
        metadata.write(row, 0, key, bold)
        metadata.write(row, 1, str(payload.get(key, "")))
    summary_row = 10
    headers = ("sheet", "run_id", "run_name", "series", "points", "colour")
    for column, label in enumerate(headers):
        metadata.write(summary_row, column, label, bold)
    metadata.col(0).width = 28 * 256
    metadata.col(1).width = 50 * 256

    def cell_value(value):
        if value is None or isinstance(value, float) and not math.isfinite(value):
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    for index, curve in enumerate(payload["series"], 1):
        variants = curve.get("x_variants", {})
        variant_names = sorted(variants)
        fields = ("sample_index", "run_id", "run_name", "series", "x", "y", "displayed_x", "point_source", *variant_names)
        for part, offset in enumerate(range(0, len(curve["y"]), 65535), 1):
            name = f"curve_{index:03d}_{part:02d}"
            sheet = workbook.add_sheet(name)
            sheet.set_panes_frozen(True)
            sheet.set_horz_split_pos(1)
            for column, label in enumerate(fields):
                sheet.write(0, column, label, bold)
                sheet.col(column).width = (22 if column in {1, 2, 3} else 18) * 256
            count = min(65535, len(curve["y"]) - offset)
            summary_row += 1
            for column, value in enumerate((name, curve.get("run_id", ""), curve.get("run_name", ""),
                                            curve.get("name", ""), count, curve.get("colour", ""))):
                metadata.write(summary_row, column, cell_value(value))
            for row, point in enumerate(range(offset, offset + count), 1):
                def at(values):
                    return values[point] if isinstance(values, list) and point < len(values) else None
                values = (point, curve.get("run_id", ""), curve.get("run_name", ""), curve.get("name", ""),
                          curve["x"][point], curve["y"][point], at(curve.get("displayed_x")),
                          at(curve.get("point_sources")), *(at(variants[key]) for key in variant_names))
                for column, value in enumerate(values):
                    sheet.write(row, column, cell_value(value))
    result = io.BytesIO()
    workbook.save(result)
    return result.getvalue()


def install(dashboard):
    original_handler = dashboard._handler_for

    def handler_for(catalog):
        base_handler = original_handler(catalog)

        class ExportHandler(base_handler):
            def do_POST(self):
                if urlparse(self.path).path != "/api/chart-export":
                    return super().do_POST()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= 128 * 1024 * 1024:
                        raise ValueError("Invalid export request size")
                    payload = json.loads(self.rfile.read(length))
                    self._send(workbook_bytes(payload), content_type="application/vnd.ms-excel")
                except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
                    self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.BAD_REQUEST)
        return ExportHandler

    dashboard._handler_for = handler_for
# ^^^ THOG
