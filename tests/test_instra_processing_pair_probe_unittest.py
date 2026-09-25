# vvv THOG exercise the lightweight pair route without loading optional GPU/chart dependencies
"""Run with python -m unittest tests.test_instra_processing_pair_probe_unittest."""

import ast
from http import HTTPStatus
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


class PairProbeRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database_path = Path(self.temporary.name) / "run" / "charts.sqlite3"
        self.database_path.parent.mkdir()
        self.state = SimpleNamespace(database_path=self.database_path)
        self.calls = []

        class BaseHandler:
            def do_GET(self):
                self.response = {"other_route": True}

            def _send_json(self, value, *, status=HTTPStatus.OK):
                self.response = value
                self.status = status

        def match(state, *, excluded_ncu_run_ids, preferred_ncu_run_id):
            self.calls.append((state, excluded_ncu_run_ids, preferred_ncu_run_id))
            return (0, 0, None, {"dashboard_run_id": "ncu-1"}, None)

        source = Path("run_thog2_local_dashboard.py").read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_handler_for_with_pair_probe")
        namespace = {
            "_handler_for_before_pair_probe":lambda catalog: BaseHandler,
            "_matching_ncu_companion":match,
            "urlparse":urlparse,
            "parse_qs":parse_qs,
            "HTTPStatus":HTTPStatus,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), source, "exec"), namespace)
        catalog = SimpleNamespace(state_for_run=lambda run_id: self.state)
        self.handler = namespace["_handler_for_with_pair_probe"](catalog)()

    def test_valid_capture_pairs_without_loading_trace_or_archive(self):
        processing = self.database_path.parent / "processing"
        processing.mkdir()
        (processing / "processing_data.json").write_text("not parsed by the pair probe")
        self.handler.path = "/api/processing-pair?run=nsys-1&exclude_ncu=claimed,other&preferred_ncu=ncu-1"
        self.handler.do_GET()
        self.assertEqual(self.handler.response["data"]["premat_compatibility_source"], {
            "nsys_dashboard_run_id":"nsys-1", "dashboard_run_id":"ncu-1",
        })
        self.assertEqual(self.calls, [(self.state, {"claimed", "other"}, "ncu-1")])

    def test_missing_capture_cannot_claim_ncu(self):
        self.handler.path = "/api/processing-pair?run=nsys-1"
        self.handler.do_GET()
        self.assertFalse(self.handler.response["available"])
        self.assertEqual(self.handler.response["data"], {})
        self.assertEqual(self.calls, [])

    def test_existing_routes_still_work(self):
        self.handler.path = "/api/runs"
        self.handler.do_GET()
        self.assertTrue(self.handler.response["other_route"])


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
