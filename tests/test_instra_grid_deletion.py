# vvv THOG Grid deletion excludes remote same-number runs and verifies local Grid identity
import io
import json
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import run_thog2_local_dashboard as dashboard


class GridDeletionTests(unittest.TestCase):
    def test_new_prefix_download_names_preserve_the_grid_prefix(self):
        class Handler:
            def do_GET(self): pass
            def do_POST(self): pass
            def do_DELETE(self): pass
            def _send_file(self, path, **kwargs): self.filename = kwargs["filename"]
        for path, expected in [
            (Path("/tmp/grid-scripts/SCR-00021.sh"), "SCR-00021.sh"),
            (Path("/tmp/grid-scripts/SCR-00021/SCR-00021_grid_bash_runner_script.sh"), "SCR-00021_grid_bash_runner_script.sh"),
            (Path("/tmp/grid-scripts/SCR-00021/manifest.json"), "SCR-00021_manifest.json"),
        ]:
            with self.subTest(path=path), patch.object(dashboard, "_handler_for_before_runner", return_value=Handler), \
                    patch.object(dashboard, "_runner_service", SimpleNamespace(file=lambda *args: path)):
                handler = dashboard._handler_for_with_runner(None)()
                handler.path = "/api/runner/file?grid_id=test&name=script&download=1"
                handler.do_GET()
                self.assertEqual(handler.filename, expected)

    def request(self, payload, *, state="completed"):
        members = [
            {"dashboard_run_id": "local-a", "runner_grid_tag": "SCR-00021", "runner_grid_id": "local-grid"},
            {"dashboard_run_id": "local-b", "runner_grid_tag": "SCR-00021", "runner_grid_id": "local-grid"},
            {"dashboard_run_id": "remote-copy", "runner_grid_tag": "SCR-00021", "runner_grid_id": "remote-grid", "remote_copy": True},
            {"dashboard_run_id": "other-grid", "runner_grid_tag": "DRE-00021", "runner_grid_id": "different-grid"},
        ]
        deleted = []
        def delete_runs(run_ids):
            deleted.extend(run_ids)
            return {"deleted_run_ids": run_ids, "errors": []}
        catalog = SimpleNamespace(runs=lambda: {"runs": members}, delete_runs=delete_runs)
        runner = SimpleNamespace(snapshot=lambda: {"grids": [{"grid_id": "local-grid", "grid_tag": "SCR-00021",
                                                               "state": state, "runs": [{"state": state}]}]})
        class Handler:
            def do_GET(self): pass
            def do_POST(self): pass
            def do_DELETE(self): raise AssertionError("Unexpected fallback")
            def _send_json(self, response, status=200):
                self.response, self.status = response, int(status)
        with patch.object(dashboard, "_handler_for_before_runner", return_value=Handler), patch.object(dashboard, "_runner_service", runner):
            kind = dashboard._handler_for_with_runner(catalog)
            handler = kind()
            body = json.dumps(payload).encode()
            handler.path = "/api/runs"
            handler.headers = {"Content-Length": str(len(body))}
            handler.rfile = io.BytesIO(body)
            handler.do_DELETE()
        return handler.status, handler.response, deleted

    def test_only_local_grid_members_are_deleted(self):
        status, result, deleted = self.request({"grid_tag": "SCR-00021", "grid_id": "local-grid"})
        self.assertEqual(status, 200)
        self.assertEqual(deleted, ["local-a", "local-b"])
        self.assertEqual(result["deleted_run_ids"], deleted)

    def test_foreign_grid_uuid_is_rejected(self):
        status, result, deleted = self.request({"grid_tag": "SCR-00021", "grid_id": "remote-grid"})
        self.assertEqual(status, 403)
        self.assertEqual(deleted, [])
        self.assertIn("not managed", result["error"])

    def test_running_grid_must_be_stopped(self):
        status, result, deleted = self.request({"grid_tag": "SCR-00021", "grid_id": "local-grid"}, state="running")
        self.assertEqual(status, 400)
        self.assertEqual(deleted, [])
        self.assertIn("Stop the Grid", result["error"])
# ^^^ THOG
