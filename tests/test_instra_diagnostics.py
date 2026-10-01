# vvv THOG health remains available while another handler is blocked
import threading
import unittest
from instra_diagnostics import instrument_handler


class DiagnosticsTests(unittest.TestCase):
    def test_health_reports_stalled_request_without_waiting_for_it(self):
        entered, release = threading.Event(), threading.Event()
        class Handler:
            command = "GET"
            def do_GET(self):
                entered.set()
                release.wait(5)
            def do_POST(self): pass
            def do_DELETE(self): pass
            def _send_json(self, payload): self.response = payload
        handler_type = instrument_handler(Handler, "/tmp/unused-instra-diagnostics-test")
        blocked = handler_type(); blocked.path = "/api/chart-group?run=private-id"
        worker = threading.Thread(target=blocked.do_GET, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        try:
            health = handler_type(); health.path = "/api/health"
            health.do_GET()
            self.assertEqual(health.response["requests"][0]["path"], "/api/chart-group")
            self.assertNotIn("private-id", str(health.response))
        finally:
            release.set(); worker.join(2)
        self.assertFalse(worker.is_alive())
# ^^^ THOG
