# vvv THOG verify bounded long-session plotting and direct canonical catalogue lookups
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from sheet.local_dashboard_wandb_charts_patch import _WandbRunScanner, _MAX_RETAINED_POINTS_PER_SERIES
from run_thog2_local_dashboard_base import DashboardCatalog


class LongSessionTests(unittest.TestCase):
    def test_metric_retention_is_bounded_with_endpoints_spikes_and_recent_samples(self):
        scanner = _WandbRunScanner(Path("/tmp/unused-retention-test.wandb"))
        for index in range(100000):
            scanner._append("train", "train/loss", "loss", "loss", index, 1e6 if index == 901 else float(index % 19),
                            "step", step=index, relative_wall_seconds=index, relative_process_seconds=index,
                            wall_time_epoch_seconds=index, default_x_axis_mode="step")
        points = scanner.series["train"]["train/loss\0loss"]
        self.assertLessEqual(len(points), _MAX_RETAINED_POINTS_PER_SERIES)
        self.assertEqual(points[0][0], 0)
        self.assertEqual(points[-1][0], 99999)
        self.assertTrue(any(point[0] == 901 and point[1] == 1e6 for point in points))
        self.assertEqual([point[0] for point in points[-1000:]], list(range(99000, 100000)))

    def test_canonical_lookup_does_not_rescan_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "charts.sqlite3"
            path.touch()
            class State:
                database_path = path
            catalog = DashboardCatalog(root=Path(directory))
            state = State()
            catalog.identity_states["canonical-run"] = state
            with patch.object(catalog, "_candidate_paths", side_effect=AssertionError("History rescan")):
                self.assertIs(catalog.state_for_run("canonical-run"), state)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG

# vvv THOG repeatedly visited historical runs cannot retain unlimited W&B scanner histories
class ScannerRetentionTests(unittest.TestCase):
    def test_scanner_cache_is_bounded_and_deleted_runs_are_forgotten(self):
        from types import SimpleNamespace
        from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog
        with tempfile.TemporaryDirectory() as directory:
            scanner_catalog = _ScannerCatalog(SimpleNamespace(root=Path(directory)))
            scanner_catalog._find_path = lambda run_id, _status: Path(directory) / (run_id + '.wandb')
            for index in range(40):
                scanner_catalog.scanner_for(SimpleNamespace(status=lambda index=index: {'wandb_run_id': str(index)}))
            self.assertEqual(len(scanner_catalog.scanners), 16)
            recent = Path(directory) / '39.wandb'
            scanner_catalog.paths['remote:producer:39'] = recent
            scanner_catalog.forget_runs(['remote:producer:39'])
            self.assertNotIn(recent, scanner_catalog.scanners)
            self.assertNotIn('remote:producer:39', scanner_catalog.paths)
# ^^^ THOG
