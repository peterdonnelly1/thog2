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
