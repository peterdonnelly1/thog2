# vvv THOG persist known run/Grid boundaries without inventing clocks for never-started legacy runs
import json
import unittest
from copy import deepcopy

from instra_runner import _capture_wall_times
from instra_duration_estimator import estimate


class WallTimeTests(unittest.TestCase):
    def test_parallel_grid_and_run_endpoints_are_recovered_from_attempts(self):
        value = {'grids': [{'state': 'completed', 'runs': [
            {'state': 'completed', 'attempts': [{'started_at': '2026-10-01T10:00:00Z', 'finished_at': '2026-10-01T10:02:00Z'}]},
            {'state': 'completed', 'attempts': [{'started_at': '2026-10-01T10:00:15Z', 'finished_at': '2026-10-01T10:03:00Z'}]},
        ]}]}
        _capture_wall_times(value)
        grid = value['grids'][0]
        self.assertEqual(grid['started_at'], '2026-10-01T10:00:00Z')
        self.assertEqual(grid['finished_at'], '2026-10-01T10:03:00Z')
        self.assertEqual(grid['runs'][0]['finished_at'], '2026-10-01T10:02:00Z')
        previous = json.loads(json.dumps(value))
        _capture_wall_times(value)
        self.assertEqual(value, previous)

    def test_retry_retains_first_start_but_has_no_end_until_latest_attempt_finishes(self):
        run = {'state': 'queued', 'finished_at': '2026-10-01T10:01:00Z', 'attempts': [
            {'started_at': '2026-10-01T10:00:00Z', 'finished_at': '2026-10-01T10:01:00Z'},
        ]}
        grid = {'state': 'queued', 'runs': [run]}
        value = {'grids': [grid]}
        _capture_wall_times(value)
        self.assertNotIn('finished_at', run)
        run['state'] = 'running'
        run['attempts'].append({'started_at': '2026-10-01T11:00:00Z'})
        _capture_wall_times(value)
        self.assertNotIn('finished_at', run)
        run['state'] = grid['state'] = 'completed'
        run['attempts'][-1]['finished_at'] = '2026-10-01T11:05:00Z'
        _capture_wall_times(value)
        self.assertEqual(run['started_at'], '2026-10-01T10:00:00Z')
        self.assertEqual(run['finished_at'], '2026-10-01T11:05:00Z')
        self.assertEqual(grid['finished_at'], run['finished_at'])

    def test_unavailable_historical_clocks_remain_unknown(self):
        value = {'grids': [{'state': 'cancelled', 'runs': [{'state': 'cancelled', 'attempts': []}]}]}
        previous = deepcopy(value)
        _capture_wall_times(value)
        self.assertEqual(value, previous)

    def test_timing_metadata_does_not_change_estimator_matching_or_launch_estimate(self):
        history = [{'state': 'completed', 'runs': [{'state': 'completed', 'duration_seconds': 100,
            'parameters': {}, 'attempts': [{'started_at': '2026-10-01T10:00:00Z', 'finished_at': '2026-10-01T10:01:40Z'}]}]}]
        baseline = estimate([{'parameters': {}}], history)
        _capture_wall_times({'grids': history})
        self.assertEqual(estimate([{'parameters': {}}], history), baseline)
# ^^^ THOG
