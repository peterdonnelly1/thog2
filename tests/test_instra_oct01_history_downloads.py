# vvv THOG persist execution timing independently of queue wait and launch-time estimates
from tests import test_instra_runner_unittest as runner_support
import instra_runner as runner
import json
import unittest

class ExecutionClockTests(unittest.TestCase):
    setUp=runner_support.RunnerTests.setUp
    def test_execution_clock_survives_completion_refresh_and_retry(self):
        saved=self.service.save_recipe(None,self.recipe)
        grid=self.service.launch(saved['recipe_id'])
        estimate=json.loads(json.dumps(grid['estimated_duration']))
        assert 'started_at' not in grid and 'finished_at' not in grid
        for iteration in range(12):
            self.service._refresh()
            for attempt in self.fake.attempts.values():
                attempt.update(state='completed',exit_code=0,finished_at=runner.now())
            current=self.service.snapshot()['grids'][0]
            if current['state']=='completed': break
        assert current['state']=='completed'
        assert current['started_at'] and current['finished_at']
        assert current['estimated_duration']==estimate
        clocks=(current['started_at'],current['finished_at'])
        self.service._refresh()
        current=self.service.snapshot()['grids'][0]
        assert (current['started_at'],current['finished_at'])==clocks
        manifest=json.loads(self.service.file(grid['grid_id'],'manifest').read_text())
        assert (manifest['started_at'],manifest['finished_at'])==clocks
        state=runner._read()
        state['grids'][0]['state']='failed'
        state['grids'][0]['runs'][0]['state']='failed'
        runner._write(state)
        retried=self.service.retry_run(grid['grid_id'],current['runs'][0]['run_id'])
        assert retried['started_at']==clocks[0] and 'finished_at' not in retried
        assert retried['estimated_duration']==estimate
# ^^^ THOG
