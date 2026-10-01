# vvv THOG regression checks for Ready, host prefixes and dynamic host-local GPU ownership
import json
import threading
import unittest
import uuid
from unittest.mock import patch

import instra_node_agent as agent
import instra_runner as runner
from instra_grid_identity import choose_prefix
from thog_grid_runner import expand
from tests import test_instra_runner_unittest as base_tests


class SchedulingTests(unittest.TestCase):
    setUp = base_tests.RunnerTests.setUp

    def test_two_hosts_receive_distinct_runs_without_shared_master(self):
        self.service.close()
        self.fake = base_tests.TwoHostNetwork()
        list_hosts = self.fake.list_hosts
        self.fake.list_hosts = lambda: {**list_hosts(), "master_id": None, "release_pending": True}
        self.service = runner.RunnerService(self.fake, start_worker=False)
        self.addCleanup(self.service.close)
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        self.service._refresh()
        current = self.service.snapshot()["grids"][0]
        running = [run for run in current["runs"] if run["state"] == "running"]
        self.assertEqual(len(running), 2)
        self.assertEqual(len({run["run_id"] for run in running}), 2)
        self.assertEqual({run["host_id"] for run in running}, {self.fake.local_id, "thog_host.remote"})
        local = next(run for run in running if run["host_id"] == self.fake.local_id)
        self.fake.host_state[self.fake.local_id][2][local["attempts"][-1]["attempt_id"]]["state"] = "completed"
        self.service._refresh()
        attempts = [attempt["attempt_id"] for run in self.service.snapshot()["grids"][0]["runs"] for attempt in run["attempts"]]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(set(attempts)), 3)
        self.service.close()
        self.service = runner.RunnerService(self.fake, start_worker=False)
        self.addCleanup(self.service.close)
        self.service._refresh()
        after_restart = [attempt["attempt_id"] for run in self.service.snapshot()["grids"][0]["runs"] for attempt in run["attempts"]]
        self.assertEqual(after_restart, attempts)

    def test_history_cannot_orphan_pending_gpu_release(self):
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        state = runner._read()
        state["grids"][0]["state"] = "completed"
        for run in state["grids"][0]["runs"]:
            run["state"] = "completed"
        runner._write(state)
        with self.assertRaisesRegex(ValueError, "GPU release is still pending"):
            self.service.delete_grid_history(grid["grid_id"])
        self.assertEqual(len(self.service.snapshot()["grids"]), 1)

    def test_save_ready_requires_no_discovered_gpu(self):
        original = self.fake.list_hosts
        self.fake.list_hosts = lambda: {**original(), "hosts": []}
        saved = self.service.save_recipe(None, self.recipe)
        self.assertEqual(saved["state"], "ready")
        self.assertEqual(self.fake.waiting_grids, {})
        self.assertEqual(self.fake.reservations, {})

    def test_ready_and_preview_do_not_enter_gpu_queues(self):
        saved = self.service.save_recipe(None, self.recipe)
        self.assertEqual(saved["state"], "ready")
        preview = self.service.preview(self.recipe)
        self.assertTrue(all(run["state"] == "ready" for run in preview["runs"]))
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"], [])
        self.assertEqual(self.fake.reservations, {})
        self.assertEqual(self.fake.waiting_grids, {})
        launched = self.service.launch(saved["recipe_id"])
        self.assertEqual(launched["state"], "queued")
        self.assertEqual(self.service.snapshot()["recipes"][0]["state"], "queued")
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["recipes"][0]["state"], "running")

    def test_host_prefix_survives_controller_restart(self):
        saved = self.service.save_recipe(None, self.recipe)
        first = self.service.launch(saved["recipe_id"])
        self.assertEqual(first["grid_tag"], "TES-00001")
        self.service.close()
        self.service = runner.RunnerService(self.fake, start_worker=False)
        self.addCleanup(self.service.close)
        second = self.service.launch(saved["recipe_id"])
        self.assertEqual(second["grid_tag"], "TES-00002")
        self.assertEqual(first["grid_owner_host_id"], self.fake.local_id)

    def test_busy_selected_gpu_joins_and_fast_gpu_drains_shared_pending_queue(self):
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        self.fake.reservations["GPU-0"] = "external-grid"
        self.service._refresh()
        current = self.service.snapshot()["grids"][0]
        first = next(run for run in current["runs"] if run["state"] == "running")
        self.assertEqual(first["gpu"]["gpu_key"], "GPU-1")
        self.assertTrue(any(item["grid_id"] == grid["grid_id"] for item in self.fake.waiting_grids[(self.fake.local_id, "GPU-0")]))
        self.fake.attempts[first["attempts"][-1]["attempt_id"]]["state"] = "completed"
        self.service._refresh()
        second = next(run for run in self.service.snapshot()["grids"][0]["runs"] if run["state"] == "running")
        self.assertEqual(second["gpu"]["gpu_key"], "GPU-1")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.fake.reservations.pop("GPU-0")
        self.service._refresh()
        current = self.service.snapshot()["grids"][0]
        self.assertEqual({run["gpu"]["gpu_key"] for run in current["runs"] if run["state"] == "running"}, {"GPU-0", "GPU-1"})
        self.assertEqual(len({attempt["attempt_id"] for run in current["runs"] for attempt in run["attempts"]}), 3)

    def test_ownership_survives_run_boundary_and_release_is_per_gpu(self):
        recipe = {**self.recipe, "parameters": {**self.recipe["parameters"], "--n-layer": [2], "DEPTH.order": [1, 2]}}
        saved = self.service.save_recipe(None, recipe)
        first = self.service.launch(saved["recipe_id"])
        waiting = self.service.launch(saved["recipe_id"])
        self.service._refresh()
        states = {grid["grid_id"]: grid for grid in self.service.snapshot()["grids"]}
        self.assertTrue(all(run["state"] == "running" for run in states[first["grid_id"]]["runs"]))
        self.assertFalse(any(run["state"] == "running" for run in states[waiting["grid_id"]]["runs"]))
        done = states[first["grid_id"]]["runs"][0]
        self.fake.attempts[done["attempts"][-1]["attempt_id"]]["state"] = "completed"
        self.service._refresh()
        states = {grid["grid_id"]: grid for grid in self.service.snapshot()["grids"]}
        self.assertEqual(states[first["grid_id"]]["state"], "running")
        self.assertEqual(self.fake.reservations[done["gpu"]["gpu_key"]], waiting["grid_id"])
        self.assertEqual(sum(run["state"] == "running" for run in states[waiting["grid_id"]]["runs"]), 1)

    def test_slow_reserve_does_not_block_save_or_overwrite_stop(self):
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        entered, release = threading.Event(), threading.Event()
        original = self.fake.runner_call
        def delayed(host_id, operation, args=None):
            if operation == "runner_reserve":
                entered.set()
                self.assertTrue(release.wait(5))
            return original(host_id, operation, args)
        self.fake.runner_call = delayed
        errors = []
        def refresh():
            try:
                self.service._refresh()
            except runner.ControllerStateChanged:
                pass
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=refresh, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        try:
            self.service.save_recipe(None, {**self.recipe, "label": "Saved during reserve"})
            self.service.stop_grid(grid["grid_id"])
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.fake.runner_call = original
        self.service._refresh()
        current = self.service.snapshot()["grids"][0]
        self.assertEqual(current["state"], "cancelled")
        self.assertFalse(any(run["attempts"] for run in current["runs"]))
        self.assertEqual(self.fake.reservations, {})

    def test_dense_only_rejects_incompatible_fields_but_mixed_reference_ignores_them(self):
        for field, value in [("DEPTH.order", 1), ("--premat", "enabled"), ("--plastic__enabled", True)]:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "dense-only"):
                    expand({"label": "dense", "parameters": {"--geometry-preset": "dense", field: value}})
                runs = expand({"label": "mixed", "parameters": {"--geometry-preset": ["dense", "depth"], field: value}})
                dense = next(run for run in runs if run["parameters"]["--geometry-preset"] == "dense")
                self.assertNotIn(field, dense["parameters"])


class LocalQueueTests(unittest.TestCase):
    setUp = base_tests.NodeReservationTests.setUp

    def test_persistent_fifo_skips_ineligible_waiter_and_retains_ownership(self):
        first, second, third = [uuid.uuid4().hex for _ in range(3)]
        request = {"gpu_key": "GPU-0", "required_mib": 4096, "headroom_mib": 512}
        agent._operation("runner_queue", {**request, "grid_id": first, "required_mib": 9000})
        agent._operation("runner_queue", {**request, "grid_id": second})
        agent._operation("runner_queue", {**request, "grid_id": third})
        with self.assertRaisesRegex(RuntimeError, "earlier"):
            agent._operation("runner_reserve", {**request, "grid_id": third})
        agent._operation("runner_reserve", {**request, "grid_id": second})
        # A state reload models another controller and an Agent restart.
        state = agent._read_state()
        self.assertEqual([item["grid_id"] for item in state["waiting_grids"]["GPU-0"]], [first, second, third])
        with self.assertRaisesRegex(RuntimeError, "reserved"):
            agent._operation("runner_reserve", {**request, "grid_id": third})
        agent._operation("runner_release", {"grid_id": second, "gpu_key": "GPU-0"})
        agent._operation("runner_reserve", {**request, "grid_id": third})

    def test_waiting_grid_can_cancel_without_releasing_another_owner(self):
        first, second = uuid.uuid4().hex, uuid.uuid4().hex
        request = {"gpu_key": "GPU-0", "required_mib": 100, "headroom_mib": 10}
        agent._operation("runner_queue", {**request, "grid_id": first})
        agent._operation("runner_reserve", {**request, "grid_id": first})
        agent._operation("runner_queue", {**request, "grid_id": second})
        agent._operation("runner_release", {"grid_id": second, "gpu_key": "GPU-0"})
        state = agent._read_state()
        self.assertEqual(state["reservations"]["GPU-0"]["grid_id"], first)
        self.assertEqual([item["grid_id"] for item in state["waiting_grids"]["GPU-0"]], [first])

    def test_random_collision_prefix_is_permanent(self):
        with patch.object(agent.socket, "gethostname", return_value="scrabble"):
            first = agent._operation("grid_identity", {"occupied": ["SCR"], "preferred": "SCR"})["grid_prefix"]
            self.assertNotEqual(first, "SCR")
            self.assertRegex(first, r"^[A-Z]{3}$")
            self.assertEqual(agent._operation("grid_identity", {"occupied": ["SCR"]})["grid_prefix"], first)
        self.assertEqual(choose_prefix("scruffy"), "SCR")
        self.assertEqual(choose_prefix("dreedle"), "DRE")

if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
