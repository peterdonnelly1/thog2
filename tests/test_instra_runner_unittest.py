# vvv THOG exercise Grid expansion, reservations, identity, recovery and standalone export without a GPU
"""Runner and Node Agent regression tests; no training runtime required."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace

import instra_network as network
import instra_node_agent as agent
import instra_runner as runner
from thog_grid_runner import command_for, expand, script_for, validate_recipe


def gpu(number):
    key = f"GPU-{number}"
    return {"gpu_key": key, "gpu_id": f"thog_host.test.gpu.{key}", "uuid": key, "ordinal": number,
            "model": "Test GPU", "memory_mib": 8192, "free_mib": 8192, "compute_cap": "8.6"}


class FakeNetwork:
    local_id = "thog_host.test"

    def __init__(self):
        self.gpus = [gpu(0), gpu(1)]
        self.reservations = {}
        self.attempts = {}
        self.activity = None
        self.release_calls = 0

    def list_hosts(self):
        return {"local_id": self.local_id, "master_id": None, "release_pending": False,
                "hosts": [{"thog_host_id": self.local_id, "display_name": "test", "local": True,
                           "execution_enabled": False, "last_discovered": {"gpus": self.gpus,
                           "execution_profiles": [{"profile_key": "current", "execution_profile_id": "thog_host.test.execution_profile.current"}]}}]}

    def runner_call(self, host_id, operation, args=None):
        if host_id != self.local_id:
            raise ValueError("Unexpected remote host")
        if operation == "runner_reconcile":
            return {"attempts": self.attempts.copy(), "reservations": self.reservations.copy(), "gpus": self.gpus}
        if operation == "runner_preflight":
            return {"resolved": True}
        if operation == "runner_log":
            return {"attempt_id": args["attempt_id"], "text": "RuntimeError: simulated training failure\n", "log_path": "/tmp/mock.log"}
        if operation == "runner_reserve":
            key, owner = args["gpu_key"], self.reservations.get(args["gpu_key"])
            if owner and owner != args["grid_id"]:
                raise RuntimeError("GPU reserved by another Grid")
            self.reservations[key] = args["grid_id"]
            return {"reservation": {"grid_id": args["grid_id"]}}
        if operation == "runner_launch":
            attempt_id = args["attempt_id"]
            if attempt_id not in self.attempts:
                self.attempts[attempt_id] = {"attempt_id": attempt_id, "state": "running", "pid": 12,
                                             "started_at": runner.now(), "log_path": "/tmp/mock.log"}
            return self.attempts[attempt_id]
        if operation == "runner_release":
            self.release_calls += 1
            self.reservations.pop(args["gpu_key"], None)
            return {"released": True}
        raise ValueError(operation)

    def set_grid_activity(self, active, queue_nonempty):
        self.activity = (active, queue_nonempty)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        for object_, attribute, value in [(runner, "STATE_DIR", root), (runner, "STATE_PATH", root / "runner.json"),
                                           (runner, "LEASE_PATH", root / "lease"), (runner, "GRID_SCRIPTS", root / "grid-scripts"),
                                           (network, "log_event", lambda *args: None)]:
            patcher = patch.object(object_, attribute, value)
            patcher.start(); self.addCleanup(patcher.stop)
        self.fake = FakeNetwork()
        self.service = runner.RunnerService(self.fake, start_worker=False)
        self.addCleanup(self.service.close)
        self.recipe = {"label": "small", "parameters": {"--max-iters": 2, "--warmup-iters": 0, "--n-layer": [2, 3],
                       "--n-embd": 64, "--n-head": 4, "--batch-size": 1, "--block-size": 32, "DEPTH.order": [1, 2],
                       "--option": ["MLP_UP.order=2", "MLP_DOWN.order=2"]}, "max_parallel": 2}

    def test_cartesian_expansion_list_and_profiler_pair(self):
        recipe = {**self.recipe, "profilers": ["nsys", "ncu"]}
        runs = expand(recipe)
        self.assertEqual(len(runs), 8)
        self.assertEqual(runs[0]["pairing_id"], runs[1]["pairing_id"])
        self.assertNotEqual(runs[0]["run_id"], runs[1]["run_id"])
        self.assertEqual(runs[0]["parameters"]["--option"], ["MLP_UP.order=2", "MLP_DOWN.order=2"])
        self.assertNotIn("--o-depth", command_for({**runs[0], "grid_tag": "G-00001"}, gpu(0)))
        self.assertIn("DEPTH.order=1", command_for({**runs[0], "grid_tag": "G-00001"}, gpu(0)))

    def test_plastic_subordinate_controls_enable_plastic_without_a_ui_master_switch(self):
        base = {"run_id": uuid.uuid4().hex, "grid_tag": "G-00001", "profiler": "none", "parameters": {}}
        disabled = command_for({**base, "parameters": {"--plastic__do_learn_layer_count": False}}, gpu(0))
        self.assertNotIn("--plastic__enabled", disabled)
        selected = command_for({**base, "parameters": {"--plastic__layers_to_sample": 3}}, gpu(0))
        self.assertIn("--plastic__enabled", selected)
        self.assertIn("--plastic__layers_to_sample", selected)

    def test_input_rejection_and_script_quoting(self):
        with self.assertRaisesRegex(ValueError, "automatic"):
            validate_recipe({"label": "bad", "parameters": {"--dtype": "float16"}})
        with self.assertRaisesRegex(ValueError, "requires a scalar"):
            validate_recipe({"label": "bad", "parameters": {"--max-iters": [2]}})
        run = {"run_id": uuid.uuid4().hex, "pairing_id": None, "grid_tag": "G-00001", "profiler": "none",
               "parameters": {"--data-dir": "data/folder with 'quotes'", "DEPTH.order": 3}, "gpu": gpu(0)}
        script = script_for([run])
        self.assertIn("'data/folder with '\"'\"'quotes'\"'\"''", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", script)
        self.assertIn("--no-activation-checkpointing", command_for({**run,"parameters":{"--activation-checkpointing":False}}, gpu(0)))

    def test_cross_parameter_validation_and_repeat_snapshot(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            expand({"label":"bad", "parameters":{"--n-embd":64,"--n-head":3}})
        with self.assertRaisesRegex(ValueError, "cannot also"):
            expand({"label":"bad", "parameters":{"DEPTH.order":3,"--option":["DEPTH.order=2"]}})
        saved=self.service.save_recipe(None,self.recipe)
        initial=self.service.launch(saved["recipe_id"])
        self.service.save_recipe(saved["recipe_id"],{**self.recipe,"parameters":{**self.recipe["parameters"],"--max-iters":99}})
        proposal=self.service.repeat_preview(initial["grid_id"],"tight")
        self.assertEqual(proposal["total_runs"],4)
        self.assertFalse(proposal["changes"])
        repeated=self.service.launch(saved["recipe_id"],source_grid_id=initial["grid_id"],repeat_mode="loose")
        self.assertNotEqual(initial["grid_id"],repeated["grid_id"])
        self.assertEqual(repeated["runs"][0]["parameters"]["--max-iters"],2)
        self.assertEqual(repeated["source_grid_id"],initial["grid_id"])

    def test_stop_preserves_completed_and_cancels_pending(self):
        saved=self.service.save_recipe(None,{**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        current=self.service.snapshot()["grids"][0]
        first=next(run for run in current["runs"] if run["state"]=="running")
        self.fake.attempts[first["attempts"][0]["attempt_id"]]["state"]="completed"
        self.service._refresh()
        current=self.service.snapshot()["grids"][0]
        self.assertEqual(current["runs"][0]["state"],"completed")
        # Remaining attempts may be running or queued. Simulate acknowledgement of named stop.
        prior=self.fake.runner_call
        def with_stop(host_id,operation,args=None):
            if operation=="runner_stop":
                self.fake.attempts[args["attempt_id"]]["state"]="failed"
                return self.fake.attempts[args["attempt_id"]]
            return prior(host_id,operation,args)
        self.fake.runner_call=with_stop
        self.service.stop_grid(grid["grid_id"])
        self.service._refresh()
        result=self.service.snapshot()["grids"][0]
        self.assertEqual(result["runs"][0]["state"],"completed")
        self.assertTrue(all(run["state"] in {"completed","cancelled"} for run in result["runs"]))

    def test_retry_preserves_run_identity_and_records_new_attempt(self):
        one={"label":"one","parameters":{"--max-iters":2,"--warmup-iters":0,"--n-embd":8,"--n-head":2,
             "--n-layer":2,"--batch-size":1,"--block-size":8}}
        saved=self.service.save_recipe(None,one)
        grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        before=self.service.snapshot()["grids"][0]["runs"][0]
        first=before["attempts"][0]["attempt_id"]
        self.fake.attempts[first]["state"]="failed"
        self.service._refresh()
        failed=self.service.snapshot()["grids"][0]["runs"][0]
        self.assertIn("simulated training failure", failed["attempts"][0]["failure_excerpt"])
        self.assertIn("simulated training failure", self.service.attempt_log(grid["grid_id"],before["run_id"],first)["text"])
        self.service.retry_run(grid["grid_id"],before["run_id"])
        self.service._refresh()
        after=self.service.snapshot()["grids"][0]["runs"][0]
        self.assertEqual(after["run_id"],before["run_id"])
        self.assertEqual(len(after["attempts"]),2)
        self.assertNotEqual(after["attempts"][1]["attempt_id"],first)

    def test_failed_attempt_explains_old_agent_log_protocol_and_can_be_read_after_upgrade(self):
        recipe = {"label":"small", "parameters":{"--max-iters":2,"--warmup-iters":0,
                  "--n-embd":8,"--n-head":2,"--n-layer":2,"--batch-size":1,"--block-size":8}}
        saved = self.service.save_recipe(None, recipe)
        grid = self.service.launch(saved["recipe_id"])
        self.service._refresh()
        run = self.service.snapshot()["grids"][0]["runs"][0]
        attempt_id = run["attempts"][0]["attempt_id"]
        self.fake.attempts[attempt_id]["state"] = "failed"
        original = self.fake.runner_call
        def old_agent(host_id, operation, args=None):
            if operation == "runner_log":
                raise network.NetworkError("operation", "Unknown Runner operation")
            return original(host_id, operation, args)
        self.fake.runner_call = old_agent
        self.service._refresh()
        result = self.service.snapshot()["grids"][0]["runs"][0]
        self.assertIn("Unknown Runner operation", result["attempts"][0]["failure_excerpt"])
        self.fake.runner_call = original
        self.assertIn("simulated training failure", self.service.attempt_log(
            grid["grid_id"], run["run_id"], attempt_id)["text"])
        self.assertIn("simulated training failure", self.service.snapshot()["grids"][0]["runs"][0]
                      ["attempts"][0]["failure_excerpt"])

    def test_tight_conversion_reassigns_pending_work_without_moving_running(self):
        saved=self.service.save_recipe(None,self.recipe)
        source=self.service.launch(saved["recipe_id"])
        state=runner._read();state["grids"][0]["state"]="completed"
        for run in state["grids"][0]["runs"]:run["state"]="completed"
        runner._write(state)
        repeated=self.service.launch(saved["recipe_id"],source_grid_id=source["grid_id"],repeat_mode="tight")
        self.fake.reservations["GPU-0"]="other-grid"
        self.service._refresh()
        before=next(grid for grid in self.service.snapshot()["grids"] if grid["grid_id"]==repeated["grid_id"])
        running=[run for run in before["runs"] if run["state"]=="running"]
        self.assertTrue(running)
        self.service.convert_to_loose(repeated["grid_id"])
        after=next(grid for grid in self.service.snapshot()["grids"] if grid["grid_id"]==repeated["grid_id"])
        self.assertEqual(after["repeat_mode"],"loose")
        self.assertEqual(after["conversion"]["moved_assignments"],3)
        self.assertTrue(all(run["gpu"]["gpu_key"]=="GPU-1" for run in after["runs"] if run["state"]=="queued"))
        self.assertEqual(next(run for run in after["runs"] if run["state"]=="running")["run_id"],running[0]["run_id"])
        self.assertTrue(self.service.file(after["grid_id"],"conversion").is_file())

    def test_recipe_snapshot_and_partial_gpu_dispatch(self):
        saved = self.service.save_recipe(None, self.recipe)
        preview = self.service.preview(self.recipe)
        self.assertEqual(preview["total_runs"], 4)
        self.assertEqual(preview["estimated_duration"]["confidence"], "insufficient")
        grid = self.service.launch(saved["recipe_id"])
        self.assertTrue(self.service.file(grid["grid_id"], "script").is_file())
        self.assertEqual(len(list(runner.GRID_SCRIPTS.glob("*.sh"))), 1)
        self.fake.reservations["GPU-0"] = "other-grid"
        self.service._refresh()
        current = self.service.snapshot()["grids"][0]
        self.assertEqual(current["runs"][0]["state"], "blocked")
        self.assertEqual(current["runs"][1]["state"], "running")
        self.assertIn("another Grid", current["runs"][0]["blocking_reason"])
        changed = {**self.recipe, "label": "edited", "parameters": {**self.recipe["parameters"], "--max-iters": 99}}
        self.service.save_recipe(saved["recipe_id"], changed)
        self.assertEqual(self.service.snapshot()["grids"][0]["recipe"]["parameters"]["--max-iters"], 2)

    def test_restart_reconciles_without_duplicate_dispatch_and_unknown_blocks(self):
        saved = self.service.save_recipe(None, {**self.recipe, "gpu_pool": [gpu(0)["gpu_id"]]})
        grid = self.service.launch(saved["recipe_id"])
        self.service._refresh()
        active = self.service.snapshot()["grids"][0]["runs"][0]
        attempt_id = active["attempts"][0]["attempt_id"]
        self.service.close()
        replacement = runner.RunnerService(self.fake, start_worker=False)
        self.addCleanup(replacement.close)
        replacement._refresh()
        restored = replacement.snapshot()["grids"][0]["runs"][0]
        self.assertEqual([attempt["attempt_id"] for attempt in restored["attempts"]], [attempt_id])
        self.fake.attempts.pop(attempt_id)
        replacement._refresh()
        unknown = replacement.snapshot()["grids"][0]["runs"][0]
        self.assertEqual(unknown["state"], "unknown")
        replacement._refresh()
        self.assertEqual(len(replacement.snapshot()["grids"][0]["runs"][0]["attempts"]), 1)

    def test_stable_poll_does_not_rewrite_state_or_release_gpu_repeatedly(self):
        saved = self.service.save_recipe(None, {**self.recipe, "parameters": {**self.recipe["parameters"],
                                                          "DEPTH.order": 1}})
        grid = self.service.launch(saved["recipe_id"])
        self.service._refresh()
        first = self.service.snapshot()["grids"][0]["runs"][0]
        self.fake.attempts[first["attempts"][0]["attempt_id"]]["state"] = "completed"
        self.service._refresh()
        before = (runner.STATE_PATH.stat().st_mtime_ns,
                  self.service.file(grid["grid_id"], "status").stat().st_mtime_ns, self.fake.release_calls)
        self.service._refresh()
        after = (runner.STATE_PATH.stat().st_mtime_ns,
                 self.service.file(grid["grid_id"], "status").stat().st_mtime_ns, self.fake.release_calls)
        self.assertEqual(before, after)

    def test_preflight_failure_blocks_without_reservation_or_attempt_and_backs_off(self):
        saved = self.service.save_recipe(None, {"label": "bad model", "parameters": {"--max-iters": 2,
                  "--warmup-iters": 0}})
        self.service.launch(saved["recipe_id"])
        original = self.fake.runner_call
        calls = []
        def reject(host_id, operation, args=None):
            if operation == "runner_preflight":
                calls.append(operation)
                raise ValueError("THOG parameter preflight rejected this run")
            return original(host_id, operation, args)
        self.fake.runner_call = reject
        self.service._refresh()
        self.service._refresh()
        run = self.service.snapshot()["grids"][0]["runs"][0]
        self.assertEqual(run["state"], "blocked")
        self.assertIn("preflight", run["blocking_reason"])
        self.assertEqual(run["attempts"], [])
        self.assertEqual(self.fake.reservations, {})
        self.assertEqual(calls, ["runner_preflight"])


class NodeReservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        for attribute, value in [("STATE_DIR", root), ("AGENT_STATE", root / "node.json")]:
            patcher = patch.object(agent, attribute, value); patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(agent, "_gpu_information", return_value=[gpu(0), gpu(1)])
        patcher.start(); self.addCleanup(patcher.stop)

    def test_atomic_owner_memory_idempotence_and_release(self):
        self.assertEqual(agent._operation("runner_capabilities", {})["cuda_preflight"], True)
        first, second = uuid.uuid4().hex, uuid.uuid4().hex
        args = {"grid_id": first, "gpu_key": "GPU-0", "required_mib": 4096, "headroom_mib": 512}
        agent._operation("runner_reserve", args)
        self.assertEqual(agent._operation("runner_reserve", args)["reservation"]["grid_id"], first)
        with self.assertRaisesRegex(RuntimeError, "reserved"):
            agent._operation("runner_reserve", {**args, "grid_id": second})
        with self.assertRaisesRegex(RuntimeError, "free"):
            agent._operation("runner_reserve", {**args, "gpu_key": "GPU-1", "required_mib": 8000})
        with self.assertRaises(PermissionError):
            agent._operation("runner_release", {"grid_id": second, "gpu_key": "GPU-0"})
        self.assertTrue(agent._operation("runner_release", {"grid_id": first, "gpu_key": "GPU-0"})["released"])
        with self.assertRaisesRegex(ValueError, "power cap"):
            agent._operation("runner_reserve", {**args,"power_cap_w":700})

    def test_attempt_identity_survives_reconcile_and_release_waits(self):
        grid_id, attempt_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex
        agent._operation("runner_reserve", {"grid_id":grid_id,"gpu_key":"GPU-0","required_mib":2048,"headroom_mib":512})
        run={"run_id":run_id,"grid_tag":"G-00001","pairing_id":None,"profiler":"none",
             "parameters":{"--max-iters":2,"--warmup-iters":0},"dtype":"bfloat16","attention_backend":"sdpa","gpu_uuid":"GPU-0"}
        args={"grid_id":grid_id,"attempt_id":attempt_id,"gpu_key":"GPU-0","run":run,
              "host_label":"local","thog_host_id":"thog_host.local","execution_profile":"current"}
        with patch.object(agent.subprocess,"Popen",return_value=SimpleNamespace(pid=os.getpid())) as spawn:
            accepted=agent._operation("runner_launch",args)
            self.assertEqual(agent._operation("runner_launch",args)["attempt_id"],attempt_id)
            spawn.assert_called_once()
        self.assertEqual(accepted["state"],"running")
        with self.assertRaisesRegex(RuntimeError,"still owns"):
            agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":"GPU-0"})
        (agent.STATE_DIR/f"attempt-{attempt_id}.exit").write_text("0\n")
        self.assertEqual(agent._operation("runner_reconcile",{})["attempts"][attempt_id]["state"],"completed")
        (agent.STATE_DIR/f"attempt-{attempt_id}.log").write_text("first line\nTraceback: meaningful failure\n")
        self.assertIn("meaningful failure", agent._operation("runner_log",{"attempt_id":attempt_id,"max_bytes":40})["text"])
        with self.assertRaises(KeyError):
            agent._operation("runner_log",{"attempt_id":uuid.uuid4().hex})
        self.assertTrue(agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":"GPU-0"})["released"])

    def test_actual_training_parser_preflight_rejects_invalid_geometry(self):
        base = {"run_id": uuid.uuid4().hex, "grid_tag": "G-00001", "pairing_id": None,
                "profiler": "none", "parameters": {"--max-iters": 2, "--warmup-iters": 0,
                "--n-layer": 2, "--n-embd": 64, "--n-head": 4, "DEPTH.order": 1},
                "dtype": "bfloat16", "attention_backend": "sdpa", "gpu_uuid": "GPU-0"}
        args = {"run": base, "gpu_key": "GPU-0", "host_label": "local"}
        with patch.object(agent, "_check_torch_cuda") as check_cuda:
            self.assertTrue(agent._operation("runner_preflight", args)["resolved"])
            check_cuda.assert_called_once()
            self.assertEqual(check_cuda.call_args.args[1]["CUDA_VISIBLE_DEVICES"], "0")
            with self.assertRaisesRegex((ValueError, RuntimeError), "preflight"):
                agent._operation("runner_preflight", {**args, "run": {**base, "parameters":
                    {**base["parameters"], "--geometry-preset": "invalid_preset"}}})
            check_cuda.assert_called_once()  # Bad parameters never probe CUDA.

    def test_cuda_preflight_reports_failure_before_reserving_or_launching(self):
        base = {"run_id": uuid.uuid4().hex, "grid_tag": "G-00001", "pairing_id": None,
                "profiler": "none", "parameters": {"--max-iters": 2, "--warmup-iters": 0,
                "--n-layer": 2, "--n-embd": 64, "--n-head": 4, "DEPTH.order": 1},
                "dtype": "bfloat16", "attention_backend": "sdpa", "gpu_uuid": "GPU-0"}
        args = {"run": base, "gpu_key": "GPU-0", "host_label": "local"}
        real_run = agent.subprocess.run

        def no_cuda(command, **options):
            if command[:2] == [agent.sys.executable, "-c"]:
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "0")
                return SimpleNamespace(returncode=1, stdout="", stderr="RuntimeError: CUDA unknown error")
            return real_run(command, **options)

        with patch.object(agent.subprocess, "run", side_effect=no_cuda):
            with self.assertRaisesRegex(RuntimeError, "PyTorch cannot initialize CUDA on GPU 0.*CUDA unknown error"):
                agent._operation("runner_preflight", args)
        self.assertEqual(agent._read_state(), {})

    def test_cuda_preflight_accepts_initialized_selected_gpu(self):
        with patch.object(agent.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="Test GPU\n", stderr="")) as check:
            agent._check_torch_cuda(gpu(1), {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(check.call_args.args[0][:2], [agent.sys.executable, "-c"])
        self.assertEqual(check.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
