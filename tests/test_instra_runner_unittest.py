# vvv THOG exercise Grid expansion, reservations, identity, recovery and standalone export without a GPU
"""Runner and Node Agent regression tests; no training runtime required."""

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
import re
from unittest.mock import patch
import uuid
from types import SimpleNamespace

import instra_network as network
import instra_node_agent as agent
import instra_runner as runner
from thog_grid_runner import classic_script_for, command_for, expand, script_for, validate_recipe


class ArtifactNamingTests(unittest.TestCase):
    def test_grid_tag_precedes_host_without_changing_classic_names(self):
        # Isolate the actual run_descriptor method; this host has no torch and
        # cannot import sheet.run_config's GPU-dependent package initialization.
        source = (Path(__file__).resolve().parents[1] / "sheet" / "run_config.py").read_text()
        configuration = next(node for node in ast.parse(source).body
                             if isinstance(node, ast.ClassDef) and node.name == "OwtRunConfig")
        descriptor = next(node for node in configuration.body
                          if isinstance(node, ast.FunctionDef) and node.name == "run_descriptor")
        namespace = {"json": json, "os": os, "re": re,
                     "normalize_component": lambda value: value}
        exec(compile(ast.Module(body=[descriptor], type_ignores=[]), "run_config.py", "exec"), namespace)
        subject = SimpleNamespace(host_label="scruffy", experiment_prefix="trial_G-00012_abc_NONE",
                                  run_start_label="260928-1030", compact_artifact_fragment=lambda:None)
        with patch.dict(os.environ, {"THOG2_RUNNER_METADATA": json.dumps({"grid_tag":"G-00012"})}):
            self.assertEqual(namespace["run_descriptor"](subject),
                             "260928-1030_G-00012_scruffy_trial_abc_NONE___DENSE")
        with patch.dict(os.environ, {"THOG2_RUNNER_METADATA": ""}):
            self.assertEqual(namespace["run_descriptor"](subject),
                             "260928-1030_scruffy_trial_G-00012_abc_NONE___DENSE")

    def test_catalogue_loss_reads_adjacent_ansi_coloured_train_log(self):
        source = (Path(__file__).resolve().parents[1] / "run_thog2_local_dashboard_base.py").read_text()
        dashboard = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.ClassDef) and node.name == "RunDashboardState")
        method = next(node for node in dashboard.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_latest_logged_loss")
        namespace = {"Optional": __import__("typing").Optional, "Dict": __import__("typing").Dict,
                     "re": re, "math": math}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "run_thog2_local_dashboard_base.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            root.mkdir()
            (root / "train.log").write_text("\x1b[32mT 10 loss=4.21\x1b[0m\n\x1b[32mT 20 loss=3.75\x1b[0m\n")
            subject = SimpleNamespace(database_path=root / "charts.sqlite3", lock=threading.RLock(), loss_log_cache=None)
            self.assertEqual(namespace["_latest_logged_loss"](subject), 3.75)


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


class TwoHostNetwork(FakeNetwork):
    def __init__(self):
        super().__init__()
        remote_gpu = {**gpu(1), "gpu_id": "thog_host.remote.gpu.GPU-1"}
        self.host_state = {self.local_id: (self.gpus[:1], {}, {}),
                           "thog_host.remote": ([remote_gpu], {}, {})}

    def list_hosts(self):
        hosts = []
        for host_id, (gpus, _reservations, _attempts) in self.host_state.items():
            hosts.append({"thog_host_id": host_id, "display_name": host_id, "local": host_id == self.local_id,
                          "execution_enabled": True, "last_discovered": {"gpus": gpus,
                          "execution_profiles": [{"execution_profile_id": f"{host_id}.execution_profile.current"}]}})
        return {"local_id": self.local_id, "master_id": self.local_id, "release_pending": False, "hosts": hosts}

    def runner_call(self, host_id, operation, args=None):
        gpus, reservations, attempts = self.host_state[host_id]
        self.gpus, self.reservations, self.attempts = gpus, reservations, attempts
        original_id = self.local_id
        try:
            self.local_id = host_id
            return super().runner_call(host_id, operation, args)
        finally:
            self.local_id = original_id


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

    def test_stable_preview_ids_mixed_dense_and_depth_unique_launches(self):
        recipe={"label":"dense plus compact", "parameters":{"--geometry-preset":["dense","depth"],
            "DEPTH.order":[1,2],"--max-iters":2,"--warmup-iters":0,"--n-layer":2,"--n-embd":64,
            "--n-head":4,"--batch-size":1,"--block-size":32},"profilers":["nsys","ncu"]}
        one=self.service.preview(recipe)
        two=self.service.preview(recipe)
        self.assertEqual([run["run_id"] for run in one["runs"]],[run["run_id"] for run in two["runs"]])
        self.assertEqual(one["total_runs"],6)  # One dense reference; two DEPTH orders, each paired.
        self.assertEqual(sum("DEPTH.order" not in run["parameters"] for run in one["runs"]),2)
        saved=self.service.save_recipe(None,recipe)
        first=self.service.launch(saved["recipe_id"])
        second=self.service.launch(saved["recipe_id"])
        self.assertFalse({run["run_id"] for run in first["runs"]} & {run["run_id"] for run in second["runs"]})
        self.assertFalse({run["run_id"] for run in first["runs"]} & {run["run_id"] for run in one["runs"]})
        self.assertEqual(first["runs"][0]["pairing_id"], first["runs"][1]["pairing_id"])

    def test_mixed_presets_use_all_dense_layers_and_largest_depth_layer(self):
        recipe = {"label": "layer comparison", "parameters": {
            "--geometry-preset": ["dense", "depth"], "--n-layer": [2, 4, 8],
            "DEPTH.order": [1, 2], "--max-iters": 2, "--warmup-iters": 0,
            "--n-embd": 64, "--n-head": 4, "--batch-size": 1, "--block-size": 32}}
        trials = expand(recipe, stable_preview=True)
        self.assertEqual(len(trials), 5)
        self.assertEqual({run["parameters"]["--n-layer"] for run in trials if
                          run["parameters"]["--geometry-preset"] == "dense"}, {2, 4, 8})
        self.assertEqual({(run["parameters"]["--n-layer"], run["parameters"]["DEPTH.order"])
                          for run in trials if run["parameters"]["--geometry-preset"] == "depth"},
                         {(8, 1), (8, 2)})
        depth_only = {**recipe, "parameters": {**recipe["parameters"], "--geometry-preset": "depth"}}
        self.assertEqual(len(expand(depth_only)), 6)
        self.assertEqual([run["run_id"] for run in trials],
                         [run["run_id"] for run in expand(recipe, stable_preview=True)])

    def test_definite_power_rejection_does_not_create_unknown_or_dispatch_another_attempt(self):
        recipe = {**self.recipe, "gpu_pool": [gpu(0)["gpu_id"]], "power_caps": {gpu(0)["gpu_id"]: 280}}
        saved = self.service.save_recipe(None, recipe)
        grid = self.service.launch(saved["recipe_id"])
        original = self.fake.runner_call
        def deny_power(host_id, operation, args=None):
            if operation == "runner_launch":
                raise network.NetworkError("operation", "nvidia-smi: insufficient permissions for 280 W")
            return original(host_id, operation, args)
        self.fake.runner_call = deny_power
        self.service._refresh()
        runs = self.service.snapshot()["grids"][0]["runs"]
        self.assertEqual(sum(len(run["attempts"]) for run in runs), 1)
        self.assertEqual(runs[0]["state"], "blocked")
        self.assertEqual(runs[0]["attempts"][0]["state"], "failed")
        self.assertIn("insufficient permissions", runs[0]["blocking_reason"])
        self.assertFalse(any(run["state"] == "unknown" for run in runs))
        self.assertEqual(self.fake.reservations, {})
        self.assertIn("280 W", self.service.file(grid["grid_id"], "log").read_text())

    def test_both_exported_scripts_reset_blank_caps_and_restore_default(self):
        selected = {**self.recipe, "gpu_pool": [gpu(0)["gpu_id"]],
                    "power_caps": {gpu(0)["gpu_id"]: 280}}
        saved = self.service.save_recipe(None, selected)
        grid = self.service.launch(saved["recipe_id"])
        for kind in ("script", "classic"):
            path = self.service.file(grid["grid_id"], kind)
            source = path.read_text()
            self.assertIn("instra-power-control GPU-0 280", source)
            self.assertIn("instra-power-control GPU-0 default", source)
            import subprocess
            subprocess.run(["bash", "-n", str(path)], check=True)
        blank = self.service.save_recipe(None, {**selected, "power_caps": {}})
        blank_grid = self.service.launch(blank["recipe_id"])
        self.assertIn("instra-power-control GPU-0 default",
                      self.service.file(blank_grid["grid_id"], "classic").read_text())

    def test_lost_acknowledgement_recovers_after_two_idle_observations(self):
        recipe = {**self.recipe, "gpu_pool": [gpu(0)["gpu_id"]]}
        saved = self.service.save_recipe(None, recipe)
        self.service.launch(saved["recipe_id"])
        original = self.fake.runner_call
        def lost(host_id, operation, args=None):
            if operation == "runner_launch":
                raise OSError("SSH acknowledgement lost")
            return original(host_id, operation, args)
        self.fake.runner_call = lost
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"][0]["runs"][0]["state"], "unknown")
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"][0]["runs"][0]["state"], "unknown")
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"][0]["runs"][0]["state"], "blocked")
        self.assertEqual(self.fake.reservations, {})

    def test_two_hosts_disjoint_grids_and_queued_remote_run(self):
        self.fake = TwoHostNetwork()
        self.service.network = self.fake
        def recipe_for(gpu_id):
            return {**self.recipe, "gpu_pool": [gpu_id], "parameters": {**self.recipe["parameters"],
                "--n-layer": 2, "DEPTH.order": 1}}
        local = self.service.launch(self.service.save_recipe(None, recipe_for(gpu(0)["gpu_id"]))["recipe_id"])
        remote_id = "thog_host.remote.gpu.GPU-1"
        remote_recipe = self.service.save_recipe(None, recipe_for(remote_id))
        first = self.service.launch(remote_recipe["recipe_id"])
        second = self.service.launch(remote_recipe["recipe_id"])
        self.service._refresh()
        grids = {grid["grid_id"]: grid for grid in self.service.snapshot()["grids"]}
        self.assertEqual(grids[local["grid_id"]]["state"], "running")
        self.assertEqual(grids[first["grid_id"]]["state"], "running")
        self.assertEqual(grids[second["grid_id"]]["runs"][0]["state"], "queued")
        remote_attempt = grids[first["grid_id"]]["runs"][0]["attempts"][0]["attempt_id"]
        self.fake.host_state["thog_host.remote"][2][remote_attempt]["state"] = "completed"
        self.service._refresh()
        self.service._refresh()
        grids = {grid["grid_id"]: grid for grid in self.service.snapshot()["grids"]}
        self.assertEqual(grids[second["grid_id"]]["state"], "running")

    def test_grid_rename_preserves_run_identity_and_updates_manifest(self):
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        renamed = self.service.rename_grid(grid["grid_id"], "New grid name")
        self.assertEqual(renamed["label"], "New grid name")
        self.assertEqual([run["run_id"] for run in renamed["runs"]],
                         [run["run_id"] for run in grid["runs"]])
        self.assertEqual(self.service.snapshot()["grids"][0]["label"], "New grid name")
        self.assertEqual(json.loads(self.service.file(grid["grid_id"], "manifest").read_text())["label"], "New grid name")
        self.assertIn("RENAME", self.service.file(grid["grid_id"], "log").read_text().upper())
        self.assertEqual(self.service.file(grid["grid_id"], "classic").name,
                         f"{grid['grid_tag']}_grid_bash_runner_script.sh")
        with self.assertRaisesRegex(ValueError, "printable"):
            self.service.rename_grid(grid["grid_id"], "bad\nname")

    def test_history_delete_rejects_active_grid_and_keeps_recipe_and_training_files(self):
        recipe = {**self.recipe, "gpu_pool": [gpu(0)["gpu_id"]],
                  "parameters": {**self.recipe["parameters"], "--n-layer": 2, "DEPTH.order": 1}}
        saved = self.service.save_recipe(None, recipe)
        grid = self.service.launch(saved["recipe_id"])
        with self.assertRaisesRegex(ValueError, "Stop the Grid"):
            self.service.delete_grid_history(grid["grid_id"])
        outside = runner.STATE_DIR / "training-output"
        outside.mkdir()
        (outside / "checkpoint.pt").write_bytes(b"retained")
        self.service.stop_grid(grid["grid_id"])
        self.service._refresh()
        self.assertEqual(self.service.delete_grid_history(grid["grid_id"]), {"deleted": grid["grid_id"]})
        self.assertEqual(self.service.snapshot()["grids"], [])
        self.assertEqual(self.service.snapshot()["recipes"][0]["recipe_id"], saved["recipe_id"])
        self.assertEqual((outside / "checkpoint.pt").read_bytes(), b"retained")
        self.assertFalse((runner.GRID_SCRIPTS / grid["grid_tag"]).exists())

    def test_launch_snapshot_survives_runtime_placement_change(self):
        saved = self.service.save_recipe(None, self.recipe)
        grid = self.service.launch(saved["recipe_id"])
        launch = grid["launch_configuration"]
        self.assertEqual(launch["recipe"], self.recipe)
        self.assertEqual(launch["runs"][0]["host_id"], grid["runs"][0]["host_id"])
        grid["runs"][0]["gpu"]["ordinal"] = 99
        grid["recipe"]["label"] = "Changed after launch"
        self.assertNotEqual(launch["runs"][0]["gpu"]["ordinal"], 99)
        self.assertEqual(launch["recipe"]["label"], "small")

    def test_stop_can_proceed_during_slow_gpu_preflight(self):
        recipe={"label":"live", "parameters":{"--max-iters":2,"--warmup-iters":0,
            "--n-layer":2,"--n-embd":64,"--n-head":4,"--batch-size":1,"--block-size":32}}
        saved=self.service.save_recipe(None,recipe)
        grid=self.service.launch(saved["recipe_id"])
        entered=threading.Event(); release=threading.Event()
        previous=self.fake.runner_call
        def slow_preflight(host_id,operation,args=None):
            if operation=="runner_preflight":
                entered.set()
                self.assertTrue(release.wait(5),"preflight did not release in test")
            return previous(host_id,operation,args)
        self.fake.runner_call=slow_preflight
        worker=threading.Thread(target=self.service._refresh,daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        try:
            self.service.stop_grid(grid["grid_id"])
            self.assertEqual(self.service.snapshot()["grids"][0]["state"],"stopping")
        finally:
            release.set();worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.service.snapshot()["grids"][0]["runs"][0]["state"],"cancelled")

    def test_node_agent_status_does_not_wait_for_gpu_preflight(self):
        entered=threading.Event(); release=threading.Event()
        def operation(_state,name,_args):
            if name=="runner_preflight":
                entered.set()
                self.assertTrue(release.wait(5))
            return {"operation":name}
        with patch.object(agent,"_read_state",return_value={}),patch.object(agent,"_runner_operation",side_effect=operation):
            worker=threading.Thread(target=lambda:agent._operation("runner_preflight",{}),daemon=True)
            worker.start()
            self.assertTrue(entered.wait(2))
            try:
                self.assertEqual(agent._operation("runner_reconcile",{}),{"operation":"runner_reconcile"})
            finally:
                release.set();worker.join(timeout=3)
            self.assertFalse(worker.is_alive())

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

    def test_classic_script_preserves_wrapper_options_and_escapes_values(self):
        run = {"run_id": uuid.uuid4().hex, "pairing_id": None, "grid_tag": "G-00001", "profiler": "none",
               "host_label": "test", "parameters": {"--max-iters": 3, "--batch-size": 2, "--geometry-preset": "depth",
               "--learning-rate": .0009, "--min-lr": .00009, "DEPTH.order": 2,
               "--data-dir": "data/folder with 'quotes'"}, "gpu": gpu(0)}
        script = classic_script_for([run])
        self.assertIn('cd "$(dirname "${BASH_SOURCE[0]}")/../.."', script)
        self.assertIn(" ./train_OWT.sh -g ", script)
        self.assertIn("-n 3 -b 2 -p depth -c 90 -f 9", script)
        self.assertIn("--select-depth --option DEPTH.order=2", script)
        self.assertIn("'data/folder with '\"'\"'quotes'\"'\"''", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", script)

    def test_large_grid_confirmation_deleted_recipe_history_and_event_log(self):
        recipe = {**self.recipe, "parameters": {"--max-iters": 2,
                       "--n-layer": list(range(11, 22)), "--warmup-iters": 0, "--n-embd": 64,
                       "--n-head": 4, "--batch-size": list(range(1, 11)), "--block-size": 32}, "max_parallel": 1000}
        saved = self.service.save_recipe(None, recipe)
        with self.assertRaisesRegex(ValueError, "110 runs"):
            self.service.launch(saved["recipe_id"])
        grid = self.service.launch(saved["recipe_id"], confirm_large=True)
        self.assertEqual(len(grid["runs"]), 110)
        self.assertEqual(len(self.service.file(grid["grid_id"], "log").read_text().splitlines()), 1)
        self.assertTrue(self.service.file(grid["grid_id"], "classic").read_text().startswith("#!/usr/bin/env bash"))
        with self.assertRaisesRegex(ValueError, "active Grids"):
            self.service.delete_recipe(saved["recipe_id"])
        self.service.stop_grid(grid["grid_id"])
        self.service._refresh()
        self.assertEqual(self.service.delete_recipe(saved["recipe_id"]), {"deleted": saved["recipe_id"]})
        self.assertEqual(self.service.snapshot()["grids"][0]["grid_id"], grid["grid_id"])
        events = [json.loads(line) for line in self.service.file(grid["grid_id"], "log").read_text().splitlines()]
        self.assertEqual(events[0]["event"], "launch")
        self.assertTrue(any(event["event"] == "stop" for event in events))
        self.assertTrue(all(event.get("time") for event in events))
        self.service.file(grid["grid_id"], "classic").unlink()
        self.service.file(grid["grid_id"], "log").unlink()
        self.assertIn("./train_OWT.sh", self.service.file(grid["grid_id"], "classic").read_text())
        historical=json.loads(self.service.file(grid["grid_id"], "log").read_text())
        self.assertEqual(historical["event"], "historical")

    def test_large_grid_cuda_failure_does_not_preflight_every_queued_run(self):
        recipe={"label":"CUDA failure", "parameters":{"--max-iters":2,"--warmup-iters":0,
                "--n-layer":list(range(2,22)),"--n-embd":64,"--n-head":4,
                "--batch-size":1,"--block-size":32},"gpu_pool":[gpu(0)["gpu_id"]]}
        saved=self.service.save_recipe(None,recipe)
        self.service.launch(saved["recipe_id"])
        calls=[]
        previous=self.fake.runner_call
        def cuda_failure(host_id,operation,args=None):
            if operation=="runner_preflight":
                calls.append(args["gpu_key"])
                raise RuntimeError("CUDA initialization failed")
            return previous(host_id,operation,args)
        self.fake.runner_call=cuda_failure
        self.service._refresh()
        self.assertEqual(len(calls),1)
        self.assertTrue(all(run["state"]=="blocked" for run in self.service.snapshot()["grids"][0]["runs"]))

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
        grace_periods=[]
        def with_stop(host_id,operation,args=None):
            if operation=="runner_stop":
                grace_periods.append(args["grace_seconds"])
                self.fake.attempts[args["attempt_id"]]["state"]="failed"
                return self.fake.attempts[args["attempt_id"]]
            return prior(host_id,operation,args)
        self.fake.runner_call=with_stop
        self.service.stop_grid(grid["grid_id"])
        self.service._refresh()
        result=self.service.snapshot()["grids"][0]
        self.assertEqual(result["runs"][0]["state"],"completed")
        self.assertTrue(all(run["state"] in {"completed","cancelled"} for run in result["runs"]))
        self.assertTrue(all(seconds == 120 for seconds in grace_periods))

    def test_running_attempt_receives_graceful_stop_period(self):
        saved=self.service.save_recipe(None,{**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        periods=[]
        previous=self.fake.runner_call
        def capture(host_id,operation,args=None):
            if operation=="runner_stop":
                periods.append(args["grace_seconds"])
                return {"state":"running"}
            return previous(host_id,operation,args)
        self.fake.runner_call=capture
        self.service.stop_grid(grid["grid_id"])
        self.assertEqual(periods,[120])

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

    def test_two_grids_run_concurrently_on_disjoint_gpus(self):
        one=self.service.save_recipe(None,{**self.recipe,"label":"GPU 0", "gpu_pool":[gpu(0)["gpu_id"]]})
        two=self.service.save_recipe(None,{**self.recipe,"label":"GPU 1", "gpu_pool":[gpu(1)["gpu_id"]]})
        first=self.service.launch(one["recipe_id"])
        second=self.service.launch(two["recipe_id"])
        self.service._refresh()
        grids={grid["grid_id"]:grid for grid in self.service.snapshot()["grids"]}
        self.assertEqual(grids[first["grid_id"]]["state"],"running")
        self.assertEqual(grids[second["grid_id"]]["state"],"running")
        self.assertEqual(self.fake.reservations["GPU-0"],first["grid_id"])
        self.assertEqual(self.fake.reservations["GPU-1"],second["grid_id"])
        self.assertTrue(any(run["state"]=="running" for run in grids[first["grid_id"]]["runs"]))
        self.assertTrue(any(run["state"]=="running" for run in grids[second["grid_id"]]["runs"]))

    def test_kill_flush_releases_only_its_gpu_and_retains_history(self):
        first_recipe=self.service.save_recipe(None,{**self.recipe,"label":"GPU 0", "gpu_pool":[gpu(0)["gpu_id"]]})
        second_recipe=self.service.save_recipe(None,{**self.recipe,"label":"GPU 1", "gpu_pool":[gpu(1)["gpu_id"]]})
        first=self.service.launch(first_recipe["recipe_id"])
        second=self.service.launch(second_recipe["recipe_id"])
        self.service._refresh()
        initial=self.service.snapshot()["grids"]
        running=next(run for run in initial[0]["runs"] if run["state"]=="running")
        prior=self.fake.runner_call
        stop_calls=[]
        def stop_target(host_id,operation,args=None):
            if operation=="runner_stop":
                stop_calls.append(args)
                self.fake.attempts[args["attempt_id"]]["state"]="failed"
                return self.fake.attempts[args["attempt_id"]]
            return prior(host_id,operation,args)
        self.fake.runner_call=stop_target
        self.service.kill_and_flush(first["grid_id"])
        self.assertEqual(self.service.snapshot()["grids"][0]["state"],"flushing")
        self.service._refresh()
        current=self.service.snapshot()
        self.assertEqual(current["grids"][0]["state"],"cancelled")
        self.assertEqual(current["grids"][1]["state"],"running")
        self.assertEqual(stop_calls[0]["attempt_id"],running["attempts"][-1]["attempt_id"])
        self.assertEqual(stop_calls[0]["grace_seconds"],0)
        self.assertNotIn("GPU-0",self.fake.reservations)
        self.assertEqual(self.fake.reservations["GPU-1"],second["grid_id"])
        self.assertEqual(len(current["recipes"]),2)
        self.assertTrue(self.service.file(first["grid_id"],"log").is_file())
        self.assertTrue(self.service.file(first["grid_id"],"manifest").is_file())

    def test_kill_flush_aborted_dispatch_recovers_gpu_without_hiding_history(self):
        saved=self.service.save_recipe(None,{**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        grid=self.service.launch(saved["recipe_id"])
        previous=self.fake.runner_call
        def lost_launch(host_id,operation,args=None):
            if operation=="runner_launch":
                raise OSError("launch acknowledgement lost")
            return previous(host_id,operation,args)
        self.fake.runner_call=lost_launch
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"][0]["state"],"blocked")
        self.assertIn("GPU-0",self.fake.reservations)
        self.service.kill_and_flush(grid["grid_id"])
        self.service._refresh()
        current=self.service.snapshot()["grids"][0]
        self.assertEqual(current["state"],"cancelled")
        self.assertTrue(current["flushed_at"])
        self.assertNotIn("GPU-0",self.fake.reservations)
        self.assertTrue(self.service.file(grid["grid_id"],"log").is_file())
        self.fake.runner_call=previous
        next_grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        self.assertEqual(next(item for item in self.service.snapshot()["grids"] if
                              item["grid_id"]==next_grid["grid_id"])["state"],"running")

    def test_kill_flush_blocked_preflight_without_an_attempt(self):
        saved=self.service.save_recipe(None,{**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        grid=self.service.launch(saved["recipe_id"])
        prior=self.fake.runner_call
        def no_cuda(host_id,operation,args=None):
            if operation=="runner_preflight":
                raise RuntimeError("CUDA unavailable")
            return prior(host_id,operation,args)
        self.fake.runner_call=no_cuda
        self.service._refresh()
        before=self.service.snapshot()["grids"][0]
        self.assertTrue(all(not run["attempts"] for run in before["runs"]))
        self.service.kill_and_flush(grid["grid_id"])
        self.service._refresh()
        after=self.service.snapshot()["grids"][0]
        self.assertEqual(after["state"],"cancelled")
        self.assertEqual(self.fake.reservations,{})
        self.assertEqual(after["recipe_id"],saved["recipe_id"])

    def test_kill_flush_keeps_uncertain_attempt_until_agent_is_reachable(self):
        saved=self.service.save_recipe(None,{**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        original=self.fake.runner_call
        def unreachable(host_id,operation,args=None):
            if operation=="runner_reconcile":
                raise OSError("host unavailable")
            return original(host_id,operation,args)
        self.fake.runner_call=unreachable
        self.service.kill_and_flush(grid["grid_id"])
        self.service._refresh()
        self.assertEqual(self.service.snapshot()["grids"][0]["state"],"flushing")
        self.assertIn("GPU-0",self.fake.reservations)
        self.fake.runner_call=original

    def test_terminal_grid_retries_a_failed_gpu_release_after_restart(self):
        recipe={**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]],
                "parameters":{**self.recipe["parameters"],"--n-layer":2,"DEPTH.order":1}}
        saved=self.service.save_recipe(None,recipe)
        grid=self.service.launch(saved["recipe_id"])
        self.service._refresh()
        attempt=self.service.snapshot()["grids"][0]["runs"][0]["attempts"][-1]["attempt_id"]
        self.fake.attempts[attempt]["state"]="completed"
        original=self.fake.runner_call
        failures=[1]
        def flaky_release(host_id,operation,args=None):
            if operation=="runner_release" and failures[0]:
                failures[0]-=1
                raise OSError("release reply unavailable")
            return original(host_id,operation,args)
        self.fake.runner_call=flaky_release
        self.service._refresh()
        terminal=self.service.snapshot()["grids"][0]
        self.assertEqual(terminal["state"],"completed")
        self.assertIn("GPU-0",self.fake.reservations)
        self.service.close()
        replacement=runner.RunnerService(self.fake,start_worker=False)
        self.addCleanup(replacement.close)
        replacement._refresh()
        repaired=replacement.snapshot()["grids"][0]
        self.assertTrue(repaired["runs"][0]["released"])
        self.assertNotIn("GPU-0",self.fake.reservations)
        self.assertEqual(repaired["state"],"completed")
        self.assertIn("Recovered release",replacement.file(grid["grid_id"],"log").read_text())

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
        patcher = patch.object(agent, "_set_gpu_power", side_effect=lambda selected, requested: {
            "before": {"current_w": 280.0, "default_w": 250.0},
            "after": {"current_w": float(requested or 250), "default_w": 250.0},
            "target_w": float(requested or 250), "changed": True})
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

    def test_power_permission_failure_exposes_helper_error_and_can_release(self):
        grid_id = uuid.uuid4().hex
        agent._operation("runner_reserve", {"grid_id":grid_id,"gpu_key":"GPU-0",
                                            "required_mib":2048,"headroom_mib":512,"power_cap_w":280})
        run = {"run_id":uuid.uuid4().hex,"grid_tag":"G-00009","pairing_id":None,"profiler":"none",
               "parameters":{"--max-iters":2,"--warmup-iters":0},"dtype":"bfloat16",
               "attention_backend":"sdpa","gpu_uuid":"GPU-0"}
        with patch.object(agent,"_set_gpu_power",side_effect=RuntimeError("sudo: a password is required")):
            with self.assertRaisesRegex(RuntimeError,"password is required"):
                agent._operation("runner_launch", {"grid_id":grid_id,"attempt_id":uuid.uuid4().hex,
                    "run":run,"gpu_key":"GPU-0","host_label":"test","thog_host_id":"thog_host.test",
                    "execution_profile":"current"})
        self.assertEqual(agent._operation("runner_reconcile",{})["attempts"],{})
        self.assertTrue(agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":"GPU-0"})["released"])

    def test_installed_power_helper_validates_range_and_readback(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "instra_power_control.py"
        spec = importlib.util.spec_from_file_location("instra_power_control_test",path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        before = {"uuid":"GPU-12345678","current_w":280.0,"default_w":250.0,
                  "minimum_w":200.0,"maximum_w":300.0}
        after = {**before,"current_w":250.0}
        with (patch.object(helper,"query",side_effect=[before,after]),
              patch.object(helper.subprocess,"run") as write,
              patch.object(helper.sys,"argv",["power","GPU-12345678","default"])):
            write.return_value.returncode = 0
            with patch("builtins.print") as printed: helper.main()
            self.assertEqual(json.loads(printed.call_args.args[0])["after"]["current_w"],250.0)
            self.assertEqual(write.call_args.args[0][-2:],["-pl","250"])
        with (patch.object(helper,"query",return_value=before),
              patch.object(helper.sys,"argv",["power","GPU-12345678","350"])):
            with self.assertRaisesRegex(ValueError,"outside GPU range"):
                helper.main()

    def test_force_stop_clears_unknown_attempt_only_after_process_group_is_gone(self):
        grid_id, attempt_id = uuid.uuid4().hex, uuid.uuid4().hex
        agent._operation("runner_reserve", {"grid_id":grid_id,"gpu_key":"GPU-0",
                                            "required_mib":2048,"headroom_mib":512})
        state=agent._read_state()
        state["attempts"]={attempt_id:{"attempt_id":attempt_id,"grid_id":grid_id,
            "run_id":uuid.uuid4().hex,"gpu_key":"GPU-0","pid":9876543,
            "pid_start_time":"former-process","started_at":runner.now(),"state":"running"}}
        agent._write_state(state)
        with patch.object(agent,"_running",return_value=False), patch.object(agent.os,"killpg",return_value=None):
            self.assertEqual(agent._operation("runner_stop",{"grid_id":grid_id,
                "attempt_id":attempt_id,"grace_seconds":0})["state"],"unknown")
            with self.assertRaisesRegex(RuntimeError,"still owns"):
                agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":"GPU-0"})
        with patch.object(agent,"_running",return_value=False), patch.object(agent.os,"killpg",side_effect=ProcessLookupError):
            self.assertEqual(agent._operation("runner_status",{"attempt_id":attempt_id})["state"],"cancelled")
            self.assertTrue(agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":"GPU-0"})["released"])

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
