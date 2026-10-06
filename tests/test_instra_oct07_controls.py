# vvv THOG protect explicit recipe edits, launch snapshots and uniform bulk deletion authority
import copy
import io
import json
from types import SimpleNamespace

import pytest

import instra_network as network
import instra_runner as runner
import run_thog2_local_dashboard as dashboard
from tests.test_instra_runner_unittest import FakeNetwork


@pytest.fixture
def recipe_service(tmp_path, monkeypatch):
    for name, value in {"STATE_DIR": tmp_path, "STATE_PATH": tmp_path / "runner.json",
                        "LEASE_PATH": tmp_path / "lease", "GRID_SCRIPTS": tmp_path / "scripts"}.items():
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(network, "log_event", lambda *args: None)
    service = runner.RunnerService(FakeNetwork(), start_worker=False)
    yield service
    service.close()


def test_explicit_edit_and_rename_preserve_identity_and_launched_facts(recipe_service):
    recipe = {"label": "Original", "parameters": {"--max-iters": 2, "--warmup-iters": 0,
              "--n-layer": 2, "--n-embd": 64, "--n-head": 4, "--batch-size": 1, "--block-size": 32}}
    saved = recipe_service.save_recipe(None, recipe)
    grid = recipe_service.launch(saved["recipe_id"])
    original_grid = copy.deepcopy(runner._read()["grids"][0])
    original_created = saved["created_at"]
    edited = copy.deepcopy(recipe)
    edited["label"] = "Edited name"
    edited["parameters"]["--max-iters"] = 9
    updated = recipe_service.save_recipe(saved["recipe_id"], edited, update_existing=True)
    assert updated["recipe_id"] == saved["recipe_id"]
    assert updated["created_at"] == original_created
    assert updated["last_grid_id"] == grid["grid_id"]
    renamed = recipe_service.rename_recipe(saved["recipe_id"], "  Renamed  ")
    assert renamed["recipe"]["label"] == "Renamed"
    assert renamed["recipe"]["parameters"]["--max-iters"] == 9
    state = runner._read()
    assert len(state["recipes"]) == 1
    assert state["grids"][0] == original_grid
    with pytest.raises(ValueError):
        recipe_service.rename_recipe(saved["recipe_id"], "\ninvalid")
    with pytest.raises(KeyError):
        recipe_service.rename_recipe("missing", "Name")
    assert runner._read() == state
    # A normal Save still makes a copy; explicit Edit does not change that workflow.
    ordinary_copy = recipe_service.save_recipe(saved["recipe_id"], {**edited, "label": "Renamed",
        "parameters": {**edited["parameters"], "--max-iters": 10}})
    assert ordinary_copy["recipe_id"] != saved["recipe_id"]
    assert ordinary_copy["recipe"]["label"] == "Copy (1) of Renamed"


def test_launch_keeps_per_run_duration_from_the_existing_grid_estimate(recipe_service):
    recipe = {"label": "Estimate", "parameters": {"--max-iters": 2, "--warmup-iters": 0,
              "--n-layer": 2, "--n-embd": 64, "--n-head": 4, "--batch-size": 1, "--block-size": 32}}
    saved = recipe_service.save_recipe(None, recipe)
    grid = recipe_service.launch(saved["recipe_id"])
    state = runner._read()
    state["grids"][0]["state"] = "completed"
    for run in state["grids"][0]["runs"]:
        run.update(state="completed", duration_seconds=12, released=True)
    runner._write(state)
    preview = recipe_service.preview(recipe)
    assert preview["estimated_duration"]["seconds"] == 12
    assert all(run["estimated_duration_seconds"] == 12 for run in preview["runs"])
    launched = recipe_service.launch(saved["recipe_id"])
    assert launched["grid_id"] != grid["grid_id"]
    assert all(run["estimated_duration_seconds"] == 12 for run in launched["runs"])


def call_delete(monkeypatch, runs, payload, fail_id=None):
    calls = []

    class Handler:
        def do_GET(self):
            pass

        def do_POST(self):
            pass

        def do_DELETE(self):
            raise AssertionError("unexpected fallback")

        def _send_json(self, value, status=200):
            self.response = value
            self.status = status

    def force_delete(run_id):
        calls.append(run_id)
        if run_id == fail_id:
            raise OSError("busy copy")
        return {"queued": True}

    catalog = SimpleNamespace(runs=lambda: {"runs": runs}, force_delete_local_copy=force_delete)
    monkeypatch.setattr(dashboard, "_handler_for_before_runner", lambda catalog: Handler)
    actual_handler = dashboard._handler_for_with_runner(catalog)
    request = actual_handler()
    data = json.dumps(payload).encode()
    request.path = "/api/runs"
    request.headers = {"Content-Length": str(len(data))}
    request.rfile = io.BytesIO(data)
    request.do_DELETE()
    return request, calls


def test_bulk_force_delete_queues_all_remote_members_once(monkeypatch):
    runs = [{"dashboard_run_id": name, "remote_copy": True} for name in ["remote_a", "remote_b"]]
    request, calls = call_delete(monkeypatch, runs,
                                {"run_ids": ["remote_a", "remote_b", "remote_a"], "force_local": True})
    assert request.status == 200
    assert calls == ["remote_a", "remote_b"]
    assert request.response == {"queued_run_ids": calls, "deleted_run_ids": [], "errors": []}


@pytest.mark.parametrize("run_ids", [["remote", "local"], ["remote", "missing"], [], [1], "remote"])
def test_bulk_force_delete_rejects_invalid_or_mixed_selection_before_any_delete(monkeypatch, run_ids):
    runs = [{"dashboard_run_id": "remote", "remote_copy": True},
            {"dashboard_run_id": "local", "remote_copy": False}]
    request, calls = call_delete(monkeypatch, runs, {"run_ids": run_ids, "force_local": True})
    assert request.status in [400, 403]
    assert calls == []


def test_bulk_force_delete_reports_individual_queue_failures(monkeypatch):
    runs = [{"dashboard_run_id": name, "remote_copy": True} for name in ["a", "b"]]
    request, calls = call_delete(monkeypatch, runs, {"run_ids": ["a", "b"], "force_local": True}, fail_id="a")
    assert calls == ["a", "b"]
    assert request.response["queued_run_ids"] == ["b"]
    assert request.response["errors"] == [{"run_id": "a", "error": "busy copy"}]
# ^^^ THOG
