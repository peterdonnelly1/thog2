# vvv THOG persist Recipes and Grids, arbitrate queues, reconcile Node Agents and export scripts
"""Instra Runner service. The Network Service owns remote transport and host authority."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import uuid
from instra_grid_identity import choose_prefix

import instra_network as network
from instra_duration_estimator import estimate
from thog_grid_runner import CATALOGUE, COMMON, classic_script_for, environment_for, expand, script_for, validate_recipe

TERMINAL = {"completed", "failed", "cancelled"}
STATE_DIR = network.STATE_DIR
STATE_PATH = STATE_DIR / "runner.json"
LEASE_PATH = STATE_DIR / "runner-controller.lock"
GRID_SCRIPTS = Path(__file__).resolve().parent / "grid-scripts"


class ControllerStateChanged(Exception):
    """A concurrent user action won while the controller was doing remote work."""


def now():
    return datetime.now(timezone.utc).isoformat()


def _read():
    try:
        return json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return {"recipes": {}, "grids": [], "next_tag": 1}


def _write(value):
    _capture_wall_times(value)                                                                                                                               # <<< THOG persist run/Grid wall-clock boundaries alongside the completed-run duration estimator history
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, STATE_PATH)


# vvv THOG recover available legacy clocks and keep run ends unknown while queued/running after retry
def _capture_wall_times(value):
    for grid in value.get("grids", []):
        starts, ends = [], []
        for run in grid.get("runs", []):
            attempts = run.get("attempts", [])
            run_starts = [attempt["started_at"] for attempt in attempts if attempt.get("started_at")]
            if run_starts:
                run.setdefault("started_at", min(run_starts))
            if run.get("started_at"):
                starts.append(run["started_at"])
            if run.get("state") in TERMINAL and attempts and attempts[-1].get("finished_at"):
                run["finished_at"] = attempts[-1]["finished_at"]
            elif run.get("state") not in TERMINAL:
                run.pop("finished_at", None)
            if run.get("finished_at"):
                ends.append(run["finished_at"])
        if starts:
            grid.setdefault("started_at", min(starts))
        if (grid.get("state") in TERMINAL and grid.get("runs")
                and all(run.get("state") in TERMINAL for run in grid["runs"])
                and all(run.get("finished_at") or not run.get("attempts") for run in grid["runs"]) and ends):
            grid.setdefault("finished_at", max(ends))
# ^^^ THOG


def _write_grid_file(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _grid_event(grid, event, detail):
    path = GRID_SCRIPTS / grid["grid_tag"] / "events.jsonl"
    record = {"time": now(), "grid_id": grid["grid_id"], "event": event, "detail": detail}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _power_detail(run, result, action):
    power = result.get("power_restoration") if action == "restoration" else result.get("power_control")
    if not power:
        return (f"{run['run_id'][:8]} {run['host_label']} GPU {run['gpu']['ordinal']} {action}: "
                f"requested={run.get('requested_power_w') or 'default'} W; "
                f"observed={result.get('observed_power_w', 'unknown')} W")
    before, after = power["before"], power["after"]
    return (f"{run['run_id'][:8]} {run['host_label']} GPU {run['gpu']['ordinal']} {action}: "
            f"requested={run.get('requested_power_w') or 'default'} W; "
            f"current={before['current_w']:g} W; default={before['default_w']:g} W; "
            f"target={power['target_w']:g} W; readback={after['current_w']:g} W; "
            f"write={'yes' if power['changed'] else 'unnecessary'}")


def _memory_peak(parameters):
    layers = int(parameters.get("--n-layer", 72))
    width = int(parameters.get("--n-embd", 768))
    batch = int(parameters.get("--batch-size", 12))
    context = int(parameters.get("--block-size", 256))
    segment = int(parameters.get("--checkpoint-segment-size", 4))
    if min(layers, width, batch, context, segment) < 1:
        raise ValueError("Model dimensions, batch and checkpoint segment must be positive")
    # Safely pessimistic approximation, exposed to the user as an estimate.
    bytes_needed = 2048 * 1048576 + 12 * layers * width * width * 12 + batch * context * width * min(layers, segment) * 12
    return (bytes_needed + 1048575) // 1048576


def _dtype_backend(gpu):
    try:
        major = int(str(gpu.get("compute_cap", "0")).split(".")[0])
    except ValueError:
        major = 0
    return ("bfloat16" if major >= 8 else "float16", "sdpa")


class RunnerService:
    def __init__(self, network_service, start_worker=True):
        self.network = network_service
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lease = LEASE_PATH.open("a+")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.controller = True
        except BlockingIOError:
            self.controller = False
        self.reconciled = False
        self._last_activity = None
        self._release_retry_at = {}
        self._lease_release_scheduled = False
        if self.controller:
            self._clean_uncommitted_files()
            persisted = _read()                                                                                                                              # <<< THOG backfill recoverable historical run/Grid clocks once, without inventing missing timestamps
            previous = json.dumps(persisted, sort_keys=True)
            _capture_wall_times(persisted)
            if json.dumps(persisted, sort_keys=True) != previous:
                _write(persisted)
        if self.controller and start_worker:
            self.worker = threading.Thread(target=self._loop, name="instra-runner-controller", daemon=True)
            self.worker.start()

    def close(self):
        self.stop_event.set()
        if self.controller and hasattr(self, "worker"):
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                if not self._lease_release_scheduled:
                    self._lease_release_scheduled = True
                    def release_after_worker():
                        self.worker.join()
                        self.lease.close()
                    threading.Thread(target=release_after_worker, name="instra-runner-lease-release", daemon=True).start()
                return
        self.lease.close()

    def _require_controller(self):
        if not self.controller:
            raise RuntimeError("Another Instra Backend owns the Runner controller lease")

    def _clean_uncommitted_files(self):
        if not GRID_SCRIPTS.is_dir():
            return
        committed = {grid["grid_tag"] for grid in _read()["grids"]}
        for marker in GRID_SCRIPTS.glob("*/.runner-created"):
            directory = marker.parent
            if directory.name in committed:
                continue
            # Only files marked as Runner-created are eligible for crash cleanup.
            import shutil
            shutil.rmtree(directory)
            (GRID_SCRIPTS / f"{directory.name}.sh").unlink(missing_ok=True)

    def snapshot(self):
        # Atomic replace makes this read safe without waiting for network I/O in the controller.
        state = _read()
        hosts = self.network.list_hosts()
        grids = {grid["grid_id"]: grid for grid in state["grids"]}
        for saved in state["recipes"].values():
            saved["state"] = grids.get(saved.get("last_grid_id"), {}).get("state", saved.get("state", "ready"))
        return {"recipes": list(state["recipes"].values()), "grids": state["grids"],
                "catalogue": CATALOGUE, "common": COMMON, "controller": self.controller,
                "reconciled": self.reconciled, "local_id": hosts["local_id"], "master_id": hosts["master_id"]}

    def save_recipe(self, recipe_id, recipe):
        self._require_controller()
        validate_recipe(recipe)
        with self.lock:
            state = _read()
            if recipe_id is not None and recipe_id not in state["recipes"]:
                raise KeyError("Unknown Recipe")
            recipe_id = recipe_id or uuid.uuid4().hex
            created_at = state["recipes"].get(recipe_id, {}).get("created_at", now())
            state["recipes"][recipe_id] = {"recipe_id": recipe_id, "recipe": recipe,
                                            "created_at": created_at, "updated_at": now(), "state": "ready"}
            _write(state)
            return state["recipes"][recipe_id]

    def delete_recipe(self, recipe_id):
        self._require_controller()
        with self.lock:
            state = _read()
            if recipe_id not in state["recipes"]:
                raise KeyError("Unknown Recipe")
            if any(grid["recipe_id"] == recipe_id and grid["state"] not in TERMINAL for grid in state["grids"]):
                raise ValueError("Stop active Grids using this Recipe before deleting it")
            del state["recipes"][recipe_id]
            _write(state)
        return {"deleted": recipe_id}

    def rename_grid(self, grid_id, label):
        self._require_controller()
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120 or any(ord(c) < 32 for c in label):
            raise ValueError("Grid name must contain 1–120 printable characters")
        with self.lock:
            state = _read()
            grid = next((item for item in state["grids"] if item["grid_id"] == grid_id), None)
            if grid is None:
                raise KeyError("Unknown Grid")
            grid["label"] = label.strip()
            _write(state)
            manifest = GRID_SCRIPTS / grid["grid_tag"] / "manifest.json"
            if manifest.exists():
                _write_grid_file(manifest, grid)
            _grid_event(grid, "rename", f"Grid renamed to {grid['label']}")
            return grid

    # vvv THOG remove only terminal Grid history after reservations and attempts have ended
    def delete_grid_history(self, grid_id):
        self._require_controller()
        with self.lock:
            state = _read()
            grid = next((item for item in state["grids"] if item["grid_id"] == grid_id), None)
            if grid is None:
                raise KeyError("Unknown Grid")
            if grid["state"] not in TERMINAL or any(run.get("state") not in TERMINAL for run in grid["runs"]):
                raise ValueError("Stop the Grid and finish all attempts before deleting its History")
            if any(not run.get("released") for run in grid["runs"]) or (grid.get("queue_mode") == "dynamic" and
                    len(grid.get("released_gpu_ids", [])) < len(grid["gpu_pool"])):
                raise ValueError("GPU release is still pending; retain History until reconciliation finishes")
            state["grids"] = [item for item in state["grids"] if item["grid_id"] != grid_id]
            _write(state)
        # The generated script directory belongs to this Grid. Training logs
        # and model artifacts elsewhere are deliberately not touched.
        import shutil
        directory = GRID_SCRIPTS / grid["grid_tag"]
        if directory.is_dir() and directory.parent == GRID_SCRIPTS:
            shutil.rmtree(directory)
        (GRID_SCRIPTS / f"{grid['grid_tag']}.sh").unlink(missing_ok=True)
        return {"deleted": grid_id}
    # ^^^ THOG

    def _pool(self, recipe, trial_count=None):
        hosts = self.network.list_hosts()
        allowed = set(recipe.get("host_ids", []))
        requested = set(recipe.get("gpu_pool", []))
        if requested and len(requested) != len(recipe.get("gpu_pool", [])):
            raise ValueError("Duplicate GPU identities")
        pool = []
        for host in hosts["hosts"]:
            host_id = host["thog_host_id"]
            if allowed and host_id not in allowed:
                continue
            if not host["local"] and not host["execution_enabled"]:
                continue
            discovery = host.get("last_discovered") or {}
            if requested and not any(gpu_id.startswith(f"{host_id}.gpu.") for gpu_id in requested):
                continue
            if not discovery.get("execution_profiles"):
                continue
            try:
                current = self.network.runner_call(host_id, "runner_reconcile")
            except (network.NetworkError, RuntimeError, OSError) as error:
                if allowed and host_id in allowed:
                    raise ValueError(f"{host_id}: live GPU preflight unavailable: {error}") from error
                continue
            identities = {gpu["gpu_key"]: gpu for gpu in discovery.get("gpus", [])}
            for live_gpu in current.get("gpus", []):
                previously = identities.get(live_gpu["gpu_key"])
                if previously is None or previously.get("uuid") != live_gpu.get("uuid"):
                    continue
                gpu = {**previously, **live_gpu}
                gpu_id = gpu.get("gpu_id", f"{host_id}.gpu.{gpu['gpu_key']}")
                if not requested or gpu_id in requested:
                    pool.append({"host_id": host_id, "host_label": host["display_name"],
                                 "execution_profile": discovery["execution_profiles"][0].get("execution_profile_id", "current"),
                                 "gpu": gpu, "reservation_owner": current.get("reservations", {}).get(gpu["gpu_key"])})
        if requested - {p["gpu"].get("gpu_id", f"{p['host_id']}.gpu.{p['gpu']['gpu_key']}") for p in pool}:
            raise ValueError("A selected GPU is unavailable or execution is disabled")
        # Caps are remembered per GPU, but only resolved placements consume them.
        # Unticked or disabled hosts must not invalidate an otherwise eligible pool.
        if not pool:
            raise ValueError("No discovered execution-enabled GPU is eligible")
        return pool[:1] if len(recipe.get("profilers", ["none"])) == 1 and (trial_count if trial_count is not None else len(expand(recipe))) == 1 and not requested else pool

    def preview(self, recipe, *, source=None, tight=False):
        trials = expand(recipe, stable_preview=True)
        pool = self._pool(recipe, len(trials))
        ready = [place for place in pool if not place.get("reservation_owner") or
                 (source and (place["reservation_owner"].get("grid_id") if isinstance(place["reservation_owner"], dict)
                              else place["reservation_owner"]) == source["grid_id"])]
        placement_pool = pool if tight or source is None or not ready else ready
        planned = []
        for index, trial in enumerate(trials):
            location = placement_pool[index % len(placement_pool)]
            if tight and source:
                prior = source["runs"][index]
                location = next((choice for choice in pool if choice["host_id"] == prior["host_id"] and
                                 choice["gpu"]["gpu_key"] == prior["gpu"]["gpu_key"]), None)
                if location is None:
                    raise ValueError(f"Tight Repeat GPU {prior['host_id']} {prior['gpu']['gpu_key']} unavailable")
            gpu = location["gpu"]
            dtype, backend = _dtype_backend(gpu)
            if tight and source:
                prior = source["runs"][index]
                if (location["execution_profile"] != prior["execution_profile"] or gpu.get("uuid") != prior["gpu"].get("uuid") or
                    dtype != prior["dtype"] or backend != prior["attention_backend"] or
                    recipe.get("power_caps", {}).get(gpu.get("gpu_id")) != prior.get("requested_power_w")):
                    raise ValueError("Tight Repeat requires its prior host, GPU, dtype and backend")
            gpu_id = gpu.get("gpu_id", f"{location['host_id']}.gpu.{gpu['gpu_key']}")
            planned.append({**trial, **location, "dtype": dtype, "attention_backend": backend,
                            "requested_power_w": recipe.get("power_caps", {}).get(gpu_id),
                            "power_request_source": f"recipe.power_caps[{gpu_id}]" if gpu_id in recipe.get("power_caps", {}) else "blank/default",
                            "default_power_w": gpu.get("default_power_w"), "current_power_w": gpu.get("power_cap_w"),
                            "required_mib": _memory_peak(trial["parameters"]), "state": "ready",
                            "attempts": [], "blocking_reason": "Ready; placement is provisional until Launch",
                            "placement_fixed": bool(tight), "placement_status": "proposed",
                            "execution_environment": {**environment_for(trial), "CUDA_VISIBLE_DEVICES": str(gpu["ordinal"])}})
        history = _read()["grids"]
        return {"runs": planned, "total_runs": len(planned), "gpu_pool": pool,
                "estimated_duration": estimate(planned, history, len(pool))}

    def _repeat_recipe(self, source, mode):
        if mode not in {"loose", "tight"}:
            raise ValueError("Invalid repeat mode")
        recipe = json.loads(json.dumps(source["recipe"]))
        if mode == "loose":
            recipe.pop("gpu_pool", None)
            recipe.pop("host_ids", None)
            recipe.pop("power_caps", None)
        else:
            recipe["gpu_pool"] = list(dict.fromkeys(run["gpu"].get("gpu_id", f"{run['host_id']}.gpu.{run['gpu']['gpu_key']}") for run in source["runs"]))
            recipe["host_ids"] = list(dict.fromkeys(run["host_id"] for run in source["runs"]))
        return recipe

    def repeat_preview(self, grid_id, mode):
        source = next((grid for grid in _read()["grids"] if grid["grid_id"] == grid_id), None)
        if source is None:
            raise KeyError("Unknown source Grid")
        recipe = self._repeat_recipe(source, mode)
        plan = self.preview(recipe, source=source, tight=mode == "tight")
        changes = []
        for old, new in zip(source["runs"], plan["runs"]):
            change = {key: [old_value, new_value] for key, old_value, new_value in (
                ("host", old["host_id"], new["host_id"]), ("gpu_uuid", old["gpu"].get("uuid"), new["gpu"].get("uuid")),
                ("dtype", old["dtype"], new["dtype"]), ("backend", old["attention_backend"], new["attention_backend"]),
                ("power_w", old.get("requested_power_w"), new.get("requested_power_w"))) if old_value != new_value}
            if change:
                changes.append({"prior_run_id": old["run_id"], "changes": change})
        return {"mode": mode, "source_grid_id": grid_id, "recipe": recipe, "total_runs": plan["total_runs"],
                "physical_executions": len(plan["runs"]), "gpu_pool": plan["gpu_pool"],
                "estimated_duration": plan["estimated_duration"], "changes": changes}

    def launch(self, recipe_id, *, confirm_large=False, source_grid_id=None, repeat_mode=None):
        self._require_controller()
        with self.lock:
            state = _read()
            saved = state["recipes"].get(recipe_id)
            if saved is None and source_grid_id is None:
                raise KeyError("Unknown Recipe")
            source = next((grid for grid in state["grids"] if grid["grid_id"] == source_grid_id), None) if source_grid_id else None
            if source_grid_id and source is None:
                raise KeyError("Unknown source Grid")
            if repeat_mode not in (None, "loose", "tight"):
                raise ValueError("Invalid repeat mode")
            recipe = self._repeat_recipe(source, repeat_mode) if source else json.loads(json.dumps(saved["recipe"]))
            self.lock.release()
            try:
                plan = self.preview(recipe, source=source, tight=repeat_mode == "tight")
            finally:
                self.lock.acquire()
            current = _read()
            if not source and current["recipes"].get(recipe_id) != saved:
                raise ValueError("Recipe changed during validation; launch the saved version again")
            state = current
            if plan["total_runs"] > 100 and not confirm_large:
                raise ValueError(f"These values will take the size of the grid to {plan['total_runs']:,} runs. Confirm before launch")
            number = state["next_tag"]
            if number > 99999:
                raise RuntimeError("Five-digit Grid tags exhausted")
            if hasattr(self.network, "grid_prefix"):
                prefix = self.network.grid_prefix()
            else:
                prefix = state.setdefault("grid_prefix", choose_prefix(self.network.local_id.removeprefix("thog_host.")))
            tag = f"{prefix}-{number:05d}"
            state["next_tag"] = number + 1
            grid_id = uuid.uuid4().hex
            pair_ids = {}
            runs = [{**run, "run_id": uuid.uuid4().hex,
                     "pairing_id": pair_ids.setdefault(run["pairing_id"], uuid.uuid4().hex) if run["pairing_id"] else None,
                     "grid_tag": tag, "grid_id": grid_id, "recipe_id": recipe_id, "state": "queued",
                     "grid_owner_host_id": self.network.local_id} for run in plan["runs"]]
            grid = {"grid_id": grid_id, "grid_tag": tag, "recipe_id": recipe_id, "label": recipe["label"],
                    "recipe": recipe, "source_grid_id": source_grid_id, "repeat_mode": repeat_mode,
                    "created_at": now(), "launched_at": now(), "state": "queued", "runs": runs, "gpu_pool": plan["gpu_pool"],
                    "grid_owner_host_id": self.network.local_id, "queue_mode": "dynamic",
                    "estimated_duration": plan["estimated_duration"],
                    "launch_configuration": json.loads(json.dumps({"recipe": recipe,
                        "runs": [{key: run.get(key) for key in ("run_id", "parameters", "host_id", "host_label",
                            "gpu", "execution_profile", "execution_environment", "dtype", "attention_backend",
                            "requested_power_w", "power_request_source", "default_power_w", "current_power_w")}
                            for run in runs]}))}
            GRID_SCRIPTS.mkdir(exist_ok=True)
            directory = GRID_SCRIPTS / tag
            directory.mkdir(exist_ok=False)
            script = GRID_SCRIPTS / f"{tag}.sh"
            try:
                (directory / ".runner-created").write_text(grid_id)
                script.write_text(script_for(runs))
                script.chmod(0o700)
                classic = directory / f"{tag}_grid_bash_runner_script.sh"
                classic.write_text(classic_script_for(runs))
                classic.chmod(0o700)
                _write_grid_file(directory / "manifest.json", grid)
                _write_grid_file(directory / "placement.json", plan["gpu_pool"])
                _write_grid_file(directory / "status.json", {"state": "queued", "created_at": grid["created_at"]})
                _grid_event(grid, "launch", f"{len(runs)} runs queued from Recipe {recipe_id}; "
                            + "; ".join(f"{r['host_label']} GPU {r['gpu']['ordinal']}: current={r.get('current_power_w')} W, "
                                        f"default={r.get('default_power_w')} W, requested={r.get('requested_power_w') or 'default'} W "
                                        f"({r.get('power_request_source')})"
                                        for r in runs[:20]))
                state["grids"].append(grid)
                if not source:
                    state["recipes"][recipe_id].update(state="queued", last_grid_id=grid_id)
                _write(state)
            except Exception:
                import shutil
                shutil.rmtree(directory)
                script.unlink(missing_ok=True)
                raise
            network.log_event("Runner", "launch", self.network.local_id, "success", f"{tag} accepted ({len(runs)} runs)")
            return grid

    def file(self, grid_id, name):
        grid = next((g for g in _read()["grids"] if g["grid_id"] == grid_id), None)
        if grid is None:
            raise KeyError("Unknown Grid")
        paths = {"script": GRID_SCRIPTS / (grid["grid_tag"] + ".sh"),
                 "classic": GRID_SCRIPTS / grid["grid_tag"] / f"{grid['grid_tag']}_grid_bash_runner_script.sh",
                 "manifest": GRID_SCRIPTS / grid["grid_tag"] / "manifest.json",
                 "placement": GRID_SCRIPTS / grid["grid_tag"] / "placement.json",
                 "status": GRID_SCRIPTS / grid["grid_tag"] / "status.json",
                 "conversion": GRID_SCRIPTS / grid["grid_tag"] / "conversion.json",
                 "log": GRID_SCRIPTS / grid["grid_tag"] / "events.jsonl"}
        if name not in paths:
            raise KeyError("Unknown Grid file")
        # Older Grids predate both exports. Reconstruct their immutable resolved
        # commands and mark the event log as historical when opened for the first time.
        if name in {"classic", "log"} and not paths[name].exists():
            with self.lock:
                path = paths[name]
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if name == "classic":
                        path.write_text(classic_script_for(grid["runs"]))
                        path.chmod(0o700)
                    else:
                        with path.open("x", encoding="utf-8") as stream:
                            stream.write(json.dumps({"time": grid["created_at"], "grid_id": grid_id,
                                "event": "historical", "detail": "Grid created before event logging was enabled; earlier events unavailable"}) + "\n")
        return paths[name]

    def attempt_log(self, grid_id, run_id, attempt_id):
        grid = next((item for item in _read()["grids"] if item["grid_id"] == grid_id), None)
        if grid is None:
            raise KeyError("Unknown Grid")
        run = next((item for item in grid["runs"] if item["run_id"] == run_id), None)
        if run is None or not any(item["attempt_id"] == attempt_id for item in run["attempts"]):
            raise KeyError("Unknown attempt in this Grid")
        result = self.network.runner_call(run["host_id"], "runner_log", {"attempt_id": attempt_id, "max_bytes": 16384})
        # A click after an Agent upgrade should repair stale generic excerpts
        # on terminal Grids as well as show the fetched log immediately.
        excerpt = str(result.get("text", ""))[-4096:]
        if excerpt:
            with self.lock:
                state = _read()
                current_grid = next((item for item in state["grids"] if item["grid_id"] == grid_id), None)
                current_run = next((item for item in current_grid["runs"] if item["run_id"] == run_id), None) if current_grid else None
                current = next((item for item in current_run["attempts"] if item["attempt_id"] == attempt_id), None) if current_run else None
                if current and current.get("state") == "failed" and current.get("failure_excerpt") != excerpt:
                    current["failure_excerpt"] = excerpt
                    _write(state)
        return result

    def stop_grid(self, grid_id, force=False):
        self._require_controller()
        with self.lock:
            state = _read()
            grid = next((item for item in state["grids"] if item["grid_id"] == grid_id), None)
            if grid is None:
                raise KeyError("Unknown Grid")
            if grid["state"] in TERMINAL:
                return grid
            for run in grid["runs"]:
                if run["state"] in {"queued", "blocked"}:
                    run.update(state="cancelled", blocking_reason="Stopped by user")
                elif run["state"] in {"running", "dispatching", "unknown"}:
                    run["stop_requested"] = True
                    run["force_stop_requested"] = bool(force)
            grid["state"] = "stopping"
            _grid_event(grid, "stop", "Force stop requested" if force else "Graceful Ctrl-C stop requested")
            _write(state)
            # The controller sends stop requests outside this lock on its next
            # reconciliation. Slow hosts must not freeze the UI action.
            return grid

    def kill_and_flush(self, grid_id):
        """Request a force stop while retaining the Recipe, logs and Grid history."""
        self._require_controller()
        with self.lock:
            state = _read()
            grid = next((item for item in state["grids"] if item["grid_id"] == grid_id), None)
            if grid is None:
                raise KeyError("Unknown Grid")
            if grid["state"] in TERMINAL:
                return grid
            grid["flush_requested"] = True
            grid.setdefault("flush_started_at", now())
            grid["state"] = "flushing"
            for run in grid["runs"]:
                if run["state"] in {"queued", "blocked"}:
                    run.update(state="cancelled", blocking_reason="Killed and flushed by user")
                elif run["state"] in {"running", "dispatching", "unknown"}:
                    run["stop_requested"] = True
                    if run["attempts"]:
                        run["attempts"][-1].pop("force_stop_sent", None)
            _grid_event(grid, "flush", "Force stop requested; awaiting verified attempt termination and GPU release")
            _write(state)
            return grid

    def retry_run(self, grid_id, run_id):
        self._require_controller()
        with self.lock:
            state = _read()
            grid = next((g for g in state["grids"] if g["grid_id"] == grid_id), None)
            if grid is None:
                raise KeyError("Unknown Grid")
            run = next((r for r in grid["runs"] if r["run_id"] == run_id), None)
            if run is None or run["state"] != "failed":
                raise ValueError("Only a failed run with known outcome can be retried")
            run.update(state="queued", blocking_reason="Retry requested")
            run.pop("released", None)
            run.pop("next_retry_at", None)
            run.pop("finished_at", None)                                                                                                                      # <<< THOG a retry opens a new end clock and must replace the previous attempt's estimator duration
            run.pop("duration_seconds", None)
            grid["state"] = "queued"
            grid.pop("finished_at", None)  # <<< THOG resume the execution clock on an explicit retry
            _grid_event(grid, "retry", f"Run {run_id} queued for a new attempt")
            _write(state)
            return grid

    def fail_grid(self, grid_id):
        self.stop_grid(grid_id)
        with self.lock:
            state = _read()
            grid = next(g for g in state["grids"] if g["grid_id"] == grid_id)
            grid["user_failed"] = True
            _write(state)
            return grid

    def convert_to_loose(self, grid_id):
        self._require_controller()
        with self.lock:
            state = _read()
            grid = next((g for g in state["grids"] if g["grid_id"] == grid_id), None)
            if grid is None or grid["repeat_mode"] != "tight" or grid["state"] in TERMINAL:
                raise ValueError("Only an active Tight Repeat can be converted")
            if any(run["state"] in {"unknown", "dispatching"} for run in grid["runs"]):
                raise RuntimeError("Resolve uncertain dispatches before changing placement")
            recipe = self._repeat_recipe(grid, "loose")
            before_preview = json.loads(json.dumps(state))
            self.lock.release()
            try:
                plan = self.preview(recipe, source=grid)
            finally:
                self.lock.acquire()
            if _read() != before_preview:
                raise RuntimeError("Grid changed during placement validation; retry conversion")
            previous = []
            for run, proposed in zip(grid["runs"], plan["runs"]):
                if run["state"] not in {"queued", "blocked"}:
                    continue
                previous.append((run["host_id"], run["gpu"]["gpu_key"]))
                for key in ("host_id", "host_label", "execution_profile", "gpu", "dtype", "attention_backend", "requested_power_w", "power_request_source", "default_power_w", "current_power_w", "required_mib", "execution_environment"):
                    run[key] = proposed[key]
                run.update(state="queued", blocking_reason="Loose placement confirmed", placement_fixed=False,
                           placement_status="proposed")
                run.pop("next_retry_at", None)
            grid["recipe"] = recipe
            grid["repeat_mode"] = "loose"
            grid["queue_mode"] = "dynamic"
            grid["gpu_pool"] = plan["gpu_pool"]
            grid["conversion"] = {"time": now(), "from": "tight", "to": "loose", "moved_assignments": len(previous)}
            _write_grid_file(GRID_SCRIPTS / grid["grid_tag"] / "conversion.json", grid["conversion"])
            _grid_event(grid, "placement", f"Tight to Loose; moved {len(previous)} queued assignments")
            _write(state)
            for host_id, gpu_key in set(previous):
                if any(run["host_id"] == host_id and run["gpu"]["gpu_key"] == gpu_key and run["state"] not in TERMINAL
                       for run in grid["runs"]):
                    continue
                try:
                    self._controller_call(state, host_id, "runner_release", {"grid_id": grid_id, "gpu_key": gpu_key})
                except (network.NetworkError, RuntimeError, OSError):
                    pass  # The owner record remains until reconciliation permits release.
            return grid

    # vvv THOG persist decisions before I/O, yield the controller lock and reject stale writeback
    def _controller_call(self, state, host_id, operation, args=None):
        if _read() != state:
            _write(state)
        committed = json.loads(json.dumps(state))
        self.lock.release()
        error = None
        try:
            try:
                result = self.network.runner_call(host_id, operation, args)
            except Exception as caught:
                error = caught
        finally:
            self.lock.acquire()
        if _read() != committed:
            raise ControllerStateChanged()
        if error is not None:
            raise error
        return result

    @staticmethod
    def _compatible(grid, run, place):
        if run.get("placement_fixed") and (run["host_id"], run["gpu"]["gpu_key"]) != (place["host_id"], place["gpu"]["gpu_key"]):
            return False
        capacity = place["gpu"].get("memory_mib")
        return capacity is None or capacity >= run["required_mib"] + grid["recipe"].get("headroom_mib", 512)

    def _dynamic_candidates(self, state, grid, snapshots, busy_gpus, excluded_gpus=()):
        selected = set()
        if grid.get("queue_mode") != "dynamic":
            return None
        pending = [run for run in grid["runs"] if run["state"] in {"queued", "blocked"}]
        for run in pending:
            if not any(self._compatible(grid, run, place) for place in grid["gpu_pool"]):
                run.update(state="blocked", blocking_reason="No selected GPU has sufficient estimated capacity for this run")
        for place in grid["gpu_pool"]:
            host_id, key = place["host_id"], place["gpu"]["gpu_key"]
            gpu_id = place["gpu"].get("gpu_id", f"{host_id}.gpu.{key}")
            if (host_id, key) in excluded_gpus or grid.get("gpu_retry_at", {}).get(gpu_id, 0) > time.time():
                continue
            snapshot = snapshots.get(host_id, {})
            if "error" in snapshot:
                continue
            if snapshot.get("protocol", 3) < 3:
                for run in pending:
                    run.update(state="blocked", blocking_reason=f"Restart the updated Node Agent on {place['host_label']} to enable local GPU queues")
                continue
            compatible = [run for run in pending if run["run_id"] not in selected and self._compatible(grid, run, place)]
            if not compatible:
                continue
            # Register every selected eligible GPU even while it is busy. Node
            # Agents retain order; updating memory requirements never requeues.
            gpu_id = place["gpu"].get("gpu_id", f"{host_id}.gpu.{key}")
            request = {"grid_id": grid["grid_id"], "gpu_key": key,
                       "required_mib": min(run["required_mib"] for run in compatible),
                       "headroom_mib": grid["recipe"].get("headroom_mib", 512),
                       "power_cap_w": grid["recipe"].get("power_caps", {}).get(gpu_id)}
            try:
                queued = self._controller_call(state, host_id, "runner_queue", request)
                snapshot.setdefault("waiting_grids", {})[key] = queued.get("waiting_grids", [])
                owner = queued.get("reservation")
                owner_id = owner.get("grid_id") if isinstance(owner, dict) else owner
                if owner_id and owner_id != grid["grid_id"] or (host_id, key) in busy_gpus:
                    continue
                live = next((gpu for gpu in snapshot.get("gpus", []) if gpu["gpu_key"] == key), place["gpu"])
                available = [run for run in compatible if run.get("next_retry_at", 0) <= time.time() and
                             live.get("free_mib") is not None and live["free_mib"] >= run["required_mib"] + request["headroom_mib"]]
                if not available:
                    continue
                run = available[0]
                reserved = self._controller_call(state, host_id, "runner_reserve", {**request, "required_mib": run["required_mib"]})
                snapshot.setdefault("reservations", {})[key] = reserved.get("reservation", {"grid_id": grid["grid_id"]})
                dtype, backend = _dtype_backend(place["gpu"])
                run.update(host_id=host_id, host_label=place["host_label"], gpu=place["gpu"],
                           execution_profile=place["execution_profile"], dtype=dtype, attention_backend=backend,
                           requested_power_w=request["power_cap_w"], default_power_w=live.get("default_power_w"),
                           current_power_w=live.get("power_cap_w"), placement_status="assigned",
                           power_request_source=f"recipe.power_caps[{gpu_id}]" if request["power_cap_w"] is not None else "blank/default",
                           execution_environment={**environment_for(run), "CUDA_VISIBLE_DEVICES": str(place["gpu"]["ordinal"])})
                selected.add(run["run_id"])
            except (network.NetworkError, RuntimeError, ValueError, OSError) as error:
                for run in compatible:
                    run["blocking_reason"] = str(error)
        return selected

    def _release_dynamic_gpus(self, state, grid, snapshots):
        if grid.get("queue_mode") != "dynamic":
            return
        pending = [run for run in grid["runs"] if run["state"] in {"queued", "blocked"}]
        released_places = grid.setdefault("released_gpu_ids", [])
        for place in grid["gpu_pool"]:
            host_id, key = place["host_id"], place["gpu"]["gpu_key"]
            gpu_id = place["gpu"].get("gpu_id", f"{host_id}.gpu.{key}")
            if any(self._compatible(grid, run, place) for run in pending):
                if gpu_id in released_places:
                    released_places.remove(gpu_id)
                continue
            if any(run["host_id"] == host_id and run["gpu"]["gpu_key"] == key and
                   run["state"] in {"running", "dispatching", "unknown"} for run in grid["runs"]):
                continue
            snapshot = snapshots.get(host_id, {})
            if "error" in snapshot or "reservations" not in snapshot:
                continue
            owner = snapshot["reservations"].get(key)
            owner_id = owner.get("grid_id") if isinstance(owner, dict) else owner
            queued = any(item["grid_id"] == grid["grid_id"] for item in snapshot.get("waiting_grids", {}).get(key, []))
            if gpu_id not in released_places or owner_id == grid["grid_id"] or queued:
                try:
                    if owner_id == grid["grid_id"] or queued:
                        self._controller_call(state, host_id, "runner_release", {"grid_id": grid["grid_id"], "gpu_key": key})
                        _grid_event(grid, "reservation", f"Released {place['host_label']} GPU {place['gpu']['ordinal']}; no compatible pending runs")
                        if owner_id == grid["grid_id"]:
                            snapshot["reservations"].pop(key, None)
                    if gpu_id not in released_places:
                        released_places.append(gpu_id)
                except (network.NetworkError, RuntimeError, OSError) as error:
                    grid["release_error"] = str(error)
                    continue
            for run in grid["runs"]:
                if run["host_id"] == host_id and run["gpu"]["gpu_key"] == key and run["state"] in TERMINAL:
                    run["released"] = True
    # ^^^ THOG

    def _refresh(self):
        self._require_controller()
        with self.lock:
            state = _read()
            active = [grid for grid in state["grids"] if grid["state"] not in TERMINAL]
            stranded = [grid for grid in state["grids"] if grid["state"] in TERMINAL and
                        (any(not run.get("released") for run in grid["runs"]) or
                         grid.get("queue_mode") == "dynamic" and len(grid.get("released_gpu_ids", [])) < len(grid["gpu_pool"]))]
            active_hosts = {run["host_id"] for grid in active for run in grid["runs"]}
            active_hosts |= {place["host_id"] for grid in active for place in grid.get("gpu_pool", [])}
            stranded_hosts = {run["host_id"] for grid in stranded for run in grid["runs"] if not run.get("released")}
            stranded_hosts |= {place["host_id"] for grid in stranded for place in grid.get("gpu_pool", [])}
            hosts = active_hosts | {host_id for host_id in stranded_hosts
                                    if self._release_retry_at.get(host_id, 0) <= time.time()}
            snapshots = {}
            before_requests = _read()
            self.lock.release()
            try:
                for host_id in hosts:
                    try:
                        snapshots[host_id] = self.network.runner_call(host_id, "runner_reconcile")
                    except (network.NetworkError, OSError) as error:
                        snapshots[host_id] = {"error": str(error)}
                        if host_id in stranded_hosts and host_id not in active_hosts:
                            self._release_retry_at[host_id] = time.time() + 30
            finally:
                self.lock.acquire()
            if _read() != before_requests:
                return  # A user action changed the Grid while remote hosts were responding.
            self.reconciled = True
            changed = False
            # A 3,200-run Grid must not monopolize the controller while CUDA or
            # argument preflights fail. This is a per-poll dispatch budget, not
            # a limit on how many runs may be active across polling cycles.
            preflight_budget = 8
            failed_gpu_preflights = {}
            busy_gpus = {(run["host_id"], run["gpu"]["gpu_key"])
                         for grid in active for run in grid["runs"]
                         if run["state"] in {"running", "dispatching", "unknown"}}
            uncertain_gpus = {(run["host_id"], run["gpu"]["gpu_key"])
                              for grid in active for run in grid["runs"] if run["state"] == "unknown"}
            for grid in active:
                previous_grid = json.dumps(grid, sort_keys=True)
                previous_state = grid["state"]
                previous_runs = {run["run_id"]: (run["state"], run.get("blocking_reason", ""), len(run["attempts"]))
                                 for run in grid["runs"]}
                for run in grid["runs"]:
                    if run["state"] not in {"running", "dispatching", "unknown"}:
                        continue
                    snapshot = snapshots.get(run["host_id"], {})
                    latest = run["attempts"][-1]
                    remote = snapshot.get("attempts", {}).get(latest["attempt_id"])
                    if remote is None:
                        gpu = next((item for item in snapshot.get("gpus", [])
                                    if item["gpu_key"] == run["gpu"]["gpu_key"]), None)
                        absent_idle = "error" not in snapshot and gpu is not None and not gpu.get("compute_pids")
                        latest["absent_idle_observations"] = latest.get("absent_idle_observations", 0) + 1 if absent_idle else 0
                        if absent_idle and (grid.get("flush_requested") or latest["absent_idle_observations"] >= 2):
                            run.update(state="cancelled" if grid.get("flush_requested") else "blocked",
                                       blocking_reason="Attempt absent from Node Agent; GPU idle; safe to retry" )
                            latest["state"] = "cancelled" if grid.get("flush_requested") else "failed"
                            if not grid.get("flush_requested"):
                                run["next_retry_at"] = time.time() + 15
                            _grid_event(grid, "reconcile", f"{run['run_id'][:8]} attempt {latest['attempt_id']} absent; GPU idle; "
                                        "launch did not take effect")
                            try:
                                released = self._controller_call(state, run["host_id"], "runner_release", {
                                    "grid_id": grid["grid_id"], "gpu_key": run["gpu"]["gpu_key"]})
                                _grid_event(grid, "power", _power_detail(run, released, "restoration"))
                                snapshot.get("reservations", {}).pop(run["gpu"]["gpu_key"], None)
                                failed_gpu_preflights[(run["host_id"], run["gpu"]["gpu_key"])] = "Recovered absent attempt; retry placement on the next poll"
                            except (network.NetworkError, RuntimeError, OSError) as release_error:
                                run["blocking_reason"] += f"; release/restoration pending: {release_error}"
                        else:
                            run["state"] = "unknown"
                            uncertain_gpus.add((run["host_id"], run["gpu"]["gpu_key"]))
                            run["blocking_reason"] = snapshot.get("error", "Attempt absent; waiting for a second idle GPU observation")
                    else:
                        stop_error = None
                        force_stop = bool(grid.get("flush_requested") or run.get("force_stop_requested"))
                        stop_key = "force_stop_sent" if force_stop else "stop_sent"
                        if run.get("stop_requested") and remote["state"] in {"running", "unknown"} and not latest.get(stop_key):
                            try:
                                remote = self._controller_call(state, run["host_id"], "runner_stop", {
                                    "grid_id": grid["grid_id"], "attempt_id": latest["attempt_id"], "grace_seconds": 0 if force_stop else 120})
                                latest[stop_key] = True
                            except (network.NetworkError, RuntimeError, OSError, KeyError) as error:
                                stop_error = str(error)
                        run["state"] = "cancelled" if run.get("stop_requested") and remote["state"] in TERMINAL else remote["state"]
                        run["blocking_reason"] = (f"Force stop pending: {stop_error}" if stop_error else
                                                  "Node Agent outcome uncertain; GPU remains reserved" if remote["state"] == "unknown" else "")
                        latest.update({key: remote.get(key) for key in ("pid", "exit_code", "log_path", "requested_power_w", "default_power_w", "observed_power_w", "power_control", "finished_at")})
                        latest["state"] = remote["state"]
                        if remote.get("started_at"):
                            latest.setdefault("dispatched_at", latest["started_at"])
                            latest["started_at"] = remote["started_at"]
                        if remote["state"] == "failed" and "failure_excerpt" not in latest:
                            try:
                                excerpt = self._controller_call(state, run["host_id"], "runner_log", {
                                    "attempt_id": latest["attempt_id"], "max_bytes": 4096})["text"]
                                latest["failure_excerpt"] = excerpt[-4096:]
                            except (network.NetworkError, RuntimeError, OSError, KeyError) as error:
                                latest["failure_excerpt"] = f"Attempt log unavailable: {type(error).__name__}: {error}"
                        if remote["state"] in TERMINAL and not run.get("duration_seconds"):
                            finished = datetime.fromisoformat(remote["finished_at"]).timestamp() if remote.get("finished_at") else time.time()
                            run["duration_seconds"] = max(0, finished - datetime.fromisoformat(latest["started_at"]).timestamp())
                busy_gpus = {(run["host_id"], run["gpu"]["gpu_key"]) for item in active for run in item["runs"]
                             if run["state"] in {"running", "dispatching", "unknown"}}
                dynamic_candidates = self._dynamic_candidates(state, grid, snapshots, busy_gpus, failed_gpu_preflights)
                for run in grid["runs"]:
                    if run["state"] not in {"queued", "blocked"}:
                        continue
                    if dynamic_candidates is not None and run["run_id"] not in dynamic_candidates:
                        continue
                    if run.get("next_retry_at", 0) > time.time():
                        continue
                    host_id, key = run["host_id"], run["gpu"]["gpu_key"]
                    snapshot = snapshots.get(host_id, {})
                    if "error" in snapshot:
                        run.update(state="blocked", blocking_reason=snapshot["error"])
                        continue
                    if (host_id, key) in failed_gpu_preflights:
                        run.update(state="blocked", blocking_reason=failed_gpu_preflights[(host_id, key)],
                                   next_retry_at=time.time() + 15)
                        continue
                    if (host_id, key) in busy_gpus:
                        if (host_id, key) in uncertain_gpus:
                            owner = next(((grid_owner, run_owner) for grid_owner in active for run_owner in grid_owner["runs"]
                                          if run_owner["host_id"] == host_id and run_owner["gpu"]["gpu_key"] == key
                                          and run_owner["state"] == "unknown"), None)
                            if owner:
                                grid_owner, run_owner = owner
                                run.update(state="blocked", blocking_reason=f"Waiting for {grid_owner['grid_tag']} "
                                           f"run {run_owner['run_id'][:8]} on {host_id} GPU {key}: "
                                           f"{run_owner.get('blocking_reason', 'launch uncertain')}; reconciliation pending")
                        else:
                            run.update(state="queued", blocking_reason="Waiting for a run on this GPU to finish")
                        continue
                    if preflight_budget <= 0:
                        continue
                    try:
                        headroom = grid["recipe"].get("headroom_mib", 512)
                        metadata = {field: run[field] for field in ("run_id", "grid_tag", "pairing_id", "profiler", "parameters", "dtype", "attention_backend")}
                        metadata["gpu_uuid"] = run["gpu"].get("uuid")
                        preflight_budget -= 1
                        before_preflight = _read()
                        preflight_error = None
                        self.lock.release()
                        try:
                            try:
                                self.network.runner_call(host_id, "runner_preflight", {"run": metadata, "gpu_key": key,
                                                                                        "host_label": run["host_label"]})
                            except (network.NetworkError, RuntimeError, ValueError, OSError) as error:
                                preflight_error = error
                        finally:
                            self.lock.acquire()
                        if _read() != before_preflight:
                            return  # Stop/Save/Launch won the race; reconcile the new state next poll.
                        if preflight_error is not None:
                            raise preflight_error
                        self._controller_call(state, host_id, "runner_reserve", {"grid_id": grid["grid_id"], "gpu_key": key,
                                                                               "required_mib": run["required_mib"], "headroom_mib": headroom,
                                                                               "power_cap_w": run.get("requested_power_w")})
                        run.pop("next_retry_at", None)
                        run.pop("released", None)
                        attempt_id = uuid.uuid4().hex
                        latest = {"attempt_id": attempt_id, "started_at": now(), "state": "dispatching"}
                        grid.setdefault("started_at", latest["started_at"])  # <<< THOG persist the first dispatch clock
                        run["attempts"].append(latest)
                        run.update(state="dispatching", blocking_reason="")
                        _write(state)  # Commit the attempt identity BEFORE sending it to the Node Agent.
                        result = self._controller_call(state, host_id, "runner_launch", {"grid_id": grid["grid_id"],
                                                "attempt_id": attempt_id, "gpu_key": key, "run": metadata,
                                                "host_label": run["host_label"], "thog_host_id": host_id,
                                                "execution_profile": run["execution_profile"], "recipe_id": grid["recipe_id"],
                                                "grid_owner_host_id": grid.get("grid_owner_host_id", self.network.local_id)})
                        latest.update({"pid": result.get("pid"), "log_path": result.get("log_path"), "state": result["state"],
                                       "requested_power_w": result.get("requested_power_w"), "default_power_w": result.get("default_power_w"),
                                       "observed_power_w": result.get("observed_power_w"), "power_control": result.get("power_control")})
                        if result.get("started_at"):
                            latest["dispatched_at"] = latest["started_at"]
                            latest["started_at"] = result["started_at"]
                        run["state"] = result["state"]
                        busy_gpus.add((host_id, key))
                        _grid_event(grid, "power", _power_detail(run, result, "launch check/write/readback"))
                    except (network.NetworkError, RuntimeError, ValueError, OSError) as error:
                        if run["attempts"] and run["state"] == "dispatching":
                            # An operation error is a definitive Node Agent rejection. A lost
                            # transport acknowledgement remains uncertain until reconciliation.
                            definite = not isinstance(error, (OSError, network.NetworkError)) or (
                                isinstance(error, network.NetworkError) and error.category == "operation")
                            if definite:
                                latest.update(state="failed", failure_excerpt=str(error), finished_at=now())
                                run.update(state="blocked", blocking_reason=f"Launch rejected before training: {error}",
                                           next_retry_at=time.time() + 15)
                                failed_gpu_preflights[(host_id, key)] = run["blocking_reason"]
                                _grid_event(grid, "launch_error", f"{run['run_id'][:8]} {host_id} GPU {key}: {error}")
                                try:
                                    released = self._controller_call(state, host_id, "runner_release", {
                                        "grid_id": grid["grid_id"], "gpu_key": key})
                                    _grid_event(grid, "power", _power_detail(run, released, "restoration"))
                                except (network.NetworkError, RuntimeError, OSError) as release_error:
                                    run["blocking_reason"] += f"; release/restoration pending: {release_error}"
                            else:
                                run.update(state="unknown", blocking_reason=f"Dispatch acknowledgement uncertain: {error}")
                                uncertain_gpus.add((host_id, key))
                                _grid_event(grid, "launch_uncertain", f"{run['run_id'][:8]} {host_id} GPU {key}: {error}")
                            busy_gpus.add((host_id, key))
                        else:
                            run.update(state="blocked", blocking_reason=str(error))
                            run["next_retry_at"] = time.time() + 15
                            if "cuda" in str(error).lower() or "preflight unavailable" in str(error).lower():
                                failed_gpu_preflights[(host_id, key)] = str(error)
                                gpu_id = run["gpu"].get("gpu_id", f"{host_id}.gpu.{key}")
                                grid.setdefault("gpu_retry_at", {})[gpu_id] = time.time() + 15
                                place = {"host_id": host_id, "gpu": run["gpu"]}
                                for pending in grid["runs"]:
                                    if pending["state"] in {"queued", "blocked"} and self._compatible(grid, pending, place):
                                        pending.update(state="blocked", blocking_reason=str(error))
                            if grid.get("queue_mode") == "dynamic":
                                try:
                                    self._controller_call(state, host_id, "runner_release", {"grid_id": grid["grid_id"], "gpu_key": key})
                                    snapshot.get("reservations", {}).pop(key, None)
                                except (network.NetworkError, RuntimeError, OSError) as release_error:
                                    run["blocking_reason"] += f"; release pending: {release_error}"
                self._release_dynamic_gpus(state, grid, snapshots)
                for run in grid["runs"]:
                    if run["state"] in TERMINAL and not run.get("released"):
                        if grid.get("queue_mode") == "dynamic":
                            continue
                        if not any(other["host_id"] == run["host_id"] and other["gpu"]["gpu_key"] == run["gpu"]["gpu_key"]
                                   and other["state"] not in TERMINAL for other in grid["runs"]):
                            try:
                                snapshot = snapshots.get(run["host_id"], {})
                                if "error" in snapshot or "reservations" not in snapshot:
                                    continue
                                owner = snapshot["reservations"].get(run["gpu"]["gpu_key"])
                                owner_id = owner.get("grid_id") if isinstance(owner, dict) else owner
                                if owner_id == grid["grid_id"]:
                                    released = self._controller_call(state, run["host_id"], "runner_release", {
                                        "grid_id": grid["grid_id"], "gpu_key": run["gpu"]["gpu_key"]})
                                    _grid_event(grid, "power", _power_detail(run, released, "restoration"))
                                    snapshot["reservations"].pop(run["gpu"]["gpu_key"], None)
                                for other in grid["runs"]:
                                    if (other["state"] in TERMINAL and other["host_id"] == run["host_id"] and
                                            other["gpu"]["gpu_key"] == run["gpu"]["gpu_key"]):
                                        other["released"] = True
                            except (network.NetworkError, RuntimeError, OSError) as release_error:
                                if run.get("last_release_error") != str(release_error):
                                    run["last_release_error"] = str(release_error)
                                    _grid_event(grid, "power_error", f"{run['run_id'][:8]} {run['host_label']} "
                                                f"GPU {run['gpu']['ordinal']} restoration/release: {release_error}")
                if grid.get("flush_requested"):
                    if all(run["state"] in TERMINAL and run.get("released") for run in grid["runs"]):
                        grid["state"] = "cancelled"
                        if not grid.get("flushed_at"):
                            grid["flushed_at"] = now()
                            _grid_event(grid, "flush", "Attempts stopped and GPU reservations released")
                    else:
                        grid["state"] = "flushing"
                elif all(run["state"] in TERMINAL for run in grid["runs"]):
                    grid["state"] = "failed" if grid.get("user_failed") or any(run["state"] == "failed" for run in grid["runs"]) else (
                        "cancelled" if any(run["state"] == "cancelled" for run in grid["runs"]) else "completed")
                elif any(run["state"] == "unknown" for run in grid["runs"]):
                    grid["state"] = "blocked"
                elif any(run.get("stop_requested") and run["state"] == "running" for run in grid["runs"]):
                    grid["state"] = "stopping"
                else:
                    grid["state"] = "running" if any(run["state"] == "running" for run in grid["runs"]) else "queued"
                for run in grid["runs"]:
                    old_state, old_reason, old_attempts = previous_runs[run["run_id"]]
                    reason = run.get("blocking_reason", "")
                    if (run["state"], reason, len(run["attempts"])) != (old_state, old_reason, old_attempts):
                        last = run["attempts"][-1] if run["attempts"] else {}
                        detail = (f"{run['run_id'][:8]} {run['host_label']} GPU {run['gpu']['ordinal']} "
                                  f"{old_state} -> {run['state']}; attempts={len(run['attempts'])}; "
                                  f"exit={last.get('exit_code', 'pending')}; {reason[:240]}")
                        _grid_event(grid, "run", detail)
                if grid["state"] in TERMINAL:
                    grid.setdefault("finished_at", now())  # <<< THOG freeze elapsed time across reloads and restarts
                if grid["state"] != previous_state:
                    _grid_event(grid, "grid", f"{previous_state} -> {grid['state']}")
                directory = GRID_SCRIPTS / grid["grid_tag"]
                changed |= json.dumps(grid, sort_keys=True) != previous_grid
                status_path = directory / "status.json"
                status = {"state": grid["state"], "runs": [{"run_id": r["run_id"],
                          "state": r["state"], "attempts": r["attempts"]} for r in grid["runs"]]}
                try:
                    old_status = json.loads(status_path.read_text())
                except (OSError, ValueError):
                    old_status = None
                if old_status != status:
                    _write_grid_file(status_path, status)
                if json.dumps(grid, sort_keys=True) != previous_grid:
                    _write_grid_file(directory / "manifest.json", grid)
            for grid in stranded:
                before_release = json.dumps(grid, sort_keys=True)
                self._release_dynamic_gpus(state, grid, snapshots)
                changed |= before_release != json.dumps(grid, sort_keys=True)
                for run in grid["runs"]:
                    if run.get("released") or run["state"] not in TERMINAL:
                        continue
                    snapshot = snapshots.get(run["host_id"], {})
                    if "error" in snapshot or "reservations" not in snapshot:
                        continue
                    key = run["gpu"]["gpu_key"]
                    owner = snapshot["reservations"].get(key)
                    owner_id = owner.get("grid_id") if isinstance(owner, dict) else owner
                    if owner_id == grid["grid_id"]:
                        try:
                            released = self._controller_call(state, run["host_id"], "runner_release", {"grid_id": grid["grid_id"],
                                                                                                     "gpu_key": key})
                            _grid_event(grid, "power", _power_detail(run, released, "restoration"))
                        except (network.NetworkError, RuntimeError, OSError) as release_error:
                            self._release_retry_at[run["host_id"]] = time.time() + 30
                            if run.get("last_release_error") != str(release_error):
                                run["last_release_error"] = str(release_error)
                                changed = True
                                _grid_event(grid, "power_error", f"{run['run_id'][:8]} {run['host_label']} "
                                            f"GPU {run['gpu']['ordinal']} restoration/release: {release_error}")
                            continue
                        snapshot["reservations"].pop(key, None)
                    for other in grid["runs"]:
                        if other["host_id"] == run["host_id"] and other["gpu"]["gpu_key"] == key:
                            other["released"] = True
                    changed = True
                    _grid_event(grid, "reservation", f"Recovered release of {run['host_label']} GPU {run['gpu']['ordinal']}")
            if changed:
                _write(state)
            activity = (any(g["state"] not in TERMINAL for g in state["grids"]),
                        any(r["state"] in {"queued", "blocked", "unknown"} for g in active for r in g["runs"]))
            if activity != self._last_activity or (not any(activity) and self.network.list_hosts().get("release_pending")):
                self.lock.release()
                try:
                    self.network.set_grid_activity(*activity)
                finally:
                    self.lock.acquire()
                self._last_activity = activity if _read() == state else None

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._refresh()
            except ControllerStateChanged:
                continue
            except Exception as error:
                network.log_event("Runner", "reconcile", self.network.local_id, "error", type(error).__name__)
            self.stop_event.wait(3)
# ^^^ THOG
