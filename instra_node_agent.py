# vvv THOG keep host discovery and permitted operations in a separate, same-user local process
"""Instra node agent: local Unix socket, with an SSH-invocable request client."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("INSTRA_STATE_DIR", Path.home() / ".local/state/instra"))
SOCKET_PATH = STATE_DIR / "node.sock"
AGENT_STATE = STATE_DIR / "node.json"
BOOTSTRAP_DIR = Path.home() / ".local/state/instra"
AGENT_ENTRY = BOOTSTRAP_DIR / "agent-request"
MAX_MESSAGE = 1024 * 1024
_lock = threading.RLock()


def _read_state():
    try:
        return json.loads(AGENT_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _write_state(state):
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = AGENT_STATE.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, AGENT_STATE)


def _running(pid):
    if not isinstance(pid, int) or pid < 2:
        return False
    try:
        os.kill(pid, 0)
        try:
            if Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] == "Z":
                return False
        except (OSError, IndexError):
            pass
        return True
    except (OSError, ValueError):
        return False


def _pid_start_time(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _gpu_information():
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,memory.free,compute_cap,power.limit", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 7)]
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        ordinal, uuid, model, memory, driver, free, compute_cap, power = fields
        gpus.append({"gpu_key": uuid if uuid.startswith("GPU-") else f"ordinal-{ordinal}", "uuid": uuid if uuid.startswith("GPU-") else None,
                     "ordinal": int(ordinal), "model": model, "memory_mib": int(memory) if memory.isdigit() else None,
                     "free_mib": int(free) if free.isdigit() else None, "compute_cap": compute_cap,
                     "power_cap_w": float(power) if re.fullmatch(r"\d+(?:\.\d+)?", power) else None, "driver": driver})
    try:
        occupied = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=8, check=True)
        for line in occupied.stdout.splitlines():
            pid, _, uuid = line.partition(",")
            if pid.strip().isdigit():
                for gpu in gpus:
                    if gpu["uuid"] == uuid.strip():
                        gpu.setdefault("compute_pids", []).append(int(pid.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    for gpu in gpus:
        gpu.setdefault("compute_pids", [])
    return gpus


def _cuda_version():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=8, check=True)
        match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
        return match.group(1) if match else None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(path):
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=4, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _discover(state):
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except OSError:
        ip_address = None
    logs_root = Path(state.get("logs_root") or os.environ.get("INSTRA_LOGS_ROOT", ROOT / "logs")).expanduser().resolve()
    wandb_root = Path(os.environ.get("WANDB_DIR", ROOT / "wandb")).expanduser().resolve()
    profile = {"profile_key": "current", "python": sys.executable, "location": str(Path(sys.executable).resolve().parent),
               "thog_entry": str(ROOT / "run_thog2_owt.py"), "shell_entry": str(ROOT / "train_OWT.sh")}
    return {"hostname": hostname, "resolved_ip": ip_address, "os": sys.platform, "observed_at": datetime.now(timezone.utc).isoformat(),
            "instra": {"root": str(ROOT), "version": _version(ROOT), "launch_command": state.get("launch_command"),
                       "running": _running(state.get("backend_pid")), "pid": state.get("backend_pid")},
            "thog": {"root": str(ROOT), "version": _version(ROOT), "train_OWT_sh": (ROOT / "train_OWT.sh").is_file(),
                     "run_thog2_owt_py": (ROOT / "run_thog2_owt.py").is_file()},
            "instra_logs_root": str(logs_root), "wandb_root": str(wandb_root), "execution_profiles": [profile],
            "cuda_version": _cuda_version(), "gpus": _gpu_information()}


def _validate_args(args, fields):
    if not isinstance(args, dict) or set(args) - fields:
        raise ValueError("invalid operation arguments")


# vvv THOG durable GPU reservations and attempt reconciliation owned by the local Node Agent
def _attempt_status(run):
    result = dict(run)
    exit_file = STATE_DIR / f"attempt-{run['attempt_id']}.exit"
    if exit_file.exists():
        try:
            result["exit_code"] = int(exit_file.read_text().strip())
            result["finished_at"] = datetime.fromtimestamp(exit_file.stat().st_mtime, timezone.utc).isoformat()
        except (OSError, ValueError):
            result["exit_code"] = None
        result["state"] = "completed" if result["exit_code"] == 0 else "failed"
    elif _running(run.get("pid")) and run.get("pid_start_time") == _pid_start_time(run["pid"]):
        result["state"] = "running"
    elif run.get("stop_requested") and isinstance(run.get("pid"), int):
        try:
            os.killpg(run["pid"], 0)
            result["state"] = "unknown"  # The supervisor exited, but a child might still be alive.
        except ProcessLookupError:
            result["state"] = "cancelled"
        except PermissionError:
            result["state"] = "unknown"
    else:
        result["state"] = "unknown"
    return result


def _known_thog_compute_pids(gpu):
    found = []
    for pid in gpu.get("compute_pids", []):
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            if b"run_thog2_owt" in command or b"train_OWT" in command:
                found.append(pid)
        except OSError:
            continue
    return found


def _check_torch_cuda(gpu, environment):
    """Check CUDA in a fresh process with the same GPU visibility as training."""
    try:
        checked = subprocess.run(
            [sys.executable, "-c", "import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))"],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"PyTorch CUDA preflight unavailable on GPU {gpu['ordinal']}: {error}") from error
    if checked.returncode:
        detail = (checked.stderr or checked.stdout).strip()[-900:] or f"exit {checked.returncode}"
        raise RuntimeError(
            f"PyTorch cannot initialize CUDA on GPU {gpu['ordinal']} ({gpu['gpu_key']}); "
            f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}: {detail}"
        )


def _runner_operation(state, name, args):
    from thog_grid_runner import command_for, environment_for, validate_recipe
    if name == "runner_capabilities":
        _validate_args(args, set())
        return {"protocol": 2, "cuda_preflight": True}
    if name == "runner_log":
        _validate_args(args, {"attempt_id", "max_bytes"})
        attempt_id = args.get("attempt_id")
        if attempt_id not in state.get("attempts", {}):
            raise KeyError("Unknown Runner attempt")
        max_bytes = args.get("max_bytes", 8192)
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16384:
            raise ValueError("Invalid log excerpt size")
        path = STATE_DIR / f"attempt-{attempt_id}.log"
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - max_bytes))
                excerpt = stream.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            excerpt = "No training output has been written yet."
        return {"attempt_id": attempt_id, "text": excerpt, "log_path": str(path)}
    if name == "runner_preflight":
        _validate_args(args, {"run", "gpu_key", "host_label"})
        run = args.get("run")
        if not isinstance(run, dict) or set(run) != {"run_id", "grid_tag", "pairing_id", "profiler", "parameters", "dtype", "attention_backend", "gpu_uuid"}:
            raise ValueError("Invalid resolved run metadata")
        validate_recipe({"label": "resolved execution", "parameters": run["parameters"], "profilers": [run["profiler"]]})
        gpu = next((item for item in _gpu_information() if item["gpu_key"] == args.get("gpu_key")), None)
        if gpu is None or gpu["uuid"] != run["gpu_uuid"]:
            raise RuntimeError("Resolved GPU unavailable or changed")
        if run["dtype"] not in {"float16", "bfloat16"} or run["attention_backend"] != "sdpa":
            raise ValueError("Unsupported resolved dtype or attention backend")
        if run["dtype"] == "bfloat16" and int(str(gpu.get("compute_cap", "0")).split(".")[0]) < 8:
            raise ValueError("Selected GPU does not support bfloat16")
        environment = os.environ.copy()
        environment.update(environment_for(run))
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu["ordinal"])
        command = command_for(run, gpu, python=sys.executable, host_label=args.get("host_label"))
        try:
            checked = subprocess.run(command + ["--print-resolved-json"], cwd=ROOT, env=environment,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"THOG parameter preflight unavailable: {type(error).__name__}") from error
        if checked.returncode:
            raise ValueError(f"THOG parameter preflight rejected this run: {checked.stderr.strip()[-450:]}")
        _check_torch_cuda(gpu, environment)
        return {"resolved": True, "gpu_uuid": gpu["uuid"]}
    if name == "runner_reconcile":
        _validate_args(args, set())
        return {"attempts": {key: _attempt_status(value) for key, value in state.get("attempts", {}).items()},
                "reservations": state.get("reservations", {}), "gpus": _gpu_information()}
    if name == "runner_reserve":
        _validate_args(args, {"grid_id", "gpu_key", "required_mib", "headroom_mib", "power_cap_w"})
        grid_id, key = args.get("grid_id"), args.get("gpu_key")
        if not isinstance(grid_id, str) or not re.fullmatch(r"[a-f0-9]{32}", grid_id) or not isinstance(key, str):
            raise ValueError("Invalid reservation identity")
        required, headroom = args.get("required_mib"), args.get("headroom_mib")
        if any(type(v) is not int or not 0 <= v <= 131072 for v in (required, headroom)):
            raise ValueError("Invalid required memory or headroom")
        cap = args.get("power_cap_w")
        if cap is not None and (type(cap) is not int or not 50 <= cap <= 600):
            raise ValueError("Invalid GPU power cap")
        gpu = next((gpu for gpu in _gpu_information() if gpu["gpu_key"] == key), None)
        if gpu is None:
            raise RuntimeError("Selected GPU unavailable")
        owner = state.setdefault("reservations", {}).get(key)
        if owner and owner["grid_id"] != grid_id:
            raise RuntimeError(f"GPU reserved by {owner['grid_id']}")
        if gpu["free_mib"] is None or gpu["free_mib"] < required + headroom:
            raise RuntimeError(f"GPU {key} free {gpu['free_mib']} MiB; needs {required}+{headroom} MiB")
        if _known_thog_compute_pids(gpu) and not any(run["gpu_key"] == key and _attempt_status(run)["state"] == "running"
                                                 and run["grid_id"] == grid_id for run in state.get("attempts", {}).values()):
            raise RuntimeError(f"GPU {key} has a THOG training process outside this Grid")
        state["reservations"][key] = {"grid_id": grid_id, "gpu_key": key, "required_mib": required,
                                        "headroom_mib": headroom, "ordinal": gpu["ordinal"], "requested_power_w": cap}
        _write_state(state)
        return {"reservation": state["reservations"][key], "gpu": gpu}
    if name == "runner_release":
        _validate_args(args, {"grid_id", "gpu_key"})
        owner = state.get("reservations", {}).get(args.get("gpu_key"))
        if owner and owner["grid_id"] != args.get("grid_id"):
            raise PermissionError("Reservation belongs to a different Grid")
        if any(run["grid_id"] == args.get("grid_id") and run["gpu_key"] == args.get("gpu_key")
               and _attempt_status(run)["state"] in {"running", "unknown"} for run in state.get("attempts", {}).values()):
            raise RuntimeError("Running or uncertain attempt still owns this GPU")
        state.get("reservations", {}).pop(args.get("gpu_key"), None)
        _write_state(state)
        return {"released": True}
    if name == "runner_launch":
        _validate_args(args, {"grid_id", "attempt_id", "run", "gpu_key", "host_label", "thog_host_id", "execution_profile", "recipe_id"})
        attempt_id = args.get("attempt_id")
        if not isinstance(attempt_id, str) or not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
            raise ValueError("Invalid attempt identity")
        old = state.setdefault("attempts", {}).get(attempt_id)
        if old:
            return _attempt_status(old)
        key, grid_id, run = args.get("gpu_key"), args.get("grid_id"), args.get("run")
        reservation = state.get("reservations", {}).get(key)
        if reservation is None or reservation["grid_id"] != grid_id:
            raise PermissionError("Run requires an owned GPU reservation")
        if not isinstance(run, dict) or set(run) != {"run_id", "grid_tag", "pairing_id", "profiler", "parameters", "dtype", "attention_backend", "gpu_uuid"}:
            raise ValueError("Invalid resolved run metadata")
        if not isinstance(run["run_id"], str) or not re.fullmatch(r"[a-f0-9]{32}", run["run_id"]):
            raise ValueError("Invalid run ID")
        validate_recipe({"label": "resolved execution", "parameters": run["parameters"], "profilers": [run["profiler"]]})
        gpu = next((gpu for gpu in _gpu_information() if gpu["gpu_key"] == key), None)
        if gpu is None or gpu["free_mib"] is None or gpu["free_mib"] < reservation["required_mib"] + reservation["headroom_mib"]:
            raise RuntimeError("GPU memory no longer sufficient")
        if _known_thog_compute_pids(gpu):
            raise RuntimeError("Selected GPU has a THOG training process")
        if gpu["uuid"] != run["gpu_uuid"] or run["attention_backend"] != "sdpa" or run["dtype"] not in {"float16", "bfloat16"}:
            raise ValueError("Resolved GPU, dtype or backend changed since preflight")
        if run["dtype"] == "bfloat16" and int(str(gpu.get("compute_cap", "0")).split(".")[0]) < 8:
            raise ValueError("Selected GPU does not support bfloat16")
        if any(other["gpu_key"] == key and _attempt_status(other)["state"] in {"running", "unknown"}
               for other in state["attempts"].values()):
            raise RuntimeError("A managed attempt is already on this GPU")
        requested_power = reservation.get("requested_power_w")
        if requested_power is not None:
            try:
                subprocess.run(["nvidia-smi", "-i", str(gpu["ordinal"]), "-pl", str(requested_power)],
                               check=True, capture_output=True, timeout=8)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(f"GPU {key} power cap {requested_power} W could not be applied") from error
            gpu = next((item for item in _gpu_information() if item["gpu_key"] == key), gpu)
        command = command_for(run, gpu, python=sys.executable, host_label=args.get("host_label"))
        log_path = STATE_DIR / f"attempt-{attempt_id}.log"
        metadata_path = STATE_DIR / f"attempt-{attempt_id}.json"
        metadata_path.write_text(json.dumps({"argv": command, "cwd": str(ROOT), "log_path": str(log_path),
                                             "exit_path": str(STATE_DIR / f"attempt-{attempt_id}.exit"),
                                             "runner_metadata": {"grid_id": grid_id, "grid_tag": run["grid_tag"],
                                                                  "recipe_id": args.get("recipe_id"),
                                                                  "run_id": run["run_id"], "pairing_id": run["pairing_id"],
                                                                  "attempt_id": attempt_id, "profiler": run["profiler"],
                                                                  "thog_host_id": args.get("thog_host_id"),
                                                                  "execution_profile": args.get("execution_profile"),
                                                                  "gpu_uuid": gpu["uuid"]},
                                             "gpu_ordinal": gpu["ordinal"], "runner_environment": environment_for(run)}))
        os.chmod(metadata_path, 0o600)
        state["attempts"][attempt_id] = {"attempt_id": attempt_id, "run_id": run["run_id"], "grid_id": grid_id,
                                         "gpu_key": key, "pid": None, "started_at": datetime.now(timezone.utc).isoformat(),
                                         "pid_start_time": None,
                                         "requested_power_w": requested_power, "observed_power_w": gpu.get("power_cap_w"),
                                         "log_path": str(log_path), "state": "accepted"}
        _write_state(state)  # Record acceptance before spawning; a crash in this window blocks relaunch.
        try:
            process = subprocess.Popen([sys.executable, str(ROOT / "instra_runner_child.py"), str(metadata_path)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            state["attempts"][attempt_id]["state"] = "failed"
            (STATE_DIR / f"attempt-{attempt_id}.exit").write_text("127\n")
            _write_state(state)
            raise RuntimeError(f"Runner child unavailable: {error}") from error
        state["attempts"][attempt_id].update(pid=process.pid, pid_start_time=_pid_start_time(process.pid), state="running")
        _write_state(state)
        return _attempt_status(state["attempts"][attempt_id])
    if name in {"runner_status", "runner_stop"}:
        _validate_args(args, {"attempt_id", "grid_id", "grace_seconds"} if name == "runner_stop" else {"attempt_id"})
        run = state.get("attempts", {}).get(args.get("attempt_id"))
        if run is None:
            raise KeyError("Unknown attempt")
        if name == "runner_stop":
            if run["grid_id"] != args.get("grid_id"):
                raise PermissionError("Attempt belongs to another Grid")
            grace = args.get("grace_seconds", 10)
            if type(grace) is not int or not 0 <= grace <= 300:
                raise ValueError("Invalid stop grace period")
            observed_state = _attempt_status(run)["state"]
            if observed_state in {"running", "unknown"}:
                run["stop_requested"] = True
                _write_state(state)
            if observed_state == "running":
                try:
                    os.killpg(run["pid"], signal.SIGKILL if grace == 0 else signal.SIGINT)
                except ProcessLookupError:
                    pass
                def force_if_still_running():
                    if _attempt_status(run)["state"] == "running":
                        try:
                            os.killpg(run["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if grace == 0:
                    force_if_still_running()
                else:
                    timer = threading.Timer(grace, force_if_still_running)
                    timer.daemon = True
                    timer.start()
        return _attempt_status(run)
    raise ValueError("Unknown Runner operation")
# ^^^ THOG


def _operation(name, args):
    # Parameter/CUDA preflight may spend twenty seconds in a child process.
    # It is read-only: never serialize discovery, status or stop behind it.
    if name == "runner_preflight":
        return _runner_operation(_read_state(), name, args)
    with _lock:
        state = _read_state()
        if name.startswith("runner_"):
            return _runner_operation(state, name, args)                                                                                                      # <<< THOG route only named Runner operations through the Node Agent
        if name == "discover":
            _validate_args(args, set())
            return _discover(state)
        if name == "state":
            _validate_args(args, set())
            runs = state.get("runs", {})
            return {"instra_running": _running(state.get("backend_pid")), "instra_pid": state.get("backend_pid"),
                    "intentional_stop": bool(state.get("intentional_stop")), "master_id": state.get("master_id"),
                    "gpus": _gpu_information(), "last_auto_restart": state.get("last_auto_restart"),
                    "runs": {key: {**run, "running": _running(run.get("pid"))} for key, run in runs.items()}}
        if name == "configure":
            _validate_args(args, {"launch_command", "logs_root", "backend_pid", "auto_restart"})
            # vvv THOG reject a second live backend before it overwrites the PID that owns this node agent
            if "backend_pid" in args and state.get("backend_pid") != args["backend_pid"] and _running(state.get("backend_pid")):
                raise RuntimeError("Instra backend is already running; use Restart Instra")
            # ^^^ THOG
            command = args.get("launch_command")
            if command is not None:
                if not isinstance(command, list) or len(command) < 2 or any(not isinstance(v, str) for v in command):
                    raise ValueError("launch_command must be a list of strings")
                if Path(command[0]).resolve() != Path(sys.executable).resolve() or Path(command[1]).resolve() != ROOT / "run_thog2_dashboard.py":
                    raise ValueError("only the local Instra launcher is permitted")
                state["launch_command"] = command
            if "logs_root" in args:
                state["logs_root"] = str(Path(args["logs_root"]).expanduser().resolve())
            if "backend_pid" in args:
                if args["backend_pid"] != os.getppid() and not _running(args["backend_pid"]):
                    raise ValueError("backend process is unavailable")
                state["backend_pid"] = args["backend_pid"]
                state["intentional_stop"] = False
            if "auto_restart" in args:
                if not isinstance(args["auto_restart"], bool):
                    raise ValueError("auto_restart must be Boolean")
                state["auto_restart"] = args["auto_restart"]
            _write_state(state)
            return {"configured": True}
        if name == "backend_exited":
            _validate_args(args, {"backend_pid"})
            if args.get("backend_pid") == state.get("backend_pid"):
                state["intentional_stop"] = True
                _write_state(state)
            return {"recorded": True}
        if name == "stop_instra":
            _validate_args(args, set())
            state["intentional_stop"] = True
            _write_state(state)
            if _running(state.get("backend_pid")):
                os.kill(state["backend_pid"], signal.SIGTERM)
            return {"stopped": True}
        if name in {"start_instra", "restart_instra"}:
            _validate_args(args, set())
            command = state.get("launch_command")
            if not command:
                raise ValueError("Instra launch command has not been configured")
            if _running(state.get("backend_pid")):
                if name == "start_instra":
                    return {"pid": state["backend_pid"], "already_running": True}
                state["intentional_stop"] = True
                _write_state(state)
                os.kill(state["backend_pid"], signal.SIGTERM)
                for _ in range(30):
                    if not _running(state["backend_pid"]):
                        break
                    time.sleep(0.1)
                if _running(state["backend_pid"]):
                    raise RuntimeError("Instra did not stop; restart refused")
            with (STATE_DIR / "backend.log").open("ab") as output:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            state.update(backend_pid=process.pid, intentional_stop=False)
            _write_state(state)
            return {"pid": process.pid}
        if name == "claim_master":
            _validate_args(args, {"master_id"})
            owner = args.get("master_id")
            if not isinstance(owner, str) or not re.fullmatch(r"thog_host\.[a-z0-9][a-z0-9_-]{0,62}", owner):
                raise ValueError("invalid Runner Master identity")
            if state.get("master_id") not in (None, owner):
                raise PermissionError("a different Runner Master is already designated")
            state["master_id"] = owner
            _write_state(state)
            return {"master_id": owner}
        if name == "release_master":
            _validate_args(args, {"master_id"})
            if not isinstance(args.get("master_id"), str) or not re.fullmatch(r"thog_host\.[a-z0-9][a-z0-9_-]{0,62}", args["master_id"]):
                raise ValueError("invalid Runner Master identity")
            if state.get("master_id") is None:
                return {"released": True, "already_released": True}
            if state.get("master_id") != args.get("master_id"):
                raise PermissionError("Runner Master identity mismatch")
            state.pop("master_id", None)
            _write_state(state)
            return {"released": True}
        if name == "launch_run":
            _validate_args(args, {"master_id", "profile_key", "argv", "run_id"})
            if state.get("master_id") != args.get("master_id"):
                raise PermissionError("Runner Master authority required")
            if args.get("profile_key") != "current" or not isinstance(args.get("argv"), list) or len(args["argv"]) > 256 or any(not isinstance(v, str) or len(v) > 4096 or "\0" in v for v in args["argv"]):
                raise ValueError("invalid resolved run")
            run_id = args.get("run_id")
            if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id) or run_id in state.get("runs", {}):
                raise ValueError("invalid or duplicate run ID")
            current_gpus = _gpu_information()                                                                                                                # <<< THOG recheck GPU processes immediately before accepting a resolved run
            if not current_gpus:
                raise RuntimeError("No CUDA GPUs are currently available")
            with (STATE_DIR / f"run-{run_id}.log").open("ab") as output:
                process = subprocess.Popen([str(ROOT / "train_OWT.sh"), *args["argv"]], cwd=ROOT,
                                           stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            state.setdefault("runs", {})[run_id] = {"pid": process.pid, "started_at": datetime.now(timezone.utc).isoformat()}
            _write_state(state)
            return {"run_id": run_id, "pid": process.pid}
        if name in {"report_run", "cancel_run"}:
            _validate_args(args, {"master_id", "run_id"})
            run = state.get("runs", {}).get(args.get("run_id"))
            if run is None:
                raise KeyError("unknown managed run")
            if name == "cancel_run":
                if state.get("master_id") != args.get("master_id"):
                    raise PermissionError("Runner Master authority required")
                if _running(run["pid"]):
                    os.killpg(run["pid"], signal.SIGTERM)
            return {**run, "running": _running(run["pid"])}
        raise ValueError("unknown operation")


def request(name, args=None, timeout=30):
    payload = json.dumps({"operation": name, "args": args or {}}).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("request too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(SOCKET_PATH))
        connection.sendall(payload + b"\n")
        with connection.makefile("rb") as stream:
            response = stream.readline(MAX_MESSAGE + 1)
    if not response or len(response) > MAX_MESSAGE:
        raise RuntimeError("agent response missing or oversized")
    result = json.loads(response)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "agent operation failed"))
    return result["result"]


def _serve_connection(connection):
    with connection:
        try:
            connection.settimeout(30)                                                                                                                       # <<< THOG bound abandoned socket readers so long-lived agents do not accumulate blocked threads
            with connection.makefile("rb") as stream:
                raw = stream.readline(MAX_MESSAGE + 1)
            if len(raw) > MAX_MESSAGE or not raw.endswith(b"\n"):
                raise ValueError("invalid request size")
            payload = json.loads(raw)
            result = _operation(payload["operation"], payload["args"])
            response = {"ok": True, "result": result}
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, PermissionError) as error:
            response = {"ok": False, "error": str(error)}
        connection.sendall(json.dumps(response).encode() + b"\n")


def _watch_backend():
    while True:
        time.sleep(3)
        state = _read_state()
        if not state.get("auto_restart") or state.get("intentional_stop") or not state.get("launch_command"):
            continue
        if _running(state.get("backend_pid")):
            continue
        try:
            _operation("start_instra", {})
            with _lock:
                state = _read_state()
                state["last_auto_restart"] = {"time": datetime.now(timezone.utc).isoformat(), "outcome": "success"}
                _write_state(state)
            with (STATE_DIR / "agent.log").open("a") as output:
                output.write(f"{datetime.now(timezone.utc).isoformat()} automatic Instra restart succeeded\n")
        except (OSError, ValueError, RuntimeError) as error:
            with _lock:
                state = _read_state()
                state["last_auto_restart"] = {"time": datetime.now(timezone.utc).isoformat(), "outcome": "failure"}
                _write_state(state)
            with (STATE_DIR / "agent.log").open("a") as output:
                output.write(f"{datetime.now(timezone.utc).isoformat()} automatic Instra restart failed: {type(error).__name__}\n")


def _install_agent_entry():
    # SSH needs a stable entry point even when THOG lives outside ~/git/thog2.
    BOOTSTRAP_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(BOOTSTRAP_DIR, 0o700)
    entry = AGENT_ENTRY.with_name(f".{AGENT_ENTRY.name}.{os.getpid()}.tmp")
    try:
        entry.write_text("#!/bin/sh\n" +
                         f"INSTRA_STATE_DIR={shlex.quote(str(STATE_DIR))} "
                         f"exec {shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} request\n")
        entry.chmod(0o700)
        os.replace(entry, AGENT_ENTRY)
    finally:
        entry.unlink(missing_ok=True)


def serve():
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    if SOCKET_PATH.exists():
        try:
            request("state", timeout=1)
        except (OSError, ConnectionError):
            SOCKET_PATH.unlink()
        else:
            _install_agent_entry()                                                                                                                           # <<< THOG refresh the SSH entry when an existing agent survives a code update
            return
    _install_agent_entry()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o600)
        listener.listen(16)
        threading.Thread(target=_watch_backend, name="instra-backend-watch", daemon=True).start()
        try:
            while True:
                connection, _ = listener.accept()
                threading.Thread(target=_serve_connection, args=(connection,), daemon=True).start()
        finally:
            SOCKET_PATH.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Instra local node agent")
    parser.add_argument("mode", choices=("serve", "request"))
    args = parser.parse_args(argv)
    if args.mode == "serve":
        serve()
        return 0
    try:
        payload = json.load(sys.stdin)
        operation = payload["operation"]
        result = request(operation, payload.get("args", {}), timeout=75 if operation == "runner_preflight" else 30)
        print(json.dumps({"ok": True, "result": result}))
        return 0
    # vvv THOG report socket failures separately so the Network view can distinguish an absent agent
    except OSError as error:
        print(json.dumps({"ok": False, "category": "agent availability", "error": str(error)}))
        return 1
    except (ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"ok": False, "category": "operation", "error": str(error)}))
        return 1
    # ^^^ THOG


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
