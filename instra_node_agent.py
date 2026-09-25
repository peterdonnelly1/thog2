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


def _gpu_information():
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5 or not fields[0].isdigit():
            continue
        ordinal, uuid, model, memory, driver = fields
        gpus.append({"gpu_key": uuid if uuid.startswith("GPU-") else f"ordinal-{ordinal}", "uuid": uuid if uuid.startswith("GPU-") else None,
                     "ordinal": int(ordinal), "model": model, "memory_mib": int(memory) if memory.isdigit() else None, "driver": driver})
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


def _operation(name, args):
    with _lock:
        state = _read_state()
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
        result = request(payload["operation"], payload.get("args", {}))
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
