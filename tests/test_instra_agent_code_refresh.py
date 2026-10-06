"""Refresh cached Runner code without stopping real child jobs or losing state.

Process fixtures use loopback TCP because some test sandboxes disable AF_UNIX
and expose /proc outside their PID namespace. Production request and startup
logic still run; credentials and /proc identity are supplied by the fixture.
Separate refusal tests cover unverified process identities.
"""
import ast
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import instra_node_agent as agent


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("instra_node_agent.py", "thog_grid_runner.py", "instra_runner_catalogue.json", "instra_grid_identity.py")
ATTEMPT = "a" * 32


def launcher(parser, subprocess_module=subprocess, socket_module=socket, path_constructor=Path):
    tree = ast.parse((ROOT / "run_thog2_dashboard.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_start_node_agent")
    namespace = {"_dashboard": SimpleNamespace(_base=SimpleNamespace(build_parser=lambda: parser)),
                 "os": os, "Path": path_constructor, "subprocess": subprocess_module, "sys": sys, "time": time,
                 "socket": socket_module, "struct": struct, "signal": signal, "errno": errno,
                 "__file__": str(Path(agent.__file__).with_name("run_thog2_dashboard.py"))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "run_thog2_dashboard.py", "exec"), namespace)
    return namespace["_start_node_agent"]


def test_fingerprint_changes_for_each_runner_dependency(tmp_path):
    for name in SOURCES:
        shutil.copyfile(ROOT / name, tmp_path / name)
    original = agent._runner_runtime_fingerprint(tmp_path)
    for name in SOURCES:
        path = tmp_path / name
        raw = path.read_bytes()
        path.write_bytes(raw + b"\n")
        assert agent._runner_runtime_fingerprint(tmp_path) != original
        path.write_bytes(raw)
        assert agent._runner_runtime_fingerprint(tmp_path) == original


def test_capability_fingerprint_is_frozen_when_files_change(monkeypatch):
    monkeypatch.setattr(agent, "_runner_runtime_fingerprint", lambda: "updated-on-disk")
    monkeypatch.setattr(agent, "_power_capability", lambda: {"ready": False})
    assert agent._runner_operation({}, "runner_capabilities", {})["runner_runtime_fingerprint"] == agent._LOADED_RUNNER_FINGERPRINT


@pytest.mark.parametrize("old_capability", ["missing", "stale"])
def test_restart_replaces_real_stale_agent_and_preserves_its_child(tmp_path, monkeypatch, old_capability):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in SOURCES:
        shutil.copyfile(ROOT / name, checkout / name)
    # Keep all processes and SSH bootstrap files inside this isolated fixture.
    # Only the initial daemon simulates a cached pre-WIDTH Runner catalogue.
    injection = '''
BOOTSTRAP_DIR = STATE_DIR / "bootstrap"
AGENT_ENTRY = BOOTSTRAP_DIR / "agent-request"
_gpu_information = lambda: []
def serve():
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _install_agent_entry()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        (STATE_DIR / "endpoint.json").write_text(json.dumps({
            "address": listener.getsockname(), "pid": os.getpid(),
            "uid": os.getuid(), "gid": os.getgid()}))
        while True:
            connection, _ = listener.accept()
            threading.Thread(target=_serve_connection, args=(connection,), daemon=True).start()
if os.environ.get("INSTRA_TEST_OLD_CAPABILITY"):
    from thog_grid_runner import CATALOGUE
    CATALOGUE.pop("--select-width", None)
    _current_runner_operation = _runner_operation
    def _runner_operation(state, name, args):
        result = _current_runner_operation(state, name, args)
        if name == "runner_capabilities":
            if os.environ["INSTRA_TEST_OLD_CAPABILITY"] == "missing":
                result.pop("runner_runtime_fingerprint")
            else:
                result["runner_runtime_fingerprint"] = "0" * 64
        return result
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             start_new_session=True)
    _write_state({"attempts": {"a" * 32: {
        "attempt_id": "a" * 32,
        "pid": child.pid, "pid_start_time": _pid_start_time(child.pid),
        "exit_path": str(STATE_DIR / "child.exit"), "state": "running"}},
        "reservations": {"GPU-0": {"grid_id": "b" * 32}}})
'''
    path = checkout / "instra_node_agent.py"
    path.write_text(path.read_text().replace('\nif __name__ == "__main__":', injection + '\nif __name__ == "__main__":'))
    state_dir = tmp_path / "state"
    endpoint_path = state_dir / "endpoint.json"
    monkeypatch.setenv("INSTRA_STATE_DIR", str(state_dir))
    for name, value in (("STATE_DIR", state_dir), ("SOCKET_PATH", state_dir / "node.sock"),
                        ("AGENT_STATE", state_dir / "node.json"), ("BOOTSTRAP_DIR", state_dir / "bootstrap"),
                        ("AGENT_ENTRY", state_dir / "bootstrap" / "agent-request"),
                        ("__file__", str(path)), ("_LOADED_RUNNER_FINGERPRINT", agent._runner_runtime_fingerprint(checkout))):
        monkeypatch.setattr(agent, name, value)

    class LocalConnection:
        def __init__(self):
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        def __enter__(self): return self
        def __exit__(self, *_args): self.connection.close()
        def __getattr__(self, name): return getattr(self.connection, name)
        def connect(self, _unix_path):
            try:
                self.endpoint = json.loads(endpoint_path.read_text())
            except (FileNotFoundError, ValueError) as error:
                raise ConnectionRefusedError("fixture agent is not ready") from error
            self.connection.connect(tuple(self.endpoint["address"]))
        def getsockopt(self, *_args):
            return struct.pack("3i", *(self.endpoint[name] for name in ("pid", "uid", "gid")))
    transport = SimpleNamespace(**{name: getattr(socket, name) for name in
        ("AF_UNIX", "AF_INET", "SOCK_STREAM", "SOL_SOCKET", "SO_PEERCRED", "SO_REUSEADDR")})
    transport.socket = lambda family, kind: LocalConnection() if family == socket.AF_UNIX else socket.socket(family, kind)
    monkeypatch.setattr(agent, "socket", transport)

    def process_path(value):
        endpoint = json.loads(endpoint_path.read_text())
        if str(value) == f"/proc/{endpoint['pid']}/cmdline":
            return SimpleNamespace(read_bytes=lambda: str(path).encode() + b"\0serve\0")
        return Path(value)
    environment = dict(os.environ, INSTRA_TEST_OLD_CAPABILITY=old_capability)
    processes = []
    child_pid = None
    with (tmp_path / "agent-output.log").open("wb") as output:
        old = subprocess.Popen([sys.executable, str(path), "serve"], cwd=checkout, env=environment,
                               stdout=output, stderr=output)
        processes.append(old)
        running = agent._running
        # Reap the real old daemon when /proc cannot reveal its zombie state.
        monkeypatch.setattr(agent, "_running", lambda pid: old.poll() is None if pid == old.pid else running(pid))
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert old.poll() is None, (tmp_path / "agent-output.log").read_text()
                try:
                    agent.request("state", timeout=.2)
                    break
                except (OSError, RuntimeError):
                    time.sleep(.05)
            else:
                pytest.fail("fixture agent did not start")
            before = agent.request("runner_reconcile")
            child_pid = before["attempts"][ATTEMPT]["pid"]
            assert before["attempts"][ATTEMPT]["state"] == "running"
            payload = {"run": {"run_id": "c" * 32, "grid_tag": "DRE-TEST", "pairing_id": None,
                       "profiler": "none", "parameters": {"--select-width": True},
                       "dtype": "float16", "attention_backend": "sdpa", "gpu_uuid": "GPU-0"},
                       "gpu_key": "GPU-0", "host_label": "dreedle"}
            with pytest.raises(RuntimeError, match="Unsupported or automatic Runner option: --select-width"):
                agent.request("runner_preflight", payload)

            def start_replacement(*args, **kwargs):
                process = subprocess.Popen(*args, **kwargs)
                processes.append(process)
                return process
            subprocess_module = SimpleNamespace(Popen=start_replacement, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)
            parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(root=state_dir / "logs", host="127.0.0.1", port=0))
            restart = launcher(parser, subprocess_module, transport, process_path)
            restart()
            old.wait(timeout=3)
            assert len(processes) == 2
            assert processes[1].poll() is None
            capabilities = agent.request("runner_capabilities")
            assert capabilities["runner_runtime_fingerprint"] == agent._LOADED_RUNNER_FINGERPRINT
            after = agent.request("runner_reconcile")
            assert after["attempts"][ATTEMPT]["state"] == "running"
            assert after["attempts"][ATTEMPT]["pid"] == child_pid
            assert after["reservations"] == before["reservations"]
            os.kill(child_pid, 0)
            # The new catalogue accepts the exact rejected selector. This
            # GPU-free fixture then reaches the independent GPU availability check.
            with pytest.raises(RuntimeError, match="Resolved GPU unavailable or changed"):
                agent.request("runner_preflight", payload)
            restart()
            assert len(processes) == 2, "a matching agent must be reused"
        finally:
            if child_pid is None and agent.AGENT_STATE.exists():
                child_pid = json.loads(agent.AGENT_STATE.read_text())["attempts"][ATTEMPT]["pid"]
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("identity", ["wrong-user", "wrong-command", "missing-proc"])
def test_stale_agent_refresh_refuses_an_unverified_peer(tmp_path, monkeypatch, identity):
    monkeypatch.setattr(agent, "request", lambda operation, *args, **kwargs: {
        "protocol": 3, "local_gpu_queue": True, "cuda_preflight": True,
        "optional_power_readback": True} if operation == "runner_capabilities" else {})
    class Peer:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def settimeout(self, _timeout): pass
        def connect(self, _path): pass
        def getsockopt(self, *_args):
            return struct.pack("3i", 12345, os.getuid() + (identity == "wrong-user"), os.getgid())
    transport = SimpleNamespace(socket=lambda *_args: Peer(), AF_UNIX=socket.AF_UNIX,
                                SOCK_STREAM=socket.SOCK_STREAM, SOL_SOCKET=socket.SOL_SOCKET,
                                SO_PEERCRED=socket.SO_PEERCRED)
    def read_command():
        if identity == "missing-proc":
            raise FileNotFoundError("process identity unavailable")
        assert identity != "wrong-user", "another user's /proc entry must not be read"
        return b"python\0some_other_server.py\0serve\0"
    def process_path(value):
        return SimpleNamespace(read_bytes=read_command) if str(value) == "/proc/12345/cmdline" else Path(value)
    monkeypatch.setattr(os, "kill", lambda *_args: pytest.fail("unverified peer must not be signalled"))
    parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(root=tmp_path, host="127.0.0.1", port=0))
    subprocess_module = SimpleNamespace(Popen=lambda *_args, **_kwargs: pytest.fail("unverified peer must not be replaced"))
    with pytest.raises(RuntimeError, match="restart its verified process manually"):
        launcher(parser, subprocess_module, transport, process_path)()
