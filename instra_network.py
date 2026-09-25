# vvv THOG own persistent host records, SSH transport, discovery and Runner Master state outside model execution
"""Instra Network Service for local and SSH-connected node agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid

import instra_node_agent as agent

STATE_DIR = agent.STATE_DIR
CONFIG_PATH = STATE_DIR / "network.json"
LOCK_PATH = STATE_DIR / "network.lock"
LOG_PATH = STATE_DIR / "events.jsonl"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
_SSH_ERROR = "SSH transport"


class NetworkError(Exception):
    def __init__(self, category, message, host_id=None):
        super().__init__(message)
        self.category = category
        self.host_id = host_id


def _one_attempt_password(passwords, host_id):
    if passwords is None:
        return None
    if not isinstance(passwords, dict) or any(not isinstance(key, str) or not isinstance(value, str) or not value
                                                   for key, value in passwords.items()):
        raise NetworkError("validation", "Invalid one-attempt SSH credentials")
    return passwords.get(host_id)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_config():
    try:
        value = json.loads(CONFIG_PATH.read_text())
        if isinstance(value, dict) and isinstance(value.get("hosts"), dict):
            return value
    except (OSError, ValueError):
        pass
    return {"hosts": {}, "restart_mode": "off", "retry_interval": 30, "master_id": None, "release_pending": False,
            "grid_active": False, "queue_nonempty": False}


@contextmanager
def _locked_config():
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE_DIR, 0o700)
    with LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = _read_config()
        yield value
        temporary = CONFIG_PATH.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with temporary.open("w") as output:
            json.dump(value, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, CONFIG_PATH)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _event(operation, host_id, category, message):
    # Caller supplies fixed operation/category and a deliberately short, secret-free message.
    record = {"time": _now(), "source": "Network", "operation": operation, "thog_host_id": host_id,
              "outcome": category, "message": message[:300]}
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    with LOG_PATH.open("a") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(json.dumps(record) + "\n")
        output.flush()
        fcntl.flock(output, fcntl.LOCK_UN)


def log_event(source, operation, host_id, category, message, duration_seconds=None):
    if source not in {"Monitoring", "Runner", "Network"}:
        raise ValueError("invalid log source")
    record = {"time": _now(), "source": source, "operation": operation, "thog_host_id": host_id,
              "outcome": category, "message": str(message)[:300]}
    if duration_seconds is not None:
        record["duration_seconds"] = round(float(duration_seconds), 3)
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    with LOG_PATH.open("a") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(json.dumps(record) + "\n")
        fcntl.flock(output, fcntl.LOCK_UN)


def _host_key(hostname):
    try:
        import ipaddress
        address = ipaddress.ip_address(hostname)
        key = "ip_" + re.sub(r"[^a-z0-9]", "_", str(address).lower())
        if len(key) > 63:
            key = "ip_" + hashlib.sha256(address.packed).hexdigest()[:32]
    except ValueError:
        name = hostname.lower()
        if "." in name:
            # Different fully qualified names must not collapse to one short host ID.
            label = re.sub(r"[^a-z0-9_-]", "_", name)[:45]
            key = f"{label}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        else:
            key = name
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", key):
        raise NetworkError("validation", "Host name must begin with a letter or digit and contain only letters, digits, '-' or '_'.")
    return key


def _ssh_target(host):
    return f"{host['ssh_user']}@{host['address']}" if host.get("ssh_user") else host["address"]


def _known_host_token(host):
    return host["address"] if host["ssh_port"] == 22 else f"[{host['address']}]:{host['ssh_port']}"


def _fingerprint(key_line):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
        stream.write(key_line + "\n")
        path = stream.name
    try:
        result = subprocess.run(["ssh-keygen", "-lf", path, "-E", "sha256"], capture_output=True, text=True, timeout=5)
        match = re.search(r"SHA256:[^\s]+", result.stdout)
        if result.returncode or not match:
            raise NetworkError("host key", "Could not inspect the host key")
        return match.group()
    finally:
        Path(path).unlink(missing_ok=True)


def _scan_key(host):
    try:
        result = subprocess.run(["ssh-keyscan", "-T", "5", "-p", str(host["ssh_port"]), host["address"]],
                                capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError) as error:
        raise NetworkError(_SSH_ERROR, "Could not retrieve the SSH host identity") from error
    candidates = [line for line in result.stdout.splitlines() if line and not line.startswith("#") and len(line.split()) >= 3]
    ed25519 = [line for line in candidates if line.split()[1] == "ssh-ed25519"]
    if not (ed25519 or candidates):
        raise NetworkError(_SSH_ERROR, "SSH host unavailable or host key missing")
    line = (ed25519 or candidates)[0]
    return line, _fingerprint(line)


def _is_known(host):
    token = _known_host_token(host)
    for path in (KNOWN_HOSTS, Path.home() / ".ssh/known_hosts"):
        if path.exists():
            result = subprocess.run(["ssh-keygen", "-F", token, "-f", str(path)], capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
    return False


def _verify_identity(host, accepted_fingerprint):
    if _is_known(host):
        return
    line, fingerprint = _scan_key(host)
    if accepted_fingerprint != fingerprint:
        raise NetworkError("host key", f"Confirm this host's SSH fingerprint before adding it: {fingerprint}")
    with KNOWN_HOSTS.open("a") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(line + "\n")
        fcntl.flock(output, fcntl.LOCK_UN)
    os.chmod(KNOWN_HOSTS, 0o600)


def _direct_route(host):
    # Check only after SSH has had an opportunity to resolve and connect to the supplied name.
    try:
        ip_address = host.get("_connected_ip")
        if not ip_address:
            addresses = socket.getaddrinfo(host["address"], host["ssh_port"], type=socket.SOCK_STREAM)
            ip_address = addresses[0][4][0]
        route = subprocess.run(["ip", "-j", "route", "get", ip_address], capture_output=True, text=True, timeout=4)
        if route.returncode == 0 and any(row.get("gateway") or row.get("via") for row in json.loads(route.stdout)):
            raise NetworkError("topology", "The route to this host uses a gateway; only directly connected hosts are supported")
        return ip_address
    except NetworkError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@contextmanager
def _ssh_askpass(password):
    if password is not None and (not isinstance(password, str) or not password):
        raise NetworkError("authentication", "Invalid SSH password")
    environment = os.environ.copy()
    askpass_file = None
    if password is not None:
        with tempfile.NamedTemporaryFile(mode="w", prefix="instra-askpass-", delete=False) as stream:
            stream.write('#!/bin/sh\nprintf "%s\\n" "$INSTRA_SSH_PASSWORD"\n')
            askpass_file = stream.name
        os.chmod(askpass_file, 0o700)
        environment.update(SSH_ASKPASS=askpass_file, SSH_ASKPASS_REQUIRE="force", INSTRA_SSH_PASSWORD=password, DISPLAY="instra:0")
    try:
        yield environment
    finally:
        if askpass_file:
            Path(askpass_file).unlink(missing_ok=True)
        environment.pop("INSTRA_SSH_PASSWORD", None)


def _ssh_request(host, operation, args, password=None, accepted_fingerprint=None):
    _verify_identity(host, accepted_fingerprint)
    ssh_options = ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={KNOWN_HOSTS} {Path.home() / '.ssh/known_hosts'}",
                   "-o", "ConnectTimeout=8", "-p", str(host["ssh_port"])]
    ssh_options += ["-o", "BatchMode=no" if password is not None else "BatchMode=yes"]
    # The remote command is fixed and never interpolates user inputs.
    remote_command = "~/.local/state/instra/agent-request"
    command = ["ssh", "-v", *ssh_options, "--", _ssh_target(host), remote_command]
    try:
        with _ssh_askpass(password) as environment:
            result = subprocess.run(command, input=json.dumps({"operation": operation, "args": args}), capture_output=True,
                                    text=True, timeout=35, env=environment, start_new_session=password is not None)
    except subprocess.TimeoutExpired as error:
        raise NetworkError(_SSH_ERROR, "SSH connection or agent request timed out") from error
    except OSError as error:
        raise NetworkError(_SSH_ERROR, "SSH client unavailable") from error
    connected = re.search(r"Connecting to [^\n]* \[([^\]]+)\] port \d+", result.stderr)
    if connected:
        host["_connected_ip"] = connected.group(1)
    try:
        response = json.loads(result.stdout)
    except ValueError:
        stderr = result.stderr.lower()
        # vvv THOG verbose SSH can report "Authentication succeeded" before an unavailable node agent; only explicit denial means bad credentials
        if "host key verification" in stderr or "remote host identification has changed" in stderr:
            raise NetworkError("host key", "SSH host identity verification failed", host["thog_host_id"])
        if "permission denied" in stderr or "authentication failed" in stderr:
            raise NetworkError("authentication", "SSH authentication failed", host["thog_host_id"])
        # ^^^ THOG
        if result.returncode == 255:
            raise NetworkError(_SSH_ERROR, "SSH connection failed", host["thog_host_id"])
        raise NetworkError("agent availability", "Node agent unavailable on this host", host["thog_host_id"])
    if not response.get("ok"):
        category = response.get("category")
        if category not in {"agent availability", "operation"}:
            category = "operation"
        raise NetworkError(category, str(response.get("error", "Node agent request failed"))[:200])
    return response["result"]


class NetworkService:
    def __init__(self, logs_root=None, start_worker=True):
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="instra-network")
        self.monitoring_provider = None                                                                                                                      # <<< THOG Monitoring installs its acquisition callback without moving file interpretation into Network
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.local_id = f"thog_host.{_host_key(socket.gethostname())}"
        with _locked_config() as config:
            config["hosts"].setdefault(self.local_id, self._new_host(socket.gethostname(), "local", 22, True))
        self.logs_root = logs_root
        if start_worker:
            threading.Thread(target=self._retry_loop, name="instra-network-retry", daemon=True).start()
            local_host = self._host(self.local_id)
            if local_host["monitoring_enabled"] or local_host["execution_enabled"]:
                self.submit("discover", self.local_id)

    def _new_host(self, address, ssh_user, ssh_port, local=False):
        host_id = f"thog_host.{_host_key(address)}"
        return {"thog_host_id": host_id, "display_name": _host_key(address), "address": address,
                "ssh_user": ssh_user, "ssh_port": ssh_port, "authentication_mode": "certificate_or_agent",
                "local": local, "monitoring_enabled": local, "execution_enabled": False, "state": "disconnected",
                "last_discovered": None, "last_success": None, "last_contact": None, "latest_error": None,
                "monitoring_status": {}, "pending": False}

    def _host(self, host_id):
        host = _read_config()["hosts"].get(host_id)
        if host is None:
            raise NetworkError("validation", "Unknown thog_host_id")
        return host

    def _agent_request(self, host, operation, args=None, password=None, accepted_fingerprint=None):
        if host["local"]:
            try:
                return agent.request(operation, args)
            except OSError as error:
                raise NetworkError("agent availability", "Local node agent unavailable") from error
            except RuntimeError as error:
                raise NetworkError("operation", str(error)[:200]) from error
        try:
            return _ssh_request(host, operation, args or {}, password, accepted_fingerprint)
        except NetworkError as error:
            error.host_id = host["thog_host_id"]
            raise

    def _record(self, host_id, **changes):
        with _locked_config() as config:
            if host_id in config["hosts"]:
                config["hosts"][host_id].update(changes)

    def _identify_discovery(self, host_id, discovery):
        # vvv THOG the saved host identity, not the remote hostname, owns its execution profiles
        result = dict(discovery)
        result["execution_profiles"] = [
            {**profile, "execution_profile_id": f"{host_id}.execution_profile.{profile['profile_key']}"}
            for profile in discovery.get("execution_profiles", [])
        ]
        result["gpus"] = [
            {**gpu, "gpu_id": f"{host_id}.gpu.{gpu['gpu_key']}"}
            for gpu in discovery.get("gpus", [])
        ]
        return result
        # ^^^ THOG

    def _retry_loop(self):
        while not self.stop_event.is_set():
            config = _read_config()
            for host_id, host in config["hosts"].items():
                if host["monitoring_enabled"] or host["execution_enabled"]:
                    if host["state"] != "discovering":
                        self.submit("discover", host_id)
            self.stop_event.wait(max(1, int(config.get("retry_interval", 30))))

    def submit(self, action, host_id=None, **args):
        job_id = uuid.uuid4().hex
        with self.jobs_lock:
            self.jobs[job_id] = {"status": "working", "action": action, "host_id": host_id}
        def work():
            try:
                result = self._execute(action, host_id, **args)
                outcome = {"status": "done", "result": result}
            except NetworkError as error:
                outcome = {"status": "error", "category": error.category, "error": str(error)}
                if error.host_id:
                    outcome["failed_host_id"] = error.host_id
                log_failure = True
                if host_id and action == "discover":
                    host = self._host(host_id)
                    previous = host.get("latest_error") or {}
                    log_failure = previous.get("category") != error.category or previous.get("message") != str(error)
                    state = "disabled" if not host["monitoring_enabled"] and not host["execution_enabled"] else (
                        "authentication required" if error.category in {"authentication", "host key"} else (
                            "unavailable" if error.category in {"agent availability", "operation"} else "disconnected"))
                    self._record(host_id, state=state, latest_error={"category": error.category, "message": str(error), "time": _now()})
                if log_failure:
                    _event(action, host_id, error.category, str(error))
            except Exception:
                outcome = {"status": "error", "category": "operation", "error": "Network operation failed"}
                _event(action, host_id, "operation", "Network operation failed")
            with self.jobs_lock:
                self.jobs[job_id].update(outcome)
                if len(self.jobs) > 300:
                    for old_id in list(self.jobs)[:100]:
                        if self.jobs[old_id]["status"] != "working":
                            self.jobs.pop(old_id, None)
        self.executor.submit(work)
        return {"job_id": job_id}

    def job(self, job_id):
        with self.jobs_lock:
            return dict(self.jobs.get(job_id, {"status": "error", "error": "Unknown operation"}))

    def list_hosts(self):
        config = _read_config()
        hosts = list(config["hosts"].values())
        return {"hosts": hosts, "local_id": self.local_id, "master_id": config["master_id"],
                "release_pending": config["release_pending"], "restart_mode": config["restart_mode"],
                "retry_interval": config["retry_interval"]}

    def events(self, host_id=None, limit=80):
        try:
            lines = LOG_PATH.read_text().splitlines()[-min(500, limit * 5):]
        except OSError:
            return []
        records = []
        for line in reversed(lines):
            try:
                item = json.loads(line)
                if host_id is None or item["thog_host_id"] == host_id:
                    records.append(item)
                if len(records) >= limit:
                    break
            except (ValueError, KeyError):
                continue
        return records

    def set_monitoring_provider(self, callback):
        self.monitoring_provider = callback

    def update_monitoring_status(self, host_id, status):
        self._host(host_id)
        if not isinstance(status, dict):
            raise NetworkError("validation", "Invalid monitoring status")
        self._record(host_id, monitoring_status={key: status.get(key) for key in (
            "refresh_interval", "activity", "last_success", "latest_error")})

    def acquire_file(self, host_id, root_kind, relative_path, destination, password=None):
        """Monitoring calls this to acquire one reported-root file over authenticated SSH."""
        host = self._host(host_id)
        if not host["monitoring_enabled"]:
            raise NetworkError("validation", "Monitoring is disabled for this host")
        root_key = {"logs": "instra_logs_root", "wandb": "wandb_root"}.get(root_kind)
        discovery = host.get("last_discovered") or {}
        if not root_key or not discovery.get(root_key):
            raise NetworkError("validation", "The selected source root has not been discovered")
        relative = PurePosixPath(relative_path)
        if not relative_path or relative.is_absolute() or any(part in {".", ".."} for part in relative_path.split("/")):
            raise NetworkError("validation", "Invalid relative run file path")
        source = str(PurePosixPath(discovery[root_key]) / relative)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            if host["local"]:
                with Path(source).open("rb") as input_stream, temporary.open("wb") as output:
                    shutil.copyfileobj(input_stream, output)
            else:
                # SFTP needs only SSH authentication and the previously reported roots;
                # acquisition remains possible while the node agent is unavailable.
                if not _is_known(host):
                    raise NetworkError("host key", "Host key is not accepted")
                def quoted(path):
                    if "\n" in path or "\r" in path or "\0" in path:
                        raise NetworkError("validation", "Invalid run file name")
                    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'
                batch = f"get {quoted(source)} {quoted(str(temporary))}\n"
                command = ["sftp", "-b", "-", "-o", "StrictHostKeyChecking=yes",
                           "-o", f"UserKnownHostsFile={KNOWN_HOSTS} {Path.home() / '.ssh/known_hosts'}",
                           "-o", "ConnectTimeout=8", "-o", "BatchMode=no" if password is not None else "BatchMode=yes",
                           "-P", str(host["ssh_port"]), "--", _ssh_target(host)]
                try:
                    with _ssh_askpass(password) as environment:
                        result = subprocess.run(command, input=batch, text=True, capture_output=True, timeout=120,
                                                env=environment, start_new_session=password is not None)
                except (OSError, subprocess.SubprocessError) as error:
                    raise NetworkError(_SSH_ERROR, "SFTP acquisition failed") from error
                if result.returncode:
                    failure = (result.stderr or "").lower()
                    if "permission denied (" in failure or "authentication failed" in failure:
                        raise NetworkError("authentication", "SFTP authentication failed", host_id)
                    if "host key verification failed" in failure or "remote host identification has changed" in failure:
                        raise NetworkError("host key", "SFTP host identity verification failed", host_id)
                    if "no such file" in failure or "couldn't stat" in failure:
                        raise NetworkError("operation", "Requested run file is unavailable", host_id)
                    raise NetworkError(_SSH_ERROR, "SFTP acquisition failed; source file or connection unavailable")
            os.replace(temporary, target)
            log_event("Monitoring", "acquire_file", host_id, "success", f"Acquired {root_kind} run file")
            return target
        finally:
            temporary.unlink(missing_ok=True)

    # vvv THOG expose root-scoped SSH listing and pull-only transfer to Monitoring; discovery and credentials remain owned by Network
    def _monitor_source(self, host_id, root_kind, relative_path=""):
        host = self._host(host_id)
        if host.get("local") or not host.get("monitoring_enabled"):
            raise NetworkError("validation", "Remote monitoring is not enabled for this host", host_id)
        discovery = host.get("last_discovered") or {}
        root_key = {"logs": "instra_logs_root", "wandb": "wandb_root"}.get(root_kind)
        root = discovery.get(root_key) if root_key else None
        if not isinstance(root, str) or not root.startswith("/") or any(character in root for character in "\0\r\n"):
            raise NetworkError("validation", "Monitoring source root is unavailable", host_id)
        if relative_path:
            if (not isinstance(relative_path, str) or relative_path.startswith("/") or "\\" in relative_path
                    or any(part in {"", ".", ".."} for part in relative_path.split("/"))):
                raise NetworkError("validation", "Invalid monitoring source path", host_id)
        if not _is_known(host):
            raise NetworkError("host key", "Host key is not accepted", host_id)
        return host, str(PurePosixPath(root) / relative_path)

    def monitor_files(self, host_id, root_kind, relative_directory=""):
        """List readable regular files without following links or requiring a running node agent."""
        host, source = self._monitor_source(host_id, root_kind, relative_directory)
        remote_command = "find " + shlex.quote(source) + " -type d ! -readable -prune -o -type f -readable -printf '%P\\0%s\\0%T@\\0'"
        command = ["ssh", "-p", str(host["ssh_port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", "StrictHostKeyChecking=yes", "-o",
                   f"UserKnownHostsFile={KNOWN_HOSTS} {Path.home() / '.ssh/known_hosts'}",
                   "--", _ssh_target(host), remote_command]
        try:
            result = subprocess.run(command, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            raise NetworkError(_SSH_ERROR, "Monitoring source listing failed", host_id) from error
        if result.returncode:
            raise NetworkError(_SSH_ERROR, "Monitoring source unavailable or SSH authentication failed", host_id)
        fields = result.stdout.split(b"\0")
        if fields[-1:] == [b""]:
            fields.pop()
        if len(fields) % 3 or len(fields) > 150000:
            raise NetworkError("operation", "Invalid or excessive monitoring source listing", host_id)
        return [(os.fsdecode(fields[index]), int(fields[index + 1]), float(fields[index + 2]))
                for index in range(0, len(fields), 3)]

    def monitor_transfer(self, host_id, root_kind, relative_path, destination, *, database=False):
        """Pull one file using rsync or SQLite's live-database protocol, never writing to the producer."""
        host, source = self._monitor_source(host_id, root_kind, relative_path)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ssh_options = ["ssh", "-p", str(host["ssh_port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                       "-o", "StrictHostKeyChecking=yes", "-o",
                       f"UserKnownHostsFile={KNOWN_HOSTS} {Path.home() / '.ssh/known_hosts'}"]
        ssh_wrapper = None
        try:
            if database:
                if not shutil.which("sqlite3_rsync"):
                    raise NetworkError("dependency", "sqlite3_rsync is required on both thog_hosts", host_id)
                with tempfile.NamedTemporaryFile(mode="w", prefix="instra-sqlite-ssh-", delete=False) as output:
                    ssh_wrapper = Path(output.name)
                    output.write("#!/bin/sh\nexec " + " ".join(shlex.quote(option) for option in ssh_options) + ' "$@"\n')
                ssh_wrapper.chmod(0o700)
                command = ["sqlite3_rsync", "--ssh", str(ssh_wrapper), f"{_ssh_target(host)}:{source}", str(destination)]
            else:
                if not shutil.which("rsync"):
                    raise NetworkError("dependency", "rsync is required on both thog_hosts", host_id)
                command = ["rsync", "--protect-args", "--no-links", "--no-owner", "--no-group", "--no-perms",
                           "--timeout=90", "-e", " ".join(shlex.quote(option) for option in ssh_options),
                           "--", f"{_ssh_target(host)}:{source}", str(destination)]
            result = subprocess.run(command, capture_output=True, timeout=180)
            if result.returncode:
                raise NetworkError("transfer", f"{'sqlite3_rsync' if database else 'rsync'} acquisition failed", host_id)
            return destination
        except (OSError, subprocess.SubprocessError) as error:
            raise NetworkError("transfer", "Monitoring transfer failed or timed out", host_id) from error
        finally:
            if ssh_wrapper is not None:
                ssh_wrapper.unlink(missing_ok=True)
    # ^^^ THOG

    def _execute(self, action, host_id, **args):
        if action == "update":
            return self.update_host(host_id, **args)
        if action == "settings":
            return self.settings(**args)
        if action == "designate_master":
            return self.designate(host_id, passwords=args.get("passwords"))
        if action == "release_master":
            return self.release(passwords=args.get("passwords"))
        if action == "remove":
            return self.remove(host_id)
        if action == "prepare_host":
            host = self._validated_host(args)
            if _is_known(host):
                return {"known": True, "fingerprint": None}
            _, fingerprint = _scan_key(host)
            return {"known": False, "fingerprint": fingerprint}
        if action == "add":
            host = self._validated_host(args)
            host_id = host["thog_host_id"]
            saved_hosts = _read_config()["hosts"]
            if host_id in saved_hosts or any((old["address"], old["ssh_user"], old["ssh_port"]) ==
                                             (host["address"], host["ssh_user"], host["ssh_port"])
                                             for old in saved_hosts.values()):
                raise NetworkError("validation", "Host key already exists; remove the existing host before adding a changed hostname")
            discovery = self._agent_request(host, "discover", password=args.get("password"), accepted_fingerprint=args.get("fingerprint"))
            resolved_ip = _direct_route(host)
            if not isinstance(discovery, dict) or not discovery.get("instra_logs_root") or not discovery.get("wandb_root"):
                raise NetworkError("operation", "Discovery did not report both monitoring source roots")
            discovery = self._identify_discovery(host_id, discovery)
            master_id = _read_config()["master_id"]
            if master_id:
                self._agent_request(host, "claim_master", {"master_id": master_id}, password=args.get("password"))
            host.update(state="disabled", last_discovered=discovery, last_success=_now(), last_contact=_now(),
                        resolved_ip=resolved_ip, latest_error=None, authentication_mode="password" if args.get("password") else "certificate_or_agent")
            host.pop("_connected_ip", None)
            with _locked_config() as config:
                if host_id in config["hosts"]:
                    raise NetworkError("validation", "Host was added concurrently")
                config["hosts"][host_id] = host
            _event(action, host_id, "success", "Host authenticated and discovered")
            return {"thog_host_id": host_id}
        if action == "discover":
            host = self._host(host_id)
            self._record(host_id, state="discovering")
            discovery = self._agent_request(host, "discover", password=args.get("password"))
            if not isinstance(discovery, dict) or not discovery.get("instra_logs_root") or not discovery.get("wandb_root"):
                raise NetworkError("operation", "Incomplete node discovery")
            discovery = self._identify_discovery(host_id, discovery)
            state = "available" if host["monitoring_enabled"] or host["execution_enabled"] else "disabled"
            self._record(host_id, state=state, last_discovered=discovery, last_success=_now(), last_contact=_now(), latest_error=None,
                         resolved_ip=None if host["local"] else _direct_route(host))
            try:
                runtime = self._agent_request(host, "state", password=args.get("password"))
                restart = runtime.get("last_auto_restart")
                if restart and restart.get("time") != host.get("last_auto_restart_id"):
                    _event("automatic_restart", host_id, restart.get("outcome", "operation"), "Automatic Instra restart attempted")
                    self._record(host_id, last_auto_restart_id=restart.get("time"))
            except NetworkError:
                pass
            mode = _read_config()["restart_mode"]
            try:
                self._agent_request(host, "configure", {"auto_restart": mode == "all" or (mode == "remote" and not host["local"])},
                                    password=args.get("password"))
            except NetworkError:
                pass
            _event(action, host_id, "success", "Host discovery completed")
            return discovery
        if action in {"start_instra", "restart_instra", "state", "report_run", "launch_run", "cancel_run"}:
            host = self._host(host_id)
            if action in {"launch_run", "cancel_run"}:
                config = _read_config()
                if config["master_id"] != self.local_id or not host["execution_enabled"]:
                    raise NetworkError("authority", "This Instra is not the Runner Master or execution is disabled")
                if action == "launch_run" and config["release_pending"]:
                    raise NetworkError("authority", "Runner Master release is pending; new remote runs are paused")
                args["master_id"] = self.local_id
            password = args.pop("password", None)
            result = self._agent_request(host, action, args, password=password)
            self._record(host_id, last_contact=_now())
            _event(action, host_id, "success", "Node operation completed")
            return result
        if action in {"monitor_refresh", "monitor_settings"}:
            host = self._host(host_id)
            if not host["monitoring_enabled"] or not host.get("last_discovered"):
                raise NetworkError("validation", "Monitoring is not enabled or discovery is incomplete")
            if self.monitoring_provider is None:
                raise NetworkError("operation", "Monitoring acquisition is not installed yet")
            return self.monitoring_provider(host_id, host["last_discovered"],
                                            refresh_interval=args.get("refresh_interval") if action == "monitor_settings" else None)
        raise NetworkError("validation", "Unknown Network action")

    def _validated_host(self, args):
        address = args.get("address", "")
        ssh_user = args.get("ssh_user", "")
        port = args.get("ssh_port", 22)
        if not isinstance(address, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.:-]{0,252}", address):
            raise NetworkError("validation", "Invalid hostname or IP address")
        if not isinstance(ssh_user, str) or (ssh_user and not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", ssh_user)):
            raise NetworkError("validation", "Invalid SSH username")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise NetworkError("validation", "SSH port must be between 1 and 65535")
        return self._new_host(address, ssh_user, port)

    def update_host(self, host_id, monitoring_enabled=None, execution_enabled=None, display_name=None):
        host = self._host(host_id)
        if monitoring_enabled is not None and monitoring_enabled and not all(host.get("last_discovered", {}).get(key) for key in ("instra_logs_root", "wandb_root")):
            raise NetworkError("validation", "Discover the monitoring source roots before enabling monitoring")
        if execution_enabled is not None and execution_enabled and not host.get("last_discovered"):
            raise NetworkError("validation", "Discover this host before enabling execution")
        if display_name is not None and (not isinstance(display_name, str) or not 1 <= len(display_name.strip()) <= 80):
            raise NetworkError("validation", "Invalid display name")
        changes = {}
        for key, value in (("monitoring_enabled", monitoring_enabled), ("execution_enabled", execution_enabled)):
            if value is not None:
                if not isinstance(value, bool):
                    raise NetworkError("validation", "Enablement must be true or false")
                changes[key] = value
        if display_name is not None:
            changes["display_name"] = display_name.strip()
        next_monitoring = changes.get("monitoring_enabled", host["monitoring_enabled"])
        next_execution = changes.get("execution_enabled", host["execution_enabled"])
        if not next_monitoring and not next_execution:
            changes["state"] = "disabled"
        became_enabled = not host["monitoring_enabled"] and not host["execution_enabled"] and (next_monitoring or next_execution)
        if became_enabled:
            changes["state"] = "discovering"
        self._record(host_id, **changes)
        if became_enabled:
            self.submit("discover", host_id)
        return self._host(host_id)

    def settings(self, restart_mode=None, retry_interval=None, passwords=None):
        if restart_mode is not None and restart_mode not in {"off", "remote", "all"}:
            raise NetworkError("validation", "Invalid automatic restart mode")
        if retry_interval is not None and (isinstance(retry_interval, bool) or not isinstance(retry_interval, int) or not 1 <= retry_interval <= 3600):
            raise NetworkError("validation", "Retry interval must be 1–3600 seconds")
        _one_attempt_password(passwords, self.local_id)
        with _locked_config() as config:
            if restart_mode is not None:
                config["restart_mode"] = restart_mode
            if retry_interval is not None:
                config["retry_interval"] = retry_interval
        if restart_mode is not None:
            for host in _read_config()["hosts"].values():
                try:
                    self._agent_request(host, "configure", {"auto_restart": restart_mode == "all" or (restart_mode == "remote" and not host["local"])},
                                        password=_one_attempt_password(passwords, host["thog_host_id"]))
                except NetworkError as error:
                    _event("configure_restart", host["thog_host_id"], error.category, str(error))
                    if error.category == "authentication":
                        raise
        return self.list_hosts()

    def designate(self, host_id, passwords=None):
        if host_id != self.local_id:
            raise NetworkError("authority", "Designate Runner Master from its own Instra instance")
        config = _read_config()
        if config["master_id"] and config["master_id"] != host_id:
            raise NetworkError("authority", "A different Runner Master is already designated")
        claimed = []
        try:
            for other in config["hosts"].values():
                self._agent_request(other, "claim_master", {"master_id": host_id},
                                    password=_one_attempt_password(passwords, other["thog_host_id"]))
                claimed.append(other)
        except NetworkError:
            if config["master_id"] != host_id:
                for other in reversed(claimed):
                    try:
                        self._agent_request(other, "release_master", {"master_id": host_id},
                                            password=_one_attempt_password(passwords, other["thog_host_id"]))
                    except NetworkError:
                        pass
            raise
        with _locked_config() as current:
            current.update(master_id=host_id, release_pending=False)
        _event("designate_master", host_id, "success", "Runner Master designated")
        return self.list_hosts()

    def release(self, passwords=None):
        with _locked_config() as config:
            if config["master_id"] != self.local_id:
                raise NetworkError("authority", "This Instra is not Runner Master")
            if config["grid_active"] or config["queue_nonempty"]:
                config["release_pending"] = True
                return {"release_pending": True}
        hosts = list(_read_config()["hosts"].values())
        # Check authentication for every host before changing any Master claim.
        for host in hosts:
            self._agent_request(host, "state", password=_one_attempt_password(passwords, host["thog_host_id"]))
        with _locked_config() as config:
            if config["grid_active"] or config["queue_nonempty"]:
                config["release_pending"] = True
                return {"release_pending": True}
            config["release_pending"] = True
        for host in hosts:
            self._agent_request(host, "release_master", {"master_id": self.local_id},
                                password=_one_attempt_password(passwords, host["thog_host_id"]))
        with _locked_config() as config:
            config.update(master_id=None, release_pending=False)
        _event("release_master", self.local_id, "success", "Runner Master released")
        return {"release_pending": False}

    def set_grid_activity(self, active, queue_nonempty):
        with _locked_config() as config:
            config.update(grid_active=bool(active), queue_nonempty=bool(queue_nonempty))
            release_pending = config["release_pending"]
        if release_pending and not active and not queue_nonempty:
            self.release()

    def remove(self, host_id):
        host = self._host(host_id)
        if host["local"]:
            raise NetworkError("validation", "The local host cannot be removed")
        if _read_config()["master_id"]:
            raise NetworkError("authority", "Release Runner Master before removing a participating host")
        with _locked_config() as config:
            config["hosts"].pop(host_id, None)
        _event("remove", host_id, "success", "Host configuration removed")
        return {"removed": host_id}

    def close(self):
        self.stop_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
# ^^^ THOG
