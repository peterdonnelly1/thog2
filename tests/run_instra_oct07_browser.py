# vvv THOG run the HTTP fixture and browser in the same execution namespace
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="instra-oct07-") as folder:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ, THOG_INSTRA_REAL_RUNNER="1", INSTRA_TEST_URL=f"http://127.0.0.1:{port}")
    environment.setdefault("NODE_PATH", os.environ.get("CODEX_PRIMARY_RUNTIME_NODE_MODULES", ""))
    with (Path(folder) / "fixture.log").open("w+") as log:
        server = subprocess.Popen([sys.executable, str(root / "tests/instra_sep30_fixture.py"), folder, str(port)],
                                  cwd=root, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Fixture startup timed out")
            program = sys.argv[1] if len(sys.argv) > 1 else "tests/instra_oct07_browser_regression.js"
            browsers = environment.get("INSTRA_TEST_BROWSERS", "chromium,firefox").split(",")
            timeout_seconds = max(300, int(environment.get("INSTRA_DEMAND_SOAK_SECONDS", "120")) * len(browsers) + 120)
            result = subprocess.run(["node", str(root / program)], cwd=root, env=environment, timeout=timeout_seconds)
            if result.returncode:
                log.seek(0)
                print(log.read(), file=sys.stderr)
            raise SystemExit(result.returncode)
        finally:
            server.terminate()
            server.wait(timeout=5)
# ^^^ THOG
