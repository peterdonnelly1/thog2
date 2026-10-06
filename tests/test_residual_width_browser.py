# vvv THOG real browser + actual HTTP reader/store acceptance inside one process namespace
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_width_runs_multiview_and_legacy_browser_acceptance(tmp_path):
    if not os.environ.get('THOG_WIDTH_BROWSER_EXECUTABLE'):
        pytest.skip('set THOG_WIDTH_BROWSER_EXECUTABLE and install Playwright to run browser acceptance')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    if 'NODE_PATH' not in environment and os.environ.get('CODEX_PRIMARY_RUNTIME_NODE_MODULES'):
        environment['NODE_PATH'] = os.environ['CODEX_PRIMARY_RUNTIME_NODE_MODULES']
    log = tmp_path / 'server.log'
    with log.open('w') as output:
        server = subprocess.Popen([sys.executable, str(ROOT/'tests/instra_width_fixture.py'), str(tmp_path/'runs'), str(port)],
                                  cwd=ROOT, stdout=output, stderr=output, env=environment)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    pytest.fail(log.read_text())
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                pytest.fail('width HTTP fixture startup timed out: ' + log.read_text())
            evidence = environment.get('THOG_WIDTH_BROWSER_EVIDENCE', str(tmp_path/'browser.json'))
            result = subprocess.run([shutil.which('node') or 'node', str(ROOT/'tests/instra_width_browser.cjs'),
                f'http://127.0.0.1:{port}', evidence], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
        finally:
            server.terminate()
            server.wait(timeout=5)
# ^^^ THOG
