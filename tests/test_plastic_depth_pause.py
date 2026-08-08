from __future__ import annotations

import fcntl
import io
import os
import pty
import termios
import threading
import time

import pytest

import sheet.plastic_depth_pause as pause_module
from sheet.plastic_depth_pause import run_plastic_coarse_review_pause


def _write_key(master_fd: int, value: bytes) -> threading.Thread:
    def writer() -> None:
        time.sleep(0.05)
        os.write(master_fd, value)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    return thread


def test_ctrl_f_starts_fine_and_restores_terminal() -> None:
    master_fd, slave_fd = pty.openpty()
    original = termios.tcgetattr(slave_fd)
    output = io.StringIO()
    thread = _write_key(master_fd, b"\x06")
    try:
        result = run_plastic_coarse_review_pause(
            duration_seconds=2.0,
            output=output,
            terminal_fd=slave_fd,
        )
        thread.join(timeout=1.0)
        assert result.disposition == "ctrl_f"
        assert result.remaining_seconds > 0.0
        assert termios.tcgetattr(slave_fd) == original
        assert "Press Ctrl-F to start FINE now" in output.getvalue()
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_ctrl_g_routes_checkpoint_and_exit() -> None:
    master_fd, slave_fd = pty.openpty()
    called = []
    thread = _write_key(master_fd, b"\x07")
    try:
        result = run_plastic_coarse_review_pause(
            duration_seconds=2.0,
            output=io.StringIO(),
            terminal_fd=slave_fd,
            checkpoint_callback=lambda: called.append(True),
        )
        thread.join(timeout=1.0)
        assert result.disposition == "checkpoint_exit"
        assert called == [True]
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_ctrl_c_propagates_and_restores_controlling_terminal() -> None:
    master_fd, slave_fd = pty.openpty()
    original = termios.tcgetattr(slave_fd)
    child_pid = os.fork()
    if child_pid == 0:
        try:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            os.close(master_fd)
            with open(os.devnull, "w", encoding="utf-8") as output:
                try:
                    run_plastic_coarse_review_pause(
                        duration_seconds=10.0,
                        output=output,
                        terminal_fd=slave_fd,
                    )
                except KeyboardInterrupt:
                    os._exit(42)
            os._exit(0)
        except BaseException:
            os._exit(43)

    try:
        deadline = time.monotonic() + 2.0
        while termios.tcgetattr(slave_fd)[3] & termios.ICANON:
            if time.monotonic() >= deadline:
                os.kill(child_pid, 9)
                pytest.fail("child did not enter cbreak mode")
            time.sleep(0.01)
        os.write(master_fd, b"\x03")
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 42
        assert termios.tcgetattr(slave_fd) == original
    finally:
        try:
            os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        os.close(master_fd)
        os.close(slave_fd)


def test_headless_timeout_does_not_busy_loop(monkeypatch) -> None:
    now = [0.0]
    sleeps = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(pause_module, "_open_controlling_terminal", lambda: None)
    result = run_plastic_coarse_review_pause(
        duration_seconds=3.0,
        clock=clock,
        sleeper=sleeper,
        output=io.StringIO(),
    )

    assert result.disposition == "timeout"
    assert result.elapsed_seconds == pytest.approx(3.0)
    assert sleeps == [1.0, 1.0, 1.0]


def test_zero_duration_times_out_immediately(monkeypatch) -> None:
    monkeypatch.setattr(pause_module, "_open_controlling_terminal", lambda: None)
    result = run_plastic_coarse_review_pause(
        duration_seconds=0.0,
        clock=lambda: 10.0,
        sleeper=lambda _: pytest.fail("zero-duration pause must not sleep"),
        output=io.StringIO(),
    )

    assert result.disposition == "timeout"
    assert result.remaining_seconds == 0.0
