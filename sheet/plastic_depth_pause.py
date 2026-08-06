from __future__ import annotations

import os
import select
import termios
import time
import tty
from dataclasses import dataclass
from typing import Callable, Optional, TextIO


PLASTIC_COARSE_REVIEW_PAUSE_SECONDS = 900


@dataclass(frozen=True)
class PlasticCoarsePauseResult:
    disposition: str
    elapsed_seconds: float
    remaining_seconds: float


Clock = Callable[[], float]
Sleeper = Callable[[float], None]
CheckpointCallback = Callable[[], None]


def _open_controlling_terminal() -> Optional[int]:
    try:
        return os.open("/dev/tty", os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None


def _render_countdown(output: TextIO, remaining: int) -> None:
    output.write(
        f"\rPhase 2 starts automatically in {remaining} seconds. "
        "Press Ctrl-F to start FINE now. "
        "Ctrl-G checkpoints and exits; Ctrl-C interrupts."
    )
    output.flush()


def run_plastic_coarse_review_pause(
    *,
    duration_seconds: float = PLASTIC_COARSE_REVIEW_PAUSE_SECONDS,
    clock: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
    output: Optional[TextIO] = None,
    terminal_fd: Optional[int] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
) -> PlasticCoarsePauseResult:
    if duration_seconds < 0.0:
        raise ValueError("duration_seconds must be non-negative")
    if output is None:
        import sys

        output = sys.stdout

    owns_terminal = terminal_fd is None
    fd = _open_controlling_terminal() if owns_terminal else terminal_fd
    original_attributes = None
    started = clock()
    deadline = started + duration_seconds
    last_rendered: Optional[int] = None

    try:
        if fd is not None:
            original_attributes = termios.tcgetattr(fd)
            tty.setcbreak(fd, termios.TCSANOW)
            current_attributes = termios.tcgetattr(fd)
            current_attributes[3] |= termios.ISIG
            termios.tcsetattr(fd, termios.TCSANOW, current_attributes)

        while True:
            now = clock()
            remaining = max(0.0, deadline - now)
            remaining_integer = int(remaining + 0.999999)
            if remaining_integer != last_rendered:
                _render_countdown(output, remaining_integer)
                last_rendered = remaining_integer
            if remaining <= 0.0:
                output.write("\n")
                output.flush()
                return PlasticCoarsePauseResult(
                    disposition="timeout",
                    elapsed_seconds=max(0.0, now - started),
                    remaining_seconds=0.0,
                )

            wait_seconds = min(1.0, remaining)
            if fd is None:
                sleeper(wait_seconds)
                continue

            readable, _, _ = select.select((fd,), (), (), wait_seconds)
            if not readable:
                continue
            value = os.read(fd, 1)
            if not value:
                continue
            key = value[0]
            now = clock()
            remaining = max(0.0, deadline - now)
            if key == 0x06:
                output.write("\n")
                output.flush()
                return PlasticCoarsePauseResult(
                    disposition="ctrl_f",
                    elapsed_seconds=max(0.0, now - started),
                    remaining_seconds=remaining,
                )
            if key == 0x07:
                if checkpoint_callback is not None:
                    checkpoint_callback()
                output.write("\n")
                output.flush()
                return PlasticCoarsePauseResult(
                    disposition="checkpoint_exit",
                    elapsed_seconds=max(0.0, now - started),
                    remaining_seconds=remaining,
                )
            if key == 0x03:
                raise KeyboardInterrupt
    finally:
        if fd is not None and original_attributes is not None:
            termios.tcsetattr(fd, termios.TCSANOW, original_attributes)
        if owns_terminal and fd is not None:
            os.close(fd)
