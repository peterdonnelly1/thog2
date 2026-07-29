# vvv THOG
from __future__ import annotations

import os
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO

from .stage6_trainer import Stage6Trainer


@dataclass
class _InterruptState:
    requested: bool = False
    handling: bool = False


def _open_controlling_terminal() -> Optional[TextIO]:
    try:
        return open("/dev/tty", "r+", encoding="utf-8", buffering=1)
    except OSError:
        return None


def _redirect_process_output_to_controlling_terminal() -> None:
    try:
        terminal_fd = os.open("/dev/tty", os.O_WRONLY)
    except OSError:
        return
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                os.dup2(terminal_fd, stream.fileno())
            except (AttributeError, OSError, ValueError):
                continue
    finally:
        os.close(terminal_fd)


def _terminal_message(message: str) -> None:
    terminal = _open_controlling_terminal()
    if terminal is None:
        return
    try:
        terminal.write(message)
        terminal.flush()
    finally:
        terminal.close()


def _prompt_save_checkpoint(
    completed_updates: int,
    *,
    reader: Optional[TextIO] = None,
    writer: Optional[TextIO] = None,
) -> bool:
    owned_terminal: Optional[TextIO] = None
    if reader is None or writer is None:
        owned_terminal = _open_controlling_terminal()
        if owned_terminal is None:
            return False
        reader = owned_terminal
        writer = owned_terminal
    try:
        while True:
            writer.write(
                f"\nSave checkpoint at completed update {completed_updates} before exit? [y/n]: "
            )
            writer.flush()
            response = reader.readline()
            if response == "":
                return False
            normalized = response.strip().lower()
            if normalized in {"y", "yes"}:
                return True
            if normalized in {"n", "no"}:
                return False
            writer.write("Please answer y or n.\n")
            writer.flush()
    finally:
        if owned_terminal is not None:
            owned_terminal.close()


def _checkpoint_after_interrupt(
    trainer: Any,
    *,
    prompt: Callable[[int], bool] = _prompt_save_checkpoint,
    message: Callable[[str], None] = _terminal_message,
) -> bool:
    decision: Optional[bool] = None
    if trainer.distributed.is_primary:
        decision = prompt(int(trainer.state.completed_updates))
    gathered = trainer.distributed.all_gather_object(decision)
    save_checkpoint = bool(gathered[0])
    if not save_checkpoint:
        if trainer.distributed.is_primary:
            message("Checkpoint skipped; finishing telemetry and exiting.\n")
        return False

    checkpoint_path = Path(trainer.config.out_dir) / "ckpt.pt"
    if trainer.distributed.is_primary:
        message(
            f"Saving checkpoint at completed update {trainer.state.completed_updates}: "
            f"{checkpoint_path}\n"
        )
    trainer.save_checkpoint(checkpoint_path)
    if trainer.distributed.is_primary:
        message(f"Checkpoint saved: {checkpoint_path}\n")
    return True


@contextmanager
def interactive_interrupt_checkpoint() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    state = _InterruptState()
    original_handler = signal.getsignal(signal.SIGINT)
    original_timed = Stage6Trainer._timed

    def handle_sigint(signum: int, frame: Any) -> None:
        del signum, frame
        if state.requested:
            raise KeyboardInterrupt
        state.requested = True
        _redirect_process_output_to_controlling_terminal()
        _terminal_message(
            "\nCtrl-C received; stopping at the next safe boundary. "
            "Press Ctrl-C again for an immediate abort.\n"
        )

    def install_handler() -> None:
        signal.signal(signal.SIGINT, handle_sigint)

    def timed_with_interrupt_checkpoint(trainer: Any, function: Callable[[], Any]):
        # vvv THOG trainer and library initialisation may replace SIGINT after program entry; reclaim it immediately before every safe timed operation
        install_handler()
        if state.requested and not state.handling:
            state.handling = True
            _checkpoint_after_interrupt(trainer)
            raise KeyboardInterrupt
        result = original_timed(trainer, function)
        if state.requested and not state.handling:
            state.handling = True
            _checkpoint_after_interrupt(trainer)
            raise KeyboardInterrupt
        return result
        # ^^^ THOG

    install_handler()
    Stage6Trainer._timed = timed_with_interrupt_checkpoint
    try:
        yield
    finally:
        Stage6Trainer._timed = original_timed
        signal.signal(signal.SIGINT, original_handler)


__all__ = [
    "interactive_interrupt_checkpoint",
]
# ^^^ THOG
