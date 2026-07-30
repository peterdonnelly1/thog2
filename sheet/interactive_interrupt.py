# vvv THOG
from __future__ import annotations

import atexit
import os
import select
import tempfile
import termios
import threading
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO

import torch

from .checkpoints import save_payload
from .stage6_trainer import Stage6Trainer
from .wandb_telemetry import WandbTelemetry


_INTERRUPTED_TRAINER: Optional[Any] = None
_CTRL_G = b"\x07"
_MIN_REQUEST_AGE_SECONDS = 0.25


class CheckpointExitRequested(RuntimeError):
    """Raised only after the requested safe-boundary checkpoint has completed."""

    def __init__(self, checkpoint_path: Path, completed_updates: int) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.completed_updates = int(completed_updates)
        super().__init__(
            f"checkpoint exit completed at update {self.completed_updates}: "
            f"{self.checkpoint_path}"
        )


def checkpoint_exit_request_path() -> Path:
    configured = os.environ.get("THOG2_CHECKPOINT_EXIT_FILE", "").strip()
    if configured:
        return Path(configured)
    distributed_key = (
        os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or str(os.getppid())
    )
    return Path(tempfile.gettempdir()) / f"thog2_checkpoint_exit_{distributed_key}"


class CheckpointExitController:
    """Read Ctrl-G from /dev/tty and expose one shared safe-boundary request file."""

    def __init__(self, *, is_primary: bool) -> None:
        self.is_primary = bool(is_primary)
        self.request_path = checkpoint_exit_request_path()
        self._terminal_fd: Optional[int] = None
        self._terminal_attributes: Optional[list[Any]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.is_primary:
            return
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        self.request_path.unlink(missing_ok=True)
        try:
            terminal_fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            terminal_attributes = termios.tcgetattr(terminal_fd)
            tty.setcbreak(terminal_fd, termios.TCSANOW)
        except (OSError, termios.error):
            try:
                os.close(terminal_fd)
            except (UnboundLocalError, OSError):
                pass
            return
        self._terminal_fd = terminal_fd
        self._terminal_attributes = terminal_attributes
        self._thread = threading.Thread(
            target=self._listen,
            name="thog2_ctrl_g_checkpoint_exit",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)
        self._write_terminal(
            "  checkpoint exit:                   Ctrl-G saves at the next safe boundary; Ctrl-C discards\n"
        )

    def requested(self) -> bool:
        try:
            age = time.time() - self.request_path.stat().st_mtime
        except OSError:
            return False
        return age >= _MIN_REQUEST_AGE_SECONDS

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._restore_terminal()
        if self.is_primary:
            self.request_path.unlink(missing_ok=True)

    def _listen(self) -> None:
        try:
            while not self._stop.is_set():
                terminal_fd = self._terminal_fd
                if terminal_fd is None:
                    return
                readable, _, _ = select.select([terminal_fd], [], [], 0.2)
                if not readable:
                    continue
                value = os.read(terminal_fd, 1)
                if value != _CTRL_G:
                    continue
                self.request_path.write_text(
                    f"requested_ns={time.time_ns()}\n",
                    encoding="utf-8",
                )
                self._write_terminal(
                    "\nCtrl-G received; checkpointing after the current timed operation.\n"
                )
                return
        except OSError:
            return
        finally:
            self._restore_terminal()

    def _write_terminal(self, message: str) -> None:
        terminal_fd = self._terminal_fd
        if terminal_fd is None:
            return
        try:
            os.write(terminal_fd, message.encode("utf-8", errors="replace"))
        except OSError:
            pass

    def _restore_terminal(self) -> None:
        with self._lock:
            terminal_fd = self._terminal_fd
            terminal_attributes = self._terminal_attributes
            self._terminal_fd = None
            self._terminal_attributes = None
        if terminal_fd is None:
            return
        try:
            if terminal_attributes is not None:
                termios.tcsetattr(
                    terminal_fd,
                    termios.TCSADRAIN,
                    terminal_attributes,
                )
        except (OSError, termios.error):
            pass
        finally:
            try:
                os.close(terminal_fd)
            except OSError:
                pass


def _open_controlling_terminal() -> Optional[TextIO]:
    try:
        return open("/dev/tty", "r+", encoding="utf-8", buffering=1)
    except OSError:
        return None


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
    completed_updates = int(trainer.state.completed_updates)
    if not prompt(completed_updates):
        message("Checkpoint skipped; finishing telemetry and exiting.\n")
        return False

    checkpoint_path = Path(trainer.config.out_dir) / "ckpt.pt"
    message(
        f"Saving checkpoint at completed update {completed_updates}: "
        f"{checkpoint_path}\n"
    )
    try:
        trainer.optimizer.zero_grad(set_to_none=True)
        device = getattr(trainer, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        save_payload(trainer.checkpoint_payload(), checkpoint_path)
    except Exception as error:
        message(f"Checkpoint save failed: {error}\n")
        return False
    message(f"Checkpoint saved: {checkpoint_path}\n")
    return True


@contextmanager
def interactive_interrupt_checkpoint() -> Iterator[None]:
    """Legacy prompt hook retained for compatibility; public runs now use Ctrl-G."""

    global _INTERRUPTED_TRAINER
    original_run_pilot = Stage6Trainer.run_pilot
    original_finish = WandbTelemetry.finish

    def run_pilot_with_interrupt_capture(trainer: Any, *args: Any, **kwargs: Any):
        global _INTERRUPTED_TRAINER
        try:
            return original_run_pilot(trainer, *args, **kwargs)
        except KeyboardInterrupt:
            _INTERRUPTED_TRAINER = trainer
            raise

    def finish_with_interrupt_checkpoint(
        telemetry: WandbTelemetry,
        *,
        exit_code: Optional[int] = None,
    ) -> None:
        global _INTERRUPTED_TRAINER
        trainer = _INTERRUPTED_TRAINER
        if exit_code == 0 and trainer is not None:
            try:
                _checkpoint_after_interrupt(trainer)
            finally:
                _INTERRUPTED_TRAINER = None
        original_finish(telemetry, exit_code=exit_code)

    Stage6Trainer.run_pilot = run_pilot_with_interrupt_capture
    WandbTelemetry.finish = finish_with_interrupt_checkpoint
    try:
        yield
    finally:
        Stage6Trainer.run_pilot = original_run_pilot
        WandbTelemetry.finish = original_finish
        _INTERRUPTED_TRAINER = None


__all__ = [
    "CheckpointExitController",
    "CheckpointExitRequested",
    "checkpoint_exit_request_path",
    "interactive_interrupt_checkpoint",
]
# ^^^ THOG
