# vvv THOG
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO

import torch

from .checkpoints import save_payload
from .stage6_trainer import Stage6Trainer
from .wandb_telemetry import WandbTelemetry


_INTERRUPTED_TRAINER: Optional[Any] = None


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
        # Do not call trainer.save_checkpoint here: rank zero reaches telemetry.finish
        # alone, and that method contains a distributed barrier. The payload is the
        # same atomic checkpoint payload but is written without introducing a DDP
        # collective mismatch during shutdown.
        save_payload(trainer.checkpoint_payload(), checkpoint_path)
    except Exception as error:
        message(f"Checkpoint save failed: {error}\n")
        return False
    message(f"Checkpoint saved: {checkpoint_path}\n")
    return True


@contextmanager
def interactive_interrupt_checkpoint() -> Iterator[None]:
    """Route the runner's proven KeyboardInterrupt path through an interactive save choice."""

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
    "interactive_interrupt_checkpoint",
]
# ^^^ THOG
