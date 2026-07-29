# vvv THOG
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import sheet.interactive_interrupt as interrupt_module
from sheet.interactive_interrupt import (
    _checkpoint_after_interrupt,
    _prompt_save_checkpoint,
    interactive_interrupt_checkpoint,
)
from sheet.stage6_trainer import Stage6Trainer
from sheet.wandb_telemetry import WandbTelemetry


class _FakeDistributed:
    is_primary = True


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = []

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_grad_calls.append(set_to_none)


class _FakeTrainer:
    def __init__(self, root: Path, completed_updates: int = 7) -> None:
        self.distributed = _FakeDistributed()
        self.state = SimpleNamespace(completed_updates=completed_updates)
        self.config = SimpleNamespace(out_dir=str(root))
        self.optimizer = _FakeOptimizer()
        self.device = torch.device("cpu")

    def checkpoint_payload(self):
        return {"completed_updates": int(self.state.completed_updates)}


class InteractiveInterruptTests(unittest.TestCase):
    def test_prompt_rejects_unknown_answer_then_accepts_yes(self) -> None:
        reader = io.StringIO("maybe\ny\n")
        writer = io.StringIO()

        self.assertTrue(
            _prompt_save_checkpoint(17, reader=reader, writer=writer)
        )
        output = writer.getvalue()
        self.assertEqual(output.count("Save checkpoint at completed update 17"), 2)
        self.assertIn("Please answer y or n.", output)

    def test_no_skips_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = _FakeTrainer(Path(temporary))
            messages = []

            saved = _checkpoint_after_interrupt(
                trainer,
                prompt=lambda completed_updates: False,
                message=messages.append,
            )

            self.assertFalse(saved)
            self.assertEqual(trainer.optimizer.zero_grad_calls, [])
            self.assertIn("Checkpoint skipped", "".join(messages))

    def test_yes_saves_current_completed_update_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = _FakeTrainer(Path(temporary), completed_updates=23)
            messages = []
            saved_payloads = []

            def save_payload(payload, path):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"checkpoint")
                saved_payloads.append((payload, target))
                return target

            with mock.patch.object(interrupt_module, "save_payload", side_effect=save_payload):
                saved = _checkpoint_after_interrupt(
                    trainer,
                    prompt=lambda completed_updates: completed_updates == 23,
                    message=messages.append,
                )

            expected = Path(temporary) / "ckpt.pt"
            self.assertTrue(saved)
            self.assertEqual(saved_payloads, [({"completed_updates": 23}, expected)])
            self.assertEqual(trainer.optimizer.zero_grad_calls, [True])
            self.assertEqual(expected.read_bytes(), b"checkpoint")
            self.assertIn("Checkpoint saved", "".join(messages))

    def test_keyboard_interrupt_is_captured_then_finish_prompts_before_sink_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = _FakeTrainer(Path(temporary), completed_updates=31)
            telemetry = object()
            order = []

            def interrupted_run_pilot(_trainer, *args, **kwargs):
                del args, kwargs
                self.assertIs(_trainer, trainer)
                raise KeyboardInterrupt

            def original_finish(_telemetry, *, exit_code=None):
                self.assertIs(_telemetry, telemetry)
                order.append(("finish", exit_code))

            with mock.patch.object(Stage6Trainer, "run_pilot", interrupted_run_pilot):
                with mock.patch.object(WandbTelemetry, "finish", original_finish):
                    with mock.patch.object(
                        interrupt_module,
                        "_checkpoint_after_interrupt",
                        side_effect=lambda value: order.append(("checkpoint", value)),
                    ):
                        with interactive_interrupt_checkpoint():
                            with self.assertRaises(KeyboardInterrupt):
                                Stage6Trainer.run_pilot(trainer)
                            WandbTelemetry.finish(telemetry, exit_code=0)

            self.assertEqual(order, [("checkpoint", trainer), ("finish", 0)])

    def test_non_interrupt_finish_does_not_prompt(self) -> None:
        trainer = object()
        telemetry = object()
        finish_calls = []

        def completed_run_pilot(_trainer, *args, **kwargs):
            del args, kwargs
            self.assertIs(_trainer, trainer)
            return {"status": "completed"}

        def original_finish(_telemetry, *, exit_code=None):
            self.assertIs(_telemetry, telemetry)
            finish_calls.append(exit_code)

        with mock.patch.object(Stage6Trainer, "run_pilot", completed_run_pilot):
            with mock.patch.object(WandbTelemetry, "finish", original_finish):
                with mock.patch.object(interrupt_module, "_checkpoint_after_interrupt") as checkpoint:
                    with interactive_interrupt_checkpoint():
                        self.assertEqual(Stage6Trainer.run_pilot(trainer), {"status": "completed"})
                        WandbTelemetry.finish(telemetry, exit_code=None)

        checkpoint.assert_not_called()
        self.assertEqual(finish_calls, [None])


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
