# vvv THOG
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sheet.interactive_interrupt import (
    _checkpoint_after_interrupt,
    _prompt_save_checkpoint,
)


class _FakeDistributed:
    is_primary = True

    @staticmethod
    def all_gather_object(value):
        return [value]


class _FakeTrainer:
    def __init__(self, root: Path, completed_updates: int = 7) -> None:
        self.distributed = _FakeDistributed()
        self.state = SimpleNamespace(completed_updates=completed_updates)
        self.config = SimpleNamespace(out_dir=str(root))
        self.saved_paths = []

    def save_checkpoint(self, path: Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
        self.saved_paths.append(target)
        return target


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
            self.assertEqual(trainer.saved_paths, [])
            self.assertIn("Checkpoint skipped", "".join(messages))

    def test_yes_saves_current_completed_update_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = _FakeTrainer(Path(temporary), completed_updates=23)
            messages = []

            saved = _checkpoint_after_interrupt(
                trainer,
                prompt=lambda completed_updates: completed_updates == 23,
                message=messages.append,
            )

            expected = Path(temporary) / "ckpt.pt"
            self.assertTrue(saved)
            self.assertEqual(trainer.saved_paths, [expected])
            self.assertEqual(expected.read_bytes(), b"checkpoint")
            self.assertIn("Checkpoint saved", "".join(messages))


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
