# vvv THOG
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HOST_WRAPPERS = (
    REPOSITORY_ROOT / "current_scruffy_train_OWT.sh",
    REPOSITORY_ROOT / "current_dreedle_train_OWT.sh",
)
INTERNAL_IMPLEMENTATIONS = (
    REPOSITORY_ROOT / ".current_scruffy_train_OWT_impl",
    REPOSITORY_ROOT / ".current_dreedle_train_OWT_impl",
)


class OptimizerEntryWrapperTests(unittest.TestCase):
    def test_familiar_host_wrappers_document_optimizer_selection(self) -> None:
        for wrapper in HOST_WRAPPERS:
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn("-y NAME, --optimizer NAME", text)
            self.assertIn("adamw | sgd | sgd_nesterov | adafactor | rmsprop", text)
            self.assertIn("capital -C remains the context length", text)
            self.assertIn("THOG2_SOURCE_OPTIMIZER_TARGET=true", text)

    def test_help_exposes_optimizer_controls_and_existing_wrapper_help(self) -> None:
        for wrapper in HOST_WRAPPERS:
            completed = subprocess.run(
                ["bash", str(wrapper), "-h"],
                cwd=REPOSITORY_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("Optimizer selection:", completed.stdout)
            self.assertIn("adamw | sgd | sgd_nesterov | adafactor | rmsprop", completed.stdout)
            self.assertIn("Model/run:", completed.stdout)
            self.assertIn(str(wrapper), completed.stdout)

    def test_internal_implementations_are_retained_and_shell_valid(self) -> None:
        paths = (*HOST_WRAPPERS, *INTERNAL_IMPLEMENTATIONS, REPOSITORY_ROOT / "optimizer_train_OWT_wrapper.sh")
        for path in paths:
            self.assertTrue(path.is_file(), path)
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_redundant_optimizer_entry_points_are_removed(self) -> None:
        self.assertFalse((REPOSITORY_ROOT / "current_scruffy_train_OWT_optimizer.sh").exists())
        self.assertFalse((REPOSITORY_ROOT / "current_dreedle_train_OWT_optimizer.sh").exists())

    def test_convenience_wrapper_comments_include_optimizer_controls(self) -> None:
        for filename in ("current_scruffy_train_DENSE_OWT.sh", "current_scruffy_train_SHEET_OWT.sh"):
            text = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("-y / --optimizer", text)
            self.assertIn("--optimizer-momentum", text)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
