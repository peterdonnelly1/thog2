# vvv THOG
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_NAMES = (
    "train_OWT.sh",
    "current_scruffy_train_OWT.sh",
    "current_dreedle_train_OWT.sh",
)


class LayerDropoutWrapperTests(unittest.TestCase):
    def _wrapper_path(self, name: str) -> Path:
        return REPOSITORY_ROOT / name

    def test_wrappers_are_valid_bash(self) -> None:
        for name in WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                subprocess.run(
                    ["bash", "-n", str(self._wrapper_path(name))],
                    check=True,
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                )

    def test_short_layer_dropout_controls_are_native_getopts(self) -> None:
        for name in WRAPPER_NAMES:
            text = self._wrapper_path(name).read_text(encoding="utf-8")
            with self.subTest(wrapper=name):
                self.assertIn("L:s:M:H", text)
                self.assertIn('s) LAYER_DROPOUT_STRATUM_SIZE="$OPTARG"', text)
                self.assertIn('M) LAYER_DROPOUT_ACTIVE_PER_STRATUM="$OPTARG"', text)

    def test_short_controls_and_long_resample_parse_before_help(self) -> None:
        for name in WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                completed = subprocess.run(
                    [
                        "bash",
                        str(self._wrapper_path(name)),
                        "-s",
                        "4",
                        "-M",
                        "2",
                        "--layer-dropout-resample-steps",
                        "10",
                        "-h",
                    ],
                    check=False,
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("-s STRATUM_SIZE=", completed.stdout)
                self.assertIn("-M N_ACTIVE_PER_STRATUM=", completed.stdout)
                self.assertIn("--layer-dropout-resample-steps", completed.stdout)

    def test_long_forms_and_forwarding_are_present(self) -> None:
        required = (
            "--layer-dropout-stratum-size",
            "--layer-dropout-active-per-stratum",
            "--layer-dropout-resample-steps",
            "optional_args+=(--layer-dropout-stratum-size",
            "optional_args+=(--layer-dropout-active-per-stratum",
            "optional_args+=(--layer-dropout-resample-steps",
        )
        for name in WRAPPER_NAMES:
            text = self._wrapper_path(name).read_text(encoding="utf-8")
            with self.subTest(wrapper=name):
                for fragment in required:
                    self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
