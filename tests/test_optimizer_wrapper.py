# vvv THOG
from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_HELPER = REPOSITORY_ROOT / "optimizer_train_OWT_wrapper.sh"


class OptimizerWrapperTests(unittest.TestCase):
    def run_wrapper(self, *arguments: str):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "output.json"
            target_path = root / "target.sh"
            target_path.write_text(
                "#!/bin/bash\n"
                "python - \"$OUTPUT_PATH\" \"$THOG2_OPTIMIZER\" "
                "\"$THOG2_OPTIMIZER_MOMENTUM\" \"$@\" <<'PY'\n"
                "import json\n"
                "import sys\n"
                "path, optimizer, momentum, *arguments = sys.argv[1:]\n"
                "with open(path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump({'optimizer': optimizer, 'momentum': momentum, 'arguments': arguments}, handle)\n"
                "PY\n",
                encoding="utf-8",
            )
            target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)
            environment = dict(os.environ)
            environment["OUTPUT_PATH"] = str(output_path)
            subprocess.run(
                ["bash", str(WRAPPER_HELPER), str(target_path), *arguments],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            return json.loads(output_path.read_text(encoding="utf-8"))

    def test_adamw_defaults_preserve_existing_learning_rates(self) -> None:
        result = self.run_wrapper("-n", "1")
        self.assertEqual(result["optimizer"], "adamw")
        self.assertEqual(result["momentum"], "0.9")
        self.assertEqual(result["arguments"], ["-n", "1", "-c", "60", "-f", "06"])

    def test_sgd_gets_optimizer_specific_defaults_and_artifact_suffix(self) -> None:
        result = self.run_wrapper("-y", "sgd", "-n", "1")
        self.assertEqual(result["optimizer"], "sgd")
        self.assertEqual(
            result["arguments"],
            [
                "-n",
                "1",
                "-c",
                "1000",
                "-f",
                "100",
                "--",
                "--artifact-suffix",
                "OPT_SGD",
            ],
        )

    def test_explicit_learning_rates_override_optimizer_defaults(self) -> None:
        result = self.run_wrapper(
            "--optimizer=sgd_nesterov",
            "-c",
            "250",
            "-f",
            "25",
        )
        self.assertEqual(result["optimizer"], "sgd_nesterov")
        self.assertIn("250", result["arguments"])
        self.assertIn("25", result["arguments"])
        self.assertNotIn("1000", result["arguments"])

    def test_existing_artifact_suffix_is_extended(self) -> None:
        result = self.run_wrapper(
            "-y",
            "rmsprop",
            "--",
            "--artifact-suffix",
            "CONTROL",
        )
        self.assertEqual(
            result["arguments"][-3:],
            ["--", "--artifact-suffix", "CONTROL_OPT_RMSPROP"],
        )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
