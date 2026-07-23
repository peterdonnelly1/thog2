# vvv THOG
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from sheet.bases import BASIS_REGISTRY


class WrapperBasisHelpTests(unittest.TestCase):
    def test_all_basis_aware_training_wrapper_help_lists_registry_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        wrappers = []

        for path in sorted(root.glob("*.sh")):
            text = path.read_text(encoding="utf-8")
            if "usage() {" in text and "BASIS_FAMILY=" in text:
                wrappers.append(path)

        self.assertGreaterEqual(len(wrappers), 2)

        for path in wrappers:
            with self.subTest(wrapper=path.name):
                completed = subprocess.run(
                    ["bash", str(path), "-h"],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, msg=completed.stderr)

                help_text = completed.stdout
                self.assertIn("canonical: chebyshev | dct | haar", help_text)
                self.assertIn("auto (recommended), or exact:", help_text)
                self.assertNotIn("registered fixed basis family", help_text)

                for definition in BASIS_REGISTRY.definitions():
                    self.assertIn(definition.family, help_text)
                    self.assertIn(definition.version, help_text)
                    for alias in definition.aliases:
                        self.assertIn(alias, help_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
