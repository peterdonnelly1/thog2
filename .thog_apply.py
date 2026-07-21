from __future__ import annotations

from pathlib import Path


OLD_HELP = """Compact geometry:
  -B BASIS_FAMILY=${BASIS_FAMILY}                   registered fixed basis family
  -v BASIS_VERSION=${BASIS_VERSION}
"""

NEW_HELP = """Compact geometry:
  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar
                                                    Chebyshev aliases: cheby | chebyshev_first_kind_qr
                                                    DCT aliases: dct_ii | dct_ii_orthonormal
                                                    Haar aliases: balanced_haar | haar_balanced
  -v BASIS_VERSION=${BASIS_VERSION}                 auto (recommended), or exact:
                                                    chebyshev_first_kind_qr_v1
                                                    dct_ii_orthonormal_v1
                                                    haar_balanced_binary_orthonormal_v1
"""

TEST_CONTENT = '''# vvv THOG
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
'''


def main() -> None:
    wrapper_paths = []

    for path in sorted(Path(".").glob("*.sh")):
        text = path.read_text(encoding="utf-8")
        if "usage() {" not in text or "BASIS_FAMILY=" not in text:
            continue

        count = text.count(OLD_HELP)
        if count != 1:
            raise RuntimeError(f"{path}: expected exactly one old basis help block; found {count}")

        path.write_text(text.replace(OLD_HELP, NEW_HELP, 1), encoding="utf-8")
        wrapper_paths.append(path)

    required = {
        "current_scruffy_train_OWT.sh",
        "current_dreedle_train_OWT.sh",
    }
    found = {path.name for path in wrapper_paths}
    missing = required - found
    if missing:
        raise RuntimeError(f"required wrappers not updated: {sorted(missing)}")

    Path("tests/test_wrapper_basis_help.py").write_text(TEST_CONTENT, encoding="utf-8")

    print("Updated wrapper basis help:")
    for path in wrapper_paths:
        print(f"  {path}")
    print("  tests/test_wrapper_basis_help.py")


if __name__ == "__main__":
    main()
