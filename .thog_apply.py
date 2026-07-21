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

OLD_FAKE_PYTHON_BLOCK = """if len(sys.argv) >= 2 and sys.argv[1] == '-c':
    payload = json.load(sys.stdin)
    code = sys.argv[2]
"""

NEW_FAKE_PYTHON_BLOCK = """if len(sys.argv) >= 2 and sys.argv[1] == '-c':
    code = sys.argv[2]
    if 'basis_artifact_tag_for_family' in code:
        family = sys.argv[3]
        aliases = {
            'cheby': 'chebyshev',
            'chebyshev_first_kind_qr': 'chebyshev',
            'dct_ii': 'dct',
            'dct_ii_orthonormal': 'dct',
            'balanced_haar': 'haar',
            'haar_balanced': 'haar',
        }
        family = aliases.get(family, family)
        print({'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}[family])
        raise SystemExit(0)
    payload = json.load(sys.stdin)
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

    fake_python_replacements = 0
    for path in sorted(Path("tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        count = text.count(OLD_FAKE_PYTHON_BLOCK)
        if count == 0:
            continue
        path.write_text(
            text.replace(OLD_FAKE_PYTHON_BLOCK, NEW_FAKE_PYTHON_BLOCK),
            encoding="utf-8",
        )
        fake_python_replacements += count

    if fake_python_replacements != 2:
        raise RuntimeError(
            "expected to update exactly two wrapper fake-Python blocks; "
            f"updated {fake_python_replacements}"
        )

    Path("tests/test_wrapper_basis_help.py").write_text(TEST_CONTENT, encoding="utf-8")

    print("Updated wrapper basis help:")
    for path in wrapper_paths:
        print(f"  {path}")
    print("Updated wrapper fake-Python test harnesses: 2")
    print("  tests/test_wrapper_basis_help.py")


if __name__ == "__main__":
    main()
