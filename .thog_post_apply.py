from __future__ import annotations

from pathlib import Path

path = Path("sheet/training_config.py")
text = path.read_text(encoding="utf-8")
old = """            canonical_basis_family = normalize_registered_basis_family(self.basis_family or \"chebyshev\")
            if canonical_basis_family == BASIS_FAMILY_LAPPED_COSINE and self.basis_version in (\"auto\", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
"""
new = """            # vvv THOG preserve the established compact-identity error contract for unknown basis names
            try:
                canonical_basis_family = normalize_registered_basis_family(self.basis_family or \"chebyshev\")
            except ValueError:
                canonical_basis_family = None
            # ^^^ THOG
            if canonical_basis_family == BASIS_FAMILY_LAPPED_COSINE and self.basis_version in (\"auto\", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
"""
if text.count(old) != 1:
    raise RuntimeError(f"expected one training_config validation target, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("preserved established invalid-basis validation path")