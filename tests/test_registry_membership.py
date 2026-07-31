# vvv THOG
from __future__ import annotations

import unittest

from sheet.bases import BASIS_REGISTRY
from sheet.recurrence_generators import RECURRENCE_GENERATOR_REGISTRY


class RegistryMembershipTests(unittest.TestCase):
    def test_unknown_basis_membership_is_false(self) -> None:
        self.assertNotIn("not_a_basis", BASIS_REGISTRY)

    def test_unknown_recurrence_generator_membership_is_false(self) -> None:
        self.assertNotIn("not_a_generator", RECURRENCE_GENERATOR_REGISTRY)

    def test_invalid_membership_key_is_false(self) -> None:
        self.assertNotIn(None, BASIS_REGISTRY)
        self.assertNotIn(None, RECURRENCE_GENERATOR_REGISTRY)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
