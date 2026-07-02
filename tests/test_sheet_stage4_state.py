# vvv THOG
from __future__ import annotations

import unittest

from tests.stage4_test_support import stage4_model


class Stage4StateTests(unittest.TestCase):
    def test_s4_06_and_s4_13_state_is_compact(self) -> None:
        model = stage4_model(checkpoint_segment_size=2)
        self.assertEqual(model.compact_state_violations(), ())
        self.assertFalse(
            any("generated" in key for key in model.state_dict())
        )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
