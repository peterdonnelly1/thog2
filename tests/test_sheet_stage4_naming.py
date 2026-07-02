# vvv THOG
from __future__ import annotations

import unittest

from sheet.run_naming import architecture_run_name
from tests.stage4_test_support import stage4_training_config


class Stage4NamingTests(unittest.TestCase):
    def test_s4_12_names_are_unambiguous(self) -> None:
        dense_name = architecture_run_name(
            stage4_training_config(model_type="dense")
        )
        sheet_name = architecture_run_name(
            stage4_training_config(model_type="thog2_sheet")
        )
        self.assertTrue(dense_name.startswith("DENSE_"))
        self.assertTrue(sheet_name.startswith("THOG2_SHEET_"))
        self.assertIn("_P3_Q8", sheet_name)
        self.assertNotEqual(dense_name, sheet_name)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
