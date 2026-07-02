# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

from sheet.run_naming import (
    architecture_output_directory,
    architecture_run_name,
)
from tests.stage4_test_support import stage4_training_config


class Stage4NamingTests(unittest.TestCase):
    def test_s4_12_names_and_paths_are_unambiguous(self) -> None:
        dense = stage4_training_config(model_type="dense")
        sheet = stage4_training_config(model_type="thog2_sheet")
        dense_name = architecture_run_name(dense)
        sheet_name = architecture_run_name(sheet)
        self.assertTrue(dense_name.startswith("DENSE_"))
        self.assertTrue(sheet_name.startswith("THOG2_SHEET_"))
        self.assertIn("_P3_Q8", sheet_name)
        self.assertNotEqual(dense_name, sheet_name)
        dense_path = architecture_output_directory(dense, "runs")
        sheet_path = architecture_output_directory(sheet, "runs")
        self.assertEqual(dense_path, Path("runs") / dense_name)
        self.assertEqual(sheet_path, Path("runs") / sheet_name)
        self.assertNotEqual(dense_path, sheet_path)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
