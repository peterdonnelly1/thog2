# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

from sheet.run_naming import (
    artifact_paths,
    bounded_filename,
    build_artifact_name,
    truncate_component,
)


class CanonicalNamingTests(unittest.TestCase):
    def test_s6_23a_prefixes_and_core_grammar_match_thog(self) -> None:
        dense = build_artifact_name(
            model_type="dense",
            host_label="scruffy",
            run_name="AKAROA",
            dataset_name="openwebtext",
            n_layer=72,
            n_head=12,
            n_embd=768,
            block_size=256,
            batch_size=12,
            gradient_accumulation_steps=160,
            max_iters=100,
        )
        sheet = build_artifact_name(
            model_type="thog2_sheet",
            host_label="scruffy",
            run_name="AKAROA",
            dataset_name="openwebtext",
            n_layer=72,
            n_head=12,
            n_embd=768,
            block_size=256,
            batch_size=12,
            gradient_accumulation_steps=160,
            max_iters=100,
            depth_order=16,
            base_row_order=64,
        )
        self.assertEqual(
            dense,
            "DENSE2_scruffy__AKAROA__owt__l_72_h_12_d_768_ctx_256__b_12_ga_160__steps_100",
        )
        self.assertEqual(
            sheet,
            "SHEET_scruffy__AKAROA__owt__l_72_h_12_d_768_ctx_256_p_16_q_64__b_12_ga_160__steps_100",
        )
        self.assertNotIn("SHEET__scruffy", sheet)

    def test_s6_23b_paths_use_thog_style_directories_and_safe_filenames(self) -> None:
        name = "SHEET_scruffy__AKAROA__owt__l_72_h_12_d_768_ctx_256_p_16_q_64__b_12_ga_160__steps_100"
        paths = artifact_paths(
            name,
            checkpoint_root=Path("checkpoints"),
            log_root=Path("logs"),
            result_root=Path("results"),
            log_timestamp="20260703_120000",
        )
        self.assertEqual(paths["checkpoint_path"], Path("checkpoints") / name / "ckpt.pt")
        self.assertEqual(paths["result_path"], Path("results") / name / "result.json")
        self.assertEqual(paths["log_path"].parent, Path("logs") / name)
        self.assertLessEqual(len(paths["log_path"].name), 255)

    def test_s6_23c_bounded_truncation_is_deterministic_and_collision_resistant(self) -> None:
        base = "SHEET_scruffy__" + "A" * 400 + "__steps_100000"
        first = truncate_component(base, max_length=120)
        second = truncate_component(base, max_length=120)
        different = truncate_component(base + "x", max_length=120)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 120)
        self.assertTrue(first.startswith("SHEET_scruffy__"))
        self.assertIn("__TRUNC_", first)
        self.assertTrue(first.endswith("__steps_100000"))
        filename = bounded_filename(first, "_train_20260703_120000.log")
        self.assertLessEqual(len(filename), 255)

    def test_s6_23d_dense_rejects_sheet_only_name_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include SHEET orders"):
            build_artifact_name(
                model_type="dense",
                host_label="scruffy",
                run_name="TEST",
                dataset_name="openwebtext",
                n_layer=2,
                n_head=2,
                n_embd=8,
                block_size=8,
                batch_size=1,
                gradient_accumulation_steps=1,
                max_iters=2,
                depth_order=2,
                base_row_order=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
