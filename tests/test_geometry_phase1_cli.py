from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import run_thog2_owt


class GeometryPhase1CliTests(unittest.TestCase):
    def parse(self, *arguments: str):
        return run_thog2_owt.build_parser().parse_args(list(arguments))

    def test_repeatable_element_and_option_arguments(self):
        arguments = self.parse(
            "--model-type", "sheet",
            "--select-element", "MLP_UP.MLP_HIDDEN",
            "--select-element", "ATTENTION_OUTPUT.ATTENTION_D_MODEL",
            "--option", "MLP_UP.compressor=dct",
            "--option", "ATTENTION_OUTPUT.compressor=dct",
        )
        self.assertEqual(len(arguments.select_element), 2)
        self.assertEqual(len(arguments.geometry_options), 2)

    def test_systematic_selection_implies_sheet_model_type(self):
        arguments = self.parse(
            "--select-depth",
            "--option", "DEPTH.compressor=chebyshev",
        )
        plan = run_thog2_owt.geometry_plan_from_arguments(arguments)
        config = run_thog2_owt.config_from_arguments(arguments, geometry_plan=plan)
        self.assertEqual(config.model_type, "sheet")
        self.assertEqual(config.geometry_preset, "depth")
        self.assertIsNotNone(config.resolved_geometry_plan)

    def test_existing_jpeg_path_is_adapted_from_systematic_ui(self):
        arguments = self.parse(
            "--select-depth",
            "--select-element", "MLP_UP.MLP_HIDDEN",
            "--option", "DEPTH.compressor=chebyshev",
            "--option", "MLP_UP.compressor=jpeg_like",
            "--option", "MLP_UP.MLP_HIDDEN.order=8",
            "--option", "MLP_UP.MLP_HIDDEN.group_size=128",
        )
        plan = run_thog2_owt.geometry_plan_from_arguments(arguments)
        config = run_thog2_owt.config_from_arguments(arguments, geometry_plan=plan)
        self.assertEqual(config.geometry_preset, "jpeg_like_v1")
        self.assertEqual(config.o_mlp_hidden, 8)
        self.assertEqual(config.mlp_hidden_group_size, 128)

    def test_valid_phase2_geometry_fails_before_model_allocation(self):
        arguments = self.parse(
            "--select-element", "MLP_UP",
            "--option", "MLP_UP.compressor=dct",
        )
        plan = run_thog2_owt.geometry_plan_from_arguments(arguments)
        with self.assertRaisesRegex(ValueError, "scheduled for Phase 2"):
            run_thog2_owt.config_from_arguments(arguments, geometry_plan=plan)

    def test_explain_geometry_does_not_touch_dataset_or_model(self):
        output = io.StringIO()
        with mock.patch.object(
            run_thog2_owt,
            "build_parser",
            return_value=mock.Mock(
                parse_args=mock.Mock(
                    return_value=self.parse(
                        "--select-element", "ATTENTION_QKV",
                        "--option", "ATTENTION_QKV.compressor=dct",
                        "--explain-geometry",
                    )
                )
            ),
        ), mock.patch.object(run_thog2_owt, "validate_dataset") as validate_dataset, contextlib.redirect_stdout(output):
            self.assertEqual(run_thog2_owt.main(), 0)
        validate_dataset.assert_not_called()
        self.assertIn("SHEET_SET", output.getvalue())
        self.assertIn("not implemented", output.getvalue())

    def test_non_default_legacy_geometry_cannot_be_mixed(self):
        arguments = self.parse(
            "--model-type", "sheet",
            "--geometry-preset", "mlp_block",
            "--select-depth",
        )
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            run_thog2_owt.geometry_plan_from_arguments(arguments)


if __name__ == "__main__":
    unittest.main()
