from __future__ import annotations

import unittest

from sheet.geometry_registry import (
    COMPRESSOR_REGISTRY,
    ELEMENT_TYPE_CURVE,
    ELEMENT_TYPE_SHEET,
    ELEMENT_TYPE_SHEET_SET,
    GEOMETRY_REGISTRY,
    format_geometry_plan,
    parse_option_assignment,
    resolve_geometry_plan,
    validate_resolved_geometry_plan,
)


ORDERS = {
    "o_depth": 32,
    "o_attn_d_model": 64,
    "o_attn_qkv_per_channel": 6,
    "o_attn_out_per_channel": 6,
    "o_mlp_d_model": 64,
    "o_mlp_hidden": 8,
}


class GeometryRegistryPhase1Tests(unittest.TestCase):
    def resolve(self, *, depth=False, elements=(), options=()):
        return resolve_geometry_plan(
            select_depth=depth,
            selected_elements=elements,
            option_assignments=options,
            legacy_orders=ORDERS,
            default_mlp_hidden_group_size=128,
        )

    def test_registry_is_complete_and_contains_implementation_flags(self):
        self.assertEqual(
            set(GEOMETRY_REGISTRY),
            {
                "MLP_UP.MLP_HIDDEN", "MLP_UP.MLP_D_MODEL", "MLP_UP",
                "MLP_DOWN.MLP_HIDDEN", "MLP_DOWN.MLP_D_MODEL", "MLP_DOWN",
                "ATTENTION_QKV.ATTENTION_D_MODEL", "ATTENTION_QKV.ATTENTION_HEAD_CHANNEL", "ATTENTION_QKV",
                "ATTENTION_OUTPUT.ATTENTION_D_MODEL", "ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL", "ATTENTION_OUTPUT",
            },
        )
        self.assertEqual({entry.implied_type for entry in GEOMETRY_REGISTRY.values()}, {ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, ELEMENT_TYPE_SHEET_SET})
        self.assertTrue(all(hasattr(entry, "implemented") for entry in GEOMETRY_REGISTRY.values()))
        self.assertTrue(all(hasattr(entry, "implemented_with_depth") for entry in GEOMETRY_REGISTRY.values()))
        self.assertTrue(all(not entry.implemented for entry in GEOMETRY_REGISTRY.values()))
        self.assertTrue(all(not entry.implemented_with_depth for entry in GEOMETRY_REGISTRY.values()))

    def test_compressor_registry_contains_only_curve_compressors(self):
        self.assertEqual(set(COMPRESSOR_REGISTRY), {"chebyshev", "dct", "haar", "lapped_cosine"})
        for capability in COMPRESSOR_REGISTRY.values():
            self.assertEqual(capability.element_types, (ELEMENT_TYPE_CURVE,))
            self.assertTrue(capability.implemented)

    def test_bare_mlp_is_registered_but_not_implemented(self):
        plan = self.resolve(elements=("MLP_UP",), options=("MLP_UP.compressor=dct",))
        self.assertEqual(plan.selections[0].implied_type, ELEMENT_TYPE_SHEET)
        self.assertFalse(plan.materializer.implemented)
        self.assertIn("cannot be used with a SHEET geometry", plan.materializer.message)
        self.assertIn("no SHEET compressor is currently implemented", plan.materializer.message)

    def test_bare_attention_is_registered_but_not_implemented(self):
        plan = self.resolve(elements=("ATTENTION_QKV",), options=("ATTENTION_QKV.compressor=dct",))
        selection = plan.selections[0]
        self.assertEqual(selection.implied_type, ELEMENT_TYPE_SHEET_SET)
        self.assertEqual(selection.independent_indices, ("QKV_ROLE", "ATTENTION_HEAD"))
        self.assertFalse(plan.materializer.implemented)
        self.assertIn("no SHEET_SET compressor is currently implemented", plan.materializer.message)

    def test_attention_axis_is_a_curve_not_curve_set(self):
        plan = self.resolve(elements=("ATTENTION_QKV.ATTENTION_D_MODEL",), options=("ATTENTION_QKV.compressor=dct",))
        self.assertEqual(plan.selections[0].implied_type, ELEMENT_TYPE_CURVE)
        self.assertIn("QKV_ROLE", plan.selections[0].independent_indices)
        self.assertFalse(plan.materializer.implemented)
        self.assertIn("registered geometry is not currently implemented", plan.materializer.message)

    def test_depth_plus_mlp_axis_is_not_mapped_to_legacy_jpeg_path(self):
        plan = self.resolve(
            depth=True,
            elements=("MLP_UP.MLP_HIDDEN",),
            options=("DEPTH.compressor=chebyshev", "MLP_UP.compressor=dct", "MLP_UP.MLP_HIDDEN.order=8"),
        )
        self.assertEqual(plan.selections[0].implied_type, ELEMENT_TYPE_SHEET)
        self.assertEqual(plan.selections[0].compressed_axes, ("DEPTH", "MLP_HIDDEN"))
        self.assertFalse(plan.materializer.implemented)
        self.assertIsNone(plan.materializer.legacy_geometry_preset)
        self.assertIn("cannot be used with a SHEET geometry", plan.materializer.message)

    def test_jpeg_like_is_rejected_as_a_systematic_compressor(self):
        with self.assertRaisesRegex(ValueError, "not currently implemented as a semantically valid SHEET compressor"):
            self.resolve(
                depth=True,
                elements=("MLP_UP.MLP_HIDDEN",),
                options=("MLP_UP.compressor=jpeg_like",),
            )

    def test_group_size_is_rejected_for_curve_compressors(self):
        with self.assertRaisesRegex(ValueError, "does not accept group_size"):
            self.resolve(
                elements=("MLP_UP.MLP_HIDDEN",),
                options=("MLP_UP.compressor=dct", "MLP_UP.MLP_HIDDEN.group_size=128"),
            )

    def test_depth_only_maps_to_existing_depth_path(self):
        plan = self.resolve(depth=True, options=("DEPTH.compressor=haar", "DEPTH.order=16"))
        self.assertTrue(plan.materializer.implemented)
        self.assertEqual(plan.materializer.legacy_geometry_preset, "depth")
        self.assertEqual(plan.depth_order, 16)

    def test_two_axis_selectors_do_not_accumulate_into_a_sheet(self):
        plan = self.resolve(elements=("MLP_UP.MLP_HIDDEN", "MLP_UP.MLP_D_MODEL"), options=("MLP_UP.compressor=dct",))
        self.assertEqual([selection.implied_type for selection in plan.selections], [ELEMENT_TYPE_CURVE, ELEMENT_TYPE_CURVE])

    def test_complete_and_axis_selector_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping geometry selectors"):
            self.resolve(elements=("MLP_UP", "MLP_UP.MLP_HIDDEN"))

    def test_depth_plus_complete_sheet_is_rejected_as_three_dimensional(self):
        with self.assertRaisesRegex(ValueError, "three-dimensional BLOCK/BLOCK_SET"):
            self.resolve(depth=True, elements=("ATTENTION_QKV",))

    def test_phase1_rejects_different_non_depth_compressors(self):
        with self.assertRaisesRegex(ValueError, "Phase 1 requires"):
            self.resolve(
                elements=("MLP_UP.MLP_HIDDEN", "ATTENTION_QKV.ATTENTION_D_MODEL"),
                options=("MLP_UP.compressor=dct", "ATTENTION_QKV.compressor=haar"),
            )

    def test_option_grammar_splits_property_from_qualified_target(self):
        option = parse_option_assignment("MLP_UP.MLP_HIDDEN.group_size=128")
        self.assertEqual(option.target, "MLP_UP.MLP_HIDDEN")
        self.assertEqual(option.property, "group_size")
        self.assertEqual(option.value, "128")

    def test_option_target_must_be_selected(self):
        with self.assertRaisesRegex(ValueError, "must target a selected element"):
            self.resolve(elements=("MLP_UP.MLP_HIDDEN",), options=("MLP_DOWN.compressor=dct",))

    def test_plan_round_trips_checkpoint_validation(self):
        plan = self.resolve(depth=True, options=("DEPTH.compressor=chebyshev",))
        self.assertEqual(validate_resolved_geometry_plan(plan.to_dict()), plan.to_dict())

    def test_console_report_uses_not_currently_implemented_wording(self):
        plan = self.resolve(elements=("ATTENTION_OUTPUT",), options=("ATTENTION_OUTPUT.compressor=dct",))
        report = format_geometry_plan(plan)
        self.assertIn("implied element type:  SHEET_SET", report)
        self.assertIn("independent instances: ATTENTION_HEAD", report)
        self.assertIn("implementation status: not currently implemented", report)


if __name__ == "__main__":
    unittest.main()
