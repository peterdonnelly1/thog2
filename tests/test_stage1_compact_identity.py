# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_CURVE,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_CONVENTIONAL,
    GEOMETRY_PRESET_CURVE,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_CURVE,
    MLP_GEOMETRY_LEGACY_SHEET_COL,
    MLP_GEOMETRY_MLP_BLOCK,
    resolve_compact_selectors,
)
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage1CompactIdentityTests(unittest.TestCase):
    def test_preset_expansion_is_central_and_deterministic(self) -> None:
        cases = (
            (
                {},
                GEOMETRY_PRESET_LEGACY_SHEET_COL,
                ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
                MLP_GEOMETRY_LEGACY_SHEET_COL,
                BASIS_FAMILY_CHEBYSHEV,
            ),
            (
                {"geometry_preset": "legacy_sheet_col"},
                GEOMETRY_PRESET_LEGACY_SHEET_COL,
                ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
                MLP_GEOMETRY_LEGACY_SHEET_COL,
                BASIS_FAMILY_CHEBYSHEV,
            ),
            (
                {"geometry_preset": "curve"},
                GEOMETRY_PRESET_CURVE,
                ATTENTION_GEOMETRY_CURVE,
                MLP_GEOMETRY_CURVE,
                BASIS_FAMILY_CHEBYSHEV,
            ),
            (
                {"geometry_preset": "mlp_block"},
                GEOMETRY_PRESET_MLP_BLOCK,
                ATTENTION_GEOMETRY_CURVE,
                MLP_GEOMETRY_MLP_BLOCK,
                BASIS_FAMILY_CHEBYSHEV,
            ),
            (
                {"geometry_preset": "block"},
                GEOMETRY_PRESET_BLOCK,
                ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
                MLP_GEOMETRY_MLP_BLOCK,
                BASIS_FAMILY_CHEBYSHEV,
            ),
        )
        for request, preset, attention, mlp, basis in cases:
            with self.subTest(request=request):
                resolved = resolve_compact_selectors(**request)
                self.assertEqual(resolved.geometry_preset, preset)
                self.assertEqual(resolved.attention_geometry, attention)
                self.assertEqual(resolved.mlp_geometry, mlp)
                self.assertEqual(resolved.basis_family, basis)

    def test_explicit_module_overrides_are_resolved_before_stage_support_validation(self) -> None:
        resolved = resolve_compact_selectors(
            geometry_preset="curve",
            mlp_geometry="mlp_block",
        )
        self.assertEqual(resolved.attention_geometry, ATTENTION_GEOMETRY_CURVE)
        self.assertEqual(resolved.mlp_geometry, MLP_GEOMETRY_MLP_BLOCK)

        resolved = resolve_compact_selectors(
            geometry_preset="mlp_block",
            attention_geometry="head_aware_block",
            basis_family="dct",
        )
        self.assertEqual(resolved.attention_geometry, ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK)
        self.assertEqual(resolved.mlp_geometry, MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(resolved.basis_family, BASIS_FAMILY_DCT)

    def test_invalid_identity_values_and_stage4_unsupported_materializations_fail(self) -> None:
        invalid_requests = (
            {"geometry_preset": "tesseract"},
            {"attention_geometry": "qkv_role_basis"},
            {"mlp_geometry": "sheet_flat"},
            {"basis_family": "fourierish"},
        )
        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    resolve_compact_selectors(**request)

        stage4_training_config(geometry_preset="curve")
        stage4_training_config(attention_geometry="curve", mlp_geometry="curve")
        unsupported_training_configs = (
            {"geometry_preset": "legacy_sheet_col", "attention_geometry": "curve"},
            {"geometry_preset": "legacy_sheet_col", "mlp_geometry": "mlp_block"},
            {"geometry_preset": "legacy_sheet_col", "basis_family": "dct"},
            {"geometry_preset": "mlp_block"},
            {"geometry_preset": "block"},
        )
        for overrides in unsupported_training_configs:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "Stage 4 supports only"):
                    stage4_training_config(**overrides)

    def test_dense_rejects_compact_fields_except_conventional(self) -> None:
        TrainingConfig(model_type="dense")
        TrainingConfig(model_type="dense", geometry_preset=GEOMETRY_PRESET_CONVENTIONAL)
        rejected = (
            {"geometry_preset": "legacy_sheet_col"},
            {"geometry_preset": "curve"},
            {"attention_geometry": "curve"},
            {"mlp_geometry": "mlp_block"},
            {"basis_family": "chebyshev"},
        )
        for overrides in rejected:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    TrainingConfig(model_type="dense", **overrides)

    def test_training_config_metadata_is_visible_and_head_aware(self) -> None:
        config = stage4_training_config()
        identity = config.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_LEGACY_SHEET_COL)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_LEGACY_SHEET_COL)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_LEGACY_SHEET_COL)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["n_head"], config.n_head)
        self.assertEqual(identity["head_dim"], config.n_embd // config.n_head)
        self.assertEqual(identity["depth_order"], config.depth_order)
        self.assertEqual(identity["base_row_order"], config.base_row_order)
        self.assertEqual(identity["qkv_role_ranges"]["query"], (0, config.n_embd))
        self.assertEqual(
            identity["attention_output_input_head_column_ranges"],
            tuple((head * identity["head_dim"], (head + 1) * identity["head_dim"]) for head in range(config.n_head)),
        )
        signature = config.compatibility_signature()
        for key in ("geometry_preset", "attention_geometry", "mlp_geometry", "basis_family"):
            self.assertIn(key, signature)

    def test_checkpoint_payload_records_compact_identity_and_resume_matches(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        trainer = Stage4Trainer(stage4_training_config(max_updates=1), train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stage1_identity.pt"
            trainer.save_checkpoint(checkpoint_path)
            payload = load_payload(checkpoint_path)
            expected_identity = trainer.config.compact_identity_metadata()
            self.assertEqual(payload["compact_identity"], expected_identity)
            self.assertEqual(payload["parameter_report"]["compact_identity"], expected_identity)
            resumed = Stage4Trainer.from_checkpoint(checkpoint_path, train_tokens, validation_tokens)
        self.assertEqual(resumed.config.compact_identity_metadata(), expected_identity)

    def test_checkpoint_resume_hard_fails_on_missing_or_mismatched_compact_identity(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        trainer = Stage4Trainer(stage4_training_config(max_updates=1), train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stage1_identity.pt"
            trainer.save_checkpoint(checkpoint_path)
            payload = load_payload(checkpoint_path)
            validate_compatibility(payload, trainer.config)

            mutations = []
            missing_identity = deepcopy(payload)
            missing_identity.pop("compact_identity")
            mutations.append(missing_identity)

            for key, value in (
                ("geometry_preset", "curve"),
                ("attention_geometry", "curve"),
                ("mlp_geometry", "mlp_block"),
                ("basis_family", "dct"),
                ("basis_version", "chebyshev_first_kind_qr_v999"),
                ("n_head", 999),
                ("n_embd", 999),
                ("depth_order", 999),
                ("base_row_order", 999),
            ):
                mutated = deepcopy(payload)
                mutated["compact_identity"][key] = value
                mutations.append(mutated)

            for index, mutated in enumerate(mutations):
                with self.subTest(index=index):
                    torch.save(mutated, checkpoint_path)
                    with self.assertRaisesRegex(ValueError, "compact_identity"):
                        Stage4Trainer.from_checkpoint(checkpoint_path, train_tokens, validation_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
