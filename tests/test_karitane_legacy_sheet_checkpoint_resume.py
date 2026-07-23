from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from run_thog2_owt import OwtTrainer
from sheet.basis import BASIS_VERSION
from sheet.compact_identity import (
    BASIS_FAMILY_CHEBYSHEV,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
)
from sheet.training_config import TrainingConfig


class KaritaneLegacySheetCheckpointResumeTests(unittest.TestCase):
    def config(self, out_dir: Path, *, geometry_preset: str) -> TrainingConfig:
        return TrainingConfig(
            model_type="thog2_sheet",
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
            depth_order=2,
            base_row_order=4,
            residual_init_policy="depth_scaled",
            residual_init_depth_source="dof_implied_depth",
            basis_version=BASIS_VERSION,
            geometry_preset=geometry_preset,
            basis_family=BASIS_FAMILY_CHEBYSHEV,
            checkpoint_segment_size=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_updates=3,
            learning_rate=6.0e-4,
            min_learning_rate=6.0e-5,
            warmup_updates=0,
            decay_updates=3,
            decay_learning_rate=True,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
            nonfinite_update_policy="skip",
            max_nonfinite_update_skips=10,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=1,
            model_seed=1337,
            data_seed=7331,
            device="cpu",
            dtype="float32",
            out_dir=str(out_dir),
        )

    def schema_one_payload(self, trainer: OwtTrainer):
        payload = trainer.checkpoint_payload()
        payload["schema_version"] = 1
        payload.pop("compact_identity", None)

        old_fields = (
            "model_type",
            "block_size",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "bias",
            "depth_order",
            "base_row_order",
            "residual_init_policy",
            "residual_init_depth_source",
            "residual_init_depth_value",
            "basis_version",
            "row_order_scaling_rule",
        )
        payload["compatibility_signature"] = {
            name: payload["compatibility_signature"][name]
            for name in old_fields
        }

        old_training_fields = {
            "model_type",
            "block_size",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "bias",
            "depth_order",
            "base_row_order",
            "residual_init_policy",
            "residual_init_depth_source",
            "residual_init_depth_value",
            "basis_version",
            "row_order_scaling_rule",
            "checkpoint_segment_size",
            "batch_size",
            "gradient_accumulation_steps",
            "max_updates",
            "learning_rate",
            "min_learning_rate",
            "warmup_updates",
            "decay_updates",
            "decay_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
            "eval_interval",
            "eval_batches",
            "checkpoint_interval",
            "log_interval",
            "model_seed",
            "data_seed",
            "device",
            "dtype",
            "out_dir",
        }
        payload["trainer_config"] = {
            name: value
            for name, value in payload["trainer_config"].items()
            if name in old_training_fields
        }
        payload["trainer_state"].pop("skipped_nonfinite_updates", None)
        payload["trainer_state"].pop("failed_update_attempts", None)
        return payload

    def test_schema_one_sheet_checkpoint_resumes_as_legacy_sheet_col_with_safe_recovery_defaults(self):
        train_tokens = torch.arange(512, dtype=torch.long) % 32
        validation_tokens = torch.arange(256, dtype=torch.long) % 32

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(
                root / "checkpoint",
                geometry_preset=GEOMETRY_PRESET_LEGACY_SHEET_COL,
            )
            source = OwtTrainer(config, train_tokens, validation_tokens)
            source.train_one_update()
            payload = self.schema_one_payload(source)
            checkpoint = root / "schema_one.pt"
            torch.save(payload, checkpoint)
            expected_model_state = {
                name: tensor.detach().clone()
                for name, tensor in source.raw_model.state_dict().items()
            }
            source.close()

            resumed = OwtTrainer.from_checkpoint(
                checkpoint,
                train_tokens,
                validation_tokens,
                expected_config=config,
                overrides={
                    "max_updates": 3,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "checkpoint_interval": 1,
                    "log_interval": 1,
                    "out_dir": str(root / "checkpoint"),
                    "device": "cpu",
                    "dtype": "float32",
                    "nonfinite_update_policy": "skip",
                    "max_nonfinite_update_skips": 10,
                },
            )
            self.assertEqual(resumed.state.completed_updates, 1)
            self.assertEqual(resumed.state.skipped_nonfinite_updates, 0)
            self.assertEqual(resumed.state.failed_update_attempts, 0)
            self.assertEqual(
                resumed.events[-1].name,
                "legacy_sheet_checkpoint_resumed",
            )
            for name, tensor in resumed.raw_model.state_dict().items():
                torch.testing.assert_close(
                    tensor,
                    expected_model_state[name],
                    rtol=0.0,
                    atol=0.0,
                )
            resumed.close()

    def test_schema_one_sheet_checkpoint_rejects_nonlegacy_geometry(self):
        train_tokens = torch.arange(512, dtype=torch.long) % 32
        validation_tokens = torch.arange(256, dtype=torch.long) % 32

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_config = self.config(
                root / "checkpoint",
                geometry_preset=GEOMETRY_PRESET_LEGACY_SHEET_COL,
            )
            source = OwtTrainer(
                legacy_config,
                train_tokens,
                validation_tokens,
            )
            checkpoint = root / "schema_one.pt"
            torch.save(self.schema_one_payload(source), checkpoint)
            source.close()

            wrong_config = self.config(
                root / "checkpoint",
                geometry_preset=GEOMETRY_PRESET_DEPTH,
            )
            with self.assertRaisesRegex(
                ValueError,
                "only as legacy_sheet_col",
            ):
                OwtTrainer.from_checkpoint(
                    checkpoint,
                    train_tokens,
                    validation_tokens,
                    expected_config=wrong_config,
                )


if __name__ == "__main__":
    unittest.main()
