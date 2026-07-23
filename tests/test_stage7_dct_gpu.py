# vvv THOG
from __future__ import annotations

import unittest

import torch
from sheet.basis import BasisCache, orthonormality_max_error
from sheet.basis_kernel import BASIS_FAMILY_DCT, DCT_BASIS_VERSION
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Stage7DctGpuTests(unittest.TestCase):
    def test_01_dct_basis_cache_builds_gpu_resident_orthonormal_buffers_and_keeps_cpu_and_gpu_entries_separate(self) -> None:
        cache = BasisCache()
        cpu_basis = cache.get(16, 8, runtime_dtype=torch.float32, device="cpu", version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        gpu_basis = cache.get(16, 8, runtime_dtype=torch.float32, device="cuda", version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(cpu_basis.device.type, "cpu")
        self.assertEqual(gpu_basis.device.type, "cuda")
        self.assertEqual(len(cache), 2)
        self.assertLess(orthonormality_max_error(gpu_basis.float()), 1.0e-6)

    def test_02_dct_full_block_gpu_bfloat16_tiny_train_step_records_dct_identity(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(512)
        config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK, basis_family=BASIS_FAMILY_DCT, device="cuda", dtype="bfloat16", max_updates=1, decay_updates=1)
        self.assertEqual(config.basis_version, DCT_BASIS_VERSION)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        metrics = trainer.train_one_update()
        self.assertTrue(torch.isfinite(torch.tensor(metrics["training_loss"])))
        payload = trainer.checkpoint_payload()
        self.assertEqual(payload["compact_identity"]["basis_family"], BASIS_FAMILY_DCT)
        self.assertEqual(payload["compact_identity"]["basis_version"], DCT_BASIS_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
