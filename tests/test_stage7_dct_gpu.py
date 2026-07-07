# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.basis import BasisCache, orthonormality_max_error
from sheet.basis_kernel import BASIS_FAMILY_DCT, DCT_BASIS_VERSION


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
