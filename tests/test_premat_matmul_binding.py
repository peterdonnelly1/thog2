# vvv THOG both PREMAT providers must preserve native matmul checkpoint metadata and gradients
from copy import deepcopy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from sheet.depth_numerical_policy import effective_depth_policy
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.premat_cpu import FAMILY_NAMES


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16, torch.float64))
@pytest.mark.parametrize("family", ("O", "QKV"))
@pytest.mark.parametrize("learned_depth", (False, True))
@pytest.mark.parametrize("original_hit,replay_hit", ((True, False), (False, True), (True, True)))
def test_gpu_premat_matmul_mixed_checkpoint_uses_native_saved_views(device, dtype, family, learned_depth, original_hit, replay_hit):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA hardware required")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU has no BF16 support")
    torch.manual_seed(812)
    config = SheetGeometryConfig(n_layer=4, n_embd=8, n_head=2, depth_order=3, base_row_order=1)
    trajectory = DepthTrajectory(config, depth_compress_layer_norm_and_bias=False).to(device=device, dtype=dtype)
    trajectory.depth_materialisation_matmul = True
    trajectory.depth_basis.requires_grad_(learned_depth)
    reference = deepcopy(trajectory)
    names = FAMILY_NAMES[family]

    def native(owner):
        row = owner._depth_row(2, owner.coefficients[names[0]])
        return torch.cat([owner.coefficients[name].reshape(-1, 3).matmul(row).reshape(owner.coefficients[name].shape[:2]) for name in names])

    with torch.no_grad():
        cached = native(trajectory)
    calls = [0]

    def forward(value):
        hit = original_hit if calls[0] == 0 else replay_hit
        calls[0] += 1
        weight = trajectory.attach_prematerialized(names, 2, cached) if hit else native(trajectory)
        return torch.nn.functional.linear(value, weight)

    value = torch.randn(2, 8, device=device, dtype=dtype, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    actual = checkpoint(forward, value, use_reentrant=False)
    expected = torch.nn.functional.linear(reference_value, native(reference))
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(value.grad, reference_value.grad)
    for name in names:
        torch.testing.assert_close(trajectory.coefficients[name].grad, reference.coefficients[name].grad)
    if learned_depth:
        torch.testing.assert_close(trajectory.depth_basis.grad, reference.depth_basis.grad)
    assert calls[0] == 2
    assert effective_depth_policy(trajectory, qualify_cpu=False).output_dtype == str(dtype).split(".")[-1]
    if dtype != torch.float32:
        with pytest.raises(ValueError):
            effective_depth_policy(trajectory)
# ^^^ THOG
