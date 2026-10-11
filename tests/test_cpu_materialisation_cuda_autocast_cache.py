# vvv THOG CUDA einsum caches leaf casts before decomposition, unlike CPU bmm autocast
import pytest
import torch
from torch.overrides import TorchFunctionMode

from sheet.depth_numerical_policy import PrematBindingContext
from sheet.premat_native_binding import bind_cached_depth
from tests.test_cpu_materialisation import policy
from tests.test_cpu_materialisation_benchmark import isolated_materializer
from tests.test_cpu_materialisation_native_binding import (
    test_cpu_autocast_whole_trainer_cached_providers_match_native_exactly as compare_whole_trainer,
)


class CudaEinsumAutocastOnCpu(TorchFunctionMode):
    """Emulate PyTorch 2.8's CUDA einsum cast/cache boundary with CPU arithmetic.

    CUDA's autocast registration casts the original leaf coefficient, sharing
    its ToCopyBackward between contractions. CPU only autocasts the inner bmm,
    after einsum has made non-leaf views, and therefore does not share that edge.
    This mode exercises the CUDA graph contract without claiming CUDA kernels.
    """

    def __init__(self):
        super().__init__()
        self.casts = {}
        self.cast_storage_bytes = []

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function is not torch.einsum or not torch.is_autocast_enabled("cpu"):
            return function(*args, **kwargs)
        dtype = torch.get_autocast_dtype("cpu")
        operands = []
        for operand in args[1:]:
            if operand.dtype != torch.float32:
                operands.append(operand)
                continue
            cacheable = (
                torch.is_autocast_cache_enabled()
                and operand.requires_grad
                and operand.is_leaf
                and not operand._is_view()
            )
            key = id(operand)
            if cacheable and key in self.casts:
                cast = self.casts[key]
            else:
                cast = operand.to(dtype=dtype)
                if operand.ndim == 3:
                    self.cast_storage_bytes.append(cast.untyped_storage().nbytes())
                if cacheable:
                    self.casts[key] = cast
            operands.append(cast)
        with torch.autocast("cpu", enabled=False):
            return function(args[0], *operands, **kwargs)


def compare_shared_cast_binding(device, dtype, hit_mask, learned_depth, emulate_cuda=False):
    torch.manual_seed(1247)
    coefficient = torch.randn(8, 8, 3, device=device, requires_grad=True)
    copy = coefficient.detach().clone().requires_grad_()
    rows = [torch.randn(3, device=device, requires_grad=learned_depth) for _ in hit_mask]
    copy_rows = [row.detach().clone().requires_grad_(learned_depth) for row in rows]
    binding = PrematBindingContext(policy("einsum", str(dtype).split(".")[-1]))
    mode = CudaEinsumAutocastOnCpu() if emulate_cuda else None
    if mode is not None:
        mode.__enter__()
    try:
        with torch.autocast(device, dtype=dtype):
            expected_parts = [torch.einsum("p,rcp->rc", row, coefficient) for row in rows]
            actual_parts = [
                bind_cached_depth(expected.detach(), binding, copy, row)
                if hit else torch.einsum("p,rcp->rc", row, copy)
                for hit, expected, row in zip(hit_mask, expected_parts, copy_rows)
            ]
        expected = torch.cat(expected_parts)
        actual = torch.cat(actual_parts)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        gradient = torch.randn_like(expected) * 256
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(copy.grad, coefficient.grad, atol=0, rtol=0)
        for row, copy_row in zip(rows, copy_rows):
            if learned_depth:
                torch.testing.assert_close(copy_row.grad, row.grad, atol=0, rtol=0)
    finally:
        if mode is not None:
            mode.__exit__(None, None, None)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("hit_mask", ((True, True, True, True), (False, True, False, True), (True, False, True, False), (False, False, False, False)))
@pytest.mark.parametrize("learned_depth", (False, True))
def test_cuda_leaf_cast_cache_contract_on_cpu(dtype, hit_mask, learned_depth):
    compare_shared_cast_binding("cpu", dtype, hit_mask, learned_depth, emulate_cuda=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA einsum shared-cast qualification")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("hit_mask", ((True, True, True, True), (False, True, False, True), (True, False, True, False)))
@pytest.mark.parametrize("learned_depth", (False, True))
def test_actual_cuda_binding_matches_shared_leaf_cast_graph(dtype, hit_mask, learned_depth):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    compare_shared_cast_binding("cuda", dtype, hit_mask, learned_depth)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("repeat", (0, 1))
def test_whole_trainer_cuda_cast_cache_contract_on_cpu(monkeypatch, isolated_materializer, dtype, repeat):
    mode = CudaEinsumAutocastOnCpu()
    original_clear = torch.clear_autocast_cache

    def clear_cache():
        mode.casts.clear()
        original_clear()

    # Match native cache lifetime, including nested disabled-autocast regions
    # and checkpoint replay, rather than treating each contraction separately.
    monkeypatch.setattr(torch, "clear_autocast_cache", clear_cache)
    with mode:
        compare_whole_trainer(monkeypatch, isolated_materializer, "einsum", dtype, repeat)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_explicitly_disabled_cast_cache_keeps_the_no_copy_binding(dtype):
    torch.manual_seed(81)
    coefficient = torch.randn(8, 8, 3, requires_grad=True)
    copy = coefficient.detach().clone().requires_grad_()
    row = torch.randn(3)
    mode = CudaEinsumAutocastOnCpu()
    binding = PrematBindingContext(policy("einsum", str(dtype).split(".")[-1]))
    with mode, torch.autocast("cpu", dtype=dtype, cache_enabled=False):
        expected = torch.einsum("p,rcp->rc", row, coefficient)
        actual = bind_cached_depth(expected.detach(), binding, copy, row)
    assert mode.cast_storage_bytes[0] == coefficient.numel() * expected.element_size()
    assert mode.cast_storage_bytes[-1] == expected.element_size()
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(copy.grad, coefficient.grad, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA order-one autocast policy")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_actual_cuda_order_one_policy_matches_native_einsum(dtype):
    from types import SimpleNamespace
    from sheet.depth_numerical_policy import effective_depth_policy

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    coefficient = torch.randn(8, 8, 1, device="cuda", requires_grad=True)
    copy = coefficient.detach().clone().requires_grad_()
    row = torch.randn(1, device="cuda")
    trajectory = SimpleNamespace(coefficients={"O": copy}, depth_materialisation_matmul=False)
    with torch.autocast("cuda", dtype=dtype):
        expected = torch.einsum("p,rcp->rc", row, coefficient)
        numerical_policy = effective_depth_policy(trajectory)
        assert numerical_policy.output_dtype == str(expected.dtype).split(".")[-1]
        actual = bind_cached_depth(expected.detach(), PrematBindingContext(numerical_policy), copy, row)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(copy.grad, coefficient.grad, atol=0, rtol=0)
# ^^^ THOG
