# vvv THOG native autograd, mixed-precision training and no-copy cache binding regressions
import gc
import json
import os
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sheet.depth_numerical_policy import CpuDepthBinding, PrematBindingContext, effective_depth_policy, materialize_cpu
from sheet.depth_trajectory import _PrematConsumerStreamAnchor
from sheet.premat_cpu import FAMILY_NAMES
from sheet.trainer import SharedTrainer
from tests.test_cpu_materialisation import policy
from tests.test_cpu_materialisation_benchmark import isolated_materializer
from tools import benchmark_cpu_materialisation as field


def native_weight(coefficient, row, backend):
    if backend == "matmul":
        with torch.autocast(coefficient.device.type, enabled=False):
            return coefficient.reshape(-1, coefficient.shape[-1]).matmul(row).reshape(coefficient.shape[:2])
    return torch.einsum("p,rcp->rc", row, coefficient)


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("learned_depth", (False, True))
def test_cached_binding_retains_the_native_backward_graph_and_rounding(backend, dtype, learned_depth):
    torch.manual_seed(822)
    coefficients = [torch.randn(8, 8, 3, requires_grad=True) for _ in range(3)]
    row = torch.randn(3, requires_grad=learned_depth)
    copies = [value.detach().clone().requires_grad_() for value in coefficients]
    copy_row = row.detach().clone().requires_grad_(learned_depth)
    numerical_policy = policy(backend, "float32" if backend == "matmul" else str(dtype).split(".")[-1])
    with torch.autocast("cpu", dtype=dtype):
        expected = torch.cat([native_weight(value, row, backend) for value in coefficients])
        actual = CpuDepthBinding.apply(expected.detach(), PrematBindingContext(numerical_policy), *(item for value in copies for item in (value, copy_row)))
    assert type(actual.grad_fn) is type(expected.grad_fn)
    gradient = torch.randn_like(expected) * 256
    expected.backward(gradient)
    actual.backward(gradient)
    for left, right in zip(coefficients, copies):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    if learned_depth:
        torch.testing.assert_close(row.grad, copy_row.grad, atol=0, rtol=0)


def test_consumer_anchor_initializes_the_main_edge_without_changing_the_graph():
    value = torch.randn(3, requires_grad=True)
    result = _PrematConsumerStreamAnchor.apply(value)
    assert result is value and result.is_leaf and result.grad_fn is None
    assert torch.autograd.graph.get_gradient_edge(value).node is torch.autograd.graph.get_gradient_edge(result).node


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_order_one_einsum_keeps_native_float32_arithmetic_under_autocast(dtype):
    coefficient = torch.randn(8, 8, 1, requires_grad=True)
    copy = coefficient.detach().clone().requires_grad_()
    row = torch.randn(1)
    trajectory = SimpleNamespace(coefficients={"O": coefficient}, depth_materialisation_matmul=False)
    with torch.autocast("cpu", dtype=dtype):
        numerical_policy = effective_depth_policy(trajectory)
        expected = native_weight(coefficient, row, "einsum")
        assert expected.dtype == torch.float32 and numerical_policy.output_dtype == "float32"
        actual = CpuDepthBinding.apply(expected.detach(), PrematBindingContext(numerical_policy), copy, row)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(coefficient.grad, copy.grad, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
@pytest.mark.parametrize("repeat", (0, 1))
def test_cpu_autocast_whole_trainer_cached_providers_match_native_exactly(monkeypatch, isolated_materializer, backend, dtype, repeat):
    # Actual CPU arithmetic/GradScaler, dropout, accumulation, checkpoint replay
    # and AdamW reproduce the field failure. No CUDA timing or copies are inferred.
    options = field.parser().parse_args(["--device", "cpu", "--threads", "1", "--warmup", "1", "--updates", "3"])
    monkeypatch.setenv("THOG2_FAST_DISCARD", "false")
    monkeypatch.setenv("THOG2_DEPTH_MATERIALISATION_MATMUL", "true" if backend == "matmul" else "false")
    isolated_materializer.install_depth_materialisation_runtime()
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(field, "memory", lambda runtime: {})
    selected_provider = ["off"]
    working_dtype = getattr(torch, dtype)

    def build(config, train_tokens, validation_tokens):
        trainer = SharedTrainer(config, train_tokens, validation_tokens)
        # Keep the public CPU configuration supported; exercise low precision
        # explicitly inside this regression rather than enabling CPU FP16 CLI.
        trainer.autocast_context = lambda: torch.autocast("cpu", dtype=working_dtype, enabled=dtype != "float32")
        if dtype == "float16":
            trainer.scaler = torch.amp.GradScaler("cpu")
        model = trainer.raw_model
        provider = selected_provider[0]
        if provider == "off":
            return trainer

        def weight(family, layer):
            if provider == "gpu":
                with torch.no_grad(), torch.autocast("cpu", dtype=working_dtype, enabled=dtype != "float32", cache_enabled=False):
                    cached = model._premat_materialize_candidate(family, layer)
                return model._premat_attach_candidate(family, layer, cached)
            numerical_policy = policy(backend, "float32" if backend == "matmul" else dtype)
            with torch.autocast("cpu", enabled=False):
                cached = torch.cat([materialize_cpu(model.trajectory.coefficients[name].detach(), model.trajectory._depth_row(layer, model.trajectory.coefficients[name]).detach(), numerical_policy) for name in FAMILY_NAMES[family]])
            return model._premat_attach_candidate(family, layer, cached, binding_context=PrematBindingContext(numerical_policy))

        model._premat_weight = weight
        return trainer

    monkeypatch.setattr(field, "SharedTrainer", build)
    results = {}
    for provider in (("off", "gpu", "cpu_and_gpu") if repeat == 0 else ("cpu_and_gpu", "off", "gpu")):
        selected_provider[0] = provider
        results[provider] = field.run(options, "off", repeat)
        assert results[provider][0]["initial_conditions"]["fast_discard"] is True
        assert os.environ["THOG2_FAST_DISCARD"] == "false"
    for provider in ("gpu", "cpu_and_gpu"):
        rows = field.compare_run_results(results["off"], results[provider], repeat=repeat, mode=provider, atol=0, rtol=0)
        failed = [row for row in rows if not row["passed"]]
        if failed:
            pytest.fail(json.dumps({"backend": backend, "dtype": dtype, "repeat": repeat, "failed_comparisons": failed}, indent=2), pytrace=False)


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
def test_native_binding_keeps_storage_and_lease_without_running_the_contraction(backend):
    class Lease:
        pass

    class PhysicalKernels(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.contractions = []

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if function in (torch.ops.aten.mv.default, torch.ops.aten.bmm.default, torch.ops.aten.mm.default):
                self.contractions.append(function)
            return function(*args, **(kwargs or {}))

    coefficient = torch.randn(8, 8, 3, requires_grad=True)
    row = torch.randn(3)
    cached = native_weight(coefficient, row, backend).detach()
    lease = Lease()
    lease_reference = weakref.ref(lease)
    binding = PrematBindingContext(policy(backend), lease)
    kernels = PhysicalKernels()
    with kernels:
        actual = CpuDepthBinding.apply(cached, binding, coefficient, row)
    assert actual.untyped_storage().data_ptr() == cached.untyped_storage().data_ptr()
    assert not kernels.contractions
    output = torch.nn.functional.linear(torch.randn(2, 8, requires_grad=True), actual)
    del actual, binding, lease
    gc.collect()
    assert lease_reference() is not None
    output.square().mean().backward()
    del output
    gc.collect()
    assert lease_reference() is None
# ^^^ THOG
