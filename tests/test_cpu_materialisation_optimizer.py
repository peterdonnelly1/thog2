# vvv THOG optimizer mutation, generation cancellation and long fused-AdamW cache regressions
from collections import deque
import threading
import time

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from sheet.depth_numerical_policy import CpuDepthBinding, PrematBindingContext, materialize_cpu
from sheet.premat_cpu import CpuPreparationProvider, shared_tensor, source_fingerprint, tensor_descriptor
from sheet.premat_cpu_config import CPU_DEFAULTS
from sheet.premat_cpu_runtime import CpuPrematRuntime
from tests.test_cpu_materialisation import policy, tiny_trajectory

NAME = "attention_output_weight"


def runtime_with_provider(trajectory, *, preparation="demand_driven", records=None):
    runtime = CpuPrematRuntime(
        trajectory=trajectory,
        cpu_configuration={**CPU_DEFAULTS, "premat_materialisation_device": "cpu_and_gpu"},
        materialize=lambda *args: None, attach=lambda *args, **kwargs: None,
        n_embd=8, n_head=2, attention_mode="fused", target_matrix=2, target_layer=0,
        stay_below_current_peak=False, gpu_memory_buffer_gb=0,
        weight_matrix_target_order="r_to_l", cuda_stream_priority="normal",
        diagnostic_layer_delay_ms=0, enable_gpu_timing_diagnostic=False,
        logging_enabled=True, shadow_mode=False,
    )
    runtime.cpu_provider = CpuPreparationProvider(
        trajectory, families=("O",), preparation=preparation,
        batch_size="all_layers", workers=1, threads=1, record=records.append if records is not None else None,
    )
    runtime.cpu_snapshot, _ = runtime.cpu_provider.begin_snapshot(policy())
    return runtime


@pytest.mark.parametrize("source", ("coefficient", "basis"))
@pytest.mark.parametrize("implementation", ("fused", "foreach", "ordinary"))
def test_optimizer_boundary_invalidates_changed_sources_even_without_version_bump(source, implementation):
    trajectory = tiny_trajectory()
    trajectory.depth_basis = torch.nn.Parameter(trajectory.depth_basis)
    runtime = runtime_with_provider(trajectory)
    provider = runtime.cpu_provider
    try:
        snapshot = runtime.cpu_snapshot
        cached = materialize_cpu(trajectory.coefficients[NAME], trajectory.depth_basis[0], policy())
        provider.cache[("O", 0)] = (cached, {"snapshot_id": snapshot.snapshot_id})
        runtime.cpu_sources = deque([("old pending copy",)])
        parameter = trajectory.coefficients[NAME] if source == "coefficient" else trajectory.depth_basis
        optimizer = torch.optim.AdamW([parameter], lr=0.01, fused=implementation == "fused", foreach=implementation == "foreach")
        before = parameter.detach().clone()
        fingerprint = source_fingerprint(trajectory, policy(), ("O",))
        version = parameter._version
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert not torch.equal(parameter, before)
        if implementation == "fused":
            assert parameter._version == version
            assert source_fingerprint(trajectory, policy(), ("O",)) == fingerprint
            # This is the stale HIT that tensor version checks alone cannot detect.
            assert provider.lookup("O", 0, policy())[0] is cached
            assert not torch.allclose(cached, materialize_cpu(trajectory.coefficients[NAME], trajectory.depth_basis[0], policy()))
        runtime.after_optimizer_step(optimizer)
        assert provider.snapshot is None and runtime.cpu_snapshot is None
        assert not provider.cache and not runtime.cpu_sources and runtime.source_paused
        assert provider.generation.value > snapshot.generation
        assert provider.lookup("O", 0, policy()) == (None, {"fallback_reason": "stale_snapshot"})
        fresh, created = provider.begin_snapshot(policy())
        assert created and fresh.snapshot_id != snapshot.snapshot_id
        assert runtime.cpu_events[-1]["snapshot_invalidated"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("case", ("missing_gradient", "unselected_parameter", "zero_learning_rate", "frozen_sources"))
def test_optimizer_boundary_preserves_snapshot_when_selected_sources_cannot_change(case):
    trajectory = tiny_trajectory()
    runtime = runtime_with_provider(trajectory)
    try:
        parameter = trajectory.coefficients[NAME]
        if case == "unselected_parameter":
            parameter = trajectory.coefficients["attention_query_weight"]
        elif case == "frozen_sources":
            trajectory.coefficients[NAME].requires_grad_(False)
            parameter = torch.nn.Parameter(torch.ones(2))
        if case != "missing_gradient":
            parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.AdamW([parameter], lr=0 if case == "zero_learning_rate" else 0.01, fused=True)
        snapshot = runtime.cpu_snapshot
        optimizer.step()
        runtime.after_optimizer_step(optimizer)
        unchanged, created = runtime.cpu_provider.begin_snapshot(policy())
        assert not created and unchanged.snapshot_id == snapshot.snapshot_id
        assert runtime.cpu_snapshot is snapshot
        assert runtime.cpu_events[-1]["snapshot_invalidated"] is False
    finally:
        runtime.close()


def test_late_worker_result_and_source_copy_cannot_revive_invalidated_snapshot():
    trajectory, records = tiny_trajectory(), []
    runtime = runtime_with_provider(trajectory, records=records)
    provider = runtime.cpu_provider
    old = runtime.cpu_snapshot
    matrix = shared_tensor(torch.ones(8, 8))
    descriptor = tensor_descriptor(matrix)
    try:
        provider.invalidate_snapshot("optimizer_mutation")
        fresh, _ = provider.begin_snapshot(policy())
        provider.publish_source(old.snapshot_id, NAME, trajectory.coefficients[NAME])
        assert not fresh.coefficients
        provider.collector = threading.Thread(target=provider._collect, daemon=True)
        provider.collector.start()
        provider.results.put({"event": "cpu_matrix_ready", "generation": old.generation, "snapshot_id": old.snapshot_id,
                              "cpu_task_id": "late", "family": "O", "layer_index": 0, "tensor": descriptor})
        deadline = time.monotonic() + 5
        while not any(row["event"] == "cpu_matrix_ready" for row in records) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert any(row["event"] == "cpu_matrix_ready" and row["discarded"] for row in records)
        assert not provider.cache and provider.snapshot is fresh
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("learned_depth", (False, True))
def test_real_worker_35_fused_updates_match_native_with_accumulation_dropout_and_checkpoint(backend, learned_depth):
    trajectory = tiny_trajectory()
    trajectory.depth_basis = torch.nn.Parameter(trajectory.depth_basis, requires_grad=learned_depth)
    coefficient = trajectory.coefficients[NAME]
    native_coefficient = torch.nn.Parameter(coefficient.detach().clone())
    native_basis = torch.nn.Parameter(trajectory.depth_basis.detach().clone(), requires_grad=learned_depth)
    cached_parameters = [coefficient, trajectory.depth_basis] if learned_depth else [coefficient]
    native_parameters = [native_coefficient, native_basis] if learned_depth else [native_coefficient]
    cached_optimizer = torch.optim.AdamW(cached_parameters, lr=0.001, fused=True)
    native_optimizer = torch.optim.AdamW(native_parameters, lr=0.001, fused=True)
    numerical_policy = policy(backend)
    runtime = runtime_with_provider(trajectory, preparation="eager")
    provider = runtime.cpu_provider
    cache_uses = 0
    identities = set()
    version = coefficient._version

    def train_microbatch(c, basis, use_cache, inputs, target):
        def segment(value, layers):
            nonlocal cache_uses
            for layer in layers:
                row = basis[layer]
                if use_cache:
                    cached, details = provider.lookup("O", layer, numerical_policy)
                    assert cached is not None, details
                    matrix = CpuDepthBinding.apply(cached, PrematBindingContext(numerical_policy), c, row)
                    cache_uses += 1
                elif backend == "matmul":
                    matrix = c.reshape(-1, 3).matmul(row).reshape(8, 8)
                else:
                    matrix = torch.einsum("p,rcp->rc", row, c)
                value = value + torch.nn.functional.dropout(torch.tanh(torch.nn.functional.linear(value, matrix)), p=0.2, training=True)
            return value
        value = inputs
        for layers in ((0, 1), (2, 3)):
            value = checkpoint(lambda item, layers=layers: segment(item, layers), value, use_reentrant=False)
        loss = (value - target).square().mean() / 2
        loss.backward()
        return loss.detach()

    try:
        provider.start()
        for update in range(35):
            snapshot, _ = provider.begin_snapshot(numerical_policy)
            runtime.cpu_snapshot = snapshot
            identities.add(snapshot.snapshot_id)
            provider.publish_source(snapshot.snapshot_id, "depth_basis", trajectory.depth_basis)
            provider.publish_source(snapshot.snapshot_id, NAME, coefficient)
            # Diagnostic preparation wait only in this correctness test, never in the benchmark.
            deadline = time.monotonic() + 30
            while len(provider.cache) < 4 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert len(provider.cache) == 4, provider.report()
            native_optimizer.zero_grad(set_to_none=True)
            cached_optimizer.zero_grad(set_to_none=True)
            for microbatch in range(2):
                generator = torch.Generator().manual_seed(update * 2 + microbatch)
                inputs, target = torch.randn(2, 8, generator=generator), torch.randn(2, 8, generator=generator)
                torch.manual_seed(700 + update * 2 + microbatch)
                expected = train_microbatch(native_coefficient, native_basis, False, inputs, target)
                native_rng = torch.get_rng_state()
                torch.manual_seed(700 + update * 2 + microbatch)
                actual = train_microbatch(coefficient, trajectory.depth_basis, True, inputs, target)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                assert torch.equal(torch.get_rng_state(), native_rng)
            for actual, expected in zip(cached_parameters, native_parameters):
                torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5)
            native_optimizer.step()
            cached_optimizer.step()
            runtime.after_optimizer_step(cached_optimizer)
            assert coefficient._version == version
            assert provider.snapshot is None and not provider.cache
            for actual, expected in zip(cached_parameters, native_parameters):
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                for key, value in native_optimizer.state[expected].items():
                    torch.testing.assert_close(cached_optimizer.state[actual][key], value, atol=3e-6, rtol=3e-5)
        assert len(identities) == 35
        assert cache_uses > 35 * 2 * 4  # Includes checkpoint recomputation, all using real worker output.
        assert not any(worker["cuda_initialized"] for worker in provider.report()["worker_health"])
    finally:
        runtime.close()
# ^^^ THOG
