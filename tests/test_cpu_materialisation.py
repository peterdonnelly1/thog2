# vvv THOG arithmetic, snapshot, shared-worker and configuration regression gates
import copy
from types import SimpleNamespace
import time

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from sheet.depth_numerical_policy import CpuDepthBinding, DepthNumericalPolicy, PrematBindingContext, materialize_cpu
from sheet.premat_cpu import CpuPreparationProvider, FAMILY_NAMES, GemmPredictor, source_fingerprint
from sheet.premat_cpu_config import CPU_DEFAULTS, normalize_cpu_batch, validate_cpu_configuration


def tiny_trajectory():
    torch.manual_seed(72)
    names = tuple(dict.fromkeys(name for values in FAMILY_NAMES.values() for name in values))
    coefficients = {}
    for name in names:
        rows, columns = (32, 8) if name == "mlp_expansion_weight" else ((8, 32) if name == "mlp_contraction_weight" else (8, 8))
        coefficients[name] = torch.nn.Parameter(torch.randn(rows, columns, 3) * 0.05)
    return SimpleNamespace(coefficients=coefficients, depth_basis=torch.randn(4, 3), config=SimpleNamespace(n_layer=4))


def policy(backend="matmul", dtype="float32"):
    return DepthNumericalPolicy(backend, "float32", dtype, "float32", dtype)


@pytest.mark.parametrize("family", tuple(FAMILY_NAMES))
@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_cpu_arithmetic_converts_operands_before_reconstruction(family, backend, dtype):
    trajectory = tiny_trajectory()
    coefficients = [trajectory.coefficients[name] for name in FAMILY_NAMES[family]]
    row = trajectory.depth_basis[2]
    generated = torch.cat([materialize_cpu(coefficient, row, policy(backend, dtype)) for coefficient in coefficients])
    converted = getattr(torch, dtype)
    expected = torch.cat([torch.einsum("p,rcp->rc", row.to(converted).float(), coefficient.to(converted).float()).to(converted) for coefficient in coefficients])
    tolerance = 2e-6 if dtype == "float32" else 1e-3 if dtype == "float16" else 1e-2
    torch.testing.assert_close(generated, expected, atol=tolerance, rtol=tolerance)
    assert generated.ndim == 2 and generated.is_contiguous()


@pytest.mark.parametrize("family", tuple(FAMILY_NAMES))
@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("learned_depth", (False, True))
def test_binding_matches_native_saved_tensors_and_gradients(family, backend, learned_depth):
    trajectory = tiny_trajectory()
    originals = [trajectory.coefficients[name] for name in FAMILY_NAMES[family]]
    cached_coefficients = [torch.nn.Parameter(coefficient.detach().clone()) for coefficient in originals]
    row = trajectory.depth_basis[2].detach().requires_grad_(learned_depth)
    bound_row = row.detach().clone().requires_grad_(learned_depth)
    native_saved, bound_saved = [], []
    def capture(destination):
        def pack(value):
            destination.append((tuple(value.shape), value.dtype, value.device))
            return value
        return torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value)
    def native(coefficient):
        return coefficient.reshape(-1, 3).matmul(row).reshape(coefficient.shape[:2]) if backend == "matmul" else torch.einsum("p,rcp->rc", row, coefficient)
    with capture(native_saved):
        expected = torch.cat([native(coefficient) for coefficient in originals])
    cached = torch.cat([materialize_cpu(coefficient, bound_row, policy(backend)) for coefficient in cached_coefficients])
    pairs = tuple(item for coefficient in cached_coefficients for item in (coefficient, bound_row))
    with capture(bound_saved):
        actual = CpuDepthBinding.apply(cached, PrematBindingContext(policy(backend)), *pairs)
    assert bound_saved == native_saved
    torch.testing.assert_close(actual, expected)
    torch.manual_seed(29)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    for first, second in zip(originals, cached_coefficients):
        torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5)
    if learned_depth:
        torch.testing.assert_close(row.grad, bound_row.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("preparation", ("eager", "scheduled", "demand_driven"))
@pytest.mark.parametrize("batch", ("single_layer", "2", "all_layers"))
def test_preparation_batch_scope_and_snapshot_reuse(preparation, batch):
    trajectory, records = tiny_trajectory(), []
    provider = CpuPreparationProvider(trajectory, families=("O",), preparation=preparation, batch_size=batch, workers=1, threads=1, record=records.append)
    try:
        snapshot, fresh = provider.begin_snapshot(policy())
        assert fresh
        provider.publish_source(snapshot.snapshot_id, "depth_basis", trajectory.depth_basis)
        provider.publish_source(snapshot.snapshot_id, "attention_output_weight", trajectory.coefficients["attention_output_weight"])
        provider.request("O", 2)
        expected = set(range(4)) if preparation == "eager" or batch == "all_layers" else {2, 3} if batch == "2" else {2}
        assert {layer for _, layer in provider.submitted} == expected
        before = len([record for record in records if record["event"] == "cpu_task_queued"])
        provider.request("O", 2)
        assert len([record for record in records if record["event"] == "cpu_task_queued"]) == before
        unchanged, fresh = provider.begin_snapshot(policy())
        assert not fresh and unchanged.snapshot_id == snapshot.snapshot_id
        with torch.no_grad():
            trajectory.coefficients["attention_output_weight"].add_(0.01)
        changed, fresh = provider.begin_snapshot(policy())
        assert fresh and changed.snapshot_id != snapshot.snapshot_id and not provider.cache
    finally:
        provider.close()


@pytest.mark.parametrize("change", ("coefficient", "basis", "replacement", "policy"))
def test_snapshot_identity_covers_actual_source_mutations(change):
    trajectory = tiny_trajectory()
    before = source_fingerprint(trajectory, policy(), ("O",))
    after_policy = policy()
    with torch.no_grad():
        if change == "coefficient":
            trajectory.coefficients["attention_output_weight"].add_(1)
        elif change == "basis":
            trajectory.depth_basis.add_(1)
        elif change == "replacement":
            trajectory.coefficients["attention_output_weight"] = torch.nn.Parameter(trajectory.coefficients["attention_output_weight"].clone())
        else:
            after_policy = policy("einsum")
    assert source_fingerprint(trajectory, after_policy, ("O",)) != before


def test_real_spawn_worker_shares_individual_matrices_without_cuda():
    trajectory, records = tiny_trajectory(), []
    provider = CpuPreparationProvider(trajectory, families=("O",), preparation="eager", batch_size="all_layers", workers=1, threads=1, record=records.append)
    try:
        provider.start()
        snapshot, _ = provider.begin_snapshot(policy())
        provider.publish_source(snapshot.snapshot_id, "depth_basis", trajectory.depth_basis)
        provider.publish_source(snapshot.snapshot_id, "attention_output_weight", trajectory.coefficients["attention_output_weight"])
        deadline = time.monotonic() + 30
        while len(provider.cache) != 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(provider.cache) == 4
        for layer in range(4):
            matrix, metadata = provider.lookup("O", layer, policy())
            assert hasattr(matrix, "_cpu_shared_owner")
            torch.testing.assert_close(matrix, materialize_cpu(trajectory.coefficients["attention_output_weight"], trajectory.depth_basis[layer], policy()))
            assert metadata["worker_pid"] != __import__("os").getpid()
        health = provider.report()["worker_health"]
        assert health and all(not worker["cuda_initialized"] and worker["native_threads"] == 1 for worker in health)
    finally:
        provider.close()
    assert all(not worker.is_alive() for worker in provider.workers)


@pytest.mark.parametrize("value", (0, 1, "0", "1", "all", "single", True, 2.5))
def test_invalid_batch_values(value):
    with pytest.raises(ValueError):
        normalize_cpu_batch(value)


@pytest.mark.parametrize("field,value", (("premat", "disabled"), ("device", "cpu"), ("geometry_preset", "WIDTH"), ("plastic__enabled", True), ("dtype", "float64"), ("premat_cpu_transfer_lead_ms", 1), ("premat_cpu_staging_limit_mb", float("nan"))))
def test_invalid_cpu_capabilities(field, value):
    config = {**CPU_DEFAULTS, "premat_materialisation_device": "cpu_and_gpu", "premat": "enabled", "device": "cuda"}
    config[field] = value
    with pytest.raises(ValueError):
        validate_cpu_configuration(config)


def test_prediction_requires_qualified_clock_and_warm_history():
    predictor = GemmPredictor()
    context = ("original_forward", 2, "O", 2, 16, "matmul")
    assert not predictor.predict(context, 100000000, 2, 0.1)["prediction_available"]
    for _ in range(3):
        predictor.observe(context, 20, None)
    assert not predictor.samples
    for _ in range(3):
        predictor.observe(context, 20, 0.1)
    prediction = predictor.predict(context, 100000000, 2, 0.1)
    assert prediction["prediction_available"] and prediction["intended_gpu_available_ns"] == 118000000
    assert not predictor.predict(("checkpoint_recompute",), 100000000, 2, 0.1)["prediction_available"]
    predictor.reset("shape_changed")
    assert not predictor.samples

def test_finite_batch_starts_at_requested_layer_without_revisiting_predecessors():
    provider=CpuPreparationProvider(tiny_trajectory(),families=("O",),preparation="scheduled",batch_size="4",workers=1,threads=1)
    try:
        assert provider.batch_layers(2)==(2,3)
        assert provider.batch_layers(1)==(1,2,3)
    finally:provider.close()
# ^^^ THOG
