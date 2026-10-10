# vvv THOG real mixed-precision arithmetic and interrupted-checkpoint regressions without CUDA
import traceback

import pytest
import torch

from sheet.checkpointing import execute_logical_layer_checkpoints, execute_logical_layers
from sheet.depth_materialisation_runtime import _depth_materialize_parameter_with_matmul
from sheet.depth_trajectory import DepthTrajectory
from sheet.model import SheetGPT, SheetGPTConfig
from tests.test_premat import _FakeCuda, _FakeTensor


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("attention_mode", ("fused", "unfused"))
def test_gpu_admission_prices_actual_depth_weights_separately_from_autocast_activations(monkeypatch, backend, dtype, attention_mode):
    # CUDA events/memory are simulated, but every materialised matrix is a real
    # FP32-storage DEPTH tensor under actual CPU autocast. This is not a GPU timing test.
    fake_cuda = _FakeCuda()
    fake_cuda.install(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    original_is_enabled = torch.is_autocast_enabled
    original_get_dtype = torch.get_autocast_dtype
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type="cuda": original_is_enabled("cpu" if device_type == "cuda" else device_type))
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device_type: original_get_dtype("cpu" if device_type == "cuda" else device_type))
    monkeypatch.setattr(DepthTrajectory, "_materialize_depth_parameter", _depth_materialize_parameter_with_matmul)
    model = SheetGPT(SheetGPTConfig(
        block_size=4, vocab_size=32, n_layer=4, n_embd=8, n_head=2,
        geometry_preset="depth", depth_order=3, base_row_order=1,
        premat="enabled", premat_attention_mode=attention_mode,
        premat_gpu_memory_buffer_gb=0, premat_headroom_stay_below_current_peak=False,
        premat_headroom_stay_within_global_buffer=True,
    ))
    model.trajectory.depth_materialisation_matmul = backend == "matmul"
    runtime = model._premat_runtime
    with torch.autocast("cpu", dtype=dtype):
        runtime.begin((0, 1, 2, 3), reference=_FakeTensor(element_bytes=4))
        try:
            runtime.layer_start(0)
            assert runtime._dtype_bytes == (4 if backend == "matmul" else 2)
            assert runtime._activation_bytes == 2 * 4 * 8 * 2
            for family in runtime._families():
                candidate = runtime._candidates[(1, family)]
                assert candidate.tensor is not None
                assert candidate.tensor.dtype == (torch.float32 if backend == "matmul" else dtype)
                assert candidate.envelope.retained_bytes == candidate.tensor.numel() * candidate.tensor.element_size()
                activation_bytes = 2 * 4 * 8 * 2
                attention_bytes = 4 * activation_bytes
                if attention_mode == "unfused":
                    attention_bytes += 2 * (2 * 2 * 4 * 4 * 2) + 4 * 4
                assert candidate.envelope.foreground_overlap_bytes == attention_bytes
        finally:
            runtime.end()


def checkpoint_execution(entry_point, value, block):
    if entry_point == "prefix":
        outputs, _ = execute_logical_layer_checkpoints(
            value, n_layer=4, segment_size=2, logical_block=block,
            training=True, layer_indices=tuple(range(4)), checkpoint_counts=(2, 4),
        )
        return outputs[-1][1]
    output, _ = execute_logical_layers(
        value, n_layer=4, segment_size=2, logical_block=block, training=True,
        layer_indices=(0, 2) if entry_point == "sparse" else None,
    )
    return output


@pytest.mark.parametrize("entry_point", ("dense", "sparse", "prefix"))
def test_interrupted_checkpoint_does_not_capture_the_next_training_graph(entry_point):
    # Pytest retains failed-test tracebacks. Retain this one explicitly to ensure
    # cleanup does not depend on garbage collection or successful forward exit.
    failure = RuntimeError("intentional admission failure")
    failed_calls = []

    def failing_block(value, layer):
        failed_calls.append(layer)
        value.sin()
        raise failure

    with pytest.raises(RuntimeError, match="intentional admission failure") as captured:
        checkpoint_execution(entry_point, torch.randn(2, 3, requires_grad=True), failing_block)
    assert captured.value is failure and failure.__traceback__ is not None
    try:
        value = torch.randn(4, 5, requires_grad=True)
        reference = value.detach().clone().requires_grad_(True)
        output = checkpoint_execution(entry_point, value, lambda x, layer: x.sin() + 0.01 * layer)
        expected = reference
        for layer in ((0, 2) if entry_point == "sparse" else range(4)):
            expected = expected.sin() + 0.01 * layer
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(value.grad, reference.grad)
        assert failed_calls == [0]
    finally:
        # Also unwind the deliberately retained upstream generator when checking
        # that this regression fails against a branch without the production fix.
        traceback.clear_frames(failure.__traceback__)
# ^^^ THOG
