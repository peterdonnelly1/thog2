# vvv THOG PLASTIC DEPTH CUDA reserve and upward-feasibility tests
from __future__ import annotations

from unittest.mock import patch

import torch

from sheet.plastic_depth_cuda import (
    GIB_BYTES,
    PlasticDepthCudaAllocatorReserve,
    is_cuda_out_of_memory,
    validate_cuda_allocator_reserve_gib,
)
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_training_config


def test_cuda_allocator_reserve_validates_and_allocates_exact_bytes() -> None:
    calls = []
    sentinel = torch.empty(1)

    def allocator(size, **kwargs):
        calls.append((size, kwargs))
        return sentinel

    reserve = PlasticDepthCudaAllocatorReserve(
        device=torch.device("cuda", 0),
        reserve_gib=0.25,
        allocator=allocator,
    )
    assert reserve.acquire()
    assert reserve.active
    assert calls == [
        (
            GIB_BYTES // 4,
            {"dtype": torch.uint8, "device": torch.device("cuda", 0)},
        )
    ]
    assert reserve.acquire()
    assert len(calls) == 1
    reserve.release()
    assert not reserve.active


def test_cuda_allocator_reserve_converts_only_cuda_oom_to_infeasible() -> None:
    def oom_allocator(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory while allocating reserve")

    reserve = PlasticDepthCudaAllocatorReserve(
        device=torch.device("cuda", 0),
        reserve_gib=0.5,
        allocator=oom_allocator,
    )
    assert not reserve.acquire()
    assert not reserve.active
    assert is_cuda_out_of_memory(RuntimeError("CUDA out of memory"))
    assert not is_cuda_out_of_memory(RuntimeError("other failure"))


def test_cuda_allocator_reserve_rejects_invalid_values_and_cpu_is_exact_noop() -> None:
    for value in (-0.1, float("inf"), float("nan"), True):
        try:
            validate_cuda_allocator_reserve_gib(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected reserve validation failure for {value!r}")

    reserve = PlasticDepthCudaAllocatorReserve(
        device=torch.device("cpu"),
        reserve_gib=1024.0,
        allocator=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("allocator called")),
    )
    assert reserve.acquire()
    assert not reserve.active


class _FakeReserve:
    instances = []
    acquire_result = True

    def __init__(self, **_kwargs):
        self.released = 0
        self.__class__.instances.append(self)

    def acquire(self):
        return self.__class__.acquire_result

    def release(self, **_kwargs):
        self.released += 1


def _learned_cpu_trainer() -> SharedTrainer:
    train_tokens, validation_tokens = token_splits(length=512)
    return SharedTrainer(
        plastic_training_config(
            n_layer=5,
            depth_order=4,
            plastic__layers_to_sample=None,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=3,
            plastic__max_permitted_layers=5,
            plastic__layer_count_update_brake=0,
            warmup_updates=0,
            gradient_accumulation_steps=1,
        ),
        train_tokens,
        validation_tokens,
    )


def test_failed_distributed_reserve_preflight_excludes_only_upward_candidate() -> None:
    trainer = _learned_cpu_trainer()
    original_device = trainer.device
    _FakeReserve.instances = []
    _FakeReserve.acquire_result = False
    try:
        trainer.device = torch.device("cuda", 0)
        with patch("sheet.trainer_step.PlasticDepthCudaAllocatorReserve", _FakeReserve):
            context = trainer._begin_plastic_depth_inline_update()
        assert context is not None
        assert context["candidate_counts"] == (2, 3)
        assert context["recoverable_upward_count"] is None
        assert context["upward_preflight_feasible"] is False
        assert trainer.raw_model._plastic_depth_update_layer_count == 3
        assert _FakeReserve.instances[0].released == 1
    finally:
        trainer._clear_plastic_depth_inline_update()
        trainer.device = original_device
        trainer.close()


def test_successful_reserve_is_released_by_update_cleanup() -> None:
    trainer = _learned_cpu_trainer()
    original_device = trainer.device
    _FakeReserve.instances = []
    _FakeReserve.acquire_result = True
    try:
        trainer.device = torch.device("cuda", 0)
        with patch("sheet.trainer_step.PlasticDepthCudaAllocatorReserve", _FakeReserve):
            context = trainer._begin_plastic_depth_inline_update()
        assert context is not None
        assert context["candidate_counts"] == (2, 3, 4)
        assert context["recoverable_upward_count"] == 4
        reserve = _FakeReserve.instances[0]
        assert reserve.released == 0
        trainer._clear_plastic_depth_inline_update()
        assert reserve.released == 1
    finally:
        trainer.device = original_device
        trainer.close()
# ^^^ THOG
