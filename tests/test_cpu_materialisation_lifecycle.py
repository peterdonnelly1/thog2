# vvv THOG executable CPU deadline, isolated-copy, cap and autograd-lifetime regressions
from types import SimpleNamespace
import threading
import time

import pytest
import torch

from tests.test_cpu_materialisation import tiny_trajectory
from tests.test_premat import _FakeCuda, _FakeTensor, _FakeEvent
from sheet.premat_cpu_config import CPU_DEFAULTS
from sheet.premat_cpu_runtime import CpuPrematRuntime
import sheet.premat_cpu_runtime as cpu_module


class Provider:
    def __init__(self, trajectory, **kwargs):
        self.trajectory, self.ready, self.requests = trajectory, True, []
    def start(self):
        pass
    def begin_snapshot(self, policy):
        return SimpleNamespace(snapshot_id="fixture_snapshot"), True
    def lookup(self, family, layer, policy):
        if not self.ready:
            return None, {"fallback_reason": "cpu_preparation_not_ready"}
        return torch.zeros(8, 8), {"cpu_matrix_id": f"fixture:{family}:{layer}", "cpu_task_id": "task", "start_ns": 1, "end_ns": 2}
    def request(self, family, layer):
        self.requests.append((family, layer))
    def report(self):
        return {}
    def close(self):
        pass


class Destination(_FakeTensor):
    def copy_(self, source, non_blocking=False):
        self.copy_source = source
        self.non_blocking = non_blocking
        return self


def setup_runtime(monkeypatch, *, target_layer=0, cap=0, transfer="as_the_code_flies", shadow=False):
    cuda, trajectory = _FakeCuda(), tiny_trajectory()
    cuda.install(monkeypatch)
    def synchronize(event):
        while not event.complete and not runtime.cpu_closed:
            time.sleep(0.001)
        if runtime.cpu_closed:
            event.complete = True
    monkeypatch.setattr(_FakeEvent, "synchronize", synchronize, raising=False)
    monkeypatch.setattr(cpu_module, "CpuPreparationProvider", Provider)
    monkeypatch.setattr(CpuPrematRuntime, "_start_sources", lambda self: None)
    monkeypatch.setattr(CpuPrematRuntime, "_start_clock", lambda self: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    monkeypatch.setattr(torch, "empty_like", lambda tensor, **kwargs: Destination(shape=tensor.shape, numel=tensor.numel(), element_bytes=tensor.element_size()))
    calls, bindings = [], []
    def materialize(family, layer):
        calls.append((family, layer))
        return Destination(numel=64)
    def attach(family, layer, tensor, *, binding_context):
        bindings.append(binding_context)
        return tensor
    runtime = CpuPrematRuntime(trajectory=trajectory, cpu_configuration={**CPU_DEFAULTS, "premat_materialisation_device": "cpu_and_gpu", "premat_cpu_staging_limit_mb": cap, "premat_cpu_transfer_timing": transfer}, materialize=materialize, attach=attach, n_embd=8, n_head=2, attention_mode="fused", target_matrix=2, stay_below_current_peak=False, gpu_memory_buffer_gb=0, target_layer=target_layer, weight_matrix_target_order="r_to_l", cuda_stream_priority="normal", diagnostic_layer_delay_ms=0, enable_gpu_timing_diagnostic=False, logging_enabled=True, shadow_mode=shadow)
    runtime.begin(tuple(range(4)), reference=_FakeTensor())
    return runtime, cuda, calls, bindings


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


def test_deadline_during_pin_copy_falls_back_without_waiting(monkeypatch):
    runtime, cuda, calls, _ = setup_runtime(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def pin(tensor):
        entered.set()
        release.wait(2)
        return tensor.clone()
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    runtime.layer_start(0)
    assert entered.wait(1)
    started = time.monotonic()
    result = runtime.acquire("O", 0)
    assert time.monotonic() - started < 0.1
    assert calls == [("O", 0)] and result is not None
    assert not cuda.main_stream.waited_events
    assert runtime._retained_bytes == 0
    release.set()
    wait_until(lambda: runtime.upload_queue.empty())
    runtime.end()
    runtime.close()


def test_late_inflight_copy_has_separate_storage_and_retains_pin(monkeypatch):
    runtime, cuda, calls, _ = setup_runtime(monkeypatch)
    runtime.layer_start(0)
    wait_until(lambda: any(upload.completion is not None for upload in runtime.cpu_uploads.values()))
    upload = next(iter(runtime.cpu_uploads.values()))
    destination, source = upload.tensor, upload.pinned
    result = runtime.acquire("O", 0)
    assert result is not destination and calls == [("O", 0)]
    assert upload.late and upload.tensor is destination and upload.pinned is source
    assert runtime._retained_bytes == 256 and not cuda.main_stream.waited_events
    upload.completion.complete = True
    runtime._refresh_available()
    assert runtime._retained_bytes == 0 and not runtime.cpu_uploads
    runtime.end()
    runtime.close()


def test_completed_copy_hit_uses_consumer_anchor_and_retains_backward_lease(monkeypatch):
    runtime, cuda, calls, bindings = setup_runtime(monkeypatch)
    cuda.complete_premat_on_record = True
    runtime.layer_start(0)
    wait_until(lambda: any(upload.completion is not None for upload in runtime.cpu_uploads.values()))
    result = runtime.acquire("O", 0)
    assert not calls and result.recorded_streams == [cuda.main_stream]
    assert not cuda.main_stream.waited_events and bindings[-1].storage_lease.tensor is result
    runtime.consumed("O", 0)
    runtime.end()
    assert runtime._retained_bytes == 256
    runtime.begin(tuple(range(4)), reference=_FakeTensor())
    assert runtime._retained_bytes == 256
    cuda.complete_main_on_record = False
    cuda.complete_premat_on_record = False
    bindings.clear()
    upload = next(iter(runtime.cpu_uploads.values()))
    assert upload.retirement is not None and not upload.retirement.query()
    runtime._refresh_available()
    assert runtime._retained_bytes == 256
    upload.retirement.complete = True
    runtime._refresh_available()
    assert runtime._retained_bytes == 0
    runtime.end()
    runtime.close()


@pytest.mark.parametrize("target,expected", ((0,{0}), (1,{1}), (2,{2}), (10,{1,0})))
def test_relative_target_offsets_reuse_existing_resolver(monkeypatch, target, expected):
    runtime, _, _, _ = setup_runtime(monkeypatch, target_layer=target)
    runtime.layer_start(0)
    assert {upload.metadata["layer_index"] for upload in runtime.cpu_uploads.values()} == expected
    for upload in runtime.cpu_uploads.values():
        if upload.completion is not None:
            upload.completion.complete = True
    runtime.end()
    runtime.close()


def test_tiny_staging_cap_preserves_ordinary_gpu_fallback(monkeypatch):
    runtime, cuda, calls, _ = setup_runtime(monkeypatch, cap=0.00001)
    runtime.layer_start(0)
    runtime.acquire("O", 0)
    assert calls == [("O",0)] and runtime._retained_bytes == 0 and not runtime.cpu_uploads
    uses = [event for event in runtime.cpu_events if event["event"] == "matrix_use"]
    assert uses[-1]["fallback_reason"] == "cpu_staging_limit"
    assert not cuda.main_stream.waited_events
    runtime.end()
    runtime.close()


def test_selector_exclusion_does_not_enter_hit_miss_totals(monkeypatch):
    runtime, _, calls, _ = setup_runtime(monkeypatch, cap=0.00001)
    runtime.layer_start(0)
    runtime.acquire("QKV",0)
    assert calls == [("QKV",0)] and runtime._aggregate["main_stream_misses"] == 0
    assert runtime._candidates[(0,"QKV")].final_outcome == "NOT TARGETED"
    runtime.end()
    runtime.close()


def test_shadow_prices_candidates_without_workers_snapshots_or_uploads(monkeypatch):
    runtime, _, calls, _ = setup_runtime(monkeypatch, shadow=True)
    runtime.layer_start(0)
    assert runtime.cpu_provider is None and not runtime.cpu_sources and not runtime.cpu_uploads
    assert any(event["event"] == "shadow_admission" for event in runtime._events)
    runtime.acquire("O",0)
    assert calls == [("O",0)]
    runtime.end()
    runtime.close()


def test_only_source_mutation_guard_adds_main_stream_dependency(monkeypatch):
    runtime, cuda, _, _ = setup_runtime(monkeypatch, cap=0.00001)
    event = torch.cuda.Event()
    runtime.source_event = event
    runtime.layer_start(0)
    runtime.acquire("O",0)
    assert not cuda.main_stream.waited_events
    runtime.before_optimizer_step()
    assert cuda.main_stream.waited_events == [event]
    runtime.end()
    runtime.close()
# ^^^ THOG
