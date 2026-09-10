from __future__ import annotations

import pytest
import torch

from sheet.premat import (
    CandidateEnvelope,
    PrematMemoryObservation,
    PrematRuntime,
    _conservative_certificate_charge_bytes,
    _inactive_allocator_bytes_from_stats,
    _largest_same_stream_inactive_block_bytes,
    decide_candidate_admission,
    validate_premat_configuration,
)

MIB = 1024 ** 2


def _validation_kwargs(mode: str) -> dict[str, object]:
    return {
        "premat": "enabled",
        "attention_mode": "fused",
        "stay_below_current_peak": False,
        "stay_within_global_buffer": True,
        "gpu_memory_buffer_gb": 0.25,
        "allocator_aware_admission": mode,
        "target_layer": 1,
        "weight_matrix_target_order": "r_to_l",
        "cuda_stream_priority": "normal",
        "diagnostic_layer_delay_ms": 0.0,
        "logging": "disabled",
        "instra": "disabled",
    }


def _blocked_case() -> tuple[PrematMemoryObservation, CandidateEnvelope]:
    return (
        PrematMemoryObservation(
            process_allocated_bytes=500 * MIB,
            process_reserved_bytes=800 * MIB,
            process_ordinary_peak_bytes=1000 * MIB,
            device_free_bytes=100 * MIB,
            device_total_bytes=1000 * MIB,
        ),
        CandidateEnvelope(
            retained_bytes=8 * MIB,
            materialisation_peak_bytes=8 * MIB,
            foreground_overlap_bytes=128 * MIB,
        ),
    )


def _runtime() -> PrematRuntime:
    runtime = PrematRuntime(
        materialize=lambda _family, _layer: torch.empty(1),
        n_embd=8,
        n_head=1,
        attention_mode="fused",
        stay_below_current_peak=False,
        gpu_memory_buffer_gb=200 * MIB / (1024 ** 3),
        allocator_aware_admission="cautious",
        target_layer=1,
        weight_matrix_target_order="r_to_l",
        cuda_stream_priority="normal",
        diagnostic_layer_delay_ms=0.0,
        logging_enabled=False,
    )
    class FakeStream:
        cuda_stream = 11
    runtime._stream = FakeStream()
    runtime._device = torch.device("cuda:0")
    return runtime


def _stats(*, reserved=500, active=100, num_device_free=7):
    return {
        "reserved_bytes.all.current": reserved * MIB,
        "active_bytes.all.current": active * MIB,
        "num_device_free": num_device_free,
    }


def _snapshot(block_mib=368):
    return [
        {
            "device": 0,
            "stream": 11,
            "segment_type": "large",
            "blocks": [{"state": "inactive", "size": block_mib * MIB}],
        }
    ]


def test_inactive_allocator_bytes_uses_reserved_minus_active() -> None:
    assert _inactive_allocator_bytes_from_stats(_stats(reserved=500, active=180)) == 320 * MIB
    with pytest.raises(ValueError):
        _inactive_allocator_bytes_from_stats(_stats(reserved=100, active=101))


def test_largest_block_filters_device_pool_stream_and_state() -> None:
    snapshot = [
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 64 * MIB},
            {"state": "active_awaiting_free", "size": 700 * MIB},
        ]},
        {"device": 0, "stream": 12, "segment_type": "large", "blocks": [{"state": "inactive", "size": 800 * MIB}]},
        {"device": 1, "stream": 11, "segment_type": "large", "blocks": [{"state": "inactive", "size": 900 * MIB}]},
    ]
    assert _largest_same_stream_inactive_block_bytes(
        snapshot, device_index=0, stream_id=11, request_bytes=8 * MIB
    ) == 64 * MIB


def test_certificate_charge_is_conservative_power_of_two_bound() -> None:
    assert _conservative_certificate_charge_bytes(0) == 0
    assert _conservative_certificate_charge_bytes(8 * MIB) == 8 * MIB
    assert _conservative_certificate_charge_bytes(12 * MIB) == 16 * MIB
    assert _conservative_certificate_charge_bytes(513) == 1024


def test_cautious_uses_stats_headroom_and_reuses_snapshot_certificate(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation, envelope=envelope,
        stay_below_current_peak=False, gpu_memory_buffer_bytes=200 * MIB,
    )
    assert not original.admitted
    runtime = _runtime()
    calls = {"snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    def snapshot():
        calls["snapshot"] += 1
        return _snapshot()
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)

    revised1, detail1 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised1.admitted
    assert detail1["snapshot_performed"] is True
    assert detail1["inactive_allocator_bytes"] == 400 * MIB
    runtime._consume_allocator_certificate(envelope.materialisation_peak_bytes)

    revised2, detail2 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised2.admitted
    assert detail2["snapshot_performed"] is False
    assert calls["snapshot"] == 1


def test_device_free_event_invalidates_certificate_and_forces_refresh(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation, envelope=envelope,
        stay_below_current_peak=False, gpu_memory_buffer_bytes=200 * MIB,
    )
    runtime = _runtime()
    state = {"free": 7, "snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats(num_device_free=state["free"]))
    def snapshot():
        state["snapshot"] += 1
        return _snapshot()
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert state["snapshot"] == 1
    state["free"] = 8
    _, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert detail["certificate_invalidated_by_device_free"] is True
    assert detail["snapshot_performed"] is True
    assert state["snapshot"] == 2


def test_insufficient_current_inactive_headroom_still_rejects(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation, envelope=envelope,
        stay_below_current_peak=False, gpu_memory_buffer_bytes=200 * MIB,
    )
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats(reserved=180, active=100))
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: _snapshot())
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert detail["reuse_qualified"] is True
    assert detail["inactive_allocator_bytes"] == 80 * MIB
    assert revised.admitted is False
    assert revised.reason == "global_device_buffer"


def test_snapshot_failure_fails_closed_when_certificate_missing(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation, envelope=envelope,
        stay_below_current_peak=False, gpu_memory_buffer_bytes=200 * MIB,
    )
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("snapshot failed")))
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised == original
    assert "snapshot failed" in str(detail["snapshot_error"])


def test_allocator_aware_mode_validation() -> None:
    validate_premat_configuration(**_validation_kwargs("disabled"))
    validate_premat_configuration(**_validation_kwargs("cautious"))
    for mode in ("normal", "aggressive"):
        with pytest.raises(ValueError, match="not implemented yet"):
            validate_premat_configuration(**_validation_kwargs(mode))
