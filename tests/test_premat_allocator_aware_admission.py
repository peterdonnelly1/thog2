from __future__ import annotations

import pytest
import torch

from sheet.premat import (
    CandidateEnvelope,
    PrematMemoryObservation,
    PrematRuntime,
    _largest_same_stream_inactive_block_bytes,
    _total_inactive_allocator_bytes,
    decide_candidate_admission,
    validate_premat_configuration,
)


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
    observation = PrematMemoryObservation(
        process_allocated_bytes=500,
        process_reserved_bytes=800,
        process_ordinary_peak_bytes=1000,
        device_free_bytes=100,
        device_total_bytes=1000,
    )
    envelope = CandidateEnvelope(
        retained_bytes=8,
        materialisation_peak_bytes=8,
        foreground_overlap_bytes=128,
    )
    return observation, envelope


def _runtime() -> PrematRuntime:
    runtime = PrematRuntime(
        materialize=lambda _family, _layer: torch.empty(1),
        n_embd=8,
        n_head=1,
        attention_mode="fused",
        stay_below_current_peak=False,
        gpu_memory_buffer_gb=200 / (1024 ** 3),
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


def test_largest_same_stream_inactive_block_filters_device_pool_stream_and_state() -> None:
    snapshot = [
        {
            "device": 0,
            "stream": 11,
            "segment_type": "small",
            "blocks": [
                {"state": "inactive", "size": 8},
                {"state": "inactive", "size": 16},
                {"state": "active_allocated", "size": 100},
                {"state": "active_awaiting_free", "size": 200},
            ],
        },
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [{"state": "inactive", "size": 1000}]},
        {"device": 0, "stream": 12, "segment_type": "small", "blocks": [{"state": "inactive", "size": 300}]},
        {"device": 1, "stream": 11, "segment_type": "small", "blocks": [{"state": "inactive", "size": 400}]},
    ]
    assert _largest_same_stream_inactive_block_bytes(
        snapshot, device_index=0, stream_id=11, request_bytes=8
    ) == 16


def test_total_inactive_allocator_bytes_counts_only_inactive_current_device() -> None:
    snapshot = [
        {
            "device": 0,
            "stream": 11,
            "segment_type": "small",
            "blocks": [
                {"state": "inactive", "size": 200},
                {"state": "active_allocated", "size": 700},
                {"state": "active_awaiting_free", "size": 800},
            ],
        },
        {"device": 0, "stream": 12, "segment_type": "large", "blocks": [{"state": "inactive", "size": 100}]},
        {"device": 1, "stream": 11, "segment_type": "small", "blocks": [{"state": "inactive", "size": 900}]},
    ]
    assert _total_inactive_allocator_bytes(snapshot, device_index=0) == 300


def test_cautious_rescue_uses_inactive_cache_as_effective_headroom(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=200,
    )
    assert original.admitted is False
    assert original.reason == "global_device_buffer"
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda,
        "memory_snapshot",
        lambda: [
            {"device": 0, "stream": 11, "segment_type": "small", "blocks": [{"state": "inactive", "size": 250}]},
            {"device": 0, "stream": 12, "segment_type": "large", "blocks": [{"state": "inactive", "size": 50}]},
        ],
    )
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert detail["reuse_qualified"] is True
    assert detail["inactive_allocator_bytes"] == 300
    assert detail["required_allocator_headroom_credit_bytes"] == 236
    assert detail["allocator_headroom_credit_bytes"] == 236
    assert detail["effective_device_free_bytes"] == 336
    assert revised.admitted is True
    assert revised.reason == "admitted"
    # The full 136-byte envelope is still charged; allocator cache increases
    # effective headroom rather than being double-counted as zero growth.
    assert revised.predicted_physical_growth_bytes == 136
    assert revised.predicted_device_used_bytes == 800


def test_cautious_rescue_remains_rejected_when_inactive_headroom_is_insufficient(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=200,
    )
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda,
        "memory_snapshot",
        lambda: [
            {"device": 0, "stream": 11, "segment_type": "small", "blocks": [{"state": "inactive", "size": 150}]},
            {"device": 0, "stream": 12, "segment_type": "large", "blocks": [{"state": "inactive", "size": 50}]},
        ],
    )
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert detail["inactive_allocator_bytes"] == 200
    assert detail["allocator_headroom_credit_bytes"] == 200
    assert revised.admitted is False
    assert revised.reason == "global_device_buffer"


def test_allocator_aware_mode_validation() -> None:
    validate_premat_configuration(**_validation_kwargs("disabled"))
    validate_premat_configuration(**_validation_kwargs("cautious"))
    for mode in ("normal", "aggressive"):
        with pytest.raises(ValueError, match="not implemented yet"):
            validate_premat_configuration(**_validation_kwargs(mode))


def test_cautious_rescue_fails_closed_on_snapshot_error(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=200,
    )
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")

    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(torch.cuda, "memory_snapshot", fail_snapshot)
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert revised == original
    assert detail["reuse_qualified"] is False
    assert "RuntimeError: snapshot failed" in str(detail["snapshot_error"])
