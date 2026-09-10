from __future__ import annotations

import pytest
import torch

from sheet.premat import (
    CandidateEnvelope,
    PrematMemoryObservation,
    PrematRuntime,
    _decide_candidate_admission_with_physical_growth,
    _largest_same_stream_inactive_block_bytes,
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
        device_free_bytes=130,
        device_total_bytes=1000,
    )
    envelope = CandidateEnvelope(
        retained_bytes=8,
        materialisation_peak_bytes=8,
        foreground_overlap_bytes=128,
    )
    return observation, envelope


def test_largest_same_stream_inactive_block_ignores_other_states_and_streams() -> None:
    snapshot = [
        {
            "stream": 11,
            "segment_type": "small",
            "blocks": [
                {"state": "inactive", "size": 8},
                {"state": "inactive", "size": 16},
                {"state": "active_allocated", "size": 100},
                {"state": "active_awaiting_free", "size": 200},
            ],
        },
        {"stream": 11, "segment_type": "large", "blocks": [{"state": "inactive", "size": 1000}]},
        {"stream": 12, "segment_type": "small", "blocks": [{"state": "inactive", "size": 300}]},
    ]
    assert _largest_same_stream_inactive_block_bytes(
        snapshot, stream_id=11, request_bytes=8
    ) == 16


def test_allocator_credit_changes_only_physical_growth_not_process_envelope() -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=0,
    )
    revised = _decide_candidate_admission_with_physical_growth(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=0,
        physical_growth_bytes=envelope.foreground_overlap_bytes,
    )
    assert original.admitted is False
    assert original.reason == "global_device_buffer"
    assert original.predicted_physical_growth_bytes == 136
    assert revised.admitted is True
    assert revised.predicted_physical_growth_bytes == 128
    assert revised.predicted_process_bytes == original.predicted_process_bytes == 636


def test_allocator_aware_mode_validation() -> None:
    validate_premat_configuration(**_validation_kwargs("disabled"))
    validate_premat_configuration(**_validation_kwargs("cautious"))
    for mode in ("normal", "aggressive"):
        with pytest.raises(ValueError, match="not implemented yet"):
            validate_premat_configuration(**_validation_kwargs(mode))


def test_cautious_rescue_uses_one_same_stream_inactive_block(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=0,
    )
    runtime = PrematRuntime(
        materialize=lambda _family, _layer: torch.empty(1),
        n_embd=8,
        n_head=1,
        attention_mode="fused",
        stay_below_current_peak=False,
        gpu_memory_buffer_gb=0.0,
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
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda,
        "memory_snapshot",
        lambda: [
            {"stream": 11, "segment_type": "small", "blocks": [{"state": "inactive", "size": 8}]},
            {"stream": 11, "segment_type": "large", "blocks": [{"state": "inactive", "size": 1000}]},
            {"stream": 12, "segment_type": "small", "blocks": [{"state": "inactive", "size": 1000}]},
        ],
    )
    revised, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert revised.admitted is True
    assert detail["reuse_qualified"] is True
    assert detail["largest_eligible_inactive_block_bytes"] == 8
    assert detail["allocator_credit_bytes"] == 8
    assert detail["revised_predicted_physical_growth_bytes"] == 128


def test_cautious_rescue_fails_closed_on_snapshot_error(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = decide_candidate_admission(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=0,
    )
    runtime = PrematRuntime(
        materialize=lambda _family, _layer: torch.empty(1),
        n_embd=8,
        n_head=1,
        attention_mode="fused",
        stay_below_current_peak=False,
        gpu_memory_buffer_gb=0.0,
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
