from __future__ import annotations

import pytest
import torch

from sheet.premat import (
    CandidateEnvelope,
    PrematMemoryObservation,
    PrematRuntime,
    _allocator_pool_reserved_bytes_from_stats,
    _conservative_certificate_charge_bytes,
    _inactive_allocator_bytes_from_stats,
    _largest_same_stream_inactive_block_bytes,
    _same_stream_inactive_block_sizes,
    _same_stream_inactive_block_summary,
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


def _stats(*, reserved=500, active=100, pool_reserved=480, num_device_free=7):
    return {
        "reserved_bytes.all.current": reserved * MIB,
        "active_bytes.all.current": active * MIB,
        "reserved_bytes.large_pool.current": pool_reserved * MIB,
        "reserved_bytes.small_pool.current": (reserved - pool_reserved) * MIB,
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


def _original_decision(observation, envelope):
    return decide_candidate_admission(
        observation=observation, envelope=envelope,
        stay_below_current_peak=False, gpu_memory_buffer_bytes=200 * MIB,
    )


def test_memory_stats_helpers() -> None:
    stats = _stats(reserved=500, active=180, pool_reserved=470)
    assert _inactive_allocator_bytes_from_stats(stats) == 320 * MIB
    assert _allocator_pool_reserved_bytes_from_stats(stats, "large") == 470 * MIB
    assert _allocator_pool_reserved_bytes_from_stats(stats, "small") == 30 * MIB
    with pytest.raises(ValueError):
        _inactive_allocator_bytes_from_stats(_stats(reserved=100, active=101, pool_reserved=90))


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


def test_cautious_certificate_is_live_reserve_not_historical_spend(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
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
    reserve = runtime._reserve_allocator_certificate(
        request_bytes=envelope.materialisation_peak_bytes, required=True
    )
    assert reserve["tracked"] is True
    assert reserve["after_available_bytes"] == 360 * MIB

    revised2, detail2 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised2.admitted
    assert detail2["snapshot_performed"] is False
    assert calls["snapshot"] == 1

    runtime._release_allocator_certificate_charge(
        generation=reserve["generation"], pool=reserve["pool"],
        block_id=reserve["block_id"], charge_bytes=reserve["charge_bytes"],
    )
    assert runtime._allocator_certificate_available_bytes("large") == 368 * MIB
    revised3, detail3 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised3.admitted
    assert detail3["snapshot_performed"] is False
    assert calls["snapshot"] == 1


def test_device_free_counter_change_does_not_invalidate_stream_certificate(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    state = {"free": 7, "snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda, "memory_stats",
        lambda _device=None: _stats(num_device_free=state["free"]),
    )
    def snapshot():
        state["snapshot"] += 1
        return _snapshot()
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    state["free"] = 999
    _, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert detail["certificate_invalidated_by_device_free"] is False
    assert detail["snapshot_performed"] is False
    assert state["snapshot"] == 1


def test_pool_reserved_shrink_invalidates_pool_until_live_charge_drains(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    state = {"pool_reserved": 480, "snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda, "memory_stats",
        lambda _device=None: _stats(pool_reserved=state["pool_reserved"]),
    )
    def snapshot():
        state["snapshot"] += 1
        return _snapshot()
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    admitted, _ = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert admitted.admitted
    reserve = runtime._reserve_allocator_certificate(
        request_bytes=envelope.materialisation_peak_bytes, required=True
    )
    state["pool_reserved"] = 380
    blocked, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert blocked == original
    assert detail["certificate_pool_reserved_shrink_debit_bytes"] == 100 * MIB
    assert detail["certificate_refresh_deferred_live_charge"] is True
    assert runtime._allocator_certificate_valid["large"] is False
    assert runtime._allocator_certificate_available_bytes("large") == 0
    assert state["snapshot"] == 1
    runtime._release_allocator_certificate_charge(
        generation=reserve["generation"], pool=reserve["pool"],
        block_id=reserve["block_id"], charge_bytes=reserve["charge_bytes"],
    )
    refreshed, detail2 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert refreshed.admitted
    assert detail2["snapshot_performed"] is True
    assert state["snapshot"] == 2


def test_no_overlapping_snapshot_while_certificate_charge_live(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    calls = {"snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    def snapshot():
        calls["snapshot"] += 1
        return _snapshot(block_mib=8)
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    revised, _ = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert revised.admitted
    reserve = runtime._reserve_allocator_certificate(
        request_bytes=envelope.materialisation_peak_bytes, required=True
    )
    blocked, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert blocked == original
    assert detail["certificate_refresh_deferred_live_charge"] is True
    assert detail["snapshot_performed"] is False
    assert calls["snapshot"] == 1
    runtime._release_allocator_certificate_charge(
        generation=reserve["generation"], pool=reserve["pool"],
        block_id=reserve["block_id"], charge_bytes=reserve["charge_bytes"],
    )
    admitted, detail2 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert admitted.admitted
    assert detail2["snapshot_performed"] is False
    assert calls["snapshot"] == 1


def test_insufficient_current_inactive_headroom_still_rejects(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda, "memory_stats",
        lambda _device=None: _stats(reserved=180, active=100, pool_reserved=160),
    )
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
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    monkeypatch.setattr(
        torch.cuda, "memory_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )
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



def test_snapshot_summary_reports_multiple_compatible_blocks() -> None:
    snapshot = [
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 64 * MIB},
            {"state": "inactive", "size": 32 * MIB},
            {"state": "inactive", "size": 16 * MIB},
            {"state": "inactive", "size": 4 * MIB},
            {"state": "active_allocated", "size": 999 * MIB},
        ]},
        {"device": 0, "stream": 12, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 800 * MIB},
        ]},
    ]
    summary = _same_stream_inactive_block_summary(
        snapshot, device_index=0, stream_id=11, request_bytes=8 * MIB
    )
    assert summary == {
        "block_count": 4,
        "sufficient_block_count": 3,
        "total_bytes": 116 * MIB,
        "largest_block_bytes": 64 * MIB,
        "second_largest_block_bytes": 32 * MIB,
        "third_largest_block_bytes": 16 * MIB,
    }


def test_snapshot_sizes_expose_full_certified_pool() -> None:
    snapshot = [
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 64 * MIB},
            {"state": "inactive", "size": 32 * MIB},
            {"state": "inactive", "size": 16 * MIB},
            {"state": "inactive", "size": 4 * MIB},
        ]},
    ]
    assert _same_stream_inactive_block_sizes(
        snapshot, device_index=0, stream_id=11, request_bytes=8 * MIB
    ) == (64 * MIB, 32 * MIB, 16 * MIB, 4 * MIB)


def test_certificate_pool_uses_individual_best_fit_blocks() -> None:
    runtime = _runtime()
    runtime._allocator_certificate_valid["large"] = True
    runtime._allocator_certificate_generation["large"] = 3
    runtime._allocator_certificate_blocks["large"] = [
        {"block_id": 1, "capacity_bytes": 64 * MIB, "live_charge_bytes": 0},
        {"block_id": 2, "capacity_bytes": 32 * MIB, "live_charge_bytes": 0},
        {"block_id": 3, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
    ]
    runtime._sync_allocator_certificate_totals("large")
    first = runtime._reserve_allocator_certificate(request_bytes=20 * MIB, required=True)
    assert first["charge_bytes"] == 32 * MIB
    assert first["block_id"] == 2
    second = runtime._reserve_allocator_certificate(request_bytes=12 * MIB, required=True)
    assert second["charge_bytes"] == 16 * MIB
    assert second["block_id"] == 3
    assert runtime._allocator_certificate_available_bytes("large") == 64 * MIB
    assert runtime._allocator_certificate_largest_available_block_bytes("large") == 64 * MIB
    runtime._release_allocator_certificate_charge(
        generation=first["generation"], pool=first["pool"], block_id=first["block_id"],
        charge_bytes=first["charge_bytes"],
    )
    assert runtime._allocator_certificate_available_bytes("large") == 96 * MIB


def test_certificate_pool_never_treats_fragmented_total_as_one_block() -> None:
    runtime = _runtime()
    runtime._allocator_certificate_valid["large"] = True
    runtime._allocator_certificate_generation["large"] = 4
    runtime._allocator_certificate_blocks["large"] = [
        {"block_id": 1, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
        {"block_id": 2, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
    ]
    runtime._sync_allocator_certificate_totals("large")
    assert runtime._allocator_certificate_available_bytes("large") == 32 * MIB
    assert runtime._allocator_certificate_sufficient_block_count("large", 32 * MIB) == 0
    with pytest.raises(RuntimeError, match="lost its certified"):
        runtime._reserve_allocator_certificate(request_bytes=20 * MIB, required=True)


def test_advance_rejection_does_not_head_of_line_block_later_matrices(monkeypatch) -> None:
    runtime = _runtime()
    runtime._allocator_aware_admission = "disabled"
    runtime._layer_indices = (0, 1)
    runtime._position = 0
    runtime._ensure_layer(1)
    observation, _ = _blocked_case()
    monkeypatch.setattr(runtime, "_refresh_available", lambda: None)
    monkeypatch.setattr(runtime, "_observe_memory", lambda: observation)
    monkeypatch.setattr(runtime, "_update_memory_aggregates", lambda _observation: None)
    monkeypatch.setattr(runtime, "_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_record_advance_return", lambda **kwargs: None)
    runtime._advance(trigger="test")
    layer_candidates = [
        candidate
        for candidate in runtime._candidates.values()
        if candidate.layer_index == 1
    ]
    assert len(layer_candidates) == 4
    assert all(candidate.first_considered_ns is not None for candidate in layer_candidates)
    assert runtime._aggregate["deferred"] == 4


def test_cautious_detail_includes_pending_release_diagnostics(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()

    class Pending:
        retained_bytes = 8 * MIB
        transient_bytes = 2 * MIB
        allocator_certificate_charge_bytes = 16 * MIB

    runtime._pending_releases = [Pending()]
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: _snapshot())
    _, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert detail["pending_release_count"] == 1
    assert detail["pending_release_bytes"] == 10 * MIB
    assert detail["certificate_bytes_waiting_for_release"] == 16 * MIB
