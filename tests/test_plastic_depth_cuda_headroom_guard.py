from __future__ import annotations

from types import SimpleNamespace

import sheet.plastic_depth_cuda_headroom_guard_patch as headroom
import sheet.plastic_depth_full_radius_oom_patch as full_radius
from sheet.plastic_depth_controller import (
    PlasticDepthPairedDirectionEvidence,
    PlasticDepthRobustCountDecision,
)
from sheet.plastic_depth_inline import PlasticDepthInlineProbeRequest


def _decision(*, current: int = 12, selected: int = 13) -> PlasticDepthRobustCountDecision:
    return PlasticDepthRobustCountDecision(
        selected_count=selected,
        current_count=current,
        update_number=10,
        brake_active=False,
        last_count_change_update=-1,
        histories={},
        evidence=(
            PlasticDepthPairedDirectionEvidence(
                candidate_count=current + 1,
                direction=1,
                paired_difference=-0.1,
                observation_count=2,
                median=None,
                mad=None,
                sigma=None,
                standardized_improvement=None,
                significant=True,
                feasible=True,
            ),
        ),
    )


def _context() -> dict:
    return {
        "current_count": 12,
        "selected_count": 13,
        "decision": _decision(),
        "plastic_same_batch_precomputed": True,
        "plastic_v055_sen_kendall_report": {"selected_count": 13},
        "plastic_directional_report": {"selected_count": 13},
    }


def _trainer(context, selected_updates):
    return SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        config=SimpleNamespace(plastic__cuda_allocator_reserve_gib=1.5),
        raw_model=SimpleNamespace(
            set_plastic_depth_update_layer_count=lambda count: selected_updates.append(int(count))
        ),
    )


def test_growth_is_held_when_grad_bearing_preflight_exceeds_reserve(monkeypatch) -> None:
    context = _context()
    selected_updates = []
    trainer = _trainer(context, selected_updates)
    releases = []
    reserve = SimpleNamespace(release=lambda **kwargs: releases.append(dict(kwargs)))

    monkeypatch.setattr(headroom, "_ORIGINAL_BEGIN_INLINE_UPDATE", lambda _self: context)
    monkeypatch.setattr(headroom, "_headroom_reserve", lambda _self, _context: (reserve, True))
    monkeypatch.setattr(
        headroom,
        "_same_batch_training_memory_preflight",
        lambda _self, _context, selected_count: False,
    )

    result = headroom._begin_plastic_depth_inline_update_with_cuda_headroom(trainer)

    assert result is context
    assert context["selected_count"] == 12
    assert context["decision"].selected_count == 12
    assert context["plastic_v055_sen_kendall_report"]["selected_count"] == 12
    assert context["plastic_directional_report"]["selected_count"] == 12
    assert context["cuda_growth_headroom_verified"] is False
    assert context["cuda_growth_headroom_reason"] == "grad_bearing_training_exceeds_cuda_reserve_barrier"
    assert selected_updates == [12]
    assert releases == [{"empty_cache": True}]


def test_growth_remains_selected_when_grad_bearing_preflight_succeeds(monkeypatch) -> None:
    context = _context()
    selected_updates = []
    trainer = _trainer(context, selected_updates)
    releases = []
    reserve = SimpleNamespace(release=lambda **kwargs: releases.append(dict(kwargs)))

    monkeypatch.setattr(headroom, "_ORIGINAL_BEGIN_INLINE_UPDATE", lambda _self: context)
    monkeypatch.setattr(headroom, "_headroom_reserve", lambda _self, _context: (reserve, True))
    monkeypatch.setattr(
        headroom,
        "_same_batch_training_memory_preflight",
        lambda _self, _context, selected_count: selected_count == 13,
    )

    result = headroom._begin_plastic_depth_inline_update_with_cuda_headroom(trainer)

    assert result is context
    assert context["selected_count"] == 13
    assert context["decision"].selected_count == 13
    assert context["cuda_growth_headroom_verified"] is True
    assert context["cuda_growth_headroom_reason"] == "grad_bearing_training_with_reserve_succeeded"
    assert releases == []
    assert selected_updates == []


def test_same_batch_stay_releases_full_radius_probe_reserve_before_training(monkeypatch) -> None:
    context = _context()
    context["selected_count"] = 12
    context["decision"] = _decision(selected=12)
    releases = []
    context["cuda_allocator_reserve"] = SimpleNamespace(
        release=lambda **kwargs: releases.append(dict(kwargs))
    )
    trainer = _trainer(context, [])

    monkeypatch.setattr(headroom, "_ORIGINAL_BEGIN_INLINE_UPDATE", lambda _self: context)

    result = headroom._begin_plastic_depth_inline_update_with_cuda_headroom(trainer)

    assert result is context
    assert releases == [{"empty_cache": True}]
    assert context["cuda_allocator_reserve"] is None
    assert context["selected_count"] == 12


def test_full_radius_reserve_stays_live_until_upward_candidate_is_rejected(monkeypatch) -> None:
    releases = []
    reserve = SimpleNamespace(
        active=True,
        release=lambda **kwargs: releases.append(dict(kwargs)),
    )
    context = {
        "current_count": 3,
        "recoverable_upward_counts": (4, 5),
        "candidate_counts": (2, 3, 4, 5),
        "decision_candidate_counts": (2, 3, 4, 5),
        "upward_candidate_feasible_by_count": {4: None, 5: None},
        "cuda_allocator_reserve": reserve,
    }
    base_request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4, 5),
        sampled_token_indices=None,
        selector=lambda candidates: candidates[0][0],
    )
    trainer = SimpleNamespace(
        distributed=SimpleNamespace(all_true=lambda value: bool(value)),
    )

    monkeypatch.setattr(
        full_radius,
        "_ORIGINAL_INLINE_PROBE_REQUEST",
        lambda _self, _targets, _context: base_request,
    )

    request = full_radius._plastic_depth_inline_probe_request_with_full_radius_oom(
        trainer,
        targets=None,
        context=context,
    )
    request.prepare_recoverable_upward_counts()
    assert releases == []
    assert request.synchronize_recoverable_upward_candidate(4, True) is True
    assert releases == []
    assert request.synchronize_recoverable_upward_candidate(5, False) is False
    assert releases == [{"empty_cache": True}]
    assert context["candidate_counts"] == (2, 3, 4)
    assert context["decision_candidate_counts"] == (2, 3, 4)
    assert context["cuda_allocator_reserve"] is None


def test_full_radius_selector_releases_reserve_for_stay(monkeypatch) -> None:
    releases = []
    reserve = SimpleNamespace(
        active=True,
        release=lambda **kwargs: releases.append(dict(kwargs)),
    )
    context = {
        "current_count": 3,
        "recoverable_upward_counts": (4, 5),
        "candidate_counts": (2, 3, 4, 5),
        "decision_candidate_counts": (2, 3, 4, 5),
        "upward_candidate_feasible_by_count": {4: None, 5: None},
        "cuda_allocator_reserve": reserve,
    }
    base_request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4, 5),
        sampled_token_indices=None,
        selector=lambda _candidates: 3,
    )
    trainer = SimpleNamespace(
        distributed=SimpleNamespace(all_true=lambda value: bool(value)),
    )
    monkeypatch.setattr(
        full_radius,
        "_ORIGINAL_INLINE_PROBE_REQUEST",
        lambda _self, _targets, _context: base_request,
    )

    request = full_radius._plastic_depth_inline_probe_request_with_full_radius_oom(
        trainer,
        targets=None,
        context=context,
    )
    assert request.selector(()) == 3
    assert releases == [{"empty_cache": True}]
    assert context["cuda_allocator_reserve"] is None


def test_full_radius_selector_keeps_reserve_for_growth(monkeypatch) -> None:
    releases = []
    reserve = SimpleNamespace(
        active=True,
        release=lambda **kwargs: releases.append(dict(kwargs)),
    )
    context = {
        "current_count": 3,
        "recoverable_upward_counts": (4, 5),
        "candidate_counts": (2, 3, 4, 5),
        "decision_candidate_counts": (2, 3, 4, 5),
        "upward_candidate_feasible_by_count": {4: None, 5: None},
        "cuda_allocator_reserve": reserve,
    }
    base_request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4, 5),
        sampled_token_indices=None,
        selector=lambda _candidates: 4,
    )
    trainer = SimpleNamespace(
        distributed=SimpleNamespace(all_true=lambda value: bool(value)),
    )
    monkeypatch.setattr(
        full_radius,
        "_ORIGINAL_INLINE_PROBE_REQUEST",
        lambda _self, _targets, _context: base_request,
    )

    request = full_radius._plastic_depth_inline_probe_request_with_full_radius_oom(
        trainer,
        targets=None,
        context=context,
    )
    assert request.selector(()) == 4
    assert releases == []
    assert context["cuda_allocator_reserve"] is reserve
