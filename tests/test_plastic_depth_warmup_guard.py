# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import torch

from sheet import plastic_depth_console_minor_patch as console_minor
from sheet import plastic_depth_probe_interval_patch as probe_interval
from sheet import plastic_depth_warmup_guard_patch as guard
from sheet.plastic_depth_controller import (
    PlasticDepthPairedDirectionEvidence,
    PlasticDepthRobustCountDecision,
)
from sheet.plastic_depth_inline import PlasticDepthInlineProbeRequest


def _trainer(*, completed_updates: int) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__freeze_geometry_during_warmup=True,
            plastic__layer_count_probe__probe_every_n_steps=4,
            plastic__layer_count_probe__window_size_as_number_of_probes=16,
            warmup_updates=100,
        ),
        state=SimpleNamespace(
            completed_updates=completed_updates,
            plastic_depth_probe_histories={"32:+1": [-0.10, -0.20, -0.30]},
        ),
    )


def test_sampled_array_is_compact_with_one_decimal_and_new_warmup_label(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        guard,
        "_ORIGINAL_FORMAT_PROGRESS_LINE",
        lambda run_id, event, payload: (
            "sampled = [  1.00,   2.04,  10.0]  <<< warmup braked enabled"
        ),
    )
    rendered = guard._format_progress_line_with_compact_sampled_array(
        "run",
        "optimizer_progress",
        {},
    )
    assert rendered == "sampled = [1.0, 2.0, 10.0]  <<< warmup brake enabled"


def test_warmup_brake_matches_the_optimizer_schedule_boundary() -> None:
    trainer = _trainer(completed_updates=99)
    assert guard._count_warmup_brake_active(trainer)
    assert console_minor._row_has_warmup_brake(trainer, 100)

    trainer.state.completed_updates = 100
    assert not guard._count_warmup_brake_active(trainer)
    assert not console_minor._row_has_warmup_brake(trainer, 101)


def test_frozen_warmup_skips_probe_construction_and_preserves_history(monkeypatch) -> None:
    trainer = _trainer(completed_updates=39)
    history_before = dict(trainer.state.plastic_depth_probe_histories)

    def unexpected_begin(_self):
        raise AssertionError("warmup must not construct a PLASTIC FINE probe")

    monkeypatch.setattr(guard, "_ORIGINAL_BEGIN_INLINE_UPDATE", unexpected_begin)
    assert guard._begin_plastic_depth_inline_update_without_warmup_probe(trainer) is None
    assert trainer.state.plastic_depth_probe_histories == history_before


def test_first_post_warmup_update_delegates_to_probe_path(monkeypatch) -> None:
    trainer = _trainer(completed_updates=100)
    sentinel = {"probe": "started"}
    monkeypatch.setattr(
        guard,
        "_ORIGINAL_BEGIN_INLINE_UPDATE",
        lambda _self: sentinel,
    )

    assert guard._begin_plastic_depth_inline_update_without_warmup_probe(trainer) is sentinel


def test_probe_interval_restarts_immediately_after_frozen_warmup(monkeypatch) -> None:
    trainer = _trainer(completed_updates=9)
    trainer.config.warmup_updates = 10
    sentinel = {"probe": "started"}
    monkeypatch.setattr(
        probe_interval,
        "_ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE",
        lambda _self: sentinel,
    )

    assert probe_interval._begin_plastic_depth_inline_update_with_interval(trainer) is None
    trainer.state.completed_updates = 10
    assert probe_interval._begin_plastic_depth_inline_update_with_interval(trainer) is sentinel
    trainer.state.completed_updates = 11
    assert probe_interval._begin_plastic_depth_inline_update_with_interval(trainer) is None
    trainer.state.completed_updates = 14
    assert probe_interval._begin_plastic_depth_inline_update_with_interval(trainer) is sentinel


def test_warmup_selector_backstop_discards_probe_evidence_if_called_directly(
    monkeypatch,
) -> None:
    trainer = _trainer(completed_updates=39)
    context = {"current_count": 32}
    evidence = PlasticDepthPairedDirectionEvidence(
        candidate_count=33,
        direction=1,
        paired_difference=-0.40,
        observation_count=4,
        median=-0.25,
        mad=0.10,
        sigma=0.15,
        standardized_improvement=1.67,
        significant=True,
        feasible=True,
    )
    decision = PlasticDepthRobustCountDecision(
        selected_count=33,
        current_count=32,
        update_number=40,
        brake_active=False,
        last_count_change_update=-1,
        histories={"32:+1": (-0.40,)},
        evidence=(evidence,),
    )

    def original_request(self, targets, request_context):
        def selector(candidates):
            request_context["selected_count"] = 33
            request_context["decision"] = decision
            request_context["paired_evidence"] = decision.report()
            return 33

        return PlasticDepthInlineProbeRequest(
            candidate_counts=(31, 32, 33),
            sampled_token_indices=None,
            selector=selector,
        )

    monkeypatch.setattr(guard, "_ORIGINAL_INLINE_PROBE_REQUEST", original_request)
    request = guard._plastic_depth_inline_probe_request_with_count_warmup_brake(
        trainer,
        torch.tensor([1]),
        context,
    )
    selected = request.selector(
        (
            (31, torch.tensor(1.0)),
            (32, torch.tensor(1.0)),
            (33, torch.tensor(1.0)),
        )
    )

    assert selected == 32
    assert context["selected_count"] == 32
    assert context["decision"].selected_count == 32
    assert context["decision"].histories == {}
    assert context["warmup_brake_active"] is True
# ^^^ THOG
