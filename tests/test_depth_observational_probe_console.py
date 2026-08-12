# vvv THOG
from __future__ import annotations

import re
from types import SimpleNamespace

from sheet import plastic_depth_console_postfix_patch as _installed_depth_overlays
from sheet import depth_observational_probe_console_patch as observational_console
from sheet import stage6_trainer as stage6


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _event():
    return SimpleNamespace(
        name="plastic_depth_count_decision",
        payload={
            "previous_active_layers": 4,
            "selected_active_layers": 4,
            "observational_only": True,
            "probe_update": 5,
            "candidates": (
                {"active_layers": 2, "validation_loss": 5.20},
                {"active_layers": 3, "validation_loss": 5.10},
                {"active_layers": 4, "validation_loss": 5.00},
                {"active_layers": 5, "validation_loss": 4.98},
                {"active_layers": 6, "validation_loss": 5.03},
            ),
        },
    )


# vvv THOG the observational event parser preserves full-radius candidate order and never invents a decision direction
def test_latest_observational_probe_report_is_full_radius_and_read_only() -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(plastic__do_learn_layer_count=False),
        events=[_event()],
    )

    report = observational_console._latest_observational_probe_report(trainer)

    assert report is not None
    assert report["probe_update"] == 5
    assert report["current_layer_count"] == 4
    assert report["offsets"] == (-2, -1, 0, 1, 2)
    assert report["losses"] == (5.20, 5.10, 5.00, 4.98, 5.03)
# ^^^ THOG


# vvv THOG a just-completed fixed-run observational probe is injected into the ordinary T-row and rendered by the established probe-delta formatter
def test_observational_probe_is_visible_on_fixed_run_console_row(monkeypatch) -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(plastic__do_learn_layer_count=False),
        state=SimpleNamespace(completed_updates=5),
        events=[_event()],
    )
    monkeypatch.setattr(
        observational_console,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event_name, payload: dict(payload),
    )
    monkeypatch.setattr(
        observational_console._depth,
        "_observational_probe_enabled",
        lambda _trainer: True,
    )

    values = observational_console._prepare_console_progress_payload_with_observational_probe(
        trainer,
        "optimizer_progress",
        {
            "completed_updates": 5,
            "cumulative_training_seconds": 10,
            "training_loss": 4.9,
            "learning_rate": 1.0e-4,
            "gradient_norm": 1.0,
        },
    )
    assert values["current_layer_count"] == 4
    assert values["plastic_probe_offsets"] == (-2, -1, 0, 1, 2)
    assert values["plastic_probe_losses"] == (5.20, 5.10, 5.00, 4.98, 5.03)

    rendered = _plain(stage6.format_progress_line("fixed", "optimizer_progress", values))
    assert "probe_Δloss [L-2 .. L+2]" in rendered
    assert "+0.200, +0.100, 5.000, -0.020, +0.030" in rendered
# ^^^ THOG


# vvv THOG stale observational evidence is not repeated on later ordinary progress rows
def test_observational_probe_console_visibility_is_update_local(monkeypatch) -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(plastic__do_learn_layer_count=False),
        state=SimpleNamespace(completed_updates=6),
        events=[_event()],
    )
    monkeypatch.setattr(
        observational_console,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event_name, payload: dict(payload),
    )
    monkeypatch.setattr(
        observational_console._depth,
        "_observational_probe_enabled",
        lambda _trainer: True,
    )

    values = observational_console._prepare_console_progress_payload_with_observational_probe(
        trainer,
        "optimizer_progress",
        {"completed_updates": 6},
    )

    assert "plastic_probe_offsets" not in values
    assert "plastic_probe_losses" not in values
# ^^^ THOG
