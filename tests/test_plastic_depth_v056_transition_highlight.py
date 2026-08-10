from __future__ import annotations

from types import SimpleNamespace

import constants

from sheet import plastic_depth_v056_transition_highlight_patch as highlight


def _trainer(*events):
    return SimpleNamespace(events=list(events))


def _decision_event(previous: int, selected: int, update: int):
    return SimpleNamespace(
        name="plastic_depth_count_decision",
        payload={
            "previous_active_layers": previous,
            "selected_active_layers": selected,
            "last_count_change_update": update if previous != selected else -1,
        },
    )


def test_transition_payload_is_marked_only_on_authoritative_optimizer_row(monkeypatch) -> None:
    trainer = _trainer(_decision_event(21, 22, 455))
    monkeypatch.setattr(
        highlight,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event, payload: dict(payload),
    )

    changed = highlight._prepare_console_progress_payload_with_transition_highlight(
        trainer,
        "optimizer_progress",
        {"completed_updates": 455},
    )
    later = highlight._prepare_console_progress_payload_with_transition_highlight(
        trainer,
        "optimizer_progress",
        {"completed_updates": 456},
    )
    validation = highlight._prepare_console_progress_payload_with_transition_highlight(
        trainer,
        "evaluation_completed",
        {"completed_updates": 455},
    )

    assert changed[highlight._TRANSITION_PAYLOAD_KEY] == 22
    assert highlight._TRANSITION_PAYLOAD_KEY not in later
    assert highlight._TRANSITION_PAYLOAD_KEY not in validation


def test_stet_does_not_create_transition_highlight(monkeypatch) -> None:
    trainer = _trainer(_decision_event(21, 21, 455))
    monkeypatch.setattr(
        highlight,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event, payload: dict(payload),
    )
    values = highlight._prepare_console_progress_payload_with_transition_highlight(
        trainer,
        "optimizer_progress",
        {"completed_updates": 455},
    )
    assert highlight._TRANSITION_PAYLOAD_KEY not in values


def test_formatter_highlights_only_numeric_new_count_once(monkeypatch) -> None:
    monkeypatch.setattr(
        highlight,
        "_ORIGINAL_FORMAT_PROGRESS_LINE",
        lambda _run_id, _event, _payload: "T 455 ... layers  22  sampled [1.0, 2.0]",
    )
    rendered = highlight._format_progress_line_with_transition_highlight(
        "run",
        "optimizer_progress",
        {highlight._TRANSITION_PAYLOAD_KEY: 22},
    )
    assert "layers  " in rendered
    assert f"{constants.BOLD}{constants.YELLOW}22{constants.R}" in rendered
    assert f"{constants.BOLD}{constants.YELLOW}layers" not in rendered

    ordinary = highlight._format_progress_line_with_transition_highlight(
        "run",
        "optimizer_progress",
        {},
    )
    assert f"{constants.BOLD}{constants.YELLOW}22{constants.R}" not in ordinary


def test_first_row_without_transition_event_never_highlights(monkeypatch) -> None:
    trainer = _trainer()
    monkeypatch.setattr(
        highlight,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event, payload: dict(payload),
    )
    values = highlight._prepare_console_progress_payload_with_transition_highlight(
        trainer,
        "optimizer_progress",
        {"completed_updates": 1},
    )
    assert highlight._TRANSITION_PAYLOAD_KEY not in values
