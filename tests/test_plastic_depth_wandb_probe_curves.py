from __future__ import annotations

from types import SimpleNamespace

import pytest

from sheet import plastic_depth_wandb_probe_curves_patch as curves
from sheet.trainer_state import TrainerEvent


def _event(*, completed_updates: int = 9, current: int = 10, selected: int = 10) -> TrainerEvent:
    return TrainerEvent(
        "plastic_depth_count_decision",
        completed_updates,
        {
            "previous_active_layers": current,
            "selected_active_layers": selected,
            "candidates": (
                {"active_layers": current - 2, "validation_loss": 5.2501234},
                {"active_layers": current - 1, "validation_loss": 5.1004567},
                {"active_layers": current, "validation_loss": 5.0000000},
                {"active_layers": current + 1, "validation_loss": 4.9987654},
                {"active_layers": current + 2, "validation_loss": 5.0043210},
            ),
        },
    )


def test_probe_record_preserves_raw_precision_and_splits_sides() -> None:
    record = curves._probe_record_from_event(_event())
    assert record is not None
    assert record["probe_id"] == "U10"
    assert record["active_layers"] == 10
    assert record["shrink"][0] == (0, 0.0, 10, 0)
    assert record["growth"][0] == (0, 0.0, 10, 0)
    assert record["shrink"][1][1] == pytest.approx(0.1004567)
    assert record["growth"][1][1] == pytest.approx(-0.0012346)
    assert record["growth"][2][1] == pytest.approx(0.0043210)


def test_curve_history_is_rolling_300_probes() -> None:
    telemetry = SimpleNamespace()
    history = curves._ensure_curve_state(telemetry)
    for index in range(305):
        history.append({"probe_id": f"P{index}"})
    assert len(history) == 300
    assert history[0]["probe_id"] == "P5"


def test_bounded_rows_drop_oldest_complete_probes_before_wandb_limit() -> None:
    points = tuple((distance, float(distance), 10 + distance, distance) for distance in range(40))
    history = tuple(
        {
            "probe_id": f"P{probe_index}",
            "optimizer_update": probe_index,
            "active_layers": 10,
            "selected_layers": 10,
            "growth": points,
        }
        for probe_index in range(300)
    )

    rows = curves._bounded_rows_for_side(history, "growth")

    assert len(rows) <= curves._MAX_TABLE_ROWS
    assert rows[-1][2] == "P299"
    assert rows[0][2] != "P0"
    first_probe = rows[0][2]
    assert sum(row[2] == first_probe for row in rows) == len(points)


class _FakeTable:
    def __init__(self, *, data, columns):
        self.data = data
        self.columns = columns


class _FakePlot:
    @staticmethod
    def line(*, table, x, y, stroke, title):
        return {
            "table": table,
            "x": x,
            "y": y,
            "stroke": stroke,
            "title": title,
        }


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step=None):
        self.calls.append((payload, step))


def test_wandb_curve_logger_emits_two_line_charts_and_never_touches_writer() -> None:
    run = _FakeRun()
    writer = SimpleNamespace(add_scalar=lambda *_args, **_kwargs: pytest.fail("TensorBoard must not be used"))
    telemetry = SimpleNamespace(
        run=run,
        module=SimpleNamespace(Table=_FakeTable, plot=_FakePlot),
        writer=writer,
    )
    history = curves._ensure_curve_state(telemetry)
    record = curves._probe_record_from_event(_event())
    assert record is not None
    history.append(record)

    curves._log_rolling_probe_charts(telemetry, step=10)

    assert len(run.calls) == 1
    payload, step = run.calls[0]
    assert step == 10
    assert set(payload) == {
        "fine/plastic_probe_shrink_curves",
        "fine/plastic_probe_growth_curves",
    }
    for chart in payload.values():
        assert chart["x"] == "distance"
        assert chart["y"] == "delta_loss"
        assert chart["stroke"] == "probe_id"
        assert chart["table"].columns == list(curves._CHART_COLUMNS)


def test_new_probe_events_are_consumed_once() -> None:
    telemetry = SimpleNamespace()
    trainer = SimpleNamespace(events=[_event(completed_updates=9), _event(completed_updates=11)])

    first = curves._consume_new_probe_records(trainer, telemetry)
    second = curves._consume_new_probe_records(trainer, telemetry)

    assert tuple(row["optimizer_update"] for row in first) == (10, 12)
    assert second == ()
    assert len(telemetry._plastic_probe_curve_history) == 2
