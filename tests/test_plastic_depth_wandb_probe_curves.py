from __future__ import annotations

from types import SimpleNamespace

import constants
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
    assert record["shrink"][1][1] == pytest.approx(0.2501234)
    assert record["shrink"][2][1] == pytest.approx(0.1004567)
    assert record["growth"][1][1] == pytest.approx(-0.0012346)
    assert record["growth"][2][1] == pytest.approx(0.0043210)


# vvv THOG combined W&B landscape must concatenate both established sides in signed layer order and retain the shared current-L point only once
def test_combined_rows_span_negative_to_positive_offsets_with_one_center() -> None:
    record = curves._probe_record_from_event(_event())
    assert record is not None

    rows = curves._rows_for_combined((record,))

    assert [row[7] for row in rows] == [-2, -1, 0, 1, 2]
    assert [row[6] for row in rows] == [8, 9, 10, 11, 12]
    assert sum(row[7] == 0 for row in rows) == 1
    assert rows[0][1] == pytest.approx(0.2501234)
    assert rows[1][1] == pytest.approx(0.1004567)
    assert rows[2][1] == pytest.approx(0.0)
    assert rows[3][1] == pytest.approx(-0.0012346)
    assert rows[4][1] == pytest.approx(0.0043210)
# ^^^ THOG


# vvv THOG the built-in W&B line preset gets an explicit zero-loss reference spanning the complete visible x domain
@pytest.mark.parametrize(
    ("side", "x", "x_index"),
    (("shrink", "distance", 0), ("growth", "distance", 0), ("combined", "offset", 7)),
)
def test_zero_loss_reference_spans_each_chart_domain(side: str, x: str, x_index: int) -> None:
    record = curves._probe_record_from_event(_event())
    assert record is not None
    rows = (
        curves._rows_for_combined((record,))
        if side == "combined"
        else curves._rows_for_side((record,), side)
    )

    rendered = curves._rows_with_zero_loss_reference(rows, x=x)
    reference = [row for row in rendered if row[2] == curves._ZERO_LOSS_REFERENCE_ID]
    data = [row for row in rendered if row[2] != curves._ZERO_LOSS_REFERENCE_ID]

    assert len(reference) == 2
    assert all(row[1] == 0.0 for row in reference)
    assert [row[x_index] for row in reference] == [
        min(row[x_index] for row in data),
        max(row[x_index] for row in data),
    ]
# ^^^ THOG


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


# vvv THOG the wider combined chart gets the same whole-probe W&B row protection without reducing the retained history used by either split chart
def test_combined_rows_drop_oldest_complete_probes_before_wandb_limit() -> None:
    shrink = tuple((abs(offset), float(offset), 100 + offset, offset) for offset in range(-20, 1))
    growth = tuple((offset, float(offset), 100 + offset, offset) for offset in range(0, 21))
    history = tuple(
        {
            "probe_id": f"P{probe_index}",
            "optimizer_update": probe_index,
            "active_layers": 100,
            "selected_layers": 100,
            "shrink": shrink,
            "growth": growth,
        }
        for probe_index in range(300)
    )

    rows = curves._bounded_rows_for_combined(history)

    assert len(rows) <= curves._MAX_TABLE_ROWS
    assert rows[-1][2] == "P299"
    assert rows[0][2] != "P0"
    first_probe = rows[0][2]
    assert sum(row[2] == first_probe for row in rows) == 41
    assert len(history) == 300
# ^^^ THOG


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


# vvv THOG retain both established split charts and add the joined signed-offset chart without ever touching TensorBoard
def test_wandb_curve_logger_emits_three_line_charts_and_never_touches_writer(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 10)
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
        "fine/plastic_probe_combined_curves",
    }
    assert payload["fine/plastic_probe_shrink_curves"]["x"] == "distance"
    assert payload["fine/plastic_probe_growth_curves"]["x"] == "distance"
    assert payload["fine/plastic_probe_combined_curves"]["x"] == "offset"
    assert "shrink/interpolation side" in payload["fine/plastic_probe_shrink_curves"]["title"]
    assert "grow/extrapolation side" in payload["fine/plastic_probe_growth_curves"]["title"]
    assert "shrink/interpolation : grow/extrapolation" in payload["fine/plastic_probe_combined_curves"]["title"]
    for chart in payload.values():
        assert chart["y"] == "delta_loss"
        assert chart["stroke"] == "probe_id"
        assert chart["table"].columns == list(curves._CHART_COLUMNS)
        assert len(chart["table"].data) <= curves._MAX_TABLE_ROWS
        reference = [
            row
            for row in chart["table"].data
            if row[2] == curves._ZERO_LOSS_REFERENCE_ID
        ]
        assert len(reference) == 2
        assert all(row[1] == 0.0 for row in reference)
# ^^^ THOG


# vvv THOG normal runs neither construct nor emit the expensive W&B PLASTIC charts
def test_wandb_curve_logger_is_disabled_at_normal_debug(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 9)
    run = _FakeRun()
    telemetry = SimpleNamespace(
        run=run,
        module=SimpleNamespace(Table=_FakeTable, plot=_FakePlot),
    )
    history = curves._ensure_curve_state(telemetry)
    record = curves._probe_record_from_event(_event())
    assert record is not None
    history.append(record)

    curves._log_rolling_probe_charts(telemetry, step=10)

    assert run.calls == []
# ^^^ THOG


def test_new_probe_events_are_consumed_once() -> None:
    telemetry = SimpleNamespace()
    trainer = SimpleNamespace(events=[_event(completed_updates=9), _event(completed_updates=11)])

    first = curves._consume_new_probe_records(trainer, telemetry)
    second = curves._consume_new_probe_records(trainer, telemetry)

    assert tuple(row["optimizer_update"] for row in first) == (10, 12)
    assert second == ()
    assert len(telemetry._plastic_probe_curve_history) == 2
