from __future__ import annotations

from types import SimpleNamespace

import constants
import pytest
import torch

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


def _coefficient_record(
    *,
    update: int,
    active_layers: int = 3,
    maximum_layers: int = 8,
):
    return {
        "step_id": f"U{update}",
        "optimizer_update": update,
        "active_layers": active_layers,
        "maximum_layers": maximum_layers,
        "capacity_layer_sample_numbers": tuple(
            1.0 + index * (maximum_layers - 1.0) / max(1, active_layers - 1)
            for index in range(active_layers)
        ),
        "coefficients": tuple(update + index / 10.0 for index in range(active_layers)),
    }


def test_coefficient_history_is_rolling_300_steps() -> None:
    telemetry = SimpleNamespace()
    history = curves._ensure_coefficient_curve_state(telemetry)
    for update in range(1, 306):
        history.append(_coefficient_record(update=update))
    assert len(history) == 300
    assert history[0]["step_id"] == "U6"


def test_coefficient_record_uses_extended_capacity_layer_sample_numbers() -> None:
    class _Lattice:
        current_active_layers = 3
        maximum_layers = 8

        @staticmethod
        def interval_report():
            return {"active_sample_layer_coordinates": (1.0, 3.75, 8.0)}

    class _Materializer:
        @staticmethod
        def direct_matrix_value(name, layer_index, output_row, row_index):
            assert name == "attention_query_weight"
            assert output_row == row_index == 0
            return torch.tensor((0.25, -0.50, 0.75)[layer_index])

    lattice = _Lattice()
    trainer = SimpleNamespace(
        _plastic_depth_lattice=lambda: lattice,
        raw_model=SimpleNamespace(semantic_materializer=_Materializer()),
    )

    record = curves._coefficient_record_from_trainer(
        trainer,
        optimizer_update=17,
    )

    assert record["step_id"] == "U17"
    assert record["capacity_layer_sample_numbers"] == (1.0, 3.75, 8.0)
    assert record["coefficients"] == pytest.approx((0.25, -0.50, 0.75))


def test_bounded_coefficient_rows_drop_oldest_complete_steps() -> None:
    history = tuple(
        _coefficient_record(update=update, active_layers=48, maximum_layers=48)
        for update in range(1, 301)
    )

    rows = curves._bounded_rows_for_coefficients(history)

    assert len(rows) <= curves._MAX_TABLE_ROWS
    assert rows[-1][2] == "U300"
    assert rows[0][2] != "U1"
    first_step = rows[0][2]
    assert sum(row[2] == first_step for row in rows) == 48
    assert [row[0] for row in rows[-48:]] == pytest.approx(
        history[-1]["capacity_layer_sample_numbers"]
    )


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


# vvv THOG retain the three probe charts, add the sampled-coefficients chart and group all four under plastic without touching TensorBoard
def test_wandb_curve_logger_emits_four_plastic_line_charts_and_never_touches_writer(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 3)
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
    coefficient_history = curves._ensure_coefficient_curve_state(telemetry)
    coefficient_history.extend(
        (_coefficient_record(update=9), _coefficient_record(update=10))
    )
    telemetry._plastic_coefficient_curve_total = 2

    curves._log_rolling_probe_charts(telemetry, step=10)

    assert len(run.calls) == 1
    payload, step = run.calls[0]
    assert step == 10
    assert set(payload) == {
        "plastic/probe_shrink_curves",
        "plastic/probe_growth_curves",
        "plastic/probe_combined_curves",
        "plastic/sampled_coefficients_curves",
    }
    assert payload["plastic/probe_shrink_curves"]["x"] == "distance"
    assert payload["plastic/probe_growth_curves"]["x"] == "distance"
    assert payload["plastic/probe_combined_curves"]["x"] == "offset"
    assert "shrink/interpolation side" in payload["plastic/probe_shrink_curves"]["title"]
    assert "grow/extrapolation side" in payload["plastic/probe_growth_curves"]["title"]
    assert "shrink/interpolation : grow/extrapolation" in payload["plastic/probe_combined_curves"]["title"]
    coefficients = payload["plastic/sampled_coefficients_curves"]
    assert coefficients["x"] == "capacity_layer_sample_number"
    assert coefficients["y"] == "sampled_coefficient"
    assert coefficients["stroke"] == "step_id"
    assert "coefficients" in coefficients["title"]
    assert {row[2] for row in coefficients["table"].data} == {"U9", "U10"}
    assert coefficients["table"].columns == list(curves._COEFFICIENT_CHART_COLUMNS)
    assert all(key.startswith("plastic/") for key in payload)
    for key, chart in payload.items():
        if key == "plastic/sampled_coefficients_curves":
            continue
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


# vvv THOG the PLASTIC chart surface is enabled exactly above DEBUG 2 and remains independent of the broader DEBUG>9 forensic telemetry gate
@pytest.mark.parametrize("debug", (0, 1, 2))
def test_wandb_curve_logger_is_disabled_at_debug_two_or_lower(monkeypatch, debug) -> None:
    monkeypatch.setattr(constants, "DEBUG", debug)
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


def test_coefficient_refresh_uses_early_then_twenty_five_step_cadence() -> None:
    telemetry = SimpleNamespace()
    history = curves._ensure_coefficient_curve_state(telemetry)
    for update in range(1, 11):
        history.append(_coefficient_record(update=update))
        telemetry._plastic_coefficient_curve_total = update
        assert curves._should_refresh_coefficient_chart(
            telemetry,
            evaluation=False,
        )
        telemetry._plastic_coefficient_curve_last_logged_total = update

    for update in range(11, 35):
        history.append(_coefficient_record(update=update))
        telemetry._plastic_coefficient_curve_total = update
        assert not curves._should_refresh_coefficient_chart(
            telemetry,
            evaluation=False,
        )

    history.append(_coefficient_record(update=35))
    telemetry._plastic_coefficient_curve_total = 35
    assert curves._should_refresh_coefficient_chart(
        telemetry,
        evaluation=False,
    )


def test_attachment_captures_every_successful_update_outside_console_cadence(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 3)

    class _Lattice:
        current_active_layers = 2
        maximum_layers = 4

        @staticmethod
        def interval_report():
            return {"active_sample_layer_coordinates": (1.0, 4.0)}

    class _Materializer:
        @staticmethod
        def direct_matrix_value(_name, layer_index, _output_row, _row_index):
            return torch.tensor((0.125, -0.25)[layer_index])

    state = SimpleNamespace(completed_updates=0)

    def train_one_update():
        state.completed_updates += 1
        return {"skipped_update": 0.0}

    def timed(function):
        return function(), 0.5

    lattice = _Lattice()
    trainer = SimpleNamespace(
        _timed=timed,
        train_one_update=train_one_update,
        _print_progress=lambda *_args, **_kwargs: None,
        _plastic_depth_lattice=lambda: lattice,
        raw_model=SimpleNamespace(semantic_materializer=_Materializer()),
        distributed=SimpleNamespace(is_primary=True),
        config=SimpleNamespace(plastic__enabled=True),
        state=state,
        events=[],
    )
    telemetry = SimpleNamespace(
        run=_FakeRun(),
        module=SimpleNamespace(Table=_FakeTable, plot=_FakePlot),
        log_event=lambda *_args, **_kwargs: None,
        log_depth_curve_figures=lambda *_args, **_kwargs: None,
    )

    curves.attach_telemetry_with_plastic_probe_curves(trainer, telemetry)
    metrics, elapsed = trainer._timed(trainer.train_one_update)

    assert metrics["skipped_update"] == 0.0
    assert elapsed == 0.5
    assert [row["step_id"] for row in telemetry._plastic_coefficient_curve_history] == ["U1"]


def test_new_probe_events_are_consumed_once() -> None:
    telemetry = SimpleNamespace()
    trainer = SimpleNamespace(events=[_event(completed_updates=9), _event(completed_updates=11)])

    first = curves._consume_new_probe_records(trainer, telemetry)
    second = curves._consume_new_probe_records(trainer, telemetry)

    assert tuple(row["optimizer_update"] for row in first) == (10, 12)
    assert second == ()
    assert len(telemetry._plastic_probe_curve_history) == 2
