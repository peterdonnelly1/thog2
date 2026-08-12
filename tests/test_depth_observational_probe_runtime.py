# vvv THOG
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import constants
from sheet import plastic_depth_console_postfix_patch as _installed_depth_overlays
from sheet import depth_observational_probe_executor_patch as observational_executor
from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet import depth_weight_curves_row_limit_patch as depth_row_limit
from sheet import plastic_depth_wandb_probe_curves_patch as probe_wandb
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.model import SheetGPTConfig
from sheet.run_config import OwtRunConfig
from sheet.trainer_step import TrainerStepMixin
from sheet.training_config import TrainingConfig
from sheet.training_model import TrainingSheetGPT


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
    def __init__(self) -> None:
        self.calls = []

    def log(self, payload, step=None) -> None:
        self.calls.append((payload, step))


# vvv THOG tiny standalone DEPTH trajectory keeps chart tests fast while exercising the real coefficient field
def _trajectory() -> DepthTrajectory:
    geometry = SheetGeometryConfig(
        n_layer=4,
        n_embd=8,
        n_head=2,
        depth_order=4,
        base_row_order=1,
        bias=True,
    )
    return DepthTrajectory(
        geometry,
        runtime_dtype=torch.float32,
        depth_compress_layer_norm_and_bias=False,
    )
# ^^^ THOG


def _chart_trainer(trajectory: DepthTrajectory):
    return SimpleNamespace(
        raw_model=SimpleNamespace(trajectory=trajectory),
        config=SimpleNamespace(model_seed=1337),
    )


def _chart_telemetry():
    return SimpleNamespace(
        name="depth-runtime-test",
        group="depth-runtime-test",
        run=_FakeRun(),
        module=SimpleNamespace(Table=_FakeTable, plot=_FakePlot),
    )


# vvv THOG the three effective read-only probe controls survive run/trainer persistence even when every PLASTIC growth switch is off
def test_observational_probe_controls_persist_with_plastic_disabled() -> None:
    run = OwtRunConfig(
        model_type="sheet",
        n_layer=4,
        n_head=2,
        n_embd=16,
        o_depth=3,
        batch_size=2,
        block_size=8,
        max_iters=10,
        warmup_iters=0,
        device="cpu",
        dtype="float32",
        plastic__enabled=False,
        plastic__do_learn_layer_count=False,
        plastic__layer_count_probe__probe_every_n_steps=7,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=5,
        plastic__layer_count_probe_radius=2,
    )

    run_persistent = run.persistent_dict()
    assert run_persistent["plastic__layer_count_probe__probe_every_n_steps"] == 7
    assert run_persistent["plastic__layer_count_probe__number_of_sampled_valid_tokens"] == 5
    assert run_persistent["plastic__layer_count_probe_radius"] == 2

    training = run.to_training_config(
        vocab_size=32,
        world_size=1,
        out_dir=Path("out-observational-probe-test"),
    )
    assert training.plastic__enabled is False
    assert training.plastic__do_learn_layer_count is False
    assert training.plastic__layer_count_probe__probe_every_n_steps == 7
    assert training.plastic__layer_count_probe__number_of_sampled_valid_tokens == 5
    assert training.plastic__layer_count_probe_radius == 2
    training_persistent = training.persistent_dict()
    assert training_persistent["plastic__layer_count_probe__probe_every_n_steps"] == 7
    assert training_persistent["plastic__layer_count_probe__number_of_sampled_valid_tokens"] == 5
    assert training_persistent["plastic__layer_count_probe_radius"] == 2
# ^^^ THOG


# vvv THOG impossible token sampling is rejected before a fixed-run observational probe reaches the trainer loop
@pytest.mark.parametrize("kind", ("run", "training"))
def test_observational_probe_token_capacity_is_validated_when_plastic_disabled(kind: str) -> None:
    common = dict(
        n_layer=4,
        n_head=2,
        n_embd=16,
        batch_size=1,
        block_size=8,
        plastic__enabled=False,
        plastic__do_learn_layer_count=False,
        plastic__layer_count_probe__probe_every_n_steps=3,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=9,
        plastic__layer_count_probe_radius=1,
    )
    with pytest.raises(
        ValueError,
        match="plastic__layer_count_probe__number_of_sampled_valid_tokens",
    ):
        if kind == "run":
            OwtRunConfig(
                model_type="sheet",
                o_depth=3,
                max_iters=10,
                warmup_iters=0,
                device="cpu",
                dtype="float32",
                **common,
            )
        else:
            TrainingConfig(
                model_type="thog2_sheet",
                geometry_preset="depth",
                basis_family="chebyshev",
                depth_order=3,
                base_row_order=8,
                max_updates=10,
                warmup_updates=0,
                device="cpu",
                dtype="float32",
                **common,
            )
# ^^^ THOG


# vvv THOG DEBUG=2 emits none of the new charts, DEBUG=3 emits only depth-group chart keys plus depth-group metadata
def test_depth_weight_charts_are_visible_iff_debug_exceeds_two(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "8")
    trajectory = _trajectory()
    trainer = _chart_trainer(trajectory)

    hidden = _chart_telemetry()
    monkeypatch.setattr(constants, "DEBUG", 2)
    depth_curves._log_depth_weight_snapshot(trainer, hidden, optimizer_update=1)
    assert hidden.run.calls == []

    visible = _chart_telemetry()
    monkeypatch.setattr(constants, "DEBUG", 3)
    depth_curves._log_depth_weight_snapshot(trainer, visible, optimizer_update=1)
    assert len(visible.run.calls) == 2
    chart_payload, chart_step = visible.run.calls[0]
    metadata_payload, metadata_step = visible.run.calls[1]
    assert chart_step == metadata_step == 1
    assert set(chart_payload) == {
        "depth/attn_q_head_N",
        "depth/attn_k_head_N",
        "depth/attn_v_head_N",
        "depth/attn_out_head_N",
        "depth/mlp_up",
        "depth/mlp_down",
    }
    assert all(key.startswith("depth/") for key in chart_payload)
    assert all(key.startswith("depth/") for key in metadata_payload)
# ^^^ THOG


# vvv THOG default accumulate mode cannot exceed the established W&B table ceiling; oldest complete snapshots are discarded first
def test_accumulated_depth_chart_rows_are_bounded_by_wandb_limit(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "3")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "256")
    trajectory = _trajectory()
    trainer = _chart_trainer(trajectory)
    telemetry = _chart_telemetry()
    snapshots = tuple(
        depth_curves._depth_weight_snapshot(
            trainer,
            telemetry,
            optimizer_update=update,
        )
        for update in range(1, 21)
    )

    rows = depth_curves._depth_chart_rows(snapshots, "mlp_up")

    assert len(rows) <= probe_wandb._MAX_TABLE_ROWS
    updates = {int(row[4]) for row in rows}
    assert 20 in updates
    assert 1 not in updates
    assert len(rows) % (3 * 256) == 0
# ^^^ THOG


# vvv THOG a four-layer fixed DEPTH model genuinely executes L-2..L+2, including six logical layers, without changing its configured depth or model state
def test_fixed_depth_observational_probe_executes_real_deeper_candidates() -> None:
    torch.manual_seed(8122026)
    model = TrainingSheetGPT(
        SheetGPTConfig(
            block_size=4,
            vocab_size=32,
            n_layer=4,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=3,
            base_row_order=8,
            geometry_preset="depth",
            basis_family="chebyshev",
            depth_compress_layer_norm_and_bias=False,
            fast_discard=True,
            direct_factorised_mlp=False,
        )
    )
    model.set_checkpoint_segment_size(0)
    tokens = torch.arange(256, dtype=torch.long) % model.config.vocab_size
    records = []
    trainer = SimpleNamespace(
        raw_model=model,
        model=model,
        config=SimpleNamespace(
            block_size=4,
            batch_size=2,
            data_seed=7331,
            model_seed=1337,
            plastic__layer_count_probe__number_of_sampled_valid_tokens=4,
            plastic__layer_count_probe_radius=2,
        ),
        batch_source=SimpleNamespace(
            train_tokens=tokens,
            validation_tokens=tokens.flip(0),
        ),
        distributed=SimpleNamespace(
            rank=0,
            world_size=1,
            mean_float=lambda value: float(value.item()),
        ),
        state=SimpleNamespace(completed_updates=5),
        device=torch.device("cpu"),
        autocast_context=lambda: nullcontext(),
        _record=lambda event, **payload: records.append((event, payload)),
    )
    trainer._plastic_depth_sampled_token_indices = lambda targets: (
        TrainerStepMixin._plastic_depth_sampled_token_indices(trainer, targets)
    )

    original_parameter_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    observational_executor._run_observational_probe_final(trainer, update=5)

    assert len(records) == 1
    event, payload = records[0]
    assert event == "plastic_depth_count_decision"
    assert payload["observational_only"] is True
    assert payload["previous_active_layers"] == payload["selected_active_layers"] == 4
    assert payload["sampled_token_count"] == 4
    measurements = tuple(payload["candidates"])
    assert tuple(row["active_layers"] for row in measurements) == (2, 3, 4, 5, 6)
    assert tuple(row["executed_logical_layers"] for row in measurements) == (2, 3, 4, 5, 6)
    assert all(row["feasible"] for row in measurements)
    assert all(torch.isfinite(torch.tensor(row["validation_loss"])).item() for row in measurements)
    assert model.config.n_layer == 4
    assert not hasattr(model.trajectory, "_thog_observational_depth_coordinates")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.detach(),
            original_parameter_state[name],
            rtol=0.0,
            atol=0.0,
        )
# ^^^ THOG