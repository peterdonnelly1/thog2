# vvv THOG
from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest
import torch

import sheet.stage6_trainer as stage6
import sheet.chaos_bump_sampling_patch as sampling_patch
from run_thog2_owt_core import build_parser
from sheet.chaos_bump_sampling import (
    CHAOS_BUMP_SAMPLING_CONFIG_FIELDS,
    chaos_bump_sampling_duration_steps,
    chaos_bump_sampling_interlude_steps,
    rattle_sampling_coordinates,
    resolve_chaos_bump_sampling_config,
)
from sheet.checkpoints import load_payload
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits
from tests.test_plastic_depth import plastic_training_config


def _resolved(**overrides):
    values = dict(
        enabled=True,
        plastic_enabled=True,
        initial_lockout_steps=1,
        maximum_bumps=2,
        interlude_min_steps=3,
        interlude_max_steps=3,
        duration_min_steps=2,
        duration_max_steps=2,
        duration_max_fraction_of_elapsed_steps=0.05,
        max_movement_fraction_of_local_gap=0.1,
    )
    values.update(overrides)
    return resolve_chaos_bump_sampling_config(**values)


def _fixed_config(**overrides):
    values = dict(
        max_updates=8,
        warmup_updates=0,
        plastic__freeze_geometry_during_warmup=False,
        chaos_bump__sampling__enabled=True,
        chaos_bump__sampling__initial_lockout__steps=1,
        chaos_bump__sampling__maximum_bumps=1,
        chaos_bump__sampling__interlude__min_steps=1,
        chaos_bump__sampling__interlude__max_steps=1,
        chaos_bump__sampling__duration__min_steps=2,
        chaos_bump__sampling__duration__max_steps=2,
        chaos_bump__sampling__duration__max_fraction_of_elapsed_steps=0.05,
        chaos_bump__sampling__max_movement_fraction_of_local_gap=0.1,
    )
    values.update(overrides)
    return plastic_training_config(**values)


def _geometry_state(trainer):
    lattice = trainer.raw_model.trajectory.plastic_sampling
    parameter = lattice.raw_intervals
    return (
        parameter.detach().clone(),
        copy.deepcopy(trainer.optimizer.state.get(parameter, {})),
        lattice.public_coordinates().detach().clone(),
    )


def test_rattle_is_deterministic_ordered_bounded_and_anchors_capacity_edges() -> None:
    base = torch.tensor([1.0, 20.0, 55.0, 80.0, 100.0])
    first = rattle_sampling_coordinates(
        base,
        maximum_fraction_of_local_gap=0.25,
        model_seed=91,
        bump_number=2,
    )
    second = rattle_sampling_coordinates(
        base,
        maximum_fraction_of_local_gap=0.25,
        model_seed=91,
        bump_number=2,
    )
    assert first == second
    assert first.coordinates[0] == 1.0
    assert first.coordinates[-1] == 100.0
    assert all(left < right for left, right in zip(first.coordinates, first.coordinates[1:]))
    assert set(first.visit_order) == {1, 2, 3}
    assert max(first.movement_fractions) <= 0.25


def test_schedule_draws_are_isolated_deterministic_and_inclusive() -> None:
    config = _resolved(interlude_min_steps=7, interlude_max_steps=7)
    assert chaos_bump_sampling_duration_steps(
        config,
        start_update=100,
        model_seed=13,
        bump_number=1,
    ) == 2
    assert chaos_bump_sampling_interlude_steps(
        config,
        model_seed=13,
        completed_bump_number=1,
    ) == 7
    before = torch.get_rng_state().clone()
    rattle_sampling_coordinates(
        [1.0, 34.0, 67.0, 100.0],
        maximum_fraction_of_local_gap=0.1,
        model_seed=13,
        bump_number=1,
    )
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"enabled": True, "plastic_enabled": False}, "requires plastic"),
        ({"interlude_min_steps": 4, "interlude_max_steps": 3}, "must not exceed"),
        ({"duration_min_steps": 3, "duration_max_steps": 2}, "must not exceed"),
        ({"max_movement_fraction_of_local_gap": 0.0}, "lie in"),
    ],
)
def test_config_rejects_invalid_controls(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _resolved(**overrides)


def test_disabled_config_and_checkpoint_surfaces_omit_every_new_field() -> None:
    config = stage3_config("thog2_sheet")
    assert all(name not in config.persistent_dict() for name in CHAOS_BUMP_SAMPLING_CONFIG_FIELDS)
    train, validation = token_splits()
    trainer = SharedTrainer(config, train, validation)
    try:
        payload = trainer.checkpoint_payload()
        assert "chaos_bump_sampling_state" not in payload
        assert "chaos_bump_sampling" not in payload["compact_identity"]
        assert not hasattr(trainer, "_chaos_bump_sampling_state")
    finally:
        trainer.close()


def test_fixed_count_bump_changes_only_execution_indices_then_restores_exactly() -> None:
    train, validation = token_splits(length=1024)
    trainer = SharedTrainer(_fixed_config(), train, validation)
    try:
        trainer.train_one_update()
        raw_before, optimizer_before, coordinates_before = _geometry_state(trainer)
        coefficients_before = {
            name: value.detach().clone()
            for name, value in trainer.raw_model.trajectory.coefficients.items()
        }

        entry = trainer.train_one_update()
        lattice = trainer.raw_model.trajectory.plastic_sampling
        state = trainer._chaos_bump_sampling_state
        assert entry["chaos_bump__sampling__transition"] == "started"
        assert state["active"]
        assert lattice.raw_intervals.grad is None
        assert not torch.equal(
            lattice.public_coordinates().detach(),
            coordinates_before,
        )

        exit_metrics = trainer.train_one_update()
        raw_after, optimizer_after, coordinates_after = _geometry_state(trainer)
        assert exit_metrics["chaos_bump__sampling__transition"] == "ended"
        assert not state["active"]
        assert torch.equal(raw_after, raw_before)
        assert torch.equal(coordinates_after, coordinates_before)
        assert set(optimizer_after) == set(optimizer_before)
        for name in optimizer_before:
            left = optimizer_after[name]
            right = optimizer_before[name]
            if isinstance(right, torch.Tensor):
                assert torch.equal(left, right)
            else:
                assert left == right
        assert any(
            not torch.equal(coefficients_before[name], value.detach())
            for name, value in trainer.raw_model.trajectory.coefficients.items()
        )
        assert [event.name for event in trainer.events].count("chaos_bump_sampling_started") == 1
        assert [event.name for event in trainer.events].count("chaos_bump_sampling_ended") == 1
    finally:
        trainer.close()


def test_learned_count_is_frozen_during_bump_and_post_bump_brake() -> None:
    train, validation = token_splits(length=1024)
    config = _fixed_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_update_brake=2,
        plastic__layer_count_probe__window_size_as_number_of_probes=2,
        n_layer=5,
        depth_order=4,
        chaos_bump__sampling__initial_lockout__steps=0,
    )
    trainer = SharedTrainer(config, train, validation)
    try:
        lattice = trainer.raw_model.trajectory.plastic_sampling
        initial_count = lattice.current_active_layers
        trainer.train_one_update()
        trainer.train_one_update()
        assert lattice.current_active_layers == initial_count
        assert not any(
            event.name == "plastic_depth_count_decision"
            for event in trainer.events
        )
        state = trainer._chaos_bump_sampling_state
        assert state["probe_lockout_until_update"] == 4
        trainer.train_one_update()
        trainer.train_one_update()
        assert lattice.current_active_layers == initial_count
        assert not any(
            event.name == "plastic_depth_count_decision"
            for event in trainer.events
        )
    finally:
        trainer.close()


def test_mid_bump_checkpoint_resume_restores_override_schedule_and_exact_exit(tmp_path: Path) -> None:
    train, validation = token_splits(length=1024)
    config = _fixed_config(chaos_bump__sampling__duration__min_steps=3, chaos_bump__sampling__duration__max_steps=3)
    source = SharedTrainer(config, train, validation)
    resumed = None
    try:
        source.train_one_update()
        base_raw, _, base_coordinates = _geometry_state(source)
        source.train_one_update()
        checkpoint = source.save_checkpoint(tmp_path / "mid_bump.pt")
        payload = load_payload(checkpoint)
        assert payload["chaos_bump_sampling_state"]["active"]

        resumed = SharedTrainer.from_checkpoint(checkpoint, train, validation)
        assert resumed._chaos_bump_sampling_state == source._chaos_bump_sampling_state
        source_coordinates = source.raw_model.trajectory.plastic_sampling.public_coordinates().detach()
        resumed_coordinates = resumed.raw_model.trajectory.plastic_sampling.public_coordinates().detach()
        assert torch.equal(resumed_coordinates, source_coordinates)

        resumed.train_one_update()
        resumed.train_one_update()
        lattice = resumed.raw_model.trajectory.plastic_sampling
        assert not resumed._chaos_bump_sampling_state["active"]
        assert torch.equal(lattice.raw_intervals.detach(), base_raw)
        assert torch.equal(lattice.public_coordinates().detach(), base_coordinates)
    finally:
        if resumed is not None:
            resumed.close()
        source.close()


def test_skipped_attempt_does_not_consume_bump_duration(monkeypatch) -> None:
    train, validation = token_splits(length=1024)
    trainer = SharedTrainer(
        _fixed_config(
            chaos_bump__sampling__initial_lockout__steps=0,
            chaos_bump__sampling__duration__min_steps=2,
            chaos_bump__sampling__duration__max_steps=2,
        ),
        train,
        validation,
    )
    original = sampling_patch._ORIGINAL_TRAIN_ONE_UPDATE
    try:
        monkeypatch.setattr(
            sampling_patch,
            "_ORIGINAL_TRAIN_ONE_UPDATE",
            lambda _trainer: {
                "completed_updates": 0.0,
                "training_loss": float("nan"),
                "learning_rate": 0.0,
                "gradient_norm": float("nan"),
                "skipped_update": 1.0,
            },
        )
        skipped = trainer.train_one_update()
        state = trainer._chaos_bump_sampling_state
        assert skipped["skipped_update"] == 1.0
        assert trainer.state.completed_updates == 0
        assert state["active"]
        assert state["end_update"] == 2
        assert skipped["chaos_bump__sampling__bump_step"] == 0
    finally:
        monkeypatch.setattr(sampling_patch, "_ORIGINAL_TRAIN_ONE_UPDATE", original)
        trainer.close()


def test_warmup_freeze_delays_first_bump_until_following_update() -> None:
    train, validation = token_splits(length=1024)
    trainer = SharedTrainer(
        _fixed_config(
            warmup_updates=2,
            plastic__freeze_geometry_during_warmup=True,
            chaos_bump__sampling__initial_lockout__steps=0,
        ),
        train,
        validation,
    )
    try:
        assert trainer.train_one_update()["chaos_bump__sampling__transition"] is None
        assert trainer.train_one_update()["chaos_bump__sampling__transition"] is None
        assert trainer.train_one_update()["chaos_bump__sampling__transition"] == "started"
        assert trainer._chaos_bump_sampling_state["start_update"] == 3
    finally:
        trainer.close()


def test_cli_exposes_only_exact_sampling_namespace() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-type",
            "sheet",
            "--plastic__enabled",
            "--chaos_bump__sampling__enabled",
            "--chaos_bump__sampling__maximum_bumps",
            "3",
        ]
    )
    assert args.chaos_bump__sampling__enabled
    assert args.chaos_bump__sampling__maximum_bumps == 3
    help_text = parser.format_help()
    assert "--chaos_bump__sampling__enabled" in help_text
    assert "depth_change" not in "\n".join(
        line for line in help_text.splitlines() if "chaos_bump" in line
    )
    wrapper = Path("train_OWT.sh").read_text(encoding="utf-8")
    assert "--chaos_bump__sampling__*" in wrapper
    assert "Non-canonical sampling chaos bump option rejected" in wrapper


def test_canonical_shell_wrapper_routes_exact_namespace_and_rejects_alias() -> None:
    accepted = subprocess.run(
        [
            "bash",
            "./train_OWT.sh",
            "--plastic__enabled",
            "--chaos_bump__sampling__enabled",
            "--chaos_bump__sampling__maximum_bumps",
            "2",
            "-h",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "--chaos_bump__sampling__maximum_bumps" in accepted.stdout
    rejected = subprocess.run(
        ["bash", "./train_OWT.sh", "--chaos-bump-sampling-enabled", "-h"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode == 2
    assert "Non-canonical sampling chaos bump option rejected" in rejected.stderr


def test_progress_formatter_appends_active_and_exit_markers() -> None:
    active = stage6.format_progress_line(
        "run",
        "optimizer_progress",
        {
            "completed_updates": "     2",
            "chaos_bump_sampling": {
                "active": True,
                "transition": "started",
                "bump_number": 1,
                "step": 1,
                "duration": 3,
            },
        },
    )
    ended = stage6.format_progress_line(
        "run",
        "optimizer_progress",
        {
            "completed_updates": "     4",
            "chaos_bump_sampling": {
                "active": False,
                "transition": "ended",
                "bump_number": 1,
                "step": None,
                "duration": 3,
            },
        },
    )
    assert "<<< chaos bump sampling B1 1/3" in active
    assert "<<< chaos bump sampling B1 ended" in ended
# ^^^ THOG
