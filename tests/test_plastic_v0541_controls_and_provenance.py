from __future__ import annotations

import pytest

import run_thog2_owt_core as runner
from sheet.plastic_depth_v0541_patch import _advance_probe_provenance, _finalize_console_v0541
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig, normalize_plastic_v0541_config_fields


def test_v0541_public_defaults_and_parser_names():
    parser = runner.build_parser()
    args = parser.parse_args([
        "--model-type", "sheet",
        "--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence", "0.91",
        "--plastic__layer_count__max_allowable_layer_change", "3",
        "--plastic__wall_time_equivalent_time_gain_discount", "0.85",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_window", "80",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations", "20",
    ])
    config = runner.config_from_arguments(args)
    assert config.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence == pytest.approx(0.91)
    assert config.plastic__layer_count__max_allowable_layer_change == 3
    assert config.plastic__wall_time_equivalent_time_gain_discount == pytest.approx(0.85)
    assert config.plastic__wall_time_equivalent_time_gain_loss_rate_window == 80
    assert config.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations == 20

    defaults = OwtRunConfig(model_type="sheet")
    assert defaults.plastic__wall_time_equivalent_time_gain_discount == pytest.approx(0.9)
    assert defaults.plastic__wall_time_equivalent_time_gain_loss_rate_window == 64
    assert defaults.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations == 16


def test_v0541_old_cli_names_are_rejected():
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-type", "sheet", "--plastic__layer_count_max_step", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-type", "sheet", "--plastic__layer_count_extrapolation_weight", "0.9"])


def test_v0541_training_config_validates_wall_time_controls():
    config = TrainingConfig(
        plastic__wall_time_equivalent_time_gain_discount=0.75,
        plastic__wall_time_equivalent_time_gain_loss_rate_window=40,
        plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=12,
    )
    assert config.plastic__wall_time_equivalent_time_gain_discount == pytest.approx(0.75)
    with pytest.raises(ValueError):
        TrainingConfig(plastic__wall_time_equivalent_time_gain_discount=1.01)
    with pytest.raises(ValueError):
        TrainingConfig(plastic__wall_time_equivalent_time_gain_loss_rate_window=8, plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=9)


@pytest.mark.parametrize("config_type", [OwtRunConfig, TrainingConfig])
def test_relative_wall_time_rejects_an_every_update_probe_schedule(config_type):
    values = {
        "plastic__enabled": True,
        "plastic__do_learn_layer_count": True,
        "plastic__initial_layer_count": 2,
        "plastic__max_permitted_layers": 4,
        "plastic__layer_count_objective": "relative_training_wall_time",
        "plastic__layer_count_probe__probe_every_n_steps": 1,
    }
    if config_type is TrainingConfig:
        values.update(
            model_type="thog2_sheet",
            batch_size=1,
            block_size=1024,
        )
    else:
        values["model_type"] = "sheet"
    with pytest.raises(ValueError, match="ordinary-training loss-rate model cannot collect observations"):
        config_type(**values)


def test_v0541_checkpoint_key_normalization_is_explicit():
    normalized = normalize_plastic_v0541_config_fields({
        "plastic__layer_count_max_step": 2,
        "plastic__layer_count_extrapolation_weight": 0.9,
        "plastic__layer_memory_budget_gib": 12.0,
        "plastic__cuda_allocator_reserve_gib": 0.5,
    })
    assert normalized["plastic__layer_count__max_allowable_layer_change"] == 2
    assert normalized["plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence"] == pytest.approx(0.9)
    assert normalized["plastic__layer_count__memory_budget_gib"] == pytest.approx(12.0)
    assert normalized["plastic__layer_count__cuda_allocator_reserve_gib"] == pytest.approx(0.5)
    assert "plastic__layer_count_max_step" not in normalized
    assert "plastic__layer_count_extrapolation_weight" not in normalized
    assert "plastic__layer_memory_budget_gib" not in normalized
    assert "plastic__cuda_allocator_reserve_gib" not in normalized


def test_v0541_probe_provenance_tracks_exact_window():
    assert _advance_probe_provenance((), probe_sequence=14, vote_total=1) == (14,)
    assert _advance_probe_provenance((14,), probe_sequence=15, vote_total=2) == (14, 15)
    assert _advance_probe_provenance((14, 15), probe_sequence=16, vote_total=3) == (14, 15, 16)


def test_v0541_console_probe_id_two_dot_label_larger_arrows_and_provenance():
    source = "T probe_Δloss [L-5 ... L+5] = [-0.1]  ↓|↑|? =[1/0/2]/3=>\x1b[1m\x1b[93m↑\x1b[0m  <<< update brake on"
    rendered = _finalize_console_v0541(
        source,
        probe_sequence=1,
        probe_provenance=(14, 15, 16),
    )
    assert "P   1  probe_Δloss [L-5 .. L+5]" in rendered
    assert "⇩|⇧|? =[1/0/2]/3=>\x1b[1m\x1b[93m⇧\x1b[0m (P14,15,16)" in rendered
    assert rendered.endswith("<<< update brake on")
