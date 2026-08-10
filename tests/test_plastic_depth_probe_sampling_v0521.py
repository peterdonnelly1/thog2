# vvv THOG
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

import constants
from run_thog2_owt_core import build_parser, config_from_arguments
from sheet import plastic_depth_console_cleanup_patch as cleanup
from sheet import plastic_depth_probe_sampling_v0521_patch as probe_v0521
from sheet.run_config import OwtRunConfig
from sheet.trainer_step import TrainerStepMixin
from sheet.training_config import TrainingConfig


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _fake_sampler(*, requested: int, completed_updates: int = 7, rank: int = 0):
    return SimpleNamespace(
        config=SimpleNamespace(
            plastic__layer_count_probe__number_of_sampled_valid_tokens=requested,
            model_seed=1337,
        ),
        state=SimpleNamespace(completed_updates=completed_updates),
        distributed=SimpleNamespace(rank=rank),
    )


def test_probe_token_default_and_runner_resolution_are_1024() -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-type", "sheet"])
    assert args.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024
    config = config_from_arguments(args)
    assert config.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024
    training = config.to_training_config(vocab_size=50304, world_size=1, out_dir=__import__("pathlib").Path("out"))
    assert training.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024


def test_probe_token_static_capacity_validation_accepts_zero_and_exact_capacity() -> None:
    OwtRunConfig(
        model_type="sheet",
        n_layer=8,
        o_depth=4,
        batch_size=2,
        block_size=8,
        plastic__enabled=True,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=4,
        plastic__max_permitted_layers=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=0,
    )
    OwtRunConfig(
        model_type="sheet",
        n_layer=8,
        o_depth=4,
        batch_size=2,
        block_size=8,
        plastic__enabled=True,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=4,
        plastic__max_permitted_layers=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
    )
    OwtRunConfig(
        model_type="dense",
        batch_size=1,
        block_size=8,
    )
    TrainingConfig(
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
    )


@pytest.mark.parametrize("value", [-1, 17])
def test_probe_token_static_capacity_validation_rejects_negative_or_too_large(value: int) -> None:
    with pytest.raises(ValueError, match="plastic__layer_count_probe__number_of_sampled_valid_tokens"):
        OwtRunConfig(
            model_type="sheet",
            n_layer=8,
            o_depth=4,
            batch_size=2,
            block_size=8,
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=4,
            plastic__max_permitted_layers=8,
            plastic__layer_count_probe__number_of_sampled_valid_tokens=value,
        )



def test_training_config_static_capacity_validation_rejects_too_large() -> None:
    with pytest.raises(ValueError, match="plastic__layer_count_probe__number_of_sampled_valid_tokens"):
        TrainingConfig(
            model_type="thog2_sheet",
            geometry_preset="depth",
            basis_family="chebyshev",
            n_layer=8,
            depth_order=4,
            batch_size=2,
            block_size=8,
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=4,
            plastic__max_permitted_layers=8,
            plastic__layer_count_probe__number_of_sampled_valid_tokens=17,
        )

def test_zero_uses_every_valid_token_without_random_subsampling() -> None:
    targets = torch.tensor([[10, 11, -1, 12], [13, -1, 14, 15]], dtype=torch.long)
    sampled = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=0),
        targets,
    )
    assert sampled.tolist() == [0, 1, 3, 4, 6, 7]


def test_positive_probe_token_count_is_exact_deterministic_random_subset() -> None:
    targets = torch.arange(20, dtype=torch.long).reshape(2, 10)
    first = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=7),
        targets,
    )
    second = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=7),
        targets,
    )
    assert first.numel() == 7
    assert len(set(first.tolist())) == 7
    assert first.tolist() == second.tolist()
    assert first.tolist() != list(range(7))


def test_runtime_rejects_request_above_actual_valid_token_count() -> None:
    targets = torch.tensor([[10, 11, -1, 12], [13, -1, 14, 15]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="requested=7, valid=6"):
        TrainerStepMixin._plastic_depth_sampled_token_indices(
            _fake_sampler(requested=7),
            targets,
        )


def test_hybrid_probe_vector_keeps_absolute_l_and_signed_candidate_deltas() -> None:
    rendered = probe_v0521._render_probe_delta_values(
        (-2, -1, 0, 1, 2),
        (4.100, 4.050, 4.077, 4.134, 4.060),
    )
    assert rendered is not None
    plain = _plain(rendered)
    assert plain == "+0.023, -0.027, 4.077, +0.057, -0.017"
    assert f"{constants.BOLD_WHITE}{constants.UNDER}4.077{constants.R}" in rendered
    assert f"{constants.BOLD_GREEN}-0.027{constants.R}" in rendered
    assert f"{cleanup._GREEN}-0.017{cleanup._RESET}" in rendered
    assert constants.BOLD_GREEN != cleanup._GREEN
    assert constants.BOLD_GREEN not in rendered.split(", ")[0]
    assert cleanup._GREEN not in rendered.split(", ")[0]
    assert cleanup._GREEN not in rendered.split(", ")[3]


def test_final_console_renames_probe_vector_to_probe_delta_loss() -> None:
    line = (
        "T 20 layers = 7\tprobe_losses [L-2 ... L+2] = [4.100, 4.050, 4.077, 4.134, 4.060]  "
        "L/R/A=[0/0/6]/6=>stet"
    )
    rendered_values = probe_v0521._render_probe_delta_values(
        (-2, -1, 0, 1, 2),
        (4.100, 4.050, 4.077, 4.134, 4.060),
    )
    assert rendered_values is not None
    replaced = probe_v0521._PROBE_VECTOR.sub(
        lambda match: f"probe_Δloss [{match.group('label')}] = [{rendered_values}]",
        line,
        count=1,
    )
    assert "probe_losses" not in _plain(replaced)
    assert "probe_Δloss [L-2 ... L+2]" in _plain(replaced)
    assert _plain(replaced).endswith("L/R/A=[0/0/6]/6=>stet")


def test_artifact_and_help_registry_name_the_new_probe_token_control() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        plastic__enabled=True,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=4,
        plastic__max_permitted_layers=8,
        n_layer=8,
        o_depth=4,
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=8,
    )
    assert "LPT_8" in config.parameter_artifact_fragment()
    help_text = build_parser().format_help()
    assert "--plastic__layer_count_probe__number_of_sampled_valid_tokens" in help_text
# ^^^ THOG

# vvv THOG v0.521 paired-token SE is a diagnostic precision estimate and never participates in count selection
def test_paired_token_standard_error_uses_paired_deltas_and_sample_standard_deviation() -> None:
    from sheet import plastic_depth_probe_se_v0521_patch as probe_se

    counts = (6, 7, 8)
    current = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    left = current + torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    right = current + torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    local = probe_se._local_paired_delta_stats(
        counts=counts,
        current_count=7,
        token_losses=(left, current, right),
    )
    standard_errors = probe_se._combine_paired_delta_standard_errors(
        counts=counts,
        current_count=7,
        gathered_stats=(local,),
    )

    expected_left = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64).std(unbiased=True).item() / 2.0
    assert standard_errors[6] == pytest.approx(expected_left)
    assert standard_errors[7] == 0.0
    assert standard_errors[8] == pytest.approx(0.0)


def test_paired_token_se_overlay_remains_beneath_final_v056_and_same_batch_wrappers() -> None:
    from sheet import plastic_depth_probe_se_v0521_patch as probe_se
    from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
    from sheet import plastic_depth_v056_objective_decision_patch as v056
    from sheet.training_model import TrainingSheetGPT

    assert TrainingSheetGPT._plastic_depth_candidate_head_loss.__module__ == probe_se.__name__
    assert TrainerStepMixin._plastic_depth_inline_probe_request.__module__ == v056.__name__
    assert v056._ORIGINAL_INLINE_PROBE_REQUEST.__module__ == same_batch.__name__
    assert same_batch._ORIGINAL_INLINE_PROBE_REQUEST.__module__ == probe_se.__name__
# ^^^ THOG
