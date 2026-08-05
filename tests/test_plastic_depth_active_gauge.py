# vvv THOG
from __future__ import annotations

import torch

from sheet.model import SheetGPTConfig
from sheet.training_model import TrainingSheetGPT


def _model() -> TrainingSheetGPT:
    return TrainingSheetGPT(
        SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=64,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=16,
            geometry_preset="depth",
            basis_family="chebyshev",
            plastic__enabled=True,
            plastic__layers_to_sample=None,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=2,
            plastic__max_permitted_layers=64,
            plastic__layer_sampling_initialisation="random",
            plastic__freeze_geometry_during_warmup=False,
        )
    )


def _active_layer_count(model: TrainingSheetGPT) -> int:
    lattice = model.trajectory.plastic_sampling
    if lattice is None:
        raise AssertionError("test model unexpectedly has no PLASTIC lattice")
    return int(lattice.current_active_layers)


def test_learned_count_growth_verifies_only_active_prefix_samples() -> None:
    model = _model()

    transition = model.prepare_plastic_depth_count_transition(3)

    assert transition.geometry.previous_active_layers == 2
    assert transition.geometry.new_active_layers == 3
    assert transition.verification_coordinate_count == 3


def test_unstable_learned_count_growth_falls_back_to_geometry_only_transition() -> None:
    model = _model()
    generator = torch.Generator(device="cpu").manual_seed(260805)
    with torch.no_grad():
        for parameter in model.trajectory.coefficients.values():
            parameter.normal_(mean=0.0, std=0.25, generator=generator)

    transition = model.prepare_plastic_depth_count_transition(3)
    report = model.commit_plastic_depth_count_transition(transition)

    assert _active_layer_count(model) == 3
    assert report["transformed_family_count"] in {0, 6}
    if report["transformed_family_count"] == 0:
        assert torch.isnan(torch.tensor(report["maximum_absolute_error"]))
# ^^^ THOG
