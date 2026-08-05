# vvv THOG
from __future__ import annotations

from sheet.model import SheetGPTConfig
from sheet.training_model import TrainingSheetGPT


def test_learned_count_growth_verifies_active_prefix_not_dormant_probe() -> None:
    model = TrainingSheetGPT(
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

    transition = model.prepare_plastic_depth_count_transition(3)

    assert transition.geometry.previous_active_layers == 2
    assert transition.geometry.new_active_layers == 3
    assert transition.verification_coordinate_count >= 3
# ^^^ THOG
