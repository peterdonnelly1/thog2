# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import run_thog2_owt


def test_model_summary_aligns_optimizer_and_reports_cosine_decay(capsys) -> None:
    optimizer = SimpleNamespace(
        param_groups=[{"lr": 9.0e-4, "thog2_optimizer_name": "adamw"}],
        defaults={"fused": True, "betas": (0.9, 0.95)},
    )
    trainer = SimpleNamespace(
        optimizer=optimizer,
        config=SimpleNamespace(decay_learning_rate=True, decay_updates=20000),
        parameter_report={
            "persistent_parameters": 100,
            "dense_equivalent_total_parameters": 200,
            "sheet_coefficients": 10,
        },
    )
    config = SimpleNamespace(
        model_type="dense",
        learning_rate=9.0e-4,
        min_lr=2.0e-4,
        warmup_iters=100,
        weight_decay=0.1,
        grad_clip=1.0,
        max_wall_minutes=0,
        nonfinite_update_policy="skip",
        max_nonfinite_update_skips=10,
        batch_size=4,
        gradient_accumulation_steps=6,
        layer_dropout_enabled=False,
        activation_checkpointing=False,
        tokens_per_iter=lambda: 36864,
    )

    run_thog2_owt._print_model_parameters_and_optimisations(config, trainer)
    output = capsys.readouterr().out

    assert "optimiser:" in output
    assert "adamw (lr=9.000e-04, fused=True, betas=(0.9, 0.95))" in output
    assert "LR decay:" in output
    assert "cosine (decay_rate=0.222222, min_lr=2.000e-04, fully_decayed_step=20000)" in output
    assert "optimiser parms:" in output
    assert "warmup=100  weight_decay=0.1  grad_clip=1" in output
# ^^^ THOG
