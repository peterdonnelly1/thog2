# vvv THOG
from types import SimpleNamespace

import run_thog2_owt


class _Optimizer:
    param_groups = [{"lr": 8.0e-5, "thog2_optimizer_name": "adamw"}]
    defaults = {"fused": True, "betas": (0.9, 0.95)}


class _RawModel:
    config = SimpleNamespace(
        bypass_semantic_qkv_adapter=True,
        vectorise_per_head_materialisation=True,
        direct_factorised_mlp=True,
        direct_factorised_hyperblock_mlp=False,
        depth_compress_layer_norm_and_bias=False,
    )
    trajectory = None

    def _supports_direct_factorised_mlp(self):
        return True


def _config():
    return SimpleNamespace(
        model_type="sheet",
        learning_rate=8.0e-5,
        min_lr=8.0e-6,
        warmup_iters=100,
        max_wall_minutes=180,
        weight_decay=0.1,
        grad_clip=1.0,
        nonfinite_update_policy="skip",
        max_nonfinite_update_skips=10,
        batch_size=2,
        gradient_accumulation_steps=32,
        layer_dropout_n_strata=1,
        layer_dropout_stratum_size=24,
        layer_dropout_active_per_stratum=24,
        n_active_layers=24,
        n_layer=24,
        layer_dropout_resample_steps=1,
        activation_checkpointing=True,
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="disabled",
        plastic__coarse_phase_roll_through=True,
        plastic__log_interval_coarse=10,
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=8,
        plastic__max_permitted_layers=24,
        plastic__layer_sampling_initialisation="equidistant",
        plastic__layer_count_objective="relative_training_wall_time",
        plastic__layer_count_update_brake=20,
        plastic__layer_count_probe__probe_every_n_steps=10,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=1024,
        plastic__layer_count_probe__window_size_as_number_of_probes=48,
        plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence=0.8,
        plastic__layer_count_probe_noise_lambda=2.0,
        plastic__layer_count_cost_weight=0.04,
        plastic__layer_count__memory_budget_gib=None,
        plastic__layer_count__cuda_allocator_reserve_gib=1.0,
        plastic__geometry_learning_rate_multiplier=0.15,
        plastic__freeze_geometry_during_warmup=True,
        plastic__initial_active_layers=8,
        tokens_per_iter=lambda: 65536,
    )


def test_startup_report_restores_full_rows_and_plastic_section(capsys):
    trainer = SimpleNamespace(
        optimizer=_Optimizer(),
        config=SimpleNamespace(decay_learning_rate=True, decay_updates=1600),
        parameter_report={
            "persistent_parameters": 124_595_735,
            "sheet_coefficients": 84_934_656,
            "dense_equivalent_total_parameters": 209_530_368,
            "plastic_depth": {
                "active_layers": 8,
                "active_public_coordinates": (1.0, 15.1, 29.3, 43.4),
                "public_coordinates": (1.0, 15.1, 29.3, 43.4, 57.6),
            },
        },
        raw_model=_RawModel(),
    )

    run_thog2_owt._print_model_parameters_and_optimisations(_config(), trainer)

    output = capsys.readouterr().out
    assert "wall stop:" in output
    assert "layer dropout:" in output
    assert "execution:" in output
    assert "plastic\n" in output
    assert "plastic__runtime_phase:" in output
    assert "plastic__coarse_phase_roll_through:" in output
    assert "plastic__log_interval_coarse:" in output
    assert "plastic__layer_count_update_brake:" in output
    assert "plastic__layer_count_probe__number_of_sampled_valid_tokens:" in output
    assert "plastic__layer_count_probe__window_size_as_number_of_probes:" in output
    assert "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence:" in output
    assert "plastic__geometry_learning_rate_multiplier:" in output
    assert "active sample_layer:" in output
    assert "capacity sample_layer:" in output
    rows = output.splitlines()
    token_row = next(
        line
        for line in rows
        if "plastic__layer_count_probe__number_of_sampled_valid_tokens:" in line
    )
    extrapolation_row = next(
        line
        for line in rows
        if "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence:" in line
    )
    initial_row = next(line for line in rows if "active sample_layer:" in line)
    capacity_row = next(line for line in rows if "capacity sample_layer:" in line)
    assert token_row.endswith("   1024")
    assert extrapolation_row.endswith("   0.8")
    assert initial_row.endswith("1.0,  15.1,  29.3,  43.4")
    assert capacity_row.endswith("1.0,  15.1,  29.3,  43.4,  57.6")
    assert initial_row.index("1.0") == capacity_row.index("1.0")
# ^^^ THOG
