# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sheet import plastic_depth_theil_sen_kendall_console_fix_patch as console_fix
from sheet import plastic_depth_theil_sen_kendall_patch as gradient
from sheet import plastic_depth_theil_sen_kendall_resume_config_patch as resume_config


def test_gradient_algorithm_is_opt_in_and_legacy_is_default(monkeypatch):
    monkeypatch.delenv(gradient._ALGORITHM_ENV, raising=False)
    assert gradient._runtime_algorithm() == gradient.LEGACY_DIRECTIONAL_ALGORITHM


def test_gradient_cli_options_are_stripped_for_core_parser():
    remaining, algorithm, tau = gradient._strip_gradient_options(
        [
            "--plastic__enabled",
            "--plastic__layer_count_decision_algorithm",
            gradient.GRADIENT_ALGORITHM,
            "--plastic__layer_count_gradient__minimum_absolute_kendall_tau=0.6",
        ]
    )
    assert remaining == ["--plastic__enabled"]
    assert algorithm == gradient.GRADIENT_ALGORITHM
    assert tau == pytest.approx(0.6)


def test_gradient_tau_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv(gradient._TAU_ENV, "1.01")
    with pytest.raises(ValueError, match="lie in \[0, 1\]"):
        gradient._runtime_minimum_absolute_kendall_tau()


def test_resume_shim_consumes_synthetic_gradient_fields(monkeypatch):
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)

    monkeypatch.setattr(resume_config, "_ORIGINAL_TRAINING_CONFIG_INIT", fake_init)
    monkeypatch.delenv(gradient._ALGORITHM_ENV, raising=False)
    monkeypatch.delenv(gradient._TAU_ENV, raising=False)
    resume_config._training_config_init_with_gradient_resume(
        SimpleNamespace(),
        123,
        plastic__layer_count_decision_algorithm=gradient.GRADIENT_ALGORITHM,
        plastic__layer_count_gradient__minimum_absolute_kendall_tau=0.7,
        plastic__enabled=True,
    )
    assert captured["args"] == (123,)
    assert captured["kwargs"] == {"plastic__enabled": True}
    assert gradient._runtime_algorithm() == gradient.GRADIENT_ALGORITHM
    assert gradient._runtime_minimum_absolute_kendall_tau() == pytest.approx(0.7)


def test_final_console_marker_accepts_fat_arrows():
    assert gradient._DIRECTION_MARKER.search("  ⇩|⇧|? =[1/0/0]/1=>⇩") is not None


def test_gradient_classifier_is_deterministic_for_identical_ddp_inputs():
    score_report = (
        {"active_layers": 9, "feasible": True, "score": 0.2, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
        {"active_layers": 10, "feasible": True, "score": 0.0, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
        {"active_layers": 11, "feasible": True, "score": -0.2, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
    )
    first = gradient._gradient_probe_classification(
        current_count=10,
        score_report=score_report,
        minimum_absolute_kendall_tau=0.5,
    )
    second = gradient._gradient_probe_classification(
        current_count=10,
        score_report=score_report,
        minimum_absolute_kendall_tau=0.5,
    )
    assert first == second


def test_console_diagnostic_units_are_seconds_per_layer():
    assert gradient._format_gradient_value(0.1834, digits=3) == "+0.183"
    assert "s/layer" in gradient._format_progress_line_with_gradient.__code__.co_consts
# ^^^ THOG
