# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import pytest

import run_thog2_owt as public_runner
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v056_objective_decision_patch as v056


def _decision_event():
    return SimpleNamespace(
        name="plastic_depth_count_decision",
        payload={
            "paired_evidence": (
                {"direction": -1, "standardized_improvement": -1.25},
                {"direction": 1, "standardized_improvement": 2.5},
            ),
        },
    )


def _trainer():
    return SimpleNamespace(
        config=SimpleNamespace(plastic__do_learn_layer_count=True),
        events=[_decision_event()],
    )


@pytest.mark.parametrize("algorithm", v056.SEN_KENDALL_ALGORITHMS)
def test_public_runner_cannot_reconstruct_change_z_for_sen_kendall(
    monkeypatch,
    algorithm: str,
) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, algorithm)
    assert public_runner._latest_plastic_change_z(_trainer()) is None


def test_public_runner_retains_change_z_for_directional_coherence(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, tsk.LEGACY_DIRECTIONAL_ALGORITHM)
    assert public_runner._latest_plastic_change_z(_trainer()) == (-1.25, 2.5)


@pytest.mark.parametrize("algorithm", v056.SEN_KENDALL_ALGORITHMS)
def test_veto_row_and_following_non_probe_row_never_regain_change_z(
    monkeypatch,
    algorithm: str,
) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, algorithm)
    monkeypatch.setattr(
        public_runner,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda _self, _event, payload: dict(payload),
    )
    trainer = _trainer()

    veto_row = public_runner._prepare_console_progress_payload_with_precise_step(
        trainer,
        "optimizer_progress",
        {
            "completed_updates": "     7",
            "plastic_cuda_growth_memory_hold": True,
            "plastic_change_z": (99.0, 99.0),
        },
    )
    following_row = public_runner._prepare_console_progress_payload_with_precise_step(
        trainer,
        "optimizer_progress",
        {
            "completed_updates": "     8",
            "plastic_change_z": (99.0, 99.0),
        },
    )

    assert "plastic_change_z" not in veto_row
    assert "plastic_change_z" not in following_row


def test_outer_public_formatter_keeps_memory_limit_after_legacy_change_z(monkeypatch) -> None:
    colour = "\033[38;2;150;220;255m"
    reset = "\033[0m"
    monkeypatch.setattr(
        public_runner,
        "_ORIGINAL_MATERIALISATION_PROGRESS_FORMAT",
        lambda _run_id, _event, _payload: (
            "T 7 layers = 38"
            f"  {colour}<<< stopped by memory limit{reset}"
        ),
    )

    rendered = public_runner._format_progress_line_with_materialisation_last(
        "run",
        "optimizer_progress",
        {"plastic_change_z": (-1.25, 2.5)},
    )

    assert "change_z [L-1, L+1] = [    -1.25,     +2.50]" in rendered
    assert rendered.index("change_z") < rendered.index("<<< stopped by memory limit")
    assert rendered.endswith(f"{colour}<<< stopped by memory limit{reset}")
# ^^^ THOG
