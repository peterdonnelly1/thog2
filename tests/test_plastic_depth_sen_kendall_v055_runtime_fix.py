# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_sen_kendall_v055_runtime_fix_patch as runtime_fix
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time


def _economic_score_report(current: int = 22):
    return tuple(
        {
            "active_layers": current + offset,
            "feasible": True,
            "score": float(offset),
            "wall_time_algorithm": wall_time.WALL_TIME_ALGORITHM,
            "wall_time_bootstrap": False,
        }
        for offset in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    )


def test_legacy_directional_snapshot_cannot_destroy_first_stratified_row(monkeypatch) -> None:
    monkeypatch.setenv(v055._tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    context = {}
    trainer = SimpleNamespace(_plastic_depth_inline_update_context=context)
    token = wall_time._ACTIVE_TRAINER.set(trainer)
    try:
        first = v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055(
            current_count=22,
            score_report=_economic_score_report(),
            histories={},
            noise_window=2,
            noise_lambda=0.0,
            update_number=37,
            last_count_change_update=-1,
            update_brake=0,
            max_step=1,
        )
        context["decision"] = first
        preserved, report = runtime_fix._updated_histories_and_direction_without_legacy_sen_kendall_ownership(
            current_count=22,
            score_report=_economic_score_report(),
            histories={},
            noise_window=2,
            extrapolation_weight=1.0,
        )
    finally:
        wall_time._ACTIVE_TRAINER.reset(token)

    assert first.selected_count == 22
    assert any("@SK_STRAT_V055" in key for key in first.histories)
    assert preserved == first.histories
    assert int(report["vote_total"]) == 1

    second = v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055(
        current_count=22,
        score_report=_economic_score_report(),
        histories=preserved,
        noise_window=2,
        noise_lambda=0.0,
        update_number=39,
        last_count_change_update=-1,
        update_brake=0,
        max_step=1,
    )
    assert second.selected_count == 21


def test_v055_console_prefers_same_batch_window_local_provenance(monkeypatch) -> None:
    monkeypatch.setenv(v055._tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    monkeypatch.setattr(
        runtime_fix,
        "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
        lambda self, event, payload: {
            "plastic_v055_algorithm": v055.STRATIFIED_ALGORITHM,
            "plastic_v055_probe_ids": (17, 18),
            "plastic_probe_provenance": (1, 2),
        },
    )
    values = runtime_fix._prepare_console_progress_payload_with_window_local_v055_provenance(
        object(),
        "optimizer_progress",
        {},
    )
    assert values["plastic_v055_probe_ids"] == (1, 2)


def test_sampled_alignment_is_identical_for_probe_and_non_probe_rows() -> None:
    non_probe = runtime_fix._align_sampled_field(
        "T     38  2239  layers  22        sampled [1.0, 2.0]"
    )
    probe = runtime_fix._align_sampled_field(
        "T     39  2239  layers  22  sampled [1.0, 2.0]  P2  probe_Δloss [L-5 .. L+5] = [0]"
    )
    assert "layers  22  sampled " in non_probe
    assert "layers  22  sampled " in probe
    assert non_probe.index("sampled") == probe.index("sampled")


def test_obsolete_bare_probe_outcome_is_removed() -> None:
    line = "P2  probe_Δloss [L-5 .. L+5] = [+0.1, 6.2, -0.1]=>▼  sen=+0.500 ken=+0.80 adj=-0.700 ∴ ▼"
    rendered = runtime_fix._remove_legacy_bare_probe_outcome(line)
    assert "]=>▼" not in rendered
    assert rendered.endswith("sen=+0.500 ken=+0.80 adj=-0.700 ∴ ▼")
# ^^^ THOG
