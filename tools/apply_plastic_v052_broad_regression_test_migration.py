#!/usr/bin/env python3
# vvv THOG
"""Migrate stale pre-v0.52 regression assertions to the final PLASTIC public contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def _replace_once(path: str, old: str, new: str) -> None:
    text = _read(path)
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one migration anchor, found {count}: {old!r}"
        )
    _write(path, text.replace(old, new, 1))


def _replace_function(path: str, old_name: str, new_source: str) -> None:
    text = _read(path)
    new_name = new_source.split("(", 1)[0].split()[-1]
    old_marker = f"def {old_name}("
    new_marker = f"def {new_name}("
    if old_marker not in text:
        if new_marker in text:
            return
        raise RuntimeError(f"{path}: missing function {old_name}")
    start = text.index(old_marker)
    next_function = text.find("\ndef ", start + len(old_marker))
    end = len(text) if next_function < 0 else next_function + 1
    _write(path, text[:start] + new_source.rstrip() + "\n\n" + text[end:])


def _migrate_console_regressions() -> None:
    path = "tests/test_console_progress_pretty_rows.py"
    _replace_once(
        path,
        '    assert "\\033[1;31mΔloss=  +0.125\\033[0m" in line\n',
        '    assert "\\033[1;31mΔ=  +0.125\\033[0m" in line\n',
    )
    _replace_function(
        path,
        "test_training_and_validation_loss_numerals_start_in_the_same_column",
        '''def test_training_and_validation_loss_numerals_use_the_same_numeric_field_width() -> None:
    line = format_progress_line(
        "OPTIMO",
        "evaluation_completed",
        {
            "completed_updates": "     2",
            "consumed_tokens": "         24576",
            "cumulative_training_seconds": "   196",
            "tok/s": "          63",
            "training_loss": "  10.8831",
            "validation_loss": "  10.7777",
        },
    )
    plain = re.sub(r"\\x1b\\[[0-9;]*m", "", line)
    training_match = re.search(r"(?<!validation )loss=(\\s+10\\.8831)", plain)
    validation_match = re.search(r"validation loss=(\\s+10\\.7777)", plain)
    assert training_match is not None
    assert validation_match is not None
    assert len(training_match.group(1)) == len(validation_match.group(1))''',
    )


def _migrate_nonfinite_default() -> None:
    path = "tests/test_picton_wrapper_defaults_and_nonfinite_policy.py"
    text = _read(path)
    old = "max_nonfinite_update_skips == 10"
    new = "max_nonfinite_update_skips == 99999"
    if old not in text:
        if text.count(new) >= 3:
            return
        raise RuntimeError(f"{path}: missing non-finite default assertions")
    if text.count(old) != 3:
        raise RuntimeError(f"{path}: expected three old non-finite defaults")
    _write(path, text.replace(old, new))


def _migrate_controller_regressions() -> None:
    path = "tests/test_plastic_depth_controller.py"
    _replace_function(
        path,
        "test_mad_gate_collects_required_observations_before_transition",
        '''def test_full_window_gate_collects_required_observations_before_transition() -> None:
    histories = {}
    for update_number in range(1, 3):
        decision = choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(lower=0.8),
            histories=histories,
            noise_window=3,
            minimum_observations=1,
            noise_lambda=1.0,
            update_number=update_number,
            last_count_change_update=-1,
            update_brake=0,
        )
        histories = decision.histories
        assert decision.selected_count == 3
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.8),
        histories=histories,
        noise_window=3,
        minimum_observations=1,
        noise_lambda=1.0,
        update_number=3,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert decision.selected_count == 2
    assert decision.evidence[0].observation_count == 3
    assert decision.evidence[0].significant''',
    )
    _replace_function(
        path,
        "test_zero_mad_uses_positive_scale_floor",
        '''def test_zero_mad_uses_positive_scale_floor() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.5),
        histories={"3:-1": (-0.5,), "3:@LRA": (-1.0,)},
        noise_window=1,
        minimum_observations=1,
        noise_lambda=1.0,
        update_number=4,
        last_count_change_update=-1,
        update_brake=0,
    )
    evidence = decision.evidence[0]
    assert evidence.mad == 0.0
    assert evidence.sigma is not None
    assert evidence.sigma >= PLASTIC_DEPTH_MAD_SIGMA_FLOOR
    assert math.isfinite(evidence.standardized_improvement)
    assert decision.selected_count == 2''',
    )
    _replace_function(
        path,
        "test_update_brake_blocks_transition_but_preserves_new_evidence",
        '''def test_update_brake_blocks_transition_but_preserves_new_evidence() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(upper=0.5),
        histories={},
        noise_window=1,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=12,
        last_count_change_update=10,
        update_brake=5,
    )
    assert decision.brake_active
    assert decision.selected_count == 3
    assert decision.histories["3:+1"] == (-0.5,)

    released = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(upper=0.5),
        histories=decision.histories,
        noise_window=1,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=15,
        last_count_change_update=10,
        update_brake=5,
    )
    assert not released.brake_active
    assert released.selected_count == 4''',
    )
    _replace_function(
        path,
        "test_exact_standardized_tie_prefers_lower_count",
        '''def test_equal_improvements_are_directionally_ambiguous_and_hold() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.5, upper=0.5),
        histories={},
        noise_window=1,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert decision.selected_count == 3
    assert all(not item.significant for item in decision.evidence)''',
    )
    _replace_function(
        path,
        "test_infeasible_direction_does_not_create_history",
        '''def test_infeasible_direction_does_not_create_history() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=float("inf"), upper=0.8),
        histories={},
        noise_window=1,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert "3:-1" not in decision.histories
    assert decision.evidence[0].feasible is False
    assert decision.selected_count == 4''',
    )
    _replace_function(
        path,
        "test_invalid_controller_inputs_are_rejected",
        '''def test_invalid_controller_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="window_size_as_number_of_probes"):
        choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(),
            histories={},
            noise_window=0,
            minimum_observations=1,
            noise_lambda=1.0,
            update_number=1,
            last_count_change_update=-1,
            update_brake=0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(),
            histories={"3:-1": (float("nan"),)},
            noise_window=8,
            minimum_observations=1,
            noise_lambda=1.0,
            update_number=1,
            last_count_change_update=-1,
            update_brake=0,
        )''',
    )


def _remove_retired_min_probe_pairs(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output = []
    index = 0
    removed = 0
    while index < len(lines):
        if '"--plastic-layer-count-probe-noise-min-observations",' in lines[index]:
            if index + 1 >= len(lines):
                raise RuntimeError("retired min-probes option has no value line")
            removed += 1
            index += 2
            continue
        output.append(lines[index])
        index += 1
    resolved = "".join(output)
    if removed == 0 and "--plastic-layer-count-probe-noise-min-observations" in resolved:
        raise RuntimeError("failed to remove retired min-probes option")
    return resolved


def _migrate_interface_regressions() -> None:
    path = "tests/test_plastic_depth_interfaces.py"
    text = _remove_retired_min_probe_pairs(_read(path))
    replacements = {
        "--plastic-enabled": "--plastic__enabled",
        "--plastic-layers-to-sample": "--plastic__layers_to_sample",
        "--plastic-do-learn-layer-count": "--plastic__do_learn_layer_count",
        "--plastic-initial-layer-count": "--plastic__initial_layer_count",
        "--plastic-max-permitted-layers": "--plastic__max_permitted_layers",
        "--plastic-layer-sampling-initialisation": "--plastic__layer_sampling_initialisation",
        "--plastic-layer-count-objective": "--plastic__layer_count_objective",
        "--plastic-layer-count-update-brake": "--plastic__layer_count_update_brake",
        "--plastic-layer-count-probe-noise-window": "--plastic__layer_count_probe__window_size_as_number_of_probes",
        "--plastic-layer-count-probe-noise-lambda": "--plastic__layer_count_probe_noise_lambda",
        "--plastic-layer-count-cost-weight": "--plastic__layer_count_cost_weight",
        "--plastic-layer-memory-budget-gib": "--plastic__layer_memory_budget_gib",
        "--plastic-cuda-allocator-reserve-gib": "--plastic__cuda_allocator_reserve_gib",
        "--plastic-geometry-learning-rate-multiplier": "--plastic__geometry_learning_rate_multiplier",
        "--no-plastic-freeze-geometry-during-warmup": "--no-plastic__freeze_geometry_during_warmup",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(
        "PLN_6_PLM_6_PLI_random_PLO_lowest_loss",
        "P__LN_6_LM_6_LI_rndm_LO_loss",
    )
    text = text.replace("PLN_2_PLM_7", "P__LN_2_LM_7")
    text = text.replace(
        "PLB_50_PLNW_20_PLNM_4_PLNL_250000",
        "LB_50_LNW_20_LNL_250000",
    )
    text = text.replace("PLASTIC DEPTH:", "PLASTIC DEPTH COARSE/FINE:")
    _write(path, text)


def _migrate_lifecycle_compatibility_regression() -> None:
    path = "tests/test_resume_and_fork_cli_compatibility.py"
    old = '''        master_long_options = set(re.findall(r"--[a-z][a-z0-9-]+", master_preparser))
'''
    new = '''        master_long_options = {
            option
            for option in re.findall(r"--[a-z][a-z0-9-]+", master_preparser)
            if option not in {"--plastic", "--no-plastic"}
        }
'''
    _replace_once(path, old, new)


def main() -> None:
    _migrate_console_regressions()
    _migrate_nonfinite_default()
    _migrate_controller_regressions()
    _migrate_interface_regressions()
    _migrate_lifecycle_compatibility_regression()


if __name__ == "__main__":
    main()
# ^^^ THOG
