from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one repair anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def repair_probe_interval_compatibility() -> None:
    replace_once(
        "sheet/plastic_depth_coarse.py",
        '    if update_brake < 1:\n'
        '        raise ValueError(\n'
        '            "plastic__layer_count_probe_interval must be supplied when "\n'
        '            "plastic__layer_count_update_brake is zero"\n'
        '        )\n'
        '    return update_brake\n',
        '    if update_brake < 0:\n'
        '        raise ValueError("plastic__layer_count_update_brake must be non-negative")\n'
        '    # Version 0.3 used brake=0 to mean every update; preserve that exact path.\n'
        '    return max(1, update_brake)\n',
    )
    replace_once(
        "tests/test_plastic_depth_coarse_config.py",
        'def test_probe_interval_requires_explicit_value_when_brake_is_zero() -> None:\n'
        '    with pytest.raises(ValueError, match="must be supplied"):\n'
        '        resolve_plastic_probe_interval(\n'
        '            probe_interval=None,\n'
        '            update_brake=0,\n'
        '            enabled=True,\n'
        '            do_learn_layer_count=True,\n'
        '        )\n',
        'def test_probe_interval_preserves_legacy_every_update_when_brake_is_zero() -> None:\n'
        '    assert resolve_plastic_probe_interval(\n'
        '        probe_interval=None,\n'
        '        update_brake=0,\n'
        '        enabled=True,\n'
        '        do_learn_layer_count=True,\n'
        '    ) == 1\n',
    )


def preserve_v03_identity_until_explicit_migration() -> None:
    replace_once(
        "sheet/plastic_depth.py",
        'PLASTIC_DEPTH_VERSION = "plastic_depth_v0_4"',
        'PLASTIC_DEPTH_VERSION = "plastic_depth_v0_3"',
    )


def repair_startup_fallbacks() -> None:
    path = "run_thog2_owt.py"
    replace_once(
        path,
        '    probe_interval = getattr(config, "plastic__layer_count_probe_interval", None)\n'
        '    probe_radius = int(config.plastic__layer_count_probe_radius)\n'
        '    max_step = int(config.plastic__layer_count_max_step)\n',
        '    probe_interval = getattr(config, "plastic__layer_count_probe_interval", None)\n'
        '    probe_radius = int(getattr(config, "plastic__layer_count_probe_radius", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS", 1)))\n'
        '    max_step = int(getattr(config, "plastic__layer_count_max_step", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_MAX_STEP", 1)))\n',
    )
    replace_once(
        path,
        '    _print_plastic_option("plastic__coarse_phase:", str(config.plastic__coarse_phase))\n'
        '    _print_plastic_option("plastic__phase_1_n_steps:", _startup_optional(config.plastic__phase_1_n_steps))\n'
        '    _print_plastic_option("plastic__phase_1_starting_layer_count:", _startup_optional(config.plastic__phase_1_starting_layer_count))\n'
        '    _print_plastic_option("plastic__phase_1__number_of_trials:", _startup_optional(config.plastic__phase_1__number_of_trials))\n'
        '    _print_plastic_option("plastic__phase_1_evaluation_steps_count:", _startup_optional(config.plastic__phase_1_evaluation_steps_count))\n'
        '    if config.plastic__coarse_phase == "enabled":\n',
        '    coarse_phase = str(getattr(config, "plastic__coarse_phase", "disabled"))\n'
        '    phase_1_n_steps = getattr(config, "plastic__phase_1_n_steps", None)\n'
        '    phase_1_starting_layer_count = getattr(config, "plastic__phase_1_starting_layer_count", None)\n'
        '    phase_1_number_of_trials = getattr(config, "plastic__phase_1__number_of_trials", None)\n'
        '    phase_1_evaluation_steps_count = getattr(config, "plastic__phase_1_evaluation_steps_count", None)\n'
        '    _print_plastic_option("plastic__coarse_phase:", coarse_phase)\n'
        '    _print_plastic_option("plastic__phase_1_n_steps:", _startup_optional(phase_1_n_steps))\n'
        '    _print_plastic_option("plastic__phase_1_starting_layer_count:", _startup_optional(phase_1_starting_layer_count))\n'
        '    _print_plastic_option("plastic__phase_1__number_of_trials:", _startup_optional(phase_1_number_of_trials))\n'
        '    _print_plastic_option("plastic__phase_1_evaluation_steps_count:", _startup_optional(phase_1_evaluation_steps_count))\n'
        '    if coarse_phase == "enabled":\n',
    )
    replace_once(
        path,
        '            coarse_phase=config.plastic__coarse_phase,\n'
        '            plastic_enabled=config.plastic__enabled,\n'
        '            do_learn_layer_count=config.plastic__do_learn_layer_count,\n'
        '            n_steps=config.plastic__phase_1_n_steps,\n'
        '            starting_layer_count=config.plastic__phase_1_starting_layer_count,\n'
        '            number_of_trials=config.plastic__phase_1__number_of_trials,\n'
        '            evaluation_steps_count=config.plastic__phase_1_evaluation_steps_count,\n',
        '            coarse_phase=coarse_phase,\n'
        '            plastic_enabled=config.plastic__enabled,\n'
        '            do_learn_layer_count=config.plastic__do_learn_layer_count,\n'
        '            n_steps=phase_1_n_steps,\n'
        '            starting_layer_count=phase_1_starting_layer_count,\n'
        '            number_of_trials=phase_1_number_of_trials,\n'
        '            evaluation_steps_count=phase_1_evaluation_steps_count,\n',
    )


def main() -> None:
    repair_probe_interval_compatibility()
    preserve_v03_identity_until_explicit_migration()
    repair_startup_fallbacks()


if __name__ == "__main__":
    main()
