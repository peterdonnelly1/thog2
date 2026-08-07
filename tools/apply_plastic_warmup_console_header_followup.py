# vvv THOG
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"expected one replacement anchor in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _update_startup_report() -> None:
    path = ROOT / "run_thog2_owt.py"
    _replace_once(
        path,
        '    "plastic__enabled:",\n    "plastic__coarse_phase:",\n',
        '    "plastic__enabled:",\n'
        '    "plastic__runtime_phase:",\n'
        '    "plastic__coarse_phase:",\n'
        '    "plastic__coarse_phase_roll_through:",\n'
        '    "plastic__log_interval_coarse:",\n',
    )
    _replace_once(
        path,
        '    "plastic__layer_count_probe__probe_every_n_steps:",\n    "plastic__layer_count_probe_radius:",\n',
        '    "plastic__layer_count_probe__probe_every_n_steps:",\n'
        '    "plastic__layer_count_probe__number_of_sampled_valid_tokens:",\n'
        '    "plastic__layer_count_probe_radius:",\n',
    )
    _replace_once(
        path,
        '    probe_interval = getattr(config, "plastic__layer_count_probe__probe_every_n_steps", None)\n'
        '    probe_radius = int(getattr(config, "plastic__layer_count_probe_radius", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS", 1)))\n',
        '    probe_interval = getattr(config, "plastic__layer_count_probe__probe_every_n_steps", None)\n'
        '    probe_token_count = int(getattr(config, "plastic__layer_count_probe__number_of_sampled_valid_tokens", 1024))\n'
        '    probe_radius = int(getattr(config, "plastic__layer_count_probe_radius", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS", 1)))\n',
    )
    _replace_once(
        path,
        '    _print_plastic_option("plastic__enabled:", _startup_bool(config.plastic__enabled))\n'
        '    coarse_phase = str(getattr(config, "plastic__coarse_phase", "disabled"))\n',
        '    _print_plastic_option("plastic__enabled:", _startup_bool(config.plastic__enabled))\n'
        '    _print_plastic_option("plastic__runtime_phase:", str(getattr(config, "plastic__runtime_phase", "fine")))\n'
        '    coarse_phase = str(getattr(config, "plastic__coarse_phase", "disabled"))\n',
    )
    _replace_once(
        path,
        '    _print_plastic_option("plastic__coarse_phase:", coarse_phase)\n'
        '    _print_plastic_option("plastic__phase_1_n_steps:", _startup_optional(phase_1_n_steps))\n',
        '    _print_plastic_option("plastic__coarse_phase:", coarse_phase)\n'
        '    _print_plastic_option("plastic__coarse_phase_roll_through:", _startup_bool(getattr(config, "plastic__coarse_phase_roll_through", False)))\n'
        '    _print_plastic_option("plastic__log_interval_coarse:", str(int(getattr(config, "plastic__log_interval_coarse", 10))))\n'
        '    _print_plastic_option("plastic__phase_1_n_steps:", _startup_optional(phase_1_n_steps))\n',
    )
    _replace_once(
        path,
        '    _print_plastic_option("plastic__layer_count_probe__probe_every_n_steps:", _startup_optional(probe_interval))\n'
        '    _print_plastic_option("plastic__layer_count_probe_radius:", str(probe_radius))\n',
        '    _print_plastic_option("plastic__layer_count_probe__probe_every_n_steps:", _startup_optional(probe_interval))\n'
        '    _print_plastic_option("plastic__layer_count_probe__number_of_sampled_valid_tokens:", str(probe_token_count))\n'
        '    _print_plastic_option("plastic__layer_count_probe_radius:", str(probe_radius))\n',
    )


def _update_probe_render_test() -> None:
    path = ROOT / "tests/test_plastic_depth_probe_sampling_v0521.py"
    _replace_once(
        path,
        '    assert f"{constants.BOLD_WHITE}4.077{constants.R}" in rendered\n'
        '    assert f"{cleanup._GREEN}-0.027{cleanup._RESET}" in rendered\n'
        '    assert f"{cleanup._GREEN}-0.017{cleanup._RESET}" in rendered\n'
        '    assert cleanup._GREEN not in rendered.split(", ")[0]\n'
        '    assert cleanup._GREEN not in rendered.split(", ")[3]\n',
        '    assert f"{constants.BOLD_WHITE}{constants.UNDER}4.077{constants.R}" in rendered\n'
        '    assert f"{constants.BOLD_MAGENTA}-0.027{constants.R}" in rendered\n'
        '    assert f"{cleanup._GREEN}-0.017{cleanup._RESET}" in rendered\n'
        '    assert constants.BOLD_MAGENTA not in rendered.split(", ")[0]\n'
        '    assert cleanup._GREEN not in rendered.split(", ")[0]\n'
        '    assert cleanup._GREEN not in rendered.split(", ")[3]\n',
    )


def _update_startup_report_test() -> None:
    path = ROOT / "tests/test_run_thog2_owt_startup_report.py"
    _replace_once(
        path,
        '    assert "plastic__layer_count_update_brake:" in output\n'
        '    assert "plastic__layer_count_probe__window_size_as_number_of_probes:" in output\n',
        '    assert "plastic__runtime_phase:" in output\n'
        '    assert "plastic__coarse_phase_roll_through:" in output\n'
        '    assert "plastic__log_interval_coarse:" in output\n'
        '    assert "plastic__layer_count_update_brake:" in output\n'
        '    assert "plastic__layer_count_probe__number_of_sampled_valid_tokens:" in output\n'
        '    assert "plastic__layer_count_probe__window_size_as_number_of_probes:" in output\n',
    )
    _replace_once(
        path,
        '    assert extrapolation_row.endswith("   0.8")\n',
        '    token_row = next(\n'
        '        line\n'
        '        for line in rows\n'
        '        if "plastic__layer_count_probe__number_of_sampled_valid_tokens:" in line\n'
        '    )\n'
        '    assert token_row.endswith("   1024")\n'
        '    assert extrapolation_row.endswith("   0.8")\n',
    )


def main() -> None:
    _update_startup_report()
    _update_probe_render_test()
    _update_startup_report_test()


if __name__ == "__main__":
    main()
# ^^^ THOG
