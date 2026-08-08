# vvv THOG
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "train_OWT_core.sh"
TEST = ROOT / "tests" / "test_plastic_v0541_wall_time_wrapper_cli.py"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:120]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    text = CORE.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="3.0"\nPLASTIC_LAYER_COUNT_COST_WEIGHT="0.0"',
        'PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="3.0"\n'
        'PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT="0.9"\n'
        'PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW=64\n'
        'PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS=16\n'
        'PLASTIC_LAYER_COUNT_COST_WEIGHT="0.0"',
    )

    text = replace_once(
        text,
        '  --plastic__layer_count_probe_noise_lambda VALUE=${PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA}\n'
        '  --plastic__layer_count_cost_weight VALUE=${PLASTIC_LAYER_COUNT_COST_WEIGHT}',
        '  --plastic__layer_count_probe_noise_lambda VALUE=${PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA}\n'
        '  --plastic__wall_time_equivalent_time_gain_discount VALUE=${PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT}\n'
        '  --plastic__wall_time_equivalent_time_gain_loss_rate_window N=${PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW}\n'
        '  --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations N=${PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS}\n'
        '  --plastic__layer_count_cost_weight VALUE=${PLASTIC_LAYER_COUNT_COST_WEIGHT}',
    )

    text = replace_once(
        text,
        '|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence|',
        '|--plastic__layer_count_probe_noise_lambda|--plastic__wall_time_equivalent_time_gain_discount|--plastic__wall_time_equivalent_time_gain_loss_rate_window|--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations|--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence|',
    )

    text = replace_once(
        text,
        '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$2" ;;\n'
        '        --plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence)',
        '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$2" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_discount) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT="$2" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_loss_rate_window) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW="$2" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS="$2" ;;\n'
        '        --plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence)',
    )

    text = replace_once(
        text,
        '|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence=*|',
        '|--plastic__layer_count_probe_noise_lambda=*|--plastic__wall_time_equivalent_time_gain_discount=*|--plastic__wall_time_equivalent_time_gain_loss_rate_window=*|--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=*|--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence=*|',
    )

    text = replace_once(
        text,
        '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$plastic_value" ;;\n'
        '        --plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence)',
        '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$plastic_value" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_discount) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT="$plastic_value" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_loss_rate_window) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW="$plastic_value" ;;\n'
        '        --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations) PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS="$plastic_value" ;;\n'
        '        --plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence)',
    )

    text = replace_once(
        text,
        'validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES" "PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES"\n'
        'case "$PLASTIC_LAYER_SAMPLING_INITIALISATION"',
        'validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES" "PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES"\n'
        'validate_positive_uint "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW" "PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW"\n'
        'validate_positive_uint "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS" "PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS"\n'
        '(( PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS <= PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW )) || { echo "PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS must not exceed PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW." >&2; exit 2; }\n'
        'python - "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT" <<\'PY\'\n'
        'import math, sys\n'
        'value = float(sys.argv[1])\n'
        'if not math.isfinite(value) or not (0.0 <= value <= 1.0):\n'
        '    raise SystemExit("PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT must lie in [0.0, 1.0]")\n'
        'PY\n'
        'case "$PLASTIC_LAYER_SAMPLING_INITIALISATION"',
    )

    text = replace_once(
        text,
        '    optional_args+=(--plastic__layer_count_probe_noise_lambda "$PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA")\n'
        '    optional_args+=(--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT")',
        '    optional_args+=(--plastic__layer_count_probe_noise_lambda "$PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA")\n'
        '    optional_args+=(--plastic__wall_time_equivalent_time_gain_discount "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT")\n'
        '    optional_args+=(--plastic__wall_time_equivalent_time_gain_loss_rate_window "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_WINDOW")\n'
        '    optional_args+=(--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations "$PLASTIC_WALL_TIME_EQUIVALENT_TIME_GAIN_LOSS_RATE_MIN_OBSERVATIONS")\n'
        '    optional_args+=(--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT")',
    )

    CORE.write_text(text, encoding="utf-8")

    TEST.write_text(
        '''# vvv THOG\nfrom __future__ import annotations\n\nimport subprocess\nfrom pathlib import Path\n\n\nROOT = Path(__file__).resolve().parents[1]\nCORE = ROOT / "train_OWT_core.sh"\nFLAGS = (\n    "--plastic__wall_time_equivalent_time_gain_discount",\n    "--plastic__wall_time_equivalent_time_gain_loss_rate_window",\n    "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",\n)\n\n\ndef _run_help(arguments: list[str]) -> subprocess.CompletedProcess[str]:\n    return subprocess.run(\n        ["bash", str(CORE), *arguments, "-h"],\n        cwd=ROOT,\n        text=True,\n        stdout=subprocess.PIPE,\n        stderr=subprocess.PIPE,\n        check=False,\n    )\n\n\ndef test_wall_time_equivalent_gain_controls_are_consumed_before_getopts() -> None:\n    result = _run_help([\n        FLAGS[0], "0.9",\n        FLAGS[1], "64",\n        FLAGS[2], "16",\n    ])\n    assert result.returncode == 0, result.stderr\n    assert "Unknown option" not in result.stderr\n    for flag in FLAGS:\n        assert flag in result.stdout\n\n\ndef test_wall_time_equivalent_gain_equals_forms_are_consumed_before_getopts() -> None:\n    result = _run_help([\n        f"{FLAGS[0]}=0.9",\n        f"{FLAGS[1]}=64",\n        f"{FLAGS[2]}=16",\n    ])\n    assert result.returncode == 0, result.stderr\n    assert "Unknown option" not in result.stderr\n\n\ndef test_wall_time_equivalent_gain_controls_are_forwarded_to_python() -> None:\n    source = CORE.read_text(encoding="utf-8")\n    for flag in FLAGS:\n        forwarding = f"optional_args+=({flag} "\n        assert forwarding in source\n# ^^^ THOG\n''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
# ^^^ THOG
