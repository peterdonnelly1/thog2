from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one wrapper anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def update_train_owt_normalization() -> None:
    path = "train_OWT.sh"
    replace_once(
        path,
        '# vvv THOG accept underscore-form long options directly in the canonical wrapper while preserving established hyphen aliases\n'
        'THOG2_UNDERSCORE_NORMALIZED_ARGS=()\n',
        '# vvv THOG accept canonical double-underscore and legacy single-underscore long options while preserving established hyphen aliases\n'
        'thog2_normalize_long_option() {\n'
        '  local option_name="$1"\n'
        '  while [[ "$option_name" == *"__"* ]]; do\n'
        '    option_name="${option_name//__/_}"\n'
        '  done\n'
        '  printf \'%s\' "${option_name//_/-}"\n'
        '}\n'
        'THOG2_UNDERSCORE_NORMALIZED_ARGS=()\n',
    )
    replace_once(
        path,
        '    --plastic_layer_count_probe_window_size|--plastic-layer-count-probe-interval)\n'
        '      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }\n'
        '      export THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE="$2"\n'
        '      shift 2\n'
        '      ;;\n'
        '    --plastic_layer_count_probe_window_size=*|--plastic-layer-count-probe-interval=*)\n'
        '      export THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE="${1#*=}"\n'
        '      shift\n'
        '      ;;\n',
        '    --plastic__layer_count_probe_window_size|--plastic_layer_count_probe_window_size|--plastic-layer-count-probe-interval)\n'
        '      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }\n'
        '      export THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE="$2"\n'
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("--plastic-layer-count-probe-interval" "$2")\n'
        '      shift 2\n'
        '      ;;\n'
        '    --plastic__layer_count_probe_window_size=*|--plastic_layer_count_probe_window_size=*|--plastic-layer-count-probe-interval=*)\n'
        '      export THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE="${1#*=}"\n'
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("--plastic-layer-count-probe-interval=${1#*=}")\n'
        '      shift\n'
        '      ;;\n',
    )
    replace_once(
        path,
        '      THOG2_UNDERSCORE_NAME="${THOG2_UNDERSCORE_NAME//_/-}"\n'
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("${THOG2_UNDERSCORE_NAME}=${THOG2_UNDERSCORE_VALUE}")\n',
        '      THOG2_UNDERSCORE_NAME="$(thog2_normalize_long_option "$THOG2_UNDERSCORE_NAME")"\n'
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("${THOG2_UNDERSCORE_NAME}=${THOG2_UNDERSCORE_VALUE}")\n',
    )
    replace_once(
        path,
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("${1//_/-}")\n',
        '      THOG2_UNDERSCORE_NORMALIZED_ARGS+=("$(thog2_normalize_long_option "$1")")\n',
    )
    replace_once(
        path,
        'unset THOG2_UNDERSCORE_NORMALIZED_ARGS THOG2_UNDERSCORE_NAME THOG2_UNDERSCORE_VALUE\n'
        '# ^^^ THOG\n',
        'unset THOG2_UNDERSCORE_NORMALIZED_ARGS THOG2_UNDERSCORE_NAME THOG2_UNDERSCORE_VALUE\n'
        'unset -f thog2_normalize_long_option\n'
        '# ^^^ THOG\n',
    )


def main() -> None:
    update_train_owt_normalization()


if __name__ == "__main__":
    main()
