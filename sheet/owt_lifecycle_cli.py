# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .bases import normalize_registered_basis_family


@dataclass(frozen=True)
class NormalizedLifecycleCli:
    argv: Tuple[str, ...]
    environment: Dict[str, str]


_SHORT_VALUE_OPTIONS = {
    "q", "g", "n", "b", "c", "f", "y", "A", "G", "u", "e", "l", "w", "k", "I",
    "F", "N", "U", "V", "p", "B", "v", "W", "i", "a", "m", "L", "s", "M", "H",
    "D", "C", "P", "Q", "J", "O", "X", "Y", "S", "E", "T", "K", "r", "z", "Z",
    "d", "t", "o", "j", "R", "x",
}
_COMPACT_PRESETS = {
    "legacy_sheet_col",
    "depth",
    "jpeg_like_v1",
    "head_aware_block",
    "mlp_block",
    "full_block",
}


def _single_value(value: str, option: str) -> str:
    if "," in value or any(character.isspace() for character in value):
        raise ValueError(f"{option} accepts one value in resume/fork mode; grid values are fresh-run only")
    return value


def _lr_value(code: str, option: str, maximum: int) -> str:
    value = _single_value(code, option)
    if not value.isdigit():
        raise ValueError(f"{option} requires an integer code")
    integer = int(value, 10)
    if integer < 1 or integer > maximum:
        raise ValueError(f"{option} requires a value in 1..{maximum}")
    return f"{integer}e-5"


def _true_false(value: str, option: str) -> str:
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"{option} requires true or false")
    return normalized


def _preset_arguments(value: str) -> List[str]:
    preset = _single_value(value, "-p")
    if preset == "dense":
        return ["--model-type", "dense"]
    if preset not in _COMPACT_PRESETS:
        raise ValueError(f"unsupported -p preset for lifecycle mode: {preset}")
    return ["--model-type", "sheet", "--geometry-preset", preset]


def _basis_family(value: str, option: str) -> str:
    family = _single_value(value, option)
    return normalize_registered_basis_family(family)


def _short_translation(option: str, value: str, environment: Dict[str, str]) -> List[str]:
    if option == "q":
        return ["-q", value]
    if option == "g":
        return ["--run-name", value, "--experiment-prefix", value]
    if option == "n":
        return ["-n", value]
    if option == "b":
        return ["--batch-size", _single_value(value, "-b")]
    if option == "c":
        return ["--learning-rate", _lr_value(value, "-c", 1000)]
    if option == "f":
        return ["--min-lr", _lr_value(value, "-f", 100)]
    if option == "y":
        return ["--optimizer", value]
    if option == "A":
        return ["--gradient-accumulation-steps", value]
    if option == "G":
        return ["-G", value]
    if option == "u":
        return ["--eval-iters", value]
    if option == "e":
        return ["--eval-interval", value]
    if option == "l":
        return ["--log-interval", value]
    if option == "w":
        return ["--warmup-iters", value]
    if option == "k":
        return ["--checkpoint-interval", value]
    if option == "I":
        return ["-I", value]
    if option == "F":
        environment["THOG2_DEPTH_CURVE_PLOTS"] = value
        return []
    if option == "N":
        environment["THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS"] = value
        return []
    if option == "U":
        environment["THOG2_DEPTH_CURVE_RENDERER"] = value
        return []
    if option == "V":
        environment["THOG2_DEPTH_CURVE_LOCAL_HTML"] = _true_false(value, "-V")
        return []
    if option == "p":
        return _preset_arguments(value)
    if option == "B":
        return ["--basis-family", _basis_family(value, "-B")]
    if option == "v":
        return ["--basis-version", value]
    if option == "W":
        return ["--lapped-cosine-window-length", value]
    if option == "i":
        return ["--lapped-cosine-overlap-fraction", value]
    if option == "a":
        return ["--attention-geometry", value]
    if option == "m":
        return ["--mlp-geometry", value]
    if option == "L":
        return ["--n-layer", value]
    if option == "s":
        return ["--layer-dropout-stratum-size", value]
    if option == "M":
        return ["--layer-dropout-active-per-stratum", value]
    if option == "H":
        return ["--n-head", value]
    if option == "D":
        return ["--n-embd", value]
    if option == "C":
        return ["--block-size", value]
    if option == "P":
        return ["--o-depth", value]
    if option == "Q":
        return ["--o-attn-d-model", value]
    if option == "J":
        return ["--o-attn-qkv-per-channel", value]
    if option == "O":
        return ["--o-attn-out-per-channel", value]
    if option == "X":
        return ["--o-mlp-d-model", value]
    if option == "Y":
        return ["--o-mlp-hidden", value]
    if option == "S":
        return ["--checkpoint-segment-size", value]
    if option == "E":
        environment["THOG2_FAST_DISCARD"] = _true_false(value, "-E")
        return []
    if option == "T":
        return ["--dtype", value]
    if option == "K":
        return ["--attention-backend", value]
    if option == "r":
        return ["--residual-init-policy", value]
    if option == "z":
        return ["--residual-init-depth-source", value]
    if option == "Z":
        return ["--residual-init-depth-value", value]
    if option == "d":
        return ["--dataset", value, "--data-dir", f"data/{value}"]
    if option == "t":
        return ["--data-dir", value]
    if option == "o":
        return ["--checkpoint-root", value]
    if option == "j":
        return ["--log-root", value]
    if option == "R":
        return ["--result-root", value]
    if option == "x":
        return ["--dry-run"] if _true_false(value, "-x") == "true" else []
    raise ValueError(f"unsupported lifecycle wrapper option: -{option}")


def normalize_lifecycle_wrapper_argv(argv: Sequence[str]) -> NormalizedLifecycleCli:
    normalized: List[str] = []
    environment: Dict[str, str] = {}
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            normalized.extend(argv[index:])
            break
        if argument == "--mlp-hidden-compressor":
            if index + 1 >= len(argv):
                raise ValueError("--mlp-hidden-compressor requires a value")
            normalized.extend((argument, _basis_family(argv[index + 1], argument)))
            index += 2
            continue
        if argument.startswith("--mlp-hidden-compressor="):
            normalized.append(
                "--mlp-hidden-compressor=" + _basis_family(argument.split("=", 1)[1], "--mlp-hidden-compressor")
            )
            index += 1
            continue
        if argument.startswith("--") or argument in ("-h",):
            normalized.append(argument)
            index += 1
            continue
        if len(argument) >= 2 and argument[0] == "-" and argument[1] in _SHORT_VALUE_OPTIONS:
            option = argument[1]
            attached_value = argument[2:]
            if attached_value:
                value = attached_value
                index += 1
            else:
                if index + 1 >= len(argv):
                    raise ValueError(f"-{option} requires a value")
                value = argv[index + 1]
                index += 2
            normalized.extend(_short_translation(option, value, environment))
            continue
        normalized.append(argument)
        index += 1
    return NormalizedLifecycleCli(tuple(normalized), environment)


__all__ = ["NormalizedLifecycleCli", "normalize_lifecycle_wrapper_argv"]
# ^^^ THOG
