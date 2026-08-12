# vvv THOG
"""Register hidden normalized aliases for canonical depth-curve instrumentation controls."""

from __future__ import annotations

import argparse

from . import depth_weight_curves_and_observational_probes_patch as _depth


# vvv THOG accept canonical, argparse-normalized double-hyphen, and train_OWT single-hyphen spellings without changing the public destination name
def _normalized_options(canonical: str) -> tuple[str, str]:
    body = canonical[2:] if canonical.startswith("--") else canonical
    parser_normalized = "--" + body.replace("_", "-")
    wrapper_body = parser_normalized[2:]
    while "--" in wrapper_body:
        wrapper_body = wrapper_body.replace("--", "-")
    wrapper_normalized = "--" + wrapper_body
    return parser_normalized, wrapper_normalized
# ^^^ THOG


# vvv THOG public boolean spelling is explicit true|false; the old valueless true form remains accepted only as a compatibility bridge for already-saved launch stanzas
def _explicit_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected true or false; got {value!r}")
# ^^^ THOG


def _ensure_cli_arguments_with_normalized_aliases(parser: argparse.ArgumentParser) -> None:
    if bool(getattr(parser, _depth._CLI_INSTALLED_ATTRIBUTE, False)):
        return

    controls = (
        (
            "instrumentation__depth_weight_curves__scalar_weights_per_matrix",
            {"type": int, "default": _depth._DEFAULT_SCALARS_PER_MATRIX},
        ),
        (
            "instrumentation__depth_weight_curves__depth_evaluation_points",
            {"type": int, "default": _depth._DEFAULT_DEPTH_POINTS},
        ),
        (
            "instrumentation__depth_weight_curves__time_mode",
            {
                "choices": ("latest", "accumulate"),
                "default": _depth._DEFAULT_TIME_MODE,
            },
        ),
        (
            "instrumentation__depth_weight_curves__history_length",
            {"type": int, "default": _depth._DEFAULT_HISTORY_LENGTH},
        ),
        (
            "instrumentation__depth_weight_curves__log_every_n_steps",
            {"type": int, "default": _depth._DEFAULT_LOG_EVERY_N_STEPS},
        ),
    )
    for destination, kwargs in controls:
        canonical = f"--{destination}"
        parser_normalized, wrapper_normalized = _normalized_options(canonical)
        parser.add_argument(
            canonical,
            parser_normalized,
            wrapper_normalized,
            dest=destination,
            **kwargs,
        )

    destination = "instrumentation__depth_weight_curves__same_coordinates_all_runs"
    canonical = f"--{destination}"
    parser_normalized, wrapper_normalized = _normalized_options(canonical)
    parser.add_argument(
        canonical,
        parser_normalized,
        wrapper_normalized,
        dest=destination,
        nargs="?",
        const=True,
        type=_explicit_bool,
        default=False,
        metavar="true|false",
    )
    setattr(parser, _depth._CLI_INSTALLED_ATTRIBUTE, True)


_depth._ensure_cli_arguments = _ensure_cli_arguments_with_normalized_aliases
# ^^^ THOG


__all__ = ["_ensure_cli_arguments_with_normalized_aliases"]
# ^^^ THOG
