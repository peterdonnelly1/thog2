# vvv THOG
"""Register hidden normalized aliases for canonical depth-curve instrumentation controls."""

from __future__ import annotations

import argparse

from . import depth_weight_curves_and_observational_probes_patch as _depth


# vvv THOG the established CLI normalizer rewrites underscores to hyphens before argparse matching; keep canonical user spelling visible and add only hidden normalized aliases
def _normalized_option(canonical: str) -> str:
    return canonical.replace("__", "--").replace("_", "-")


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
        parser.add_argument(
            canonical,
            _normalized_option(canonical),
            dest=destination,
            **kwargs,
        )

    destination = "instrumentation__depth_weight_curves__same_coordinates_all_runs"
    canonical = f"--{destination}"
    normalized = _normalized_option(canonical)
    parser.add_argument(
        canonical,
        normalized,
        dest=destination,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    setattr(parser, _depth._CLI_INSTALLED_ATTRIBUTE, True)


_depth._ensure_cli_arguments = _ensure_cli_arguments_with_normalized_aliases
# ^^^ THOG


__all__ = ["_ensure_cli_arguments_with_normalized_aliases"]
# ^^^ THOG
