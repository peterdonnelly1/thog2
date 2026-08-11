# vvv THOG
"""PLASTIC v0.56 objective/decision matrix and operator interpretation help."""

from __future__ import annotations

import argparse

from . import help_registry_descriptor_patch as _registry
from . import plastic_depth_v056_objective_decision_patch as _v056


_SECTION = "PLASTIC DEPTH decision algorithms"
_GROWTH_DISCOUNT_OPTION = "--plastic__layer_count_decision_algorithm__growth_side_discount VALUE"
_ROWS = (
    (
        "LDA",
        "--plastic__layer_count_decision_algorithm ALGORITHM",
        "directional_coherence | theil_sen_kendall_LRA | sen_kendall__tau__stratified; selected independently of the objective",
    ),
    ("", "    For Sen slope:", ""),
    (
        "",
        "        sen < 0",
        "y tends to decrease as x increases. Adding layers tends to improve the wall-time-equivalent economic score",
    ),
    (
        "",
        "        sen = 0",
        "no overall trend. Little directional economic preference",
    ),
    (
        "",
        "        sen > 0",
        "y tends to increase as x increases. Removing layers tends to improve the wall-time-equivalent economic score",
    ),
    ("", "    For Kendall tau:", ""),
    (
        "",
        "        tau = -1",
        "perfectly decreasing ordering. Extremely strong/near-perfect indication toward adding layers",
    ),
    ("", "        tau ≈ -0.8", "strongly decreasing. Strong indication toward adding layers"),
    (
        "",
        "        tau ≈ -0.5",
        "moderately decreasing. Moderate / minimum accepted indication toward adding layer",
    ),
    ("", "        tau ≈  0", "little/no monotonic association. No reliable directional indication"),
    (
        "",
        "        tau ≈ +0.5",
        "moderately increasing. Moderate / minimum accepted indication toward removing layers",
    ),
    ("", "        tau ≈ +0.8", "strongly increasing. Strong indication toward removing layers"),
    (
        "",
        "        tau = +1",
        "perfectly increasing ordering. Extremely strong/near-perfect indication toward removing layers",
    ),
    (
        "—",
        "objective × decision compatibility",
        "all 4 objectives are permitted with all 3 decision algorithms (4 × 3 full matrix)",
    ),
    (
        "—",
        "lowest_loss decision score y",
        "probe loss; lower is better",
    ),
    (
        "—",
        "layer_efficiency decision score y",
        "existing layer-efficiency objective score; lower is better",
    ),
    (
        "—",
        "relative_training_wall_time decision score y",
        "wall-time-equivalent economic score in equivalent seconds; lower is better",
    ),
    (
        "—",
        "memory_budget decision score y",
        "probe loss among memory-feasible candidates; infeasible candidates are excluded",
    ),
    (
        "—",
        "directional_coherence",
        "base non-TSK algorithm: candidate score histories use robust median/MAD/sigma, change_z/score_z, newest-favourable and strict-majority gates plus L/R/A directional coherence",
    ),
    (
        "—",
        "theil_sen_kendall_LRA",
        "compute Sen/Kendall/adj per probe, classify L/R/A, then require a strict full-window directional majority; no legacy z machinery",
    ),
    (
        "—",
        "sen_kendall__tau__stratified",
        "pool only within-probe Sen slopes and Kendall pair evidence over the complete probe window; no L/R/A vote and no legacy z machinery",
    ),
    (
        "—",
        "TSK acceptance threshold",
        "fixed |tau| >= 0.5; grow requires tau <= -0.5, shrink requires tau >= +0.5",
    ),
    (
        "—",
        "TSK adjacent action check",
        "the exact adjacent candidate in the Sen-indicated direction must have undiscounted objective-score delta adj < 0 before committing ±1",
    ),
    (
        "—",
        _GROWTH_DISCOUNT_OPTION,
        "TSK only: credit [0,1] of beneficial growth-side objective evidence before Sen/Kendall; adverse growth evidence and the exact adjacent check remain undiscounted",
    ),
)


def _descriptor_sections_v056():
    sections = []
    for section, rows in _registry._DESCRIPTOR_SECTIONS:
        if section == _SECTION:
            continue
        filtered_rows = tuple(
            row for row in rows
            if row[1] != _GROWTH_DISCOUNT_OPTION
        )
        sections.append((section, filtered_rows))
    insertion_index = next(
        (index + 1 for index, (section, _rows) in enumerate(sections) if section == "PLASTIC DEPTH"),
        len(sections),
    )
    sections.insert(insertion_index, (_SECTION, _ROWS))
    return tuple(sections)


_registry._DESCRIPTOR_SECTIONS = _descriptor_sections_v056()


# vvv THOG final argparse help uses all three v0.56 decision names even when older layered formatters have already rewritten the selector description
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _format_help_v056(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    rendered = rendered.replace("wall_time__theil_sen_kendall_LRA", _v056.LRA_ALGORITHM)
    rendered = rendered.replace("wall_time__sen_kendall__tau__stratified", _v056.STRATIFIED_ALGORITHM)
    rendered = rendered.replace(
        f"directional_coherence (default) or {_v056.LRA_ALGORITHM}",
        f"directional_coherence (default), {_v056.LRA_ALGORITHM}, or {_v056.STRATIFIED_ALGORITHM}",
    )
    rendered = rendered.replace(
        "v0.55 Sen/Kendall only: credit X of beneficial growth-side economic evidence",
        "Sen/Kendall only: credit X of beneficial growth-side objective evidence",
    )
    if (
        any(action.dest == "plastic__enabled" for action in self._actions)
        and _v056.STRATIFIED_ALGORITHM not in rendered
    ):
        rendered = (
            rendered.rstrip()
            + "\n  --plastic__layer_count_decision_algorithm ALGORITHM\n"
            + f"                        directional_coherence (default), {_v056.LRA_ALGORITHM}, or {_v056.STRATIFIED_ALGORITHM}\n"
        )
    return rendered


argparse.ArgumentParser.format_help = _format_help_v056
# ^^^ THOG


__all__ = ["_descriptor_sections_v056", "_format_help_v056"]
# ^^^ THOG
