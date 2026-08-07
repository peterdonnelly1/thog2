# vvv THOG
"""PLASTIC v0.521 configurable probe-token sampling and hybrid probe-delta console rendering."""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

import torch

import constants as _constants

from . import plastic_depth_console_cleanup_patch as _cleanup
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_PROBE_VECTOR = re.compile(
    r"probe_losses \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]*)\]"
)
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _plastic_depth_sampled_token_indices_v0521(self: Any, targets: torch.Tensor) -> torch.Tensor:
    flattened = targets.reshape(-1)
    valid = torch.nonzero(flattened != -1, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise RuntimeError("PLASTIC DEPTH inline probe found no non-ignored target tokens")
    requested = int(self.config.plastic__layer_count_probe__number_of_sampled_valid_tokens)
    if requested == 0:
        return valid
    if requested > int(valid.numel()):
        raise RuntimeError(
            "plastic__layer_count_probe__number_of_sampled_valid_tokens exceeds the actual valid-token count "
            f"in this probe microbatch: requested={requested}, valid={int(valid.numel())}"
        )
    if requested == int(valid.numel()):
        return valid
    generator = torch.Generator(device="cpu")
    seed = (
        int(self.config.model_seed)
        + 1_000_003 * int(self.state.completed_updates)
        + 97_409 * int(self.distributed.rank)
    )
    generator.manual_seed(seed)
    positions = torch.randperm(int(valid.numel()), generator=generator)[:requested]
    return valid.index_select(0, positions.to(device=valid.device))


def _format_probe_delta(value: Optional[float]) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:+.3f}"


def _format_probe_absolute(value: Optional[float]) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.3f}"


def _render_probe_delta_values(
    offsets: Sequence[Any],
    losses: Sequence[Any],
) -> Optional[str]:
    resolved_offsets = tuple(int(value) for value in offsets)
    resolved_losses = tuple(None if value is None else float(value) for value in losses)
    if len(resolved_offsets) != len(resolved_losses) or 0 not in resolved_offsets:
        return None
    current_index = resolved_offsets.index(0)
    current_loss = resolved_losses[current_index]
    if current_loss is None or not math.isfinite(current_loss):
        return None
    rendered = []
    for offset, loss in zip(resolved_offsets, resolved_losses):
        if offset == 0:
            rendered.append(
                f"{_constants.BOLD_WHITE}{_format_probe_absolute(loss)}{_constants.R}"
            )
            continue
        delta = None if loss is None else float(loss) - float(current_loss)
        text = _format_probe_delta(delta)
        if delta is not None and math.isfinite(delta) and delta < 0.0:
            text = f"{_cleanup._GREEN}{text}{_cleanup._RESET}"
        rendered.append(text)
    return ", ".join(rendered)


def _format_progress_line_with_probe_deltas(
    run_id: str,
    event: str,
    payload: dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    offsets = payload.get("plastic_probe_offsets")
    losses = payload.get("plastic_probe_losses")
    if offsets is None or losses is None:
        return line
    rendered = _render_probe_delta_values(offsets, losses)
    if rendered is None:
        return line

    def replace(match: re.Match[str]) -> str:
        return f"probe_Δloss [{match.group('label')}] = [{rendered}]"

    return _PROBE_VECTOR.sub(replace, line, count=1)


_trainer_step.TrainerStepMixin._plastic_depth_sampled_token_indices = (
    _plastic_depth_sampled_token_indices_v0521
)
_stage6.format_progress_line = _format_progress_line_with_probe_deltas


__all__ = [
    "_format_probe_delta",
    "_plastic_depth_sampled_token_indices_v0521",
    "_render_probe_delta_values",
]
# ^^^ THOG
