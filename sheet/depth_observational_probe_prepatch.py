# vvv THOG
"""Import-order shim for the fixed-run observational DEPTH probe overlay."""

from __future__ import annotations

from typing import Tuple

from .training_model import TrainingSheetGPT


# vvv THOG the first observational overlay historically expected this helper name; provide a no-op compatibility surface before that module imports, while the final probe executor does not rely on it
def _layer_indices_for_current_forward_compat(self: TrainingSheetGPT) -> Tuple[int, ...]:
    return self._optimizer_update_layer_indices()


if not hasattr(TrainingSheetGPT, "_layer_indices_for_current_forward"):
    TrainingSheetGPT._layer_indices_for_current_forward = _layer_indices_for_current_forward_compat
# ^^^ THOG


__all__ = ["_layer_indices_for_current_forward_compat"]
# ^^^ THOG
