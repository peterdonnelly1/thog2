# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class LayerDropoutConfig:
    n_layer: int
    stratum_size: int
    active_per_stratum: int
    resample_steps: int = 1
    seed: int = 1337

    def __post_init__(self) -> None:
        for name in ("n_layer", "stratum_size", "active_per_stratum", "resample_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer; got {self.seed!r}")
        if self.n_layer % self.stratum_size != 0:
            raise ValueError(
                "n_layer must be divisible by stratum_size; "
                f"got n_layer={self.n_layer}, stratum_size={self.stratum_size}"
            )
        if self.active_per_stratum > self.stratum_size:
            raise ValueError(
                "active_per_stratum must not exceed stratum_size; "
                f"got active_per_stratum={self.active_per_stratum}, stratum_size={self.stratum_size}"
            )

    @property
    def n_strata(self) -> int:
        return self.n_layer // self.stratum_size

    @property
    def n_active_layers(self) -> int:
        return self.n_strata * self.active_per_stratum

    @property
    def all_layers_active(self) -> bool:
        return self.active_per_stratum == self.stratum_size

    def resample_bucket(self, completed_updates: int) -> int:
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError(f"completed_updates must be a non-negative integer; got {completed_updates!r}")
        return completed_updates // self.resample_steps

    def active_layer_indices(self, completed_updates: int) -> Optional[Tuple[int, ...]]:
        # The all-active case is deliberately a zero-sampling fast path.                      # <<< THOG layer-dropout degenerate path preserves current execution
        if self.all_layers_active:
            return None

        bucket = self.resample_bucket(completed_updates)
        mixed_seed = (
            (int(self.seed) * 6364136223846793005)
            + (bucket * 1442695040888963407)
        ) & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)

        selected = []
        for stratum_start in range(0, self.n_layer, self.stratum_size):
            offsets = torch.randperm(self.stratum_size, generator=generator)[: self.active_per_stratum]
            selected.extend(stratum_start + int(offset) for offset in offsets.tolist())
        selected.sort()
        return tuple(selected)
# ^^^ THOG
