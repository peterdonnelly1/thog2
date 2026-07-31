# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from torch import Tensor


MaterializeAt = Callable[[Tensor, int], Tensor]
InitializeParameters = Callable[[Tensor, str, float, int], None]
RescaleOutput = Callable[[Tensor, float], None]


@dataclass(frozen=True)
class RecurrenceGeneratorDefinition:
    family: str
    aliases: Tuple[str, ...]
    version: str
    artifact_tag: str
    persistent_widths: Tuple[int, ...]
    supported_targets: Tuple[str, ...]
    option_names: Tuple[str, ...]
    description: str
    materialize_at: MaterializeAt
    initialize_parameters: InitializeParameters
    rescale_output: RescaleOutput

    def __post_init__(self) -> None:
        family = self.family.strip().lower()
        aliases = tuple(alias.strip().lower() for alias in self.aliases)
        version = self.version.strip().lower()
        artifact_tag = self.artifact_tag.strip().upper()
        if not family:
            raise ValueError("recurrence generator family must be non-empty")
        if not version:
            raise ValueError("recurrence generator version must be non-empty")
        if not artifact_tag:
            raise ValueError("recurrence generator artifact_tag must be non-empty")
        if not self.persistent_widths or any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in self.persistent_widths):
            raise ValueError("recurrence generator persistent_widths must contain positive integers")
        if not self.supported_targets:
            raise ValueError("recurrence generator must advertise at least one supported target")
        if any(not name.startswith("generator_") for name in self.option_names):
            raise ValueError("recurrence generator option names must use the generator_ prefix")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "artifact_tag", artifact_tag)

    def accepts_persistent_width(self, width: int) -> bool:
        return int(width) in self.persistent_widths

    def validate_options(self, options: Dict[str, str]) -> Dict[str, str]:
        unknown = tuple(sorted(set(options) - set(self.option_names)))
        if unknown:
            raise ValueError(
                f"recurrence generator {self.family}@{self.version} does not accept options: {', '.join(unknown)}"
            )
        return dict(options)
# ^^^ THOG