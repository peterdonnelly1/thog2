# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import torch


@dataclass(frozen=True)
class MemorySample:
    phase: str
    cuda: bool
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MemoryTelemetry:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.samples: List[MemorySample] = []

    @property
    def cuda_enabled(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def reset_peak(self) -> None:
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self, phase: str) -> MemorySample:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("memory phase must be a non-empty string")
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            allocated = 0
            reserved = 0
            peak_allocated = 0
            peak_reserved = 0
        sample = MemorySample(
            phase=phase,
            cuda=self.cuda_enabled,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
        self.samples.append(sample)
        return sample

    def report(self) -> Dict[str, object]:
        return {
            "device": str(self.device),
            "cuda": self.cuda_enabled,
            "samples": [sample.to_dict() for sample in self.samples],
        }
# ^^^ THOG
