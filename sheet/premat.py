# vvv THOG dynamic pre-materialisation scheduler, admission policy, lifecycle, and telemetry
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


PREMAT_SWITCHES = ("enabled", "disabled")
PREMAT_ATTENTION_MODES = ("fused", "unfused")
PREMAT_DEFAULT_GPU_MEMORY_BUFFER_GB = 1.0
PREMAT_TELEMETRY_VERSION = 1


def validate_premat_configuration(
    *,
    premat: str,
    attention_mode: str,
    stay_below_current_peak: bool,
    stay_within_global_buffer: bool,
    gpu_memory_buffer_gb: float,
    logging: str,
    instra: str,
) -> None:
    for name, value in (("premat", premat), ("premat_logging", logging), ("premat_instra", instra)):
        if value not in PREMAT_SWITCHES:
            raise ValueError(f"{name} must be one of {PREMAT_SWITCHES}; got {value!r}")
    if attention_mode not in PREMAT_ATTENTION_MODES:
        raise ValueError(
            f"premat_attention_mode must be one of {PREMAT_ATTENTION_MODES}; "
            f"got {attention_mode!r}"
        )
    if not isinstance(stay_below_current_peak, bool):
        raise ValueError("premat_headroom_stay_below_current_peak must be bool")
    if not isinstance(stay_within_global_buffer, bool):
        raise ValueError("premat_headroom_stay_within_global_buffer must be bool")
    if stay_below_current_peak and stay_within_global_buffer:
        raise ValueError(
            "premat_headroom_stay_below_current_peak and "
            "premat_headroom_stay_within_global_buffer are mutually exclusive"
        )
    if (
        isinstance(gpu_memory_buffer_gb, bool)
        or not isinstance(gpu_memory_buffer_gb, (int, float))
        or not math.isfinite(float(gpu_memory_buffer_gb))
        or float(gpu_memory_buffer_gb) < 0.0
    ):
        raise ValueError("premat_gpu_memory_buffer_gb must be finite and non-negative")


class CandidateState(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    MATERIALISING = "MATERIALISING"
    AVAILABLE = "AVAILABLE"
    CONSUMING = "CONSUMING"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True)
class PrematMemoryObservation:
    process_allocated_bytes: int
    process_reserved_bytes: int
    process_ordinary_peak_bytes: int
    device_free_bytes: int
    device_total_bytes: int

    @property
    def reusable_allocator_bytes(self) -> int:
        return max(0, self.process_reserved_bytes - self.process_allocated_bytes)

    @property
    def device_used_bytes(self) -> int:
        return max(0, self.device_total_bytes - self.device_free_bytes)


@dataclass(frozen=True)
class CandidateEnvelope:
    retained_bytes: int
    materialisation_peak_bytes: int
    foreground_overlap_bytes: int

    @property
    def process_envelope_bytes(self) -> int:
        return max(
            self.materialisation_peak_bytes,
            self.retained_bytes + self.foreground_overlap_bytes,
        )


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    process_guard_passed: bool
    device_guard_passed: bool
    predicted_process_bytes: int
    predicted_device_used_bytes: int
    device_ceiling_bytes: int


def decide_candidate_admission(
    *,
    observation: PrematMemoryObservation,
    envelope: CandidateEnvelope,
    stay_below_current_peak: bool,
    gpu_memory_buffer_bytes: int,
) -> AdmissionDecision:
    predicted_process = observation.process_allocated_bytes + envelope.process_envelope_bytes
    physical_growth = max(
        0,
        envelope.process_envelope_bytes - observation.reusable_allocator_bytes,
    )
    predicted_device_used = observation.device_used_bytes + physical_growth
    device_ceiling = max(0, observation.device_total_bytes - gpu_memory_buffer_bytes)
    process_ok = (
        not stay_below_current_peak
        or predicted_process <= observation.process_ordinary_peak_bytes
    )
    device_ok = predicted_device_used <= device_ceiling
    if not process_ok:
        reason = "ordinary_process_peak"
    elif not device_ok:
        reason = "global_device_buffer"
    else:
        reason = "admitted"
    return AdmissionDecision(
        admitted=process_ok and device_ok,
        reason=reason,
        process_guard_passed=process_ok,
        device_guard_passed=device_ok,
        predicted_process_bytes=predicted_process,
        predicted_device_used_bytes=predicted_device_used,
        device_ceiling_bytes=device_ceiling,
    )


@dataclass
class _Candidate:
    layer_index: int
    family: str
    sequence: int
    envelope: CandidateEnvelope
    state: CandidateState = CandidateState.UNAVAILABLE
    owner: str = "none"
    tensor: Optional[Tensor] = None
    completion_event: Optional[torch.cuda.Event] = None
    launch_ns: Optional[int] = None
    available_ns: Optional[int] = None
    deadline_ns: Optional[int] = None
    critical_path_miss: bool = False
    admission_reason: str = "not_checked"


MaterializeCandidate = Callable[[str, int], Tensor]


class PrematRuntime:
    """One-model, one-stream dynamic pre-materialisation runtime.

    Only the current and immediately following logical layer are represented.
    Candidate launch is strictly ordered, and admission always precedes launch.
    """

    _FUSED_FAMILIES = ("QKV", "O", "UP", "DOWN")
    _UNFUSED_FAMILIES = ("QK", "V", "O", "UP", "DOWN")

    def __init__(
        self,
        *,
        materialize: MaterializeCandidate,
        n_embd: int,
        attention_mode: str,
        stay_below_current_peak: bool,
        gpu_memory_buffer_gb: float,
        logging_enabled: bool,
    ) -> None:
        self._materialize = materialize
        self._n_embd = int(n_embd)
        self._attention_mode = attention_mode
        self._stay_below_current_peak = bool(stay_below_current_peak)
        self._buffer_bytes = int(float(gpu_memory_buffer_gb) * (1024 ** 3))
        self._logging_enabled = bool(logging_enabled)
        self._stream: Optional[torch.cuda.Stream] = None
        self._device: Optional[torch.device] = None
        self._layer_indices: Tuple[int, ...] = ()
        self._position = -1
        self._candidates: Dict[Tuple[int, str], _Candidate] = {}
        self._sequence = 0
        self._event_sequence = 0
        self._events: List[Dict[str, object]] = []
        self._ordinary_peak_bytes = 0
        self._retained_bytes = 0
        self._activation_bytes = 0
        self._dtype_bytes = 0
        self._active = False
        self._aggregate = {
            "admitted": 0,
            "deferred": 0,
            "available_hits": 0,
            "waited_hits": 0,
            "main_stream_misses": 0,
        }

    @property
    def active(self) -> bool:
        return self._active

    def begin(
        self,
        layer_indices: Sequence[int],
        *,
        reference: Tensor,
    ) -> None:
        if self._active:
            raise RuntimeError("a pre-materialisation pass is already active")
        if reference.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "--premat enabled requires CUDA and must fail before the first logical layer"
            )
        resolved = tuple(int(value) for value in layer_indices)
        if not resolved:
            raise ValueError("pre-materialisation requires at least one logical layer")
        self._device = reference.device
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=reference.device)
        self._layer_indices = resolved
        self._position = -1
        self._candidates.clear()
        self._retained_bytes = 0
        self._activation_bytes = int(reference.numel() * reference.element_size())
        self._dtype_bytes = int(reference.element_size())
        self._active = True
        observation = self._observe_memory()
        self._ordinary_peak_bytes = max(
            self._ordinary_peak_bytes,
            observation.process_allocated_bytes,
        )
        self._record("pass_begin", layer=resolved[0], detail={"layer_count": len(resolved)})

    def end(self) -> None:
        if not self._active:
            return
        for candidate in tuple(self._candidates.values()):
            if candidate.state not in (CandidateState.CONSUMED, CandidateState.UNAVAILABLE):
                self._record(
                    "pass_end_release",
                    candidate=candidate,
                    detail={"prior_state": candidate.state.value},
                )
            candidate.tensor = None
            candidate.completion_event = None
        self._candidates.clear()
        self._retained_bytes = 0
        self._layer_indices = ()
        self._position = -1
        self._active = False
        self._record("pass_end")

    def layer_start(self, layer_index: int) -> None:
        self._require_active()
        try:
            position = self._layer_indices.index(int(layer_index))
        except ValueError as exc:
            raise RuntimeError(f"logical layer {layer_index} is outside the active premat pass") from exc
        if position < self._position:
            raise RuntimeError("premat logical layers must execute in pass order")
        self._position = position
        permitted = set(self._layer_indices[position : position + 2])
        stale = [key for key in self._candidates if key[0] not in permitted]
        for key in stale:
            candidate = self._candidates.pop(key)
            if candidate.state not in (CandidateState.CONSUMED, CandidateState.UNAVAILABLE):
                raise RuntimeError(
                    f"premat candidate left layer window before consumption: {key}={candidate.state.value}"
                )
        for permitted_layer in self._layer_indices[position : position + 2]:
            self._ensure_layer(permitted_layer)
        self._record("layer_start", layer=layer_index)
        self._update_ordinary_peak()
        self._advance()

    def event(self, name: str, *, layer_index: int) -> None:
        self._require_active()
        self._record(name, layer=layer_index)
        self._refresh_available()
        self._update_ordinary_peak()
        self._advance()

    def acquire(self, family: str, layer_index: int) -> Tensor:
        self._require_active()
        self.event(f"deadline_{family.lower()}", layer_index=layer_index)
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None:
            raise RuntimeError(f"premat candidate is outside the two-layer window: {key}")
        candidate.deadline_ns = time.perf_counter_ns()
        current_stream = torch.cuda.current_stream(device=self._device)
        if candidate.state == CandidateState.AVAILABLE:
            self._aggregate["available_hits"] += 1
        elif candidate.state == CandidateState.MATERIALISING:
            if candidate.completion_event is None:
                raise RuntimeError(f"materialising candidate {key} has no completion event")
            current_stream.wait_event(candidate.completion_event)
            candidate.critical_path_miss = True
            self._aggregate["waited_hits"] += 1
            self._record("critical_path_wait", candidate=candidate)
        elif candidate.state == CandidateState.UNAVAILABLE:
            candidate.owner = "main"
            candidate.critical_path_miss = True
            candidate.tensor = self._materialize(candidate.family, candidate.layer_index)
            self._aggregate["main_stream_misses"] += 1
            self._record("critical_path_materialize", candidate=candidate)
        else:
            raise RuntimeError(f"premat candidate {key} cannot be acquired from {candidate.state.value}")
        if candidate.tensor is None:
            raise RuntimeError(f"premat candidate {key} has no tensor at its deadline")
        candidate.tensor.record_stream(current_stream)
        candidate.state = CandidateState.CONSUMING
        self._record("consuming", candidate=candidate)
        return candidate.tensor

    def consumed(self, family: str, layer_index: int) -> None:
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None or candidate.state != CandidateState.CONSUMING:
            raise RuntimeError(f"premat candidate {key} was not consuming")
        candidate.state = CandidateState.CONSUMED
        self._retained_bytes = max(0, self._retained_bytes - candidate.envelope.retained_bytes)
        candidate.tensor = None
        candidate.completion_event = None
        self._record("consumed", candidate=candidate)
        self._update_ordinary_peak()
        self._advance()

    def report(self) -> Dict[str, object]:
        candidates = []
        for candidate in sorted(self._candidates.values(), key=lambda item: item.sequence):
            candidates.append(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "state": candidate.state.value,
                    "owner": candidate.owner,
                    "critical_path_miss": candidate.critical_path_miss,
                    "admission_reason": candidate.admission_reason,
                    "envelope": asdict(candidate.envelope),
                }
            )
        memory = None
        if self._device is not None and torch.cuda.is_available():
            memory = asdict(self._observe_memory())
            memory["global_buffer_bytes"] = self._buffer_bytes
            memory["premat_retained_bytes"] = self._retained_bytes
        return {
            "version": PREMAT_TELEMETRY_VERSION,
            "enabled": True,
            "attention_mode": self._attention_mode,
            "headroom_mode": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
            "current_layer_index": (
                self._layer_indices[self._position]
                if self._active and self._position >= 0
                else None
            ),
            "next_layer_index": (
                self._layer_indices[self._position + 1]
                if self._active and self._position + 1 < len(self._layer_indices)
                else None
            ),
            "candidates": candidates,
            "events": list(self._events[-256:]),
            "memory": memory,
            "aggregate": dict(self._aggregate),
        }

    def _families(self) -> Tuple[str, ...]:
        return self._FUSED_FAMILIES if self._attention_mode == "fused" else self._UNFUSED_FAMILIES

    def _ensure_layer(self, layer_index: int) -> None:
        for family in self._families():
            key = (layer_index, family)
            if key in self._candidates:
                continue
            self._sequence += 1
            self._candidates[key] = _Candidate(
                layer_index=layer_index,
                family=family,
                sequence=self._sequence,
                envelope=self._candidate_envelope(family),
            )

    def _candidate_envelope(self, family: str) -> CandidateEnvelope:
        width = self._n_embd
        rows = {
            "QKV": 3 * width,
            "QK": 2 * width,
            "V": width,
            "O": width,
            "UP": 4 * width,
            "DOWN": width,
        }[family]
        columns = 4 * width if family == "DOWN" else width
        retained = rows * columns * self._dtype_bytes
        materialisation_peak = 2 * retained if family in ("QKV", "QK") else retained
        activation_multiple = {
            "QKV": 2,
            "QK": 3,
            "V": 4,
            "O": 5,
            "UP": 3,
            "DOWN": 5,
        }[family]
        return CandidateEnvelope(
            retained_bytes=retained,
            materialisation_peak_bytes=materialisation_peak,
            foreground_overlap_bytes=activation_multiple * self._activation_bytes,
        )

    def _observe_memory(self) -> PrematMemoryObservation:
        if self._device is None:
            raise RuntimeError("premat CUDA device is not initialized")
        allocated = int(torch.cuda.memory_allocated(self._device))
        reserved = int(torch.cuda.memory_reserved(self._device))
        free, total = torch.cuda.mem_get_info(self._device)
        return PrematMemoryObservation(
            process_allocated_bytes=allocated,
            process_reserved_bytes=reserved,
            process_ordinary_peak_bytes=max(self._ordinary_peak_bytes, allocated - self._retained_bytes),
            device_free_bytes=int(free),
            device_total_bytes=int(total),
        )

    def _update_ordinary_peak(self) -> None:
        observation = self._observe_memory()
        ordinary_allocated = max(0, observation.process_allocated_bytes - self._retained_bytes)
        self._ordinary_peak_bytes = max(self._ordinary_peak_bytes, ordinary_allocated)

    def _refresh_available(self) -> None:
        for candidate in self._candidates.values():
            if candidate.state != CandidateState.MATERIALISING:
                continue
            if candidate.completion_event is not None and candidate.completion_event.query():
                candidate.state = CandidateState.AVAILABLE
                candidate.available_ns = time.perf_counter_ns()
                self._record("available", candidate=candidate)

    def _advance(self) -> None:
        self._refresh_available()
        if any(item.state == CandidateState.MATERIALISING for item in self._candidates.values()):
            return
        ordered = sorted(self._candidates.values(), key=lambda item: item.sequence)
        candidate = next(
            (item for item in ordered if item.state == CandidateState.UNAVAILABLE),
            None,
        )
        if candidate is None:
            return
        observation = self._observe_memory()
        decision = decide_candidate_admission(
            observation=observation,
            envelope=candidate.envelope,
            stay_below_current_peak=self._stay_below_current_peak,
            gpu_memory_buffer_bytes=self._buffer_bytes,
        )
        candidate.admission_reason = decision.reason
        if not decision.admitted:
            self._aggregate["deferred"] += 1
            self._record("admission_deferred", candidate=candidate, detail=asdict(decision))
            return
        if self._stream is None or self._device is None:
            raise RuntimeError("premat stream is not initialized")
        candidate.state = CandidateState.MATERIALISING
        candidate.owner = "premat"
        candidate.launch_ns = time.perf_counter_ns()
        self._aggregate["admitted"] += 1
        self._record("materialising", candidate=candidate, detail=asdict(decision))
        current_stream = torch.cuda.current_stream(device=self._device)
        self._stream.wait_stream(current_stream)
        with torch.cuda.stream(self._stream):
            candidate.tensor = self._materialize(candidate.family, candidate.layer_index)
            candidate.tensor.record_stream(self._stream)
            candidate.completion_event = torch.cuda.Event(enable_timing=False)
            candidate.completion_event.record(self._stream)
        self._retained_bytes += candidate.envelope.retained_bytes

    def _record(
        self,
        event: str,
        *,
        layer: Optional[int] = None,
        candidate: Optional[_Candidate] = None,
        detail: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._logging_enabled and event not in {
            "materialising",
            "available",
            "consuming",
            "consumed",
            "critical_path_wait",
            "critical_path_materialize",
        }:
            return
        self._event_sequence += 1
        payload: Dict[str, object] = {
            "sequence": self._event_sequence,
            "host_time_ns": time.perf_counter_ns(),
            "event": event,
        }
        if layer is not None:
            payload["layer_index"] = int(layer)
        if candidate is not None:
            payload.update(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "state": candidate.state.value,
                    "critical_path_miss": candidate.critical_path_miss,
                }
            )
        if detail:
            payload["detail"] = detail
        self._events.append(payload)
        if len(self._events) > 2048:
            del self._events[:1024]

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("pre-materialisation pass is not active")


def plastic_memory_budget_gib(*, device: torch.device, gpu_memory_buffer_gb: float) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PLASTIC memory_budget objective requires CUDA device capacity")
    _, total = torch.cuda.mem_get_info(device)
    usable_bytes = int(total) - int(float(gpu_memory_buffer_gb) * (1024 ** 3))
    if usable_bytes <= 0:
        raise RuntimeError("premat_gpu_memory_buffer_gb leaves no usable CUDA capacity")
    return usable_bytes / float(1024 ** 3)
# ^^^ THOG
