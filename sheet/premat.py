# vvv THOG dynamic pre-materialisation scheduler, admission policy, lifecycle, and telemetry
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


PREMAT_SWITCHES = ("enabled", "disabled")
PREMAT_ATTENTION_MODES = ("fused", "unfused")
PREMAT_DEFAULT_GPU_MEMORY_BUFFER_GB = 1.0
PREMAT_TELEMETRY_VERSION = 2


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
    predicted_physical_growth_bytes: int
    process_headroom_bytes: int
    device_headroom_bytes: int


def decide_candidate_admission(
    *,
    observation: PrematMemoryObservation,
    envelope: CandidateEnvelope,
    stay_below_current_peak: bool,
    gpu_memory_buffer_bytes: int,
) -> AdmissionDecision:
    predicted_process = observation.process_allocated_bytes + envelope.process_envelope_bytes
    # A CUDA allocator's reserved-but-unused total is not a promise that a
    # suitably sized block is reusable, especially across streams.  Charge the
    # complete envelope to physical device use so fragmentation cannot turn an
    # admitted prefetch into an OOM.
    physical_growth = envelope.process_envelope_bytes
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
        predicted_physical_growth_bytes=physical_growth,
        process_headroom_bytes=max(
            0,
            observation.process_ordinary_peak_bytes
            - observation.process_allocated_bytes,
        ),
        device_headroom_bytes=max(
            0,
            observation.device_free_bytes - gpu_memory_buffer_bytes,
        ),
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
    materialisation_start_event: Optional[torch.cuda.Event] = None
    completion_event: Optional[torch.cuda.Event] = None
    launch_ns: Optional[int] = None
    available_ns: Optional[int] = None
    deadline_ns: Optional[int] = None
    critical_path_miss: bool = False
    admission_reason: str = "not_checked"
    retained_counted: bool = False


@dataclass
class _PendingCudaTiming:
    kind: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    event_payload: Optional[Dict[str, object]]


MaterializeCandidate = Callable[[str, int], Tensor]


class PrematRuntime:
    """One-model, one-stream dynamic pre-materialisation runtime.

    Only the current and immediately following logical layer are represented.
    Candidate launch is strictly ordered, and admission always precedes launch.
    """

    _FUSED_FAMILIES = ("QKV", "O", "UP", "DOWN")
    _UNFUSED_FAMILIES = ("QK", "V", "O", "UP", "DOWN")
    _LEGAL_TRANSITIONS = {
        CandidateState.UNAVAILABLE: (CandidateState.MATERIALISING,),
        CandidateState.MATERIALISING: (
            CandidateState.AVAILABLE,
            CandidateState.CONSUMING,
        ),
        CandidateState.AVAILABLE: (CandidateState.CONSUMING,),
        CandidateState.CONSUMING: (CandidateState.CONSUMED,),
        CandidateState.CONSUMED: (),
    }

    def __init__(
        self,
        *,
        materialize: MaterializeCandidate,
        n_embd: int,
        n_head: int,
        attention_mode: str,
        stay_below_current_peak: bool,
        gpu_memory_buffer_gb: float,
        logging_enabled: bool,
    ) -> None:
        self._materialize = materialize
        self._n_embd = int(n_embd)
        self._n_head = int(n_head)
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
        self._pending_timings: List[_PendingCudaTiming] = []
        self._display_layer_pair: Optional[Tuple[int, Optional[int]]] = None
        self._display_candidates: List[Dict[str, object]] = []
        self._ordinary_peak_bytes = 0
        self._retained_bytes = 0
        self._activation_bytes = 0
        self._dtype_bytes = 0
        self._batch_size = 0
        self._sequence_length = 0
        self._pass_start_ns: Optional[int] = None
        self._active = False
        self._aggregate = {
            "admitted": 0,
            "deferred": 0,
            "available_hits": 0,
            "fully_hidden_hits": 0,
            "waited_hits": 0,
            "main_stream_misses": 0,
            "ordinary_deadline_materialisations": 0,
            "premat_materialisation_ms_total": 0.0,
            "main_stream_wait_ms_total": 0.0,
            "main_stream_materialisation_ms_total": 0.0,
            "captured_pass_host_ms_total": 0.0,
            "observed_peak_allocated_bytes": 0,
            "observed_peak_reserved_bytes": 0,
            "minimum_headroom_bytes": None,
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
        if self._device is not None and self._device != reference.device:
            raise RuntimeError(
                "a PrematRuntime cannot move its persistent CUDA stream between devices"
            )
        self._device = reference.device
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=reference.device)
        self._layer_indices = resolved
        self._position = -1
        self._candidates.clear()
        self._display_layer_pair = None
        self._display_candidates = []
        self._retained_bytes = 0
        self._activation_bytes = int(reference.numel() * reference.element_size())
        self._dtype_bytes = max(4, int(reference.element_size()))
        self._batch_size = int(reference.shape[0]) if reference.ndim >= 1 else 1
        self._sequence_length = int(reference.shape[1]) if reference.ndim >= 2 else 1
        self._pass_start_ns = time.perf_counter_ns()
        self._active = True
        observation = self._observe_memory()
        self._ordinary_peak_bytes = max(
            self._ordinary_peak_bytes,
            observation.process_allocated_bytes,
        )
        self._update_memory_aggregates(observation)
        self._record("pass_begin", layer=resolved[0], detail={"layer_count": len(resolved)})

    def end(self) -> None:
        if not self._active:
            return
        self._resolve_pending_timings()
        for candidate in tuple(self._candidates.values()):
            if candidate.state not in (CandidateState.CONSUMED, CandidateState.UNAVAILABLE):
                self._record(
                    "pass_end_release",
                    candidate=candidate,
                    outcome="released_at_pass_end",
                    detail={"prior_state": candidate.state.value},
                )
        self._record("pass_end")
        self._capture_display_snapshot()
        if self._pass_start_ns is not None:
            self._aggregate["captured_pass_host_ms_total"] += max(
                0.0,
                (time.perf_counter_ns() - self._pass_start_ns) / 1_000_000.0,
            )
        for candidate in tuple(self._candidates.values()):
            candidate.tensor = None
            candidate.materialisation_start_event = None
            candidate.completion_event = None
        self._retained_bytes = 0
        self._active = False

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
        self._update_ordinary_peak()
        self._record("layer_start", layer=layer_index, outcome="reconsider")
        self._advance()

    def event(self, name: str, *, layer_index: int) -> None:
        self._require_active()
        self._refresh_available()
        self._update_ordinary_peak()
        self._record(name, layer=layer_index, outcome="reconsider")
        self._advance()

    def acquire(self, family: str, layer_index: int) -> Tensor:
        self._require_active()
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None:
            raise RuntimeError(f"premat candidate is outside the two-layer window: {key}")
        # vvv THOG a deadline observes completion but cannot launch an UNAVAILABLE head on the premat stream
        self._refresh_available()
        self._update_ordinary_peak()
        self._record(
            f"deadline_{family.lower()}",
            candidate=candidate,
            outcome="deadline",
        )
        # ^^^ THOG
        candidate.deadline_ns = time.perf_counter_ns()
        current_stream = torch.cuda.current_stream(device=self._device)
        if (
            candidate.state
            in (CandidateState.AVAILABLE, CandidateState.MATERIALISING)
            and candidate.tensor is None
        ):
            raise RuntimeError(f"premat candidate {key} has no owned tensor")
        if candidate.state == CandidateState.AVAILABLE:
            self._aggregate["available_hits"] += 1
            self._aggregate["fully_hidden_hits"] += 1
            candidate.tensor.record_stream(current_stream)
            self._transition(
                candidate,
                CandidateState.CONSUMING,
                "consuming",
                outcome="fully_hidden",
            )
        elif candidate.state == CandidateState.MATERIALISING:
            if candidate.completion_event is None:
                raise RuntimeError(f"materialising candidate {key} has no completion event")
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_end = torch.cuda.Event(enable_timing=True)
            wait_start.record(current_stream)
            current_stream.wait_event(candidate.completion_event)
            wait_end.record(current_stream)
            candidate.critical_path_miss = True
            self._aggregate["waited_hits"] += 1
            candidate.tensor.record_stream(current_stream)
            wait_payload = self._transition(
                candidate,
                CandidateState.CONSUMING,
                "critical_path_wait",
                outcome="waited_for_premat",
                reason="materialising_at_deadline",
            )
            self._pending_timings.append(
                _PendingCudaTiming(
                    kind="main_stream_wait",
                    start_event=wait_start,
                    end_event=wait_end,
                    event_payload=wait_payload,
                )
            )
        elif candidate.state == CandidateState.UNAVAILABLE:
            candidate.owner = "main"
            candidate.critical_path_miss = True
            candidate.launch_ns = time.perf_counter_ns()
            fallback_payload = self._transition(
                candidate,
                CandidateState.MATERIALISING,
                "materialising_on_critical_path",
                decision="main_claim",
                outcome="ordinary_deadline_materialisation",
                reason="unavailable_at_deadline",
            )
            materialise_start = torch.cuda.Event(enable_timing=True)
            materialise_end = torch.cuda.Event(enable_timing=True)
            materialise_start.record(current_stream)
            try:
                candidate.tensor = self._materialize(candidate.family, candidate.layer_index)
            except BaseException as error:
                self._record(
                    "main_materialisation_failed",
                    candidate=candidate,
                    outcome="failure",
                    reason=type(error).__name__,
                    detail={"error": str(error)},
                )
                raise RuntimeError(
                    "premat main-stream fallback materialisation failed; "
                    f"layer={candidate.layer_index}, family={candidate.family}, "
                    f"mode={self._attention_mode}, memory={self._memory_summary()}"
                ) from error
            materialise_end.record(current_stream)
            self._pending_timings.append(
                _PendingCudaTiming(
                    kind="main_stream_materialisation",
                    start_event=materialise_start,
                    end_event=materialise_end,
                    event_payload=fallback_payload,
                )
            )
            candidate.available_ns = time.perf_counter_ns()
            self._transition(
                candidate,
                CandidateState.AVAILABLE,
                "available_on_critical_path",
                outcome="ordinary_deadline_materialisation",
            )
            self._aggregate["main_stream_misses"] += 1
            self._aggregate["ordinary_deadline_materialisations"] += 1
            candidate.tensor.record_stream(current_stream)
            self._transition(
                candidate,
                CandidateState.CONSUMING,
                "consuming",
                outcome="main_stream_fallback",
            )
        else:
            raise RuntimeError(f"premat candidate {key} cannot be acquired from {candidate.state.value}")
        if candidate.tensor is None:
            raise RuntimeError(f"premat candidate {key} has no tensor at its deadline")
        return candidate.tensor

    def consumed(self, family: str, layer_index: int) -> None:
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None or candidate.state != CandidateState.CONSUMING:
            raise RuntimeError(f"premat candidate {key} was not consuming")
        if candidate.retained_counted:
            self._retained_bytes = max(
                0,
                self._retained_bytes - candidate.envelope.retained_bytes,
            )
            candidate.retained_counted = False
        candidate.tensor = None
        candidate.materialisation_start_event = None
        candidate.completion_event = None
        self._update_ordinary_peak()
        self._transition(
            candidate,
            CandidateState.CONSUMED,
            "consumed",
            outcome="consumption_complete",
        )
        self._advance()

    def report(self) -> Dict[str, object]:
        self._resolve_pending_timings()
        candidates = [
            self._candidate_payload(candidate)
            for candidate in sorted(
                self._candidates.values(),
                key=lambda item: item.sequence,
            )
        ]
        current_layer = (
            self._layer_indices[self._position]
            if self._position >= 0 and self._position < len(self._layer_indices)
            else None
        )
        next_layer = (
            self._layer_indices[self._position + 1]
            if self._position + 1 < len(self._layer_indices)
            else None
        )
        if not self._active and self._display_layer_pair is not None:
            current_layer, next_layer = self._display_layer_pair
            candidates = list(self._display_candidates)
        memory: Optional[Dict[str, object]] = None
        if self._device is not None and torch.cuda.is_available():
            memory = self._memory_summary()
        aggregate = dict(self._aggregate)
        aggregate["premat_hidden_ms_estimate"] = max(
            0.0,
            float(aggregate["premat_materialisation_ms_total"])
            - float(aggregate["main_stream_wait_ms_total"]),
        )
        pass_ms = float(aggregate["captured_pass_host_ms_total"])
        aggregate["premat_stream_busy_fraction_estimate"] = (
            min(1.0, float(aggregate["premat_materialisation_ms_total"]) / pass_ms)
            if pass_ms > 0.0
            else None
        )
        return {
            "version": PREMAT_TELEMETRY_VERSION,
            "schema_version": PREMAT_TELEMETRY_VERSION,
            "enabled": True,
            "attention_mode": self._attention_mode,
            "headroom_mode": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
            "current_layer_index": current_layer,
            "next_layer_index": next_layer,
            "lookahead_layer_limit": 1,
            "effective_fast_discard": True,
            "queue_head": self._queue_head_payload(),
            "candidates": candidates,
            "events": list(self._events[-256:]),
            "memory": memory,
            "aggregate": aggregate,
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
        # vvv THOG conservatively cover the largest known foreground interval through
        # the deadline, including explicit unfused [B,H,T,T] score/probability tensors.
        score_bytes = (
            self._batch_size
            * self._n_head
            * self._sequence_length
            * self._sequence_length
            * self._dtype_bytes
        )
        causal_mask_bytes = self._sequence_length * self._sequence_length
        attention_peak = 4 * self._activation_bytes + 2 * score_bytes
        if self._attention_mode == "unfused":
            attention_peak += causal_mask_bytes
        foreground_overlap = max(8 * self._activation_bytes, attention_peak)
        # ^^^ THOG
        return CandidateEnvelope(
            retained_bytes=retained,
            materialisation_peak_bytes=materialisation_peak,
            foreground_overlap_bytes=foreground_overlap,
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
        ordinary_allocated = max(
            0,
            observation.process_allocated_bytes - self._retained_bytes,
        )
        self._ordinary_peak_bytes = max(self._ordinary_peak_bytes, ordinary_allocated)
        self._update_memory_aggregates(observation)

    def _refresh_available(self) -> None:
        self._resolve_pending_timings()
        for candidate in self._candidates.values():
            if (
                candidate.state != CandidateState.MATERIALISING
                or candidate.owner != "premat"
            ):
                continue
            if (
                candidate.completion_event is not None
                and candidate.completion_event.query()
            ):
                candidate.available_ns = time.perf_counter_ns()
                self._transition(
                    candidate,
                    CandidateState.AVAILABLE,
                    "available",
                    outcome="premat_complete",
                )

    def _advance(self) -> None:
        self._refresh_available()
        if any(
            item.state == CandidateState.MATERIALISING
            for item in self._candidates.values()
        ):
            return
        ordered = sorted(self._candidates.values(), key=lambda item: item.sequence)
        candidate = next(
            (
                item
                for item in ordered
                if item.state == CandidateState.UNAVAILABLE
            ),
            None,
        )
        if candidate is None:
            return
        observation = self._observe_memory()
        self._update_memory_aggregates(observation)
        decision = decide_candidate_admission(
            observation=observation,
            envelope=candidate.envelope,
            stay_below_current_peak=self._stay_below_current_peak,
            gpu_memory_buffer_bytes=self._buffer_bytes,
        )
        candidate.admission_reason = decision.reason
        if not decision.admitted:
            self._aggregate["deferred"] += 1
            self._record(
                "admission_deferred",
                candidate=candidate,
                decision="defer",
                outcome="not_launched",
                reason=decision.reason,
                detail=asdict(decision),
            )
            return
        if self._stream is None or self._device is None:
            raise RuntimeError("premat stream is not initialized")
        candidate.owner = "premat"
        candidate.launch_ns = time.perf_counter_ns()
        self._aggregate["admitted"] += 1
        launch_payload = self._transition(
            candidate,
            CandidateState.MATERIALISING,
            "materialising",
            decision="admit",
            outcome="premat_launched",
            reason=decision.reason,
            detail=asdict(decision),
        )
        current_stream = torch.cuda.current_stream(device=self._device)
        self._stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(self._stream):
                candidate.materialisation_start_event = torch.cuda.Event(
                    enable_timing=True
                )
                candidate.completion_event = torch.cuda.Event(enable_timing=True)
                candidate.materialisation_start_event.record(self._stream)
                candidate.tensor = self._materialize(
                    candidate.family,
                    candidate.layer_index,
                )
                actual_retained_bytes = int(
                    candidate.tensor.numel() * candidate.tensor.element_size()
                )
                if actual_retained_bytes > candidate.envelope.retained_bytes:
                    raise RuntimeError(
                        "materialised tensor exceeds its pre-admission retained estimate: "
                        f"actual={actual_retained_bytes}, "
                        f"predicted={candidate.envelope.retained_bytes}"
                    )
                candidate.tensor.record_stream(self._stream)
                candidate.completion_event.record(self._stream)
        except BaseException as error:
            candidate.tensor = None
            candidate.materialisation_start_event = None
            candidate.completion_event = None
            self._record(
                "materialisation_failed",
                candidate=candidate,
                outcome="failure",
                reason=type(error).__name__,
                detail={"error": str(error), "admission": asdict(decision)},
            )
            raise RuntimeError(
                "premat candidate materialisation failed after admission; "
                f"layer={candidate.layer_index}, family={candidate.family}, "
                f"mode={self._attention_mode}, envelope={asdict(candidate.envelope)}, "
                f"admission={asdict(decision)}, memory={self._memory_summary()}"
            ) from error
        self._pending_timings.append(
            _PendingCudaTiming(
                kind="premat_materialisation",
                start_event=candidate.materialisation_start_event,
                end_event=candidate.completion_event,
                event_payload=launch_payload,
            )
        )
        self._retained_bytes += candidate.envelope.retained_bytes
        candidate.retained_counted = True

    def _transition(
        self,
        candidate: _Candidate,
        new_state: CandidateState,
        event: str,
        *,
        decision: Optional[str] = None,
        outcome: Optional[str] = None,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        old_state = candidate.state
        if new_state not in self._LEGAL_TRANSITIONS[old_state]:
            raise RuntimeError(
                "illegal premat candidate transition: "
                f"layer={candidate.layer_index}, family={candidate.family}, "
                f"{old_state.value}->{new_state.value}"
            )
        candidate.state = new_state
        return self._record(
            event,
            candidate=candidate,
            previous_state=old_state,
            decision=decision,
            outcome=outcome,
            reason=reason,
            detail=detail,
        )

    def _record(
        self,
        event: str,
        *,
        layer: Optional[int] = None,
        candidate: Optional[_Candidate] = None,
        previous_state: Optional[CandidateState] = None,
        decision: Optional[str] = None,
        outcome: Optional[str] = None,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        self._capture_display_snapshot()
        if not self._logging_enabled:
            return None
        self._event_sequence += 1
        now_ns = time.perf_counter_ns()
        payload: Dict[str, object] = {
            "schema_version": PREMAT_TELEMETRY_VERSION,
            "sequence": self._event_sequence,
            "host_time_ns": now_ns,
            "elapsed_ms": (
                max(0.0, (now_ns - self._pass_start_ns) / 1_000_000.0)
                if self._pass_start_ns is not None
                else 0.0
            ),
            "event": event,
            "event_type": event,
            "headroom_policy": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
        }
        if layer is not None:
            payload["layer_index"] = int(layer)
        if candidate is not None:
            payload.update(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "old_state": (
                        previous_state.value
                        if previous_state is not None
                        else candidate.state.value
                    ),
                    "new_state": candidate.state.value,
                    "state": candidate.state.value,
                    "owner": candidate.owner,
                    "critical_path_miss": candidate.critical_path_miss,
                    "predicted_retained_bytes": candidate.envelope.retained_bytes,
                    "predicted_materialisation_peak_bytes": (
                        candidate.envelope.materialisation_peak_bytes
                    ),
                    "predicted_foreground_overlap_bytes": (
                        candidate.envelope.foreground_overlap_bytes
                    ),
                    "predicted_envelope_bytes": (
                        candidate.envelope.process_envelope_bytes
                    ),
                }
            )
        if self._device is not None and torch.cuda.is_available():
            memory = self._memory_summary()
            for name in (
                "process_allocated_bytes",
                "process_reserved_bytes",
                "process_ordinary_peak_bytes",
                "device_free_bytes",
                "device_used_bytes",
                "device_total_bytes",
                "global_buffer_bytes",
                "device_ceiling_bytes",
                "process_headroom_bytes",
                "device_headroom_bytes",
                "premat_headroom_bytes",
            ):
                payload[name] = memory[name]
        if decision is not None:
            payload["decision"] = decision
        if outcome is not None:
            payload["outcome"] = outcome
        if reason is not None:
            payload["reason"] = reason
        if detail:
            payload["detail"] = detail
        self._events.append(payload)
        if len(self._events) > 2048:
            del self._events[:1024]
        return payload

    def _candidate_payload(self, candidate: _Candidate) -> Dict[str, object]:
        return {
            "sequence": candidate.sequence,
            "layer_index": candidate.layer_index,
            "family": candidate.family,
            "state": candidate.state.value,
            "owner": candidate.owner,
            "critical_path_miss": candidate.critical_path_miss,
            "admission_reason": candidate.admission_reason,
            "envelope": asdict(candidate.envelope),
        }

    def _queue_head_payload(self) -> Optional[Dict[str, object]]:
        head = next(
            (
                candidate
                for candidate in sorted(
                    self._candidates.values(),
                    key=lambda item: item.sequence,
                )
                if candidate.state
                in (CandidateState.UNAVAILABLE, CandidateState.MATERIALISING)
            ),
            None,
        )
        if head is None:
            return None
        payload = self._candidate_payload(head)
        payload["deferred"] = head.admission_reason not in (
            "not_checked",
            "admitted",
        )
        return payload

    def _capture_display_snapshot(self) -> None:
        if not self._active or not (0 <= self._position < len(self._layer_indices)):
            return
        current_layer = self._layer_indices[self._position]
        next_layer = (
            self._layer_indices[self._position + 1]
            if self._position + 1 < len(self._layer_indices)
            else None
        )
        self._display_layer_pair = (current_layer, next_layer)
        displayed_layers = {current_layer}
        if next_layer is not None:
            displayed_layers.add(next_layer)
        self._display_candidates = [
            self._candidate_payload(candidate)
            for candidate in sorted(
                self._candidates.values(),
                key=lambda item: item.sequence,
            )
            if candidate.layer_index in displayed_layers
        ]

    def _memory_summary(
        self,
        observation: Optional[PrematMemoryObservation] = None,
    ) -> Dict[str, object]:
        resolved = self._observe_memory() if observation is None else observation
        process_headroom = max(
            0,
            resolved.process_ordinary_peak_bytes
            - resolved.process_allocated_bytes,
        )
        device_headroom = max(
            0,
            resolved.device_free_bytes - self._buffer_bytes,
        )
        premat_headroom = (
            min(process_headroom, device_headroom)
            if self._stay_below_current_peak
            else device_headroom
        )
        self._update_memory_aggregates(resolved, headroom_bytes=premat_headroom)
        queue_head = self._queue_head_payload()
        return {
            **asdict(resolved),
            "device_used_bytes": resolved.device_used_bytes,
            "reusable_allocator_bytes": resolved.reusable_allocator_bytes,
            "global_buffer_bytes": self._buffer_bytes,
            "device_ceiling_bytes": max(
                0,
                resolved.device_total_bytes - self._buffer_bytes,
            ),
            "process_ceiling_bytes": (
                resolved.process_ordinary_peak_bytes
                if self._stay_below_current_peak
                else None
            ),
            "active_ceiling_bytes": (
                resolved.process_ordinary_peak_bytes
                if self._stay_below_current_peak
                else max(0, resolved.device_total_bytes - self._buffer_bytes)
            ),
            "active_ceiling_scope": (
                "process_ordinary_peak"
                if self._stay_below_current_peak
                else "device_used"
            ),
            "process_headroom_bytes": process_headroom,
            "device_headroom_bytes": device_headroom,
            "premat_headroom_bytes": premat_headroom,
            "premat_retained_bytes": self._retained_bytes,
            "next_candidate": queue_head,
            "next_candidate_defer_reason": (
                queue_head.get("admission_reason")
                if queue_head is not None and queue_head.get("deferred")
                else None
            ),
        }

    def _update_memory_aggregates(
        self,
        observation: PrematMemoryObservation,
        *,
        headroom_bytes: Optional[int] = None,
    ) -> None:
        self._aggregate["observed_peak_allocated_bytes"] = max(
            int(self._aggregate["observed_peak_allocated_bytes"]),
            observation.process_allocated_bytes,
        )
        self._aggregate["observed_peak_reserved_bytes"] = max(
            int(self._aggregate["observed_peak_reserved_bytes"]),
            observation.process_reserved_bytes,
        )
        if headroom_bytes is None:
            process_headroom = max(
                0,
                observation.process_ordinary_peak_bytes
                - observation.process_allocated_bytes,
            )
            device_headroom = max(
                0,
                observation.device_free_bytes - self._buffer_bytes,
            )
            headroom_bytes = (
                min(process_headroom, device_headroom)
                if self._stay_below_current_peak
                else device_headroom
            )
        prior = self._aggregate["minimum_headroom_bytes"]
        self._aggregate["minimum_headroom_bytes"] = (
            int(headroom_bytes)
            if prior is None
            else min(int(prior), int(headroom_bytes))
        )

    def _resolve_pending_timings(self) -> None:
        remaining: List[_PendingCudaTiming] = []
        for timing in self._pending_timings:
            try:
                if not timing.end_event.query():
                    remaining.append(timing)
                    continue
                elapsed_ms = max(
                    0.0,
                    float(timing.start_event.elapsed_time(timing.end_event)),
                )
            except RuntimeError:
                remaining.append(timing)
                continue
            if timing.kind == "premat_materialisation":
                aggregate_name = "premat_materialisation_ms_total"
                payload_name = "materialisation_ms"
            elif timing.kind == "main_stream_wait":
                aggregate_name = "main_stream_wait_ms_total"
                payload_name = "wait_ms"
            elif timing.kind == "main_stream_materialisation":
                aggregate_name = "main_stream_materialisation_ms_total"
                payload_name = "main_stream_materialisation_ms"
            else:
                raise RuntimeError(f"unknown premat CUDA timing kind: {timing.kind}")
            self._aggregate[aggregate_name] += elapsed_ms
            if timing.event_payload is not None:
                timing.event_payload[payload_name] = elapsed_ms
        self._pending_timings = remaining

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("pre-materialisation pass is not active")


def plastic_memory_budget_gib(*, device: torch.device, gpu_memory_buffer_gb: float) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PLASTIC memory_budget objective requires CUDA device capacity")
    free, _ = torch.cuda.mem_get_info(device)
    reserved = int(torch.cuda.memory_reserved(device))
    usable_bytes = (
        reserved
        + int(free)
        - int(float(gpu_memory_buffer_gb) * (1024 ** 3))
    )
    if usable_bytes <= 0:
        raise RuntimeError("premat_gpu_memory_buffer_gb leaves no usable CUDA capacity")
    return usable_bytes / float(1024 ** 3)
# ^^^ THOG
