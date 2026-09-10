# vvv THOG dynamic pre-materialisation scheduler, admission policy, lifecycle, and telemetry
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


PREMAT_SWITCHES = ("enabled", "disabled")
PREMAT_ATTENTION_MODES = ("fused", "unfused")
PREMAT_TARGET_LAYERS = (0, 1, 2)
PREMAT_WEIGHT_MATRIX_TARGET_ORDERS = ("l_to_r", "r_to_l")
PREMAT_CUDA_STREAM_PRIORITIES = ("normal", "high")
PREMAT_ALLOCATOR_AWARE_ADMISSION_MODES = (
    "disabled",
    "cautious",
    "normal",
    "aggressive",
)
PREMAT_HIGHEST_CUDA_STREAM_PRIORITY_REQUEST = -(2 ** 31)
PREMAT_DEFAULT_GPU_MEMORY_BUFFER_GB = 1.0
PREMAT_TELEMETRY_VERSION = 3


def validate_premat_configuration(
    *,
    premat: str,
    attention_mode: str,
    stay_below_current_peak: bool,
    stay_within_global_buffer: bool,
    gpu_memory_buffer_gb: float,
    allocator_aware_admission: str = "disabled",
    target_layer: int,
    weight_matrix_target_order: str,
    cuda_stream_priority: str,
    diagnostic_layer_delay_ms: float,
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
    if allocator_aware_admission not in PREMAT_ALLOCATOR_AWARE_ADMISSION_MODES:
        raise ValueError(
            "premat_allocator_aware_admission must be one of "
            f"{PREMAT_ALLOCATOR_AWARE_ADMISSION_MODES}; "
            f"got {allocator_aware_admission!r}"
        )
    if allocator_aware_admission in ("normal", "aggressive"):
        raise ValueError(
            "premat_allocator_aware_admission mode "
            f"{allocator_aware_admission!r} is not implemented yet"
        )
    if isinstance(target_layer, bool) or target_layer not in PREMAT_TARGET_LAYERS:
        raise ValueError(
            f"premat_target_layer must be one of {PREMAT_TARGET_LAYERS}; "
            f"got {target_layer!r}"
        )
    if weight_matrix_target_order not in PREMAT_WEIGHT_MATRIX_TARGET_ORDERS:
        raise ValueError(
            "premat_weight_matrix_target_order must be one of "
            f"{PREMAT_WEIGHT_MATRIX_TARGET_ORDERS}; got {weight_matrix_target_order!r}"
        )
    if cuda_stream_priority not in PREMAT_CUDA_STREAM_PRIORITIES:
        raise ValueError(
            "premat_cuda_stream_priority must be one of "
            f"{PREMAT_CUDA_STREAM_PRIORITIES}; got {cuda_stream_priority!r}"
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
    if (
        isinstance(diagnostic_layer_delay_ms, bool)
        or not isinstance(diagnostic_layer_delay_ms, (int, float))
        or not math.isfinite(float(diagnostic_layer_delay_ms))
        or float(diagnostic_layer_delay_ms) < 0.0
    ):
        raise ValueError(
            "premat_diagnostic_layer_delay_ms must be finite and non-negative"
        )


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

    @property
    def candidate_transient_bytes(self) -> int:
        """Candidate-owned workspace above the retained output size.

        ``foreground_overlap_bytes`` is a shared Main Stream safety envelope,
        not storage owned by each queued candidate.  It belongs in every
        admission decision, but must not be multiplied into the cumulative
        charge once per candidate.
        """
        return max(0, self.materialisation_peak_bytes - self.retained_bytes)


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


def _decide_candidate_admission_with_physical_growth(
    *,
    observation: PrematMemoryObservation,
    envelope: CandidateEnvelope,
    stay_below_current_peak: bool,
    gpu_memory_buffer_bytes: int,
    physical_growth_bytes: int,
) -> AdmissionDecision:
    predicted_process = observation.process_allocated_bytes + envelope.process_envelope_bytes
    physical_growth = max(0, int(physical_growth_bytes))
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


def decide_candidate_admission(
    *,
    observation: PrematMemoryObservation,
    envelope: CandidateEnvelope,
    stay_below_current_peak: bool,
    gpu_memory_buffer_bytes: int,
) -> AdmissionDecision:
    # A CUDA allocator's reserved-but-unused total is not a promise that a
    # suitably sized block is reusable, especially across streams.  The
    # ordinary path therefore charges the complete envelope to physical device
    # use.  Optional allocator-aware admission may later rescue only a rejected
    # device-headroom decision using stronger evidence about one reusable block.
    return _decide_candidate_admission_with_physical_growth(
        observation=observation,
        envelope=envelope,
        stay_below_current_peak=stay_below_current_peak,
        gpu_memory_buffer_bytes=gpu_memory_buffer_bytes,
        physical_growth_bytes=envelope.process_envelope_bytes,
    )


def _allocator_pool_for_request(size_bytes: int) -> str:
    # PyTorch's native CUDA allocator defines the largest small allocation as
    # 1 MiB (kSmallSize).  Derive the pool from the actual candidate size so
    # model width/dtype changes cannot invalidate cautious admission.
    return "small" if int(size_bytes) <= 1024 ** 2 else "large"


def _largest_same_stream_inactive_block_bytes(
    snapshot: Sequence[Mapping[str, object]],
    *,
    stream_id: int,
    request_bytes: int,
) -> int:
    """Return the largest compatible inactive block for one CUDA stream.

    Cautious admission deliberately does not sum fragmented blocks.  A block
    must be in the allocator pool used by the request and individually cover
    the complete premat materialisation peak before we claim that
    candidate-owned allocation need not grow physical device usage.
    """
    largest = 0
    required_pool = _allocator_pool_for_request(request_bytes)
    for segment in snapshot:
        if not isinstance(segment, Mapping):
            raise ValueError("CUDA allocator snapshot contains a non-mapping segment")
        if int(segment.get("stream", -1)) != int(stream_id):
            continue
        if segment.get("segment_type") != required_pool:
            continue
        blocks = segment.get("blocks")
        if not isinstance(blocks, Sequence):
            raise ValueError("CUDA allocator snapshot segment has no block sequence")
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("CUDA allocator snapshot contains a non-mapping block")
            if block.get("state") != "inactive":
                continue
            size = block.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("CUDA allocator snapshot contains an invalid block size")
            largest = max(largest, int(size))
    return largest


@dataclass
class _Candidate:
    layer_index: int
    family: str
    sequence: int
    order_position: int
    envelope: CandidateEnvelope
    state: CandidateState = CandidateState.UNAVAILABLE
    owner: str = "none"
    tensor: Optional[Tensor] = None
    materialisation_start_event: Optional[torch.cuda.Event] = None
    completion_event: Optional[torch.cuda.Event] = None
    launch_ns: Optional[int] = None
    available_ns: Optional[int] = None
    deadline_ns: Optional[int] = None
    consumed_ns: Optional[int] = None
    first_considered_ns: Optional[int] = None
    first_observed_admissible_ns: Optional[int] = None
    final_outcome: Optional[str] = None
    critical_path_miss: bool = False
    admission_reason: str = "not_checked"
    retained_counted: bool = False
    transient_counted: bool = False


@dataclass
class _PendingCudaTiming:
    kind: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    event_payload: Optional[Dict[str, object]]


@dataclass
class _PendingRelease:
    end_event: torch.cuda.Event
    retained_bytes: int
    transient_bytes: int
    tensor: Tensor


MaterializeCandidate = Callable[[str, int], Tensor]
AttachCandidate = Callable[[str, int, Tensor], Tensor]
PrematLiveReporter = Callable[[Mapping[str, object]], None]
PrematLiveCapturePredicate = Callable[[int], bool]


class PrematRuntime:
    """One-model Main Stream/Premat Stream materialisation runtime.

    The current layer and configured exact target are represented. Candidate
    submission is strictly ordered, cumulatively charged, and admission-gated.
    """

    _FUSED_FAMILIES = {
        "l_to_r": ("QKV", "O", "UP", "DOWN"),
        "r_to_l": ("DOWN", "UP", "O", "QKV"),
    }
    _UNFUSED_FAMILIES = {
        "l_to_r": ("QK", "V", "O", "UP", "DOWN"),
        "r_to_l": ("DOWN", "UP", "O", "V", "QK"),
    }
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
        attach: Optional[AttachCandidate] = None,
        n_embd: int,
        n_head: int,
        attention_mode: str,
        stay_below_current_peak: bool,
        gpu_memory_buffer_gb: float,
        allocator_aware_admission: str = "disabled",
        target_layer: int,
        weight_matrix_target_order: str,
        cuda_stream_priority: str,
        diagnostic_layer_delay_ms: float,
        logging_enabled: bool,
    ) -> None:
        self._materialize = materialize
        self._attach = attach or (lambda _family, _layer_index, tensor: tensor)
        self._n_embd = int(n_embd)
        self._n_head = int(n_head)
        self._attention_mode = attention_mode
        self._stay_below_current_peak = bool(stay_below_current_peak)
        self._buffer_bytes = int(float(gpu_memory_buffer_gb) * (1024 ** 3))
        self._allocator_aware_admission = allocator_aware_admission
        self._target_layer = int(target_layer)
        self._weight_matrix_target_order = weight_matrix_target_order
        self._cuda_stream_priority = cuda_stream_priority
        self._diagnostic_layer_delay_ms = float(diagnostic_layer_delay_ms)
        self._logging_enabled = bool(logging_enabled)
        self._stream: Optional[torch.cuda.Stream] = None
        self._device: Optional[torch.device] = None
        self._layer_indices: Tuple[int, ...] = ()
        self._position = -1
        self._candidates: Dict[Tuple[int, str], _Candidate] = {}
        self._sequence = 0
        self._event_sequence = 0
        self._advance_sequence = 0
        self._pass_sequence = 0
        self._events: List[Dict[str, object]] = []
        # vvv THOG Instra receives bounded transition snapshots on the training
        # thread; no polling thread or CUDA synchronization enters the scheduler.
        self._live_reporter: Optional[PrematLiveReporter] = None
        self._live_capture_enabled: Optional[PrematLiveCapturePredicate] = None
        self._live_publish_error: Optional[str] = None
        # ^^^ THOG
        self._pending_timings: List[_PendingCudaTiming] = []
        self._pending_releases: List[_PendingRelease] = []
        self._display_layer_pair: Optional[Tuple[int, Optional[int]]] = None
        self._display_candidates: List[Dict[str, object]] = []
        self._ordinary_peak_bytes = 0
        self._retained_bytes = 0
        self._transient_bytes = 0
        self._activation_bytes = 0
        self._dtype_bytes = 0
        self._batch_size = 0
        self._sequence_length = 0
        self._pass_start_ns: Optional[int] = None
        self._active = False
        self._aggregate = {
            "admitted": 0,
            "deferred": 0,
            "admission_rejections_by_reason": {},
            "queue_depth_high_water": 0,
            "cumulative_charged_bytes_high_water": 0,
            "observed_admission_lag_ms_total": 0.0,
            "observed_admission_lag_ms_max": 0.0,
            "available_hits": 0,
            "fully_hidden_hits": 0,
            "waited_hits": 0,
            "main_stream_misses": 0,
            "ordinary_deadline_materialisations": 0,
            "premat_materialisation_ms_total": 0.0,
            "main_stream_wait_ms_total": 0.0,
            "main_stream_materialisation_ms_total": 0.0,
            "captured_pass_host_ms_total": 0.0,
            "diagnostic_layer_delay_count": 0,
            "diagnostic_layer_delay_ms_requested_total": 0.0,
            "diagnostic_layer_delay_ms_actual_total": 0.0,
            "observed_peak_allocated_bytes": 0,
            "observed_peak_reserved_bytes": 0,
            "minimum_headroom_bytes": None,
        }

    @property
    def active(self) -> bool:
        return self._active

    # vvv THOG the reporter is observational and deliberately absent unless
    # --premat_instra is enabled on the primary process.
    def set_live_reporter(
        self,
        reporter: Optional[PrematLiveReporter],
        capture_enabled: Optional[PrematLiveCapturePredicate] = None,
    ) -> None:
        self._live_reporter = reporter
        self._live_capture_enabled = capture_enabled
        self._live_publish_error = None
    # ^^^ THOG

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
            priority = (
                PREMAT_HIGHEST_CUDA_STREAM_PRIORITY_REQUEST
                if self._cuda_stream_priority == "high"
                else 0
            )
            self._stream = torch.cuda.Stream(
                device=reference.device,
                priority=priority,
            )
        self._layer_indices = resolved
        self._position = -1
        # One forward pass is one accumulation microstep's complete Premat
        # timeline.  Never carry event fragments across microsteps.
        self._events.clear()
        self._pass_sequence += 1
        self._candidates.clear()
        self._pending_releases.clear()
        self._display_layer_pair = None
        self._display_candidates = []
        self._retained_bytes = 0
        self._transient_bytes = 0
        # Price the tensors that Premat will actually allocate.  The pass
        # reference can remain FP32 under CUDA autocast even though matrix
        # materialisation and its overlapping Main Stream activations are
        # BF16/FP16.  Charging the reference dtype therefore doubles the
        # admission envelope for mixed-precision training.
        if torch.is_autocast_enabled("cuda"):
            autocast_dtype = torch.get_autocast_dtype("cuda")
            if autocast_dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "Premat CUDA autocast requires float16 or bfloat16; "
                    f"got {autocast_dtype}"
                )
            self._dtype_bytes = 2
        else:
            self._dtype_bytes = int(reference.element_size())
        self._activation_bytes = int(reference.numel() * self._dtype_bytes)
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
        self._record(
            "pass_begin",
            layer=resolved[0],
            detail={
                "layer_count": len(resolved),
                "materialisation_element_bytes": self._dtype_bytes,
                "foreground_activation_bytes": self._activation_bytes,
            },
        )

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
        self._transient_bytes = 0
        self._pending_releases.clear()
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
        window_width = max(2, self._target_layer + 1)
        permitted = set(self._layer_indices[position : position + window_width])
        stale = [key for key in self._candidates if key[0] not in permitted]
        for key in stale:
            candidate = self._candidates.pop(key)
            if candidate.state not in (CandidateState.CONSUMED, CandidateState.UNAVAILABLE):
                raise RuntimeError(
                    f"premat candidate left layer window before consumption: {key}={candidate.state.value}"
                )
        for permitted_layer in self._layer_indices[position : position + window_width]:
            self._ensure_layer(permitted_layer)
        self._update_ordinary_peak()
        self._record("layer_start", layer=layer_index, outcome="reconsider")
        self._advance(trigger="layer_start")

    def layer_complete(self, layer_index: int) -> None:
        """Apply the optional diagnostic host-dispatch delay after a layer.

        This deliberately does not synchronize the main CUDA stream: all work
        for the layer has been submitted, and pausing the host prevents the next
        layer from being submitted while the Premat Stream continues. The fixed
        reconsideration loop may retry a memory-blocked candidate; ordinary
        candidate-to-candidate submission never waits for completion polling.
        """
        self._require_active()
        if self._diagnostic_layer_delay_ms <= 0.0:
            return
        if self._position < 0 or self._layer_indices[self._position] != int(layer_index):
            raise RuntimeError(
                f"logical layer {layer_index} completed outside the active premat position"
            )
        if self._position + 1 >= len(self._layer_indices):
            return
        requested_ms = self._diagnostic_layer_delay_ms
        start_ns = time.perf_counter_ns()
        deadline_ns = start_ns + int(requested_ms * 1_000_000.0)
        self._aggregate["diagnostic_layer_delay_count"] += 1
        self._aggregate["diagnostic_layer_delay_ms_requested_total"] += requested_ms
        self._record(
            "diagnostic_layer_delay_begin",
            layer=layer_index,
            outcome="host_dispatch_paused",
            detail={"requested_ms": requested_ms},
        )
        while True:
            self._advance(trigger="diagnostic_layer_delay_poll")
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                break
            time.sleep(min(0.00025, remaining_ns / 1_000_000_000.0))
        self._advance(trigger="diagnostic_layer_delay_final")
        actual_ms = max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        self._aggregate["diagnostic_layer_delay_ms_actual_total"] += actual_ms
        self._record(
            "diagnostic_layer_delay_end",
            layer=layer_index,
            outcome="host_dispatch_resumed",
            detail={"requested_ms": requested_ms, "actual_ms": actual_ms},
        )

    def event(self, name: str, *, layer_index: int) -> None:
        self._require_active()
        self._refresh_available()
        self._update_ordinary_peak()
        self._record(name, layer=layer_index, outcome="reconsider")
        self._advance(trigger=name)

    def acquire(self, family: str, layer_index: int) -> Tensor:
        self._require_active()
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None:
            raise RuntimeError(f"premat candidate is outside the two-layer window: {key}")
        # vvv THOG a deadline observes completion but cannot launch an UNAVAILABLE head on the Premat Stream
        candidate.deadline_ns = time.perf_counter_ns()
        self._refresh_available()
        self._update_ordinary_peak()
        self._record(
            f"deadline_{family.lower()}",
            candidate=candidate,
            outcome="deadline",
        )
        # ^^^ THOG
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
            candidate.final_outcome = "FULL HIT"
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
            candidate.final_outcome = "PARTIAL HIT"
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
            candidate.final_outcome = "COMPLETE MISS"
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
                # The admission miss is the ordinary critical-path operation.
                # Keep its native autograd graph instead of routing it through
                # the no-grad Premat binding used only by Premat Stream hits.
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
                    "Premat Main Stream fallback materialisation failed; "
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
        if candidate.owner == "main":
            return candidate.tensor
        return self._attach(candidate.family, candidate.layer_index, candidate.tensor)

    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:
        # Checkpoint replay has no schedulable lookahead lifetime.  Recreate the
        # exact ordinary differentiable materialisation on its execution stream.
        return self._materialize(family, layer_index)

    def consumed(self, family: str, layer_index: int) -> None:
        key = (int(layer_index), str(family))
        candidate = self._candidates.get(key)
        if candidate is None or candidate.state != CandidateState.CONSUMING:
            raise RuntimeError(f"premat candidate {key} was not consuming")
        # vvv THOG callers discard their final Python weight reference before
        # entering here.  Keep an admitted Premat tensor charged and strongly
        # referenced until a Main Stream event proves its consuming kernel has
        # finished.  This pending storage participates in the ordinary memory
        # guards, but does not globally block a later independently affordable
        # candidate.  Main-path materialisations have no Premat storage to
        # release and must never create a zero-byte scheduler gate.
        if candidate.retained_counted:
            if candidate.tensor is None:
                raise RuntimeError(f"premat candidate {key} lost its retained tensor")
            current_stream = torch.cuda.current_stream(device=self._device)
            release_event = torch.cuda.Event(enable_timing=False)
            release_event.record(current_stream)
            self._pending_releases.append(
                _PendingRelease(
                    end_event=release_event,
                    retained_bytes=candidate.envelope.retained_bytes,
                    transient_bytes=(
                        self._candidate_transient_bytes(candidate)
                        if candidate.transient_counted
                        else 0
                    ),
                    tensor=candidate.tensor,
                )
            )
        candidate.retained_counted = False
        candidate.transient_counted = False
        # ^^^ THOG
        candidate.consumed_ns = time.perf_counter_ns()
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
        self._resolve_pending_releases()
        self._advance(trigger="consumed")

    def report(self) -> Dict[str, object]:
        self._resolve_pending_timings()
        self._resolve_pending_releases()
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
        pass_complete = bool(
            self._events
            and self._events[-1].get("event") == "pass_end"
        )
        return {
            "version": PREMAT_TELEMETRY_VERSION,
            "schema_version": PREMAT_TELEMETRY_VERSION,
            "enabled": True,
            "attention_mode": self._attention_mode,
            "materialisation_element_bytes": self._dtype_bytes,
            "target_offset": self._target_layer,
            "target_layer": self._target_layer,
            "matrix_order": self._weight_matrix_target_order,
            "weight_matrix_target_order": self._weight_matrix_target_order,
            "cuda_stream_priority": self._cuda_stream_priority,
            "allocator_aware_admission": self._allocator_aware_admission,
            "diagnostic_layer_delay_ms": self._diagnostic_layer_delay_ms,
            "headroom_mode": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
            "current_layer_index": current_layer,
            "next_layer_index": next_layer,
            "lookahead_layer_limit": self._target_layer,
            "target_scope": f"relative_layer_{self._target_layer}",
            "target_order": self._weight_matrix_target_order,
            "effective_fast_discard": True,
            "pass_sequence": self._pass_sequence,
            "pass_complete": pass_complete,
            "layer_indices": list(self._layer_indices),
            "queue_head": self._queue_head_payload(),
            "candidates": candidates,
            "events": [dict(event) for event in self._events],
            "event_count": len(self._events),
            "latest_event_sequence": self._event_sequence,
            "event_window_limit": "complete_pass",
            "memory": memory,
            "aggregate": aggregate,
            "live_publish_error": self._live_publish_error,
        }

    def _families(self) -> Tuple[str, ...]:
        families = (
            self._FUSED_FAMILIES
            if self._attention_mode == "fused"
            else self._UNFUSED_FAMILIES
        )
        return families[self._weight_matrix_target_order]

    def _ensure_layer(self, layer_index: int) -> None:
        for order_position, family in enumerate(self._families()):
            key = (layer_index, family)
            if key in self._candidates:
                continue
            self._sequence += 1
            self._candidates[key] = _Candidate(
                layer_index=layer_index,
                family=family,
                sequence=self._sequence,
                order_position=order_position,
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
        # vvv THOG the fused Premat path does not explicitly materialise
        # [B,H,T,T] score/probability tensors.  Charge those quadratic tensors
        # only to the deliberately unfused path; the fused foreground bound is
        # the Q/K/V/result activation quartet.
        attention_peak = 4 * self._activation_bytes
        if self._attention_mode == "unfused":
            score_bytes = (
                self._batch_size
                * self._n_head
                * self._sequence_length
                * self._sequence_length
                * self._dtype_bytes
            )
            causal_mask_bytes = self._sequence_length * self._sequence_length
            attention_peak += 2 * score_bytes + causal_mask_bytes
        foreground_overlap = max(4 * self._activation_bytes, attention_peak)
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
        self._resolve_pending_releases()
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
                self._release_candidate_transient_charge(candidate)
                self._transition(
                    candidate,
                    CandidateState.AVAILABLE,
                    "available",
                    outcome="premat_complete",
                )

    def _cautious_allocator_aware_rescue(
        self,
        *,
        observation: PrematMemoryObservation,
        envelope: CandidateEnvelope,
        original_decision: AdmissionDecision,
    ) -> Tuple[AdmissionDecision, Dict[str, object]]:
        detail: Dict[str, object] = {
            "mode": "cautious",
            "attempted": True,
            "original_admitted": original_decision.admitted,
            "original_reason": original_decision.reason,
            "original_predicted_physical_growth_bytes": (
                original_decision.predicted_physical_growth_bytes
            ),
            "predicted_envelope_bytes": envelope.process_envelope_bytes,
            "premat_materialisation_peak_bytes": envelope.materialisation_peak_bytes,
            "foreground_overlap_bytes": envelope.foreground_overlap_bytes,
            "premat_stream_id": None,
            "largest_eligible_inactive_block_bytes": 0,
            "reuse_qualified": False,
            "allocator_credit_bytes": 0,
            "revised_predicted_physical_growth_bytes": (
                original_decision.predicted_physical_growth_bytes
            ),
            "final_admitted": original_decision.admitted,
            "final_reason": original_decision.reason,
            "snapshot_error": None,
        }
        if self._stream is None:
            detail["snapshot_error"] = "Premat Stream is not initialized"
            return original_decision, detail

        stream_id = int(self._stream.cuda_stream)
        detail["premat_stream_id"] = stream_id
        try:
            allocator_backend = torch.cuda.memory.get_allocator_backend()
            detail["allocator_backend"] = allocator_backend
            if allocator_backend != "native":
                detail["snapshot_error"] = (
                    "cautious allocator-aware admission requires the native CUDA allocator"
                )
                return original_decision, detail
            snapshot = torch.cuda.memory_snapshot()
            largest_inactive_block = _largest_same_stream_inactive_block_bytes(
                snapshot,
                stream_id=stream_id,
                request_bytes=envelope.materialisation_peak_bytes,
            )
        except Exception as error:
            detail["snapshot_error"] = f"{type(error).__name__}: {error}"
            return original_decision, detail

        detail["largest_eligible_inactive_block_bytes"] = largest_inactive_block
        reuse_qualified = largest_inactive_block >= envelope.materialisation_peak_bytes
        detail["reuse_qualified"] = reuse_qualified
        if not reuse_qualified:
            return original_decision, detail

        # The allocator evidence covers only candidate-owned materialisation.
        # Keep the foreground Main Stream safety allowance fully charged as
        # possible physical growth.  This preserves the existing protection for
        # ordinary memory consumed between prematerialisation and its GEMM.
        revised_physical_growth = envelope.foreground_overlap_bytes
        revised_decision = _decide_candidate_admission_with_physical_growth(
            observation=observation,
            envelope=envelope,
            stay_below_current_peak=self._stay_below_current_peak,
            gpu_memory_buffer_bytes=self._buffer_bytes,
            physical_growth_bytes=revised_physical_growth,
        )
        detail["allocator_credit_bytes"] = max(
            0,
            original_decision.predicted_physical_growth_bytes
            - revised_decision.predicted_physical_growth_bytes,
        )
        detail["revised_predicted_physical_growth_bytes"] = (
            revised_decision.predicted_physical_growth_bytes
        )
        detail["final_admitted"] = revised_decision.admitted
        detail["final_reason"] = revised_decision.reason
        return revised_decision, detail

    def _advance(self, *, trigger: str) -> None:
        """Queue every consecutively admissible candidate for the exact target."""
        self._refresh_available()
        self._advance_sequence += 1
        invocation = self._advance_sequence
        submitted: List[Dict[str, object]] = []
        target_position = self._position + self._target_layer
        target_layer_index = (
            self._layer_indices[target_position]
            if 0 <= target_position < len(self._layer_indices)
            else None
        )
        first_candidate = self._next_premat_candidate()
        self._record(
            "advance_begin",
            layer=(
                self._layer_indices[self._position]
                if 0 <= self._position < len(self._layer_indices)
                else None
            ),
            reason=trigger,
            detail={
                "invocation": invocation,
                "trigger": trigger,
                "target_offset": self._target_layer,
                "target_layer_index": target_layer_index,
                "matrix_order": self._weight_matrix_target_order,
                "candidate_start_index": (
                    first_candidate.order_position
                    if first_candidate is not None
                    else None
                ),
            },
        )
        if target_layer_index is None:
            self._record_advance_return(
                invocation=invocation,
                trigger=trigger,
                submitted=submitted,
                reason="target_out_of_range",
            )
            return
        if self._stream is None or self._device is None:
            raise RuntimeError("Premat Stream is not initialized")

        while True:
            candidate = self._next_premat_candidate()
            if candidate is None:
                self._record_advance_return(
                    invocation=invocation,
                    trigger=trigger,
                    submitted=submitted,
                    reason="target_exhausted",
                )
                return
            considered_ns = time.perf_counter_ns()
            if candidate.first_considered_ns is None:
                candidate.first_considered_ns = considered_ns
            raw_observation = self._observe_memory()
            observation = self._observation_with_cumulative_charge(raw_observation)
            self._update_memory_aggregates(raw_observation)
            original_decision = decide_candidate_admission(
                observation=observation,
                envelope=candidate.envelope,
                stay_below_current_peak=self._stay_below_current_peak,
                gpu_memory_buffer_bytes=self._buffer_bytes,
            )
            decision = original_decision
            allocator_aware_detail: Optional[Dict[str, object]] = None
            if (
                self._allocator_aware_admission == "cautious"
                and not original_decision.admitted
                and original_decision.reason == "global_device_buffer"
            ):
                decision, allocator_aware_detail = self._cautious_allocator_aware_rescue(
                    observation=observation,
                    envelope=candidate.envelope,
                    original_decision=original_decision,
                )
            candidate.admission_reason = decision.reason
            raw_memory = {
                **asdict(raw_observation),
                "reusable_allocator_bytes": raw_observation.reusable_allocator_bytes,
                "device_used_bytes": raw_observation.device_used_bytes,
                "device_free_minus_buffer_bytes": (
                    raw_observation.device_free_bytes - self._buffer_bytes
                ),
            }
            charged_memory = {
                **asdict(observation),
                "reusable_allocator_bytes": observation.reusable_allocator_bytes,
                "device_used_bytes": observation.device_used_bytes,
                "device_free_minus_buffer_bytes": (
                    observation.device_free_bytes - self._buffer_bytes
                ),
            }
            decision_detail = {
                **asdict(decision),
                "invocation": invocation,
                "trigger": trigger,
                "target_offset": self._target_layer,
                "target_layer_index": target_layer_index,
                "matrix_order": self._weight_matrix_target_order,
                "order_position": candidate.order_position,
                "queue_depth": self._queue_depth(),
                "cumulative_charged_bytes": self._cumulative_charged_bytes(),
                "raw_memory": raw_memory,
                "charged_memory": charged_memory,
            }
            if allocator_aware_detail is not None:
                decision_detail["allocator_aware_admission"] = allocator_aware_detail
            if decision.admitted:
                candidate.first_observed_admissible_ns = considered_ns
            self._record(
                "admission_considered",
                candidate=candidate,
                decision="admit" if decision.admitted else "defer",
                outcome="first_observed_admissible" if decision.admitted else "rejected",
                reason=decision.reason,
                detail=decision_detail,
            )
            if not decision.admitted:
                self._aggregate["deferred"] += 1
                rejection_counts = self._aggregate["admission_rejections_by_reason"]
                rejection_counts[decision.reason] = int(
                    rejection_counts.get(decision.reason, 0)
                ) + 1
                self._record(
                    "admission_deferred",
                    candidate=candidate,
                    decision="defer",
                    outcome="not_submitted",
                    reason=decision.reason,
                    detail=decision_detail,
                )
                self._record_advance_return(
                    invocation=invocation,
                    trigger=trigger,
                    submitted=submitted,
                    reason="admission_rejected",
                    blocking_candidate=candidate,
                    blocking_reason=decision.reason,
                )
                return

            candidate.owner = "premat"
            candidate.launch_ns = time.perf_counter_ns()
            self._charge_candidate(candidate)
            current_stream = torch.cuda.current_stream(device=self._device)
            self._stream.wait_stream(current_stream)
            try:
                with torch.cuda.stream(self._stream):
                    candidate.materialisation_start_event = torch.cuda.Event(
                        enable_timing=True
                    )
                    candidate.completion_event = torch.cuda.Event(enable_timing=True)
                    candidate.materialisation_start_event.record(self._stream)
                    # Autocast caches lower-precision casts of FP32 parameter
                    # leaves for the enclosing forward.  A cast produced here
                    # belongs to the Premat Stream and, under PyTorch 2.8
                    # no_grad(), is detached.  Publishing it through the shared
                    # autocast cache lets ordinary Main Stream materialisation
                    # reuse storage with neither a dependency nor a gradient
                    # edge.  Keep autocast's dtype policy, but make Premat casts
                    # private to this submission.
                    autocast_cache_enabled = torch.is_autocast_cache_enabled()
                    torch.set_autocast_cache_enabled(False)
                    try:
                        with torch.no_grad():
                            candidate.tensor = self._materialize(
                                candidate.family,
                                candidate.layer_index,
                            )
                    finally:
                        torch.set_autocast_cache_enabled(autocast_cache_enabled)
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
                self._rollback_candidate_charge(candidate)
                candidate.owner = "none"
                candidate.tensor = None
                candidate.materialisation_start_event = None
                candidate.completion_event = None
                self._record(
                    "materialisation_failed",
                    candidate=candidate,
                    outcome="failure",
                    reason=type(error).__name__,
                    detail={"error": str(error), "admission": decision_detail},
                )
                raise RuntimeError(
                    "Premat Stream candidate materialisation failed after admission; "
                    f"layer={candidate.layer_index}, family={candidate.family}, "
                    f"mode={self._attention_mode}, envelope={asdict(candidate.envelope)}, "
                    f"admission={asdict(decision)}, memory={self._memory_summary()}"
                ) from error
            self._aggregate["admitted"] += 1
            observed_admission_lag_ms = max(
                0.0,
                (candidate.launch_ns - candidate.first_observed_admissible_ns)
                / 1_000_000.0,
            )
            self._aggregate["observed_admission_lag_ms_total"] += observed_admission_lag_ms
            self._aggregate["observed_admission_lag_ms_max"] = max(
                float(self._aggregate["observed_admission_lag_ms_max"]),
                observed_admission_lag_ms,
            )
            launch_detail = {
                **decision_detail,
                "queue_depth_after_submission": self._queue_depth(),
                "cumulative_charged_bytes_after_submission": (
                    self._cumulative_charged_bytes()
                ),
                "observed_admission_lag_ms": observed_admission_lag_ms,
            }
            launch_payload = self._transition(
                candidate,
                CandidateState.MATERIALISING,
                "materialising",
                decision="admit",
                outcome="submitted_to_premat_stream",
                reason=decision.reason,
                detail=launch_detail,
            )
            self._pending_timings.append(
                _PendingCudaTiming(
                    kind="premat_materialisation",
                    start_event=candidate.materialisation_start_event,
                    end_event=candidate.completion_event,
                    event_payload=launch_payload,
                )
            )
            self._update_queue_aggregates()
            submitted.append(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "order_position": candidate.order_position,
                }
            )

    def _next_premat_candidate(self) -> Optional[_Candidate]:
        target_position = self._position + self._target_layer
        if self._position < 0 or target_position >= len(self._layer_indices):
            return None
        target_layer_index = self._layer_indices[target_position]
        return next(
            (
                item
                for item in sorted(
                    self._candidates.values(),
                    key=lambda candidate: candidate.sequence,
                )
                if item.state == CandidateState.UNAVAILABLE
                and item.layer_index == target_layer_index
            ),
            None,
        )

    def _candidate_transient_bytes(self, candidate: _Candidate) -> int:
        return candidate.envelope.candidate_transient_bytes

    def _cumulative_charged_bytes(self) -> int:
        return self._retained_bytes + self._transient_bytes

    def _queue_depth(self) -> int:
        return sum(
            1
            for candidate in self._candidates.values()
            if candidate.owner == "premat"
            and candidate.state != CandidateState.CONSUMED
        )

    def _charge_candidate(self, candidate: _Candidate) -> None:
        if candidate.retained_counted or candidate.transient_counted:
            raise RuntimeError(
                "Premat candidate received a duplicate admission charge: "
                f"layer={candidate.layer_index}, family={candidate.family}"
            )
        self._retained_bytes += candidate.envelope.retained_bytes
        self._transient_bytes += self._candidate_transient_bytes(candidate)
        candidate.retained_counted = True
        candidate.transient_counted = True

    def _rollback_candidate_charge(self, candidate: _Candidate) -> None:
        if candidate.retained_counted:
            self._retained_bytes = max(
                0,
                self._retained_bytes - candidate.envelope.retained_bytes,
            )
            candidate.retained_counted = False
        if candidate.transient_counted:
            self._transient_bytes = max(
                0,
                self._transient_bytes - self._candidate_transient_bytes(candidate),
            )
            candidate.transient_counted = False

    def _release_candidate_transient_charge(self, candidate: _Candidate) -> None:
        if not candidate.transient_counted:
            return
        self._transient_bytes = max(
            0,
            self._transient_bytes - self._candidate_transient_bytes(candidate),
        )
        candidate.transient_counted = False

    def _observation_with_cumulative_charge(
        self,
        observation: PrematMemoryObservation,
    ) -> PrematMemoryObservation:
        ordinary_process_bytes = max(
            0,
            observation.process_allocated_bytes - self._retained_bytes,
        )
        charged_process_bytes = max(
            observation.process_allocated_bytes,
            ordinary_process_bytes + self._cumulative_charged_bytes(),
        )
        ordinary_device_bytes = max(
            0,
            observation.device_used_bytes - self._retained_bytes,
        )
        charged_device_bytes = max(
            observation.device_used_bytes,
            ordinary_device_bytes + self._cumulative_charged_bytes(),
        )
        return PrematMemoryObservation(
            process_allocated_bytes=charged_process_bytes,
            process_reserved_bytes=max(
                observation.process_reserved_bytes,
                charged_process_bytes,
            ),
            process_ordinary_peak_bytes=observation.process_ordinary_peak_bytes,
            device_free_bytes=max(
                0,
                observation.device_total_bytes - charged_device_bytes,
            ),
            device_total_bytes=observation.device_total_bytes,
        )

    def _update_queue_aggregates(self) -> None:
        self._aggregate["queue_depth_high_water"] = max(
            int(self._aggregate["queue_depth_high_water"]),
            self._queue_depth(),
        )
        self._aggregate["cumulative_charged_bytes_high_water"] = max(
            int(self._aggregate["cumulative_charged_bytes_high_water"]),
            self._cumulative_charged_bytes(),
        )

    def _record_advance_return(
        self,
        *,
        invocation: int,
        trigger: str,
        submitted: Sequence[Mapping[str, object]],
        reason: str,
        blocking_candidate: Optional[_Candidate] = None,
        blocking_reason: Optional[str] = None,
    ) -> None:
        self._record(
            "advance_return",
            candidate=blocking_candidate,
            reason=reason,
            detail={
                "invocation": invocation,
                "trigger": trigger,
                "return_reason": reason,
                "target_offset": self._target_layer,
                "matrix_order": self._weight_matrix_target_order,
                "submitted": [dict(item) for item in submitted],
                "submitted_count": len(submitted),
                "blocking_candidate": (
                    {
                        "layer_index": blocking_candidate.layer_index,
                        "family": blocking_candidate.family,
                        "order_position": blocking_candidate.order_position,
                    }
                    if blocking_candidate is not None
                    else None
                ),
                "blocking_reason": blocking_reason,
                "queue_depth": self._queue_depth(),
                "cumulative_charged_bytes": self._cumulative_charged_bytes(),
            },
        )

    def _resolve_pending_releases(self) -> None:
        remaining: List[_PendingRelease] = []
        for release in self._pending_releases:
            try:
                complete = bool(release.end_event.query())
            except RuntimeError:
                complete = False
            if not complete:
                remaining.append(release)
                continue
            self._retained_bytes = max(
                0,
                self._retained_bytes - release.retained_bytes,
            )
            self._transient_bytes = max(
                0,
                self._transient_bytes - release.transient_bytes,
            )
        self._pending_releases = remaining

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
            "pass_sequence": self._pass_sequence,
            "headroom_policy": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
            "target_offset": self._target_layer,
            "target_order": self._weight_matrix_target_order,
            "queue_depth": self._queue_depth(),
            "cumulative_charged_bytes": self._cumulative_charged_bytes(),
            "charged_retained_bytes": self._retained_bytes,
            "charged_transient_bytes": self._transient_bytes,
        }
        # Every retained event carries its exact display window.  The browser
        # can therefore replay a sampled update layer by layer even when the
        # live SQLite row has already advanced to the end of the forward.
        display_position = self._position if self._position >= 0 else 0
        if 0 <= display_position < len(self._layer_indices):
            payload["current_layer_index"] = self._layer_indices[display_position]
            payload["next_layer_index"] = (
                self._layer_indices[display_position + 1]
                if display_position + 1 < len(self._layer_indices)
                else None
            )
        if layer is not None:
            payload["layer_index"] = int(layer)
        if candidate is not None:
            payload.update(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "target_order_position": candidate.order_position,
                    "old_state": (
                        previous_state.value
                        if previous_state is not None
                        else candidate.state.value
                    ),
                    "new_state": candidate.state.value,
                    "state": candidate.state.value,
                    "owner": candidate.owner,
                    "admission_reason": candidate.admission_reason,
                    "critical_path_miss": candidate.critical_path_miss,
                    "first_considered_ns": candidate.first_considered_ns,
                    "first_observed_admissible_ns": (
                        candidate.first_observed_admissible_ns
                    ),
                    "submission_ns": candidate.launch_ns,
                    "cuda_completion_observed_ns": candidate.available_ns,
                    "deadline_ns": candidate.deadline_ns,
                    "consumption_ns": candidate.consumed_ns,
                    "final_outcome": candidate.final_outcome,
                    "observed_admission_lag_ms": (
                        max(
                            0.0,
                            (
                                candidate.launch_ns
                                - candidate.first_observed_admissible_ns
                            )
                            / 1_000_000.0,
                        )
                        if candidate.launch_ns is not None
                        and candidate.first_observed_admissible_ns is not None
                        else None
                    ),
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
                "reusable_allocator_bytes",
                "process_ordinary_peak_bytes",
                "device_free_bytes",
                "device_used_bytes",
                "device_total_bytes",
                "global_buffer_bytes",
                "device_ceiling_bytes",
                "device_free_minus_buffer_bytes",
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
        # vvv THOG publish only the completed sampled microstep.  Partial live
        # rows cannot be mistaken for coherent snapshots, and the browser owns
        # all deliberately slowed playback.
        reporter = self._live_reporter
        capture_enabled = self._live_capture_enabled
        publish_due = (
            reporter is not None
            and event == "pass_end"
            and (
                capture_enabled is None
                or capture_enabled(self._pass_sequence)
            )
        )
        if publish_due:
            try:
                reporter(self.report())
            except Exception as error:  # pragma: no cover - sink failures are environment-specific
                self._live_publish_error = f"{type(error).__name__}: {error}"
                self._live_reporter = None
        # ^^^ THOG
        return payload

    def _candidate_payload(self, candidate: _Candidate) -> Dict[str, object]:
        return {
            "sequence": candidate.sequence,
            "layer_index": candidate.layer_index,
            "family": candidate.family,
            "target_order_position": candidate.order_position,
            "state": candidate.state.value,
            "owner": candidate.owner,
            "critical_path_miss": candidate.critical_path_miss,
            "first_considered_ns": candidate.first_considered_ns,
            "first_observed_admissible_ns": candidate.first_observed_admissible_ns,
            "submission_ns": candidate.launch_ns,
            "cuda_completion_observed_ns": candidate.available_ns,
            "deadline_ns": candidate.deadline_ns,
            "consumption_ns": candidate.consumed_ns,
            "final_outcome": candidate.final_outcome,
            "observed_admission_lag_ms": (
                max(
                    0.0,
                    (candidate.launch_ns - candidate.first_observed_admissible_ns)
                    / 1_000_000.0,
                )
                if candidate.launch_ns is not None
                and candidate.first_observed_admissible_ns is not None
                else None
            ),
            "admission_reason": candidate.admission_reason,
            "envelope": asdict(candidate.envelope),
        }

    def _queue_head_payload(self) -> Optional[Dict[str, object]]:
        materialising = next(
            (
                candidate
                for candidate in self._candidates.values()
                if candidate.state == CandidateState.MATERIALISING
            ),
            None,
        )
        head = materialising or self._next_premat_candidate()
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
            "device_free_minus_buffer_bytes": (
                resolved.device_free_bytes - self._buffer_bytes
            ),
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
            "premat_transient_bytes": self._transient_bytes,
            "premat_cumulative_charged_bytes": self._cumulative_charged_bytes(),
            "premat_queue_depth": self._queue_depth(),
            "materialisation_element_bytes": self._dtype_bytes,
            "foreground_activation_bytes": self._activation_bytes,
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
                timing.event_payload["cuda_elapsed_ms"] = elapsed_ms
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
