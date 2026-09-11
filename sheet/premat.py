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
    # A CUDA allocator's reserved-but-unused total is not a promise that its
    # storage is actually reusable.  The ordinary path therefore charges the
    # complete envelope against driver-visible free memory.  Optional cautious
    # admission may rescue a rejected device-headroom decision only from native
    # allocator blocks explicitly reported inactive by a live snapshot.
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


def _same_stream_inactive_block_sizes(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> Tuple[int, ...]:
    """Return every certified inactive block for one CUDA stream/pool."""
    required_pool = _allocator_pool_for_request(request_bytes)
    sizes: List[int] = []
    for segment in snapshot:
        if not isinstance(segment, Mapping):
            raise ValueError("CUDA allocator snapshot contains a non-mapping segment")
        segment_device = segment.get("device")
        if isinstance(segment_device, bool) or not isinstance(segment_device, int):
            raise ValueError("CUDA allocator snapshot contains an invalid device index")
        if int(segment_device) != int(device_index):
            continue
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
            sizes.append(int(size))
    return tuple(sorted(sizes, reverse=True))


def _same_stream_inactive_block_summary(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> Dict[str, int]:
    """Summarise compatible inactive blocks for one CUDA stream/pool."""
    sizes = _same_stream_inactive_block_sizes(
        snapshot,
        device_index=device_index,
        stream_id=stream_id,
        request_bytes=request_bytes,
    )
    padded = list(sizes[:3]) + [0] * max(0, 3 - len(sizes))
    return {
        "block_count": len(sizes),
        "sufficient_block_count": sum(
            1 for size in sizes if size >= int(request_bytes)
        ),
        "total_bytes": sum(sizes),
        "largest_block_bytes": padded[0],
        "second_largest_block_bytes": padded[1],
        "third_largest_block_bytes": padded[2],
    }


def _largest_same_stream_inactive_block_bytes(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> int:
    return _same_stream_inactive_block_summary(
        snapshot,
        device_index=device_index,
        stream_id=stream_id,
        request_bytes=request_bytes,
    )["largest_block_bytes"]

def _memory_stat_nonnegative_int(
    stats: Mapping[str, object],
    key: str,
) -> int:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"CUDA allocator memory_stats has invalid {key!r}: {value!r}")
    return int(value)


def _inactive_allocator_bytes_from_stats(stats: Mapping[str, object]) -> int:
    """Return currently cached bytes that are no longer allocator-active.

    ``active_bytes`` includes allocations awaiting completion on another stream,
    so ``reserved - active`` deliberately gives those blocks no cautious credit.
    """
    reserved = _memory_stat_nonnegative_int(stats, "reserved_bytes.all.current")
    active = _memory_stat_nonnegative_int(stats, "active_bytes.all.current")
    if active > reserved:
        raise ValueError(
            "CUDA allocator memory_stats reports active bytes above reserved bytes"
        )
    return reserved - active


def _allocator_pool_reserved_bytes_from_stats(
    stats: Mapping[str, object],
    pool: str,
) -> int:
    if pool not in ("small", "large"):
        raise ValueError(f"unknown CUDA allocator pool {pool!r}")
    stat_pool = "small_pool" if pool == "small" else "large_pool"
    return _memory_stat_nonnegative_int(
        stats, f"reserved_bytes.{stat_pool}.current"
    )


def _conservative_certificate_charge_bytes(request_bytes: int) -> int:
    """Upper-bound native allocator rounding for one cached-block request.

    PyTorch's configurable native rounding cannot round a request beyond the
    next power of two.  Using that bound intentionally depletes a cached
    contiguous-space certificate faster than the normal allocator would.
    """
    size = max(0, int(request_bytes))
    if size == 0:
        return 0
    rounded = 1 << (size - 1).bit_length()
    return max(512, rounded)


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
    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_block_id: Optional[int] = None
    allocator_certificate_charge_bytes: int = 0


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
    tensor: Optional[Tensor]
    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_block_id: Optional[int] = None
    allocator_certificate_charge_bytes: int = 0


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
        # Cautious allocator admission certifies the complete set of inactive
        # blocks observed in each exact Premat-stream allocator pool.  The native
        # allocator is best-fit, so reservations mirror that policy against the
        # smallest certified block that can satisfy each conservatively rounded
        # request.  Per-block live charges prevent fragmented cache from being
        # treated as one fictitious contiguous allocation and prevent double use.
        self._allocator_certificate_valid = {"small": False, "large": False}
        self._allocator_certificate_generation = {"small": 0, "large": 0}
        self._allocator_certificate_blocks: Dict[str, List[Dict[str, int]]] = {
            "small": [],
            "large": [],
        }
        self._allocator_certificate_capacity_bytes = {"small": 0, "large": 0}
        self._allocator_certificate_live_charge_bytes = {"small": 0, "large": 0}
        self._allocator_certificate_pool_reserved_floor_bytes: Dict[str, Optional[int]] = {
            "small": None,
            "large": None,
        }
        self._allocator_snapshot_count = 0
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
            "allocator_snapshot_count": 0,
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
        # Pending releases may cross a microstep boundary.  They remain charged
        # and strongly referenced until their Main Stream event has completed;
        # clearing them here used to discard exact lifetime information.
        self._resolve_pending_releases()
        self._candidates.clear()
        self._display_layer_pair = None
        self._display_candidates = []
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
            # Anything not transferred to _PendingRelease has no Main Stream
            # lifetime left at pass end.  Remove both scheduler and allocator
            # certificate charges before dropping the final tensor reference.
            if candidate.retained_counted or candidate.transient_counted:
                self._rollback_candidate_charge(candidate)
            self._release_allocator_certificate_candidate(candidate)
            candidate.tensor = None
            candidate.materialisation_start_event = None
            candidate.completion_event = None
        # Do not clear _pending_releases: unlike the old record_stream fallback,
        # explicit strong-reference lifetime may legitimately span microsteps.
        self._resolve_pending_releases()
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
            # Explicit lifetime is stronger than record_stream here: the runtime
            # retains the tensor until a Main Stream event recorded after its
            # consuming GEMM completes.  Avoiding record_stream means final
            # release returns directly to the tensor's Premat allocator stream.
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
            # candidate.tensor remains strongly referenced through consumed().
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
                    allocator_certificate_generation=(
                        candidate.allocator_certificate_generation
                    ),
                    allocator_certificate_pool=candidate.allocator_certificate_pool,
                    allocator_certificate_block_id=(
                        candidate.allocator_certificate_block_id
                    ),
                    allocator_certificate_charge_bytes=(
                        candidate.allocator_certificate_charge_bytes
                    ),
                )
            )
            candidate.allocator_certificate_generation = None
            candidate.allocator_certificate_pool = None
            candidate.allocator_certificate_block_id = None
            candidate.allocator_certificate_charge_bytes = 0
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

    def _pending_release_diagnostics(self) -> Dict[str, int]:
        return {
            "pending_release_count": len(self._pending_releases),
            "pending_release_bytes": sum(
                int(release.retained_bytes) + int(release.transient_bytes)
                for release in self._pending_releases
            ),
            "certificate_bytes_waiting_for_release": sum(
                int(release.allocator_certificate_charge_bytes)
                for release in self._pending_releases
            ),
        }

    def _sync_allocator_certificate_totals(self, pool: str) -> None:
        blocks = self._allocator_certificate_blocks.get(pool, ())
        self._allocator_certificate_capacity_bytes[pool] = sum(
            int(block["capacity_bytes"]) for block in blocks
        )
        self._allocator_certificate_live_charge_bytes[pool] = sum(
            int(block["live_charge_bytes"]) for block in blocks
        )

    def _allocator_certificate_available_bytes(self, pool: str) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        return sum(
            max(0, int(block["capacity_bytes"]) - int(block["live_charge_bytes"]))
            for block in self._allocator_certificate_blocks.get(pool, ())
        )

    def _allocator_certificate_largest_available_block_bytes(self, pool: str) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        return max(
            (
                max(0, int(block["capacity_bytes"]) - int(block["live_charge_bytes"]))
                for block in self._allocator_certificate_blocks.get(pool, ())
            ),
            default=0,
        )

    def _allocator_certificate_sufficient_block_count(
        self,
        pool: str,
        charge_bytes: int,
    ) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        charge = max(0, int(charge_bytes))
        return sum(
            1
            for block in self._allocator_certificate_blocks.get(pool, ())
            if int(block["capacity_bytes"]) - int(block["live_charge_bytes"]) >= charge
        )

    def _allocator_certificate_best_fit_block(
        self,
        pool: str,
        charge_bytes: int,
    ) -> Optional[Dict[str, int]]:
        if not self._allocator_certificate_valid.get(pool, False):
            return None
        charge = max(0, int(charge_bytes))
        eligible = [
            block
            for block in self._allocator_certificate_blocks.get(pool, ())
            if int(block["capacity_bytes"]) - int(block["live_charge_bytes"]) >= charge
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda block: (
                int(block["capacity_bytes"]) - int(block["live_charge_bytes"]),
                int(block["capacity_bytes"]),
                int(block["block_id"]),
            ),
        )

    def _debit_allocator_certificate_for_pool_shrink(
        self,
        *,
        pool: str,
        current_reserved_bytes: int,
    ) -> int:
        """Invalidate future use if the certified allocator pool has shrunk.

        A pool-level shrink does not identify which certified block disappeared.
        Guessing would make per-block accounting unsafe, so retain the old block
        records only until outstanding charges can be released and fail closed
        for new admissions.  With no live charge the caller may immediately take
        a fresh snapshot and build a new generation.
        """
        floor = self._allocator_certificate_pool_reserved_floor_bytes.get(pool)
        if floor is None:
            return 0
        current = max(0, int(current_reserved_bytes))
        if current >= int(floor):
            return 0
        shrink = int(floor) - current
        self._allocator_certificate_valid[pool] = False
        self._allocator_certificate_pool_reserved_floor_bytes[pool] = current
        return shrink

    def _reserve_allocator_certificate(
        self,
        *,
        request_bytes: int,
        required: bool,
    ) -> Dict[str, object]:
        pool = _allocator_pool_for_request(request_bytes)
        charge = _conservative_certificate_charge_bytes(request_bytes)
        generation = int(self._allocator_certificate_generation.get(pool, 0))
        before_live = int(self._allocator_certificate_live_charge_bytes.get(pool, 0))
        before_available = self._allocator_certificate_available_bytes(pool)
        block = self._allocator_certificate_best_fit_block(pool, charge)
        tracked = block is not None
        block_id: Optional[int] = None
        block_capacity = 0
        block_available_before = 0
        block_available_after = 0
        if required and not tracked:
            raise RuntimeError(
                "allocator-aware admission lost its certified Premat-stream reserve "
                "between decision and launch"
            )
        if tracked:
            block_id = int(block["block_id"])
            block_capacity = int(block["capacity_bytes"])
            block_available_before = max(
                0, block_capacity - int(block["live_charge_bytes"])
            )
            block["live_charge_bytes"] = int(block["live_charge_bytes"]) + charge
            block_available_after = max(
                0, block_capacity - int(block["live_charge_bytes"])
            )
            self._sync_allocator_certificate_totals(pool)
            after_available = self._allocator_certificate_available_bytes(pool)
        else:
            after_available = 0
            # An ordinary admission does not need allocator evidence, but it may
            # consume the same Premat-stream cache.  If the request cannot be
            # represented by one certified block, stop using the certificate
            # until all tracked charges drain and a fresh snapshot proves state.
            if self._allocator_certificate_valid.get(pool, False):
                self._allocator_certificate_valid[pool] = False
        return {
            "tracked": tracked,
            "pool": pool,
            "generation": generation if tracked else None,
            "block_id": block_id,
            "block_capacity_bytes": block_capacity,
            "block_available_before_bytes": block_available_before,
            "block_available_after_bytes": block_available_after,
            "charge_bytes": charge if tracked else 0,
            "before_live_charge_bytes": before_live,
            "after_live_charge_bytes": int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            ),
            "before_available_bytes": before_available,
            "after_available_bytes": after_available,
            "largest_available_block_bytes": (
                self._allocator_certificate_largest_available_block_bytes(pool)
            ),
        }

    def _release_allocator_certificate_charge(
        self,
        *,
        generation: Optional[int],
        pool: Optional[str],
        block_id: Optional[int],
        charge_bytes: int,
    ) -> bool:
        if (
            generation is None
            or pool not in ("small", "large")
            or block_id is None
            or charge_bytes <= 0
        ):
            return False
        if int(generation) != int(self._allocator_certificate_generation.get(pool, -1)):
            return False
        block = next(
            (
                item
                for item in self._allocator_certificate_blocks.get(pool, ())
                if int(item["block_id"]) == int(block_id)
            ),
            None,
        )
        if block is None:
            return False
        before = int(block["live_charge_bytes"])
        block["live_charge_bytes"] = max(0, before - int(charge_bytes))
        self._sync_allocator_certificate_totals(pool)
        return True

    def _release_allocator_certificate_candidate(self, candidate: _Candidate) -> bool:
        released = self._release_allocator_certificate_charge(
            generation=candidate.allocator_certificate_generation,
            pool=candidate.allocator_certificate_pool,
            block_id=candidate.allocator_certificate_block_id,
            charge_bytes=candidate.allocator_certificate_charge_bytes,
        )
        candidate.allocator_certificate_generation = None
        candidate.allocator_certificate_pool = None
        candidate.allocator_certificate_block_id = None
        candidate.allocator_certificate_charge_bytes = 0
        return released

    def _cautious_allocator_aware_rescue(
        self,
        *,
        observation: PrematMemoryObservation,
        envelope: CandidateEnvelope,
        original_decision: AdmissionDecision,
    ) -> Tuple[AdmissionDecision, Dict[str, object]]:
        pool = _allocator_pool_for_request(envelope.materialisation_peak_bytes)
        required_certificate_charge = _conservative_certificate_charge_bytes(
            envelope.materialisation_peak_bytes
        )
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
            "device_index": None,
            "premat_stream_id": None,
            "allocator_pool": pool,
            "certificate_required_charge_bytes": required_certificate_charge,
            "certificate_generation": int(
                self._allocator_certificate_generation.get(pool, 0)
            ),
            "certificate_valid": bool(
                self._allocator_certificate_valid.get(pool, False)
            ),
            "certificate_capacity_bytes": int(
                self._allocator_certificate_capacity_bytes.get(pool, 0)
            ),
            "certificate_block_count": len(
                self._allocator_certificate_blocks.get(pool, ())
            ),
            "certificate_sufficient_block_count": self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            ),
            "certificate_largest_available_block_bytes": self._allocator_certificate_largest_available_block_bytes(pool),
            "certificate_live_charge_bytes": int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            ),
            "certificate_available_bytes": self._allocator_certificate_available_bytes(pool),
            "certificate_pool_reserved_floor_bytes": (
                self._allocator_certificate_pool_reserved_floor_bytes.get(pool)
            ),
            "certificate_pool_reserved_shrink_debit_bytes": 0,
            "certificate_refresh_deferred_live_charge": False,
            "snapshot_eligible_inactive_block_count": 0,
            "snapshot_eligible_sufficient_block_count": 0,
            "snapshot_eligible_inactive_total_bytes": 0,
            "snapshot_eligible_largest_block_bytes": 0,
            "snapshot_eligible_second_largest_block_bytes": 0,
            "snapshot_eligible_third_largest_block_bytes": 0,
            **self._pending_release_diagnostics(),
            "largest_eligible_inactive_block_bytes": 0,
            "certified_premat_stream_cache_bytes": 0,
            "certified_premat_stream_available_bytes": 0,
            # Retained for CSV compatibility. Device-wide free events no longer
            # invalidate a stream certificate; pool shrink is used instead.
            "certificate_invalidated_by_device_free": False,
            "reuse_qualified": False,
            "inactive_allocator_bytes": 0,
            "inactive_allocator_bytes_source": "memory_stats_reserved_minus_active",
            "memory_stats_reserved_bytes": 0,
            "memory_stats_active_bytes": 0,
            "memory_stats_num_device_free": 0,
            "memory_stats_pool_reserved_bytes": 0,
            "required_allocator_headroom_credit_bytes": 0,
            "allocator_headroom_credit_bytes": 0,
            "allocator_credit_bytes": 0,
            "effective_device_free_bytes": observation.device_free_bytes,
            "snapshot_performed": False,
            "allocator_snapshot_count": self._allocator_snapshot_count,
            "revised_predicted_physical_growth_bytes": (
                original_decision.predicted_physical_growth_bytes
            ),
            "final_admitted": original_decision.admitted,
            "final_reason": original_decision.reason,
            "snapshot_error": None,
        }
        if self._stream is None or self._device is None:
            detail["snapshot_error"] = "Premat Stream or CUDA device is not initialized"
            return original_decision, detail

        device_index = self._device.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        stream_id = int(self._stream.cuda_stream)
        detail["device_index"] = int(device_index)
        detail["premat_stream_id"] = stream_id

        try:
            allocator_backend = torch.cuda.memory.get_allocator_backend()
            detail["allocator_backend"] = allocator_backend
            if allocator_backend != "native":
                detail["snapshot_error"] = (
                    "cautious allocator-aware admission requires the native CUDA allocator"
                )
                return original_decision, detail

            stats = torch.cuda.memory_stats(self._device)
            reserved_bytes = _memory_stat_nonnegative_int(
                stats, "reserved_bytes.all.current"
            )
            active_bytes = _memory_stat_nonnegative_int(
                stats, "active_bytes.all.current"
            )
            inactive_allocator_bytes = _inactive_allocator_bytes_from_stats(stats)
            num_device_free = _memory_stat_nonnegative_int(stats, "num_device_free")
            pool_reserved_bytes = _allocator_pool_reserved_bytes_from_stats(stats, pool)
            shrink_debit = self._debit_allocator_certificate_for_pool_shrink(
                pool=pool, current_reserved_bytes=pool_reserved_bytes
            )
            detail["memory_stats_reserved_bytes"] = reserved_bytes
            detail["memory_stats_active_bytes"] = active_bytes
            detail["memory_stats_num_device_free"] = num_device_free
            detail["memory_stats_pool_reserved_bytes"] = pool_reserved_bytes
            detail["inactive_allocator_bytes"] = inactive_allocator_bytes
            detail["certificate_pool_reserved_shrink_debit_bytes"] = shrink_debit

            certificate_available = self._allocator_certificate_available_bytes(pool)
            certificate_valid = bool(self._allocator_certificate_valid.get(pool, False))
            certificate_sufficient = self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            )
            live_charge = int(self._allocator_certificate_live_charge_bytes.get(pool, 0))
            if (not certificate_valid) or certificate_sufficient == 0:
                if live_charge > 0:
                    # Never take a new snapshot while allocations charged to the
                    # current certificate are still live: the new inactive block
                    # could be a remainder of the same physical block, creating
                    # overlapping certificates. Wait for safe release instead.
                    detail["certificate_refresh_deferred_live_charge"] = True
                else:
                    floor = self._allocator_certificate_pool_reserved_floor_bytes.get(pool)
                    snapshot_needed = (
                        not certificate_valid
                        or floor is None
                        or int(pool_reserved_bytes) != int(floor)
                    )
                    if snapshot_needed:
                        snapshot = torch.cuda.memory_snapshot()
                        detail["snapshot_performed"] = True
                        self._allocator_snapshot_count += 1
                        self._aggregate["allocator_snapshot_count"] = self._allocator_snapshot_count
                        block_sizes = _same_stream_inactive_block_sizes(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
                        block_summary = _same_stream_inactive_block_summary(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
                        detail["snapshot_eligible_inactive_block_count"] = int(
                            block_summary["block_count"]
                        )
                        detail["snapshot_eligible_sufficient_block_count"] = int(
                            block_summary["sufficient_block_count"]
                        )
                        detail["snapshot_eligible_inactive_total_bytes"] = int(
                            block_summary["total_bytes"]
                        )
                        detail["snapshot_eligible_largest_block_bytes"] = int(
                            block_summary["largest_block_bytes"]
                        )
                        detail["snapshot_eligible_second_largest_block_bytes"] = int(
                            block_summary["second_largest_block_bytes"]
                        )
                        detail["snapshot_eligible_third_largest_block_bytes"] = int(
                            block_summary["third_largest_block_bytes"]
                        )
                        self._allocator_certificate_generation[pool] = (
                            int(self._allocator_certificate_generation.get(pool, 0)) + 1
                        )
                        self._allocator_certificate_blocks[pool] = [
                            {
                                "block_id": index,
                                "capacity_bytes": int(size),
                                "live_charge_bytes": 0,
                            }
                            for index, size in enumerate(block_sizes, start=1)
                        ]
                        self._sync_allocator_certificate_totals(pool)
                        self._allocator_certificate_pool_reserved_floor_bytes[pool] = (
                            pool_reserved_bytes
                        )
                        self._allocator_certificate_valid[pool] = True

            certificate_capacity = int(
                self._allocator_certificate_capacity_bytes.get(pool, 0)
            )
            certificate_available = self._allocator_certificate_available_bytes(pool)
            detail["allocator_snapshot_count"] = self._allocator_snapshot_count
            detail["certificate_generation"] = int(
                self._allocator_certificate_generation.get(pool, 0)
            )
            detail["certificate_valid"] = bool(
                self._allocator_certificate_valid.get(pool, False)
            )
            detail["certificate_capacity_bytes"] = certificate_capacity
            detail["certificate_block_count"] = len(
                self._allocator_certificate_blocks.get(pool, ())
            )
            detail["certificate_sufficient_block_count"] = (
                self._allocator_certificate_sufficient_block_count(
                    pool, required_certificate_charge
                )
            )
            detail["certificate_largest_available_block_bytes"] = (
                self._allocator_certificate_largest_available_block_bytes(pool)
            )
            detail["certificate_live_charge_bytes"] = int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            )
            detail["certificate_available_bytes"] = certificate_available
            detail["certificate_pool_reserved_floor_bytes"] = (
                self._allocator_certificate_pool_reserved_floor_bytes.get(pool)
            )
            detail["largest_eligible_inactive_block_bytes"] = (
                self._allocator_certificate_largest_available_block_bytes(pool)
            )
            detail["certified_premat_stream_cache_bytes"] = certificate_capacity
            detail["certified_premat_stream_available_bytes"] = certificate_available
        except Exception as error:
            detail["snapshot_error"] = f"{type(error).__name__}: {error}"
            return original_decision, detail

        reuse_qualified = (
            bool(self._allocator_certificate_valid.get(pool, False))
            and self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            ) > 0
        )
        detail["reuse_qualified"] = reuse_qualified
        if not reuse_qualified:
            return original_decision, detail

        # Current global headroom comes from cheap allocator counters.  Only
        # reserved bytes no longer classed active are credited, which excludes
        # active-awaiting-free blocks.  The complete existing candidate envelope
        # remains charged, so allocator cache is not double-counted.
        required_credit = max(
            0,
            self._buffer_bytes
            + envelope.process_envelope_bytes
            - observation.device_free_bytes,
        )
        headroom_credit = min(inactive_allocator_bytes, required_credit)
        effective_device_free = min(
            observation.device_total_bytes,
            observation.device_free_bytes + headroom_credit,
        )
        detail["required_allocator_headroom_credit_bytes"] = required_credit
        detail["allocator_headroom_credit_bytes"] = headroom_credit
        detail["allocator_credit_bytes"] = headroom_credit
        detail["effective_device_free_bytes"] = effective_device_free

        adjusted_observation = PrematMemoryObservation(
            process_allocated_bytes=observation.process_allocated_bytes,
            process_reserved_bytes=observation.process_reserved_bytes,
            process_ordinary_peak_bytes=observation.process_ordinary_peak_bytes,
            device_free_bytes=effective_device_free,
            device_total_bytes=observation.device_total_bytes,
        )
        revised_decision = decide_candidate_admission(
            observation=adjusted_observation,
            envelope=envelope,
            stay_below_current_peak=self._stay_below_current_peak,
            gpu_memory_buffer_bytes=self._buffer_bytes,
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
        deferred_sequences: set[int] = set()
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
            candidate = self._next_premat_candidate(
                excluded_sequences=deferred_sequences
            )
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
                **self._pending_release_diagnostics(),
                "raw_memory": raw_memory,
                "charged_memory": charged_memory,
            }
            if decision.admitted and self._allocator_aware_admission == "cautious":
                certificate_consumption = self._reserve_allocator_certificate(
                    request_bytes=candidate.envelope.materialisation_peak_bytes,
                    required=allocator_aware_detail is not None,
                )
                if bool(certificate_consumption["tracked"]):
                    candidate.allocator_certificate_generation = int(
                        certificate_consumption["generation"]
                    )
                    candidate.allocator_certificate_pool = str(
                        certificate_consumption["pool"]
                    )
                    candidate.allocator_certificate_block_id = int(
                        certificate_consumption["block_id"]
                    )
                    candidate.allocator_certificate_charge_bytes = int(
                        certificate_consumption["charge_bytes"]
                    )
                if allocator_aware_detail is not None:
                    allocator_aware_detail["certificate_consumption"] = certificate_consumption
                    allocator_aware_detail["certified_premat_stream_cache_bytes_after_admission"] = (
                        certificate_consumption["after_available_bytes"]
                    )
                else:
                    decision_detail["allocator_certificate_consumption"] = (
                        certificate_consumption
                    )
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
                deferred_sequences.add(candidate.sequence)
                # A rejection is local to this matrix for this scheduler
                # invocation. Later matrices in the configured order still get
                # their own admission test. This candidate remains UNAVAILABLE
                # and will be reconsidered on the next invocation.
                continue

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
                self._release_allocator_certificate_candidate(candidate)
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

    def _next_premat_candidate(
        self,
        *,
        excluded_sequences: Optional[set[int]] = None,
    ) -> Optional[_Candidate]:
        target_position = self._position + self._target_layer
        if self._position < 0 or target_position >= len(self._layer_indices):
            return None
        target_layer_index = self._layer_indices[target_position]
        excluded = excluded_sequences or set()
        return next(
            (
                item
                for item in sorted(
                    self._candidates.values(),
                    key=lambda candidate: candidate.sequence,
                )
                if item.state == CandidateState.UNAVAILABLE
                and item.layer_index == target_layer_index
                and item.sequence not in excluded
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
                complete = release.end_event.query()
            except RuntimeError:
                complete = False
            if not complete:
                remaining.append(release)
                continue
            # The tensor was deliberately kept alive until this Main Stream
            # event.  PREMAT hits no longer call record_stream(current_stream),
            # so dropping the last reference now returns the allocation directly
            # to its original Premat-stream allocator pool.
            release.tensor = None
            self._retained_bytes = max(
                0,
                self._retained_bytes - release.retained_bytes,
            )
            self._transient_bytes = max(
                0,
                self._transient_bytes - release.transient_bytes,
            )
            self._release_allocator_certificate_charge(
                generation=release.allocator_certificate_generation,
                pool=release.allocator_certificate_pool,
                block_id=release.allocator_certificate_block_id,
                charge_bytes=release.allocator_certificate_charge_bytes,
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
