# vvv THOG dynamic pre-materialisation scheduler, admission policy, lifecycle, and telemetry
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from .premat_processing import processing_operation_range                                                     # <<< THOG semantic processing ranges are inert outside the selected capture
from torch import Tensor


PREMAT_SWITCHES = ("enabled", "disabled")
PREMAT_ATTENTION_MODES = ("fused", "unfused")
PREMAT_TARGET_LAYERS = (0, 1, 2, 10)
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
PREMAT_TELEMETRY_VERSION = 4


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
    enable_gpu_timing_diagnostic: bool = False,
    shadow_mode: bool = False,
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
    # vvv THOG PREMAT CUDA timestamping is an explicit diagnostic, never a default execution cost
    if not isinstance(enable_gpu_timing_diagnostic, bool):
        raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")
    # ^^^ THOG
    # vvv THOG shadow PREMAT is an explicit diagnostic and never a default execution path
    if not isinstance(shadow_mode, bool):
        raise ValueError("premat_enable_shadow_mode must be bool")
    if shadow_mode and premat != "enabled":
        raise ValueError("premat_enable_shadow_mode requires premat enabled")
    # ^^^ THOG


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


# vvv THOG GPU-forensic intervals are retained only while the explicit timing diagnostic is enabled
@dataclass
class _ForensicLayerInterval:
    pass_sequence: int
    layer_index: int
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None


@dataclass
class _ForensicCandidateInterval:
    pass_sequence: int
    layer_index: int
    family: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    dependency_event: Optional[torch.cuda.Event] = None
    wait_end_event: Optional[torch.cuda.Event] = None
    consumed: bool = False


@dataclass
class _ForensicMainWorkInterval:
    pass_sequence: int
    layer_index: int
    family: str
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None


@dataclass
class _PendingForensicPass:
    pass_sequence: int
    origin_event: torch.cuda.Event
    layer_intervals: Tuple[_ForensicLayerInterval, ...]
    candidate_intervals: Tuple[_ForensicCandidateInterval, ...]
    main_work_intervals: Tuple[_ForensicMainWorkInterval, ...] = ()
# ^^^ THOG


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
    # vvv THOG zero-cost host counters remain available even when CUDA forensic timing is disabled
    real_premat_launched: bool = False
    real_premat_consumed: bool = False
    forensic_interval: Optional[_ForensicCandidateInterval] = None
    # ^^^ THOG


@dataclass
class _PendingCudaTiming:
    kind: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    event_payload: Optional[Dict[str, object]]
    pass_sequence: int = 0
    layer_index: Optional[int] = None
    family: Optional[str] = None
    dependency_event: Optional[torch.cuda.Event] = None
    candidate: Optional[_Candidate] = None


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
        enable_gpu_timing_diagnostic: bool = False,
        shadow_mode: bool = False,
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
        # vvv THOG keep correctness/synchronisation events but make timestamp diagnostics opt-in
        if not isinstance(enable_gpu_timing_diagnostic, bool):
            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")
        self._enable_gpu_timing_diagnostic = enable_gpu_timing_diagnostic
        # ^^^ THOG
        # vvv THOG shadow mode runs admission/scheduling but suppresses side-stream weight materialisation
        if not isinstance(shadow_mode, bool):
            raise ValueError("premat_enable_shadow_mode must be bool")
        self._shadow_mode = shadow_mode
        # ^^^ THOG
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
        # vvv THOG explicit GPU timing diagnostic also captures Main/PREMAT overlap without synchronising training
        self._forensic_pass_origin_event: Optional[torch.cuda.Event] = None
        self._forensic_current_layers: List[_ForensicLayerInterval] = []
        self._forensic_current_candidates: List[_ForensicCandidateInterval] = []
        self._forensic_current_main_work: List[_ForensicMainWorkInterval] = []
        self._pending_forensic_passes: List[_PendingForensicPass] = []
        self._latest_forensic_pass: Optional[Dict[str, object]] = None
        # ^^^ THOG
        # Completed sampled passes are retained only until all CUDA timings for
        # that pass have actually resolved. This lets Instra receive a final
        # GPU-timeline classification without synchronising the training stream.
        self._pending_completed_live_reports: Dict[int, Dict[str, object]] = {}
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
            "shadow_main_materialisations": 0,
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
            # vvv THOG forensic counters separate real side work, Main work, overlap and dependency shortfall
            "real_premat_materialisations_launched": 0,
            "real_premat_materialisations_consumed": 0,
            "real_premat_materialisations_unused": 0,
            "duplicate_main_materialisations_after_premat": 0,
            "forensic_resolved_passes": 0,
            "forensic_main_layer_count": 0,
            "forensic_main_layer_gpu_ms_total": 0.0,
            "forensic_main_wait_marker_ms_total": 0.0,
            "forensic_main_nonwait_gpu_ms_total": 0.0,
            "forensic_main_consume_count": 0,
            "forensic_main_consume_gpu_ms_total": 0.0,
            "forensic_premat_overlap_during_main_consume_ms_total": 0.0,
            "forensic_premat_gpu_ms_total": 0.0,
            "forensic_premat_temporal_overlap_ms_total": 0.0,
            "forensic_premat_wait_overlap_ms_total": 0.0,
            "forensic_premat_useful_overlap_ms_total": 0.0,
            "forensic_premat_outside_main_layer_ms_total": 0.0,
            "forensic_dependency_count": 0,
            "forensic_dependency_shortfall_count": 0,
            "forensic_dependency_shortfall_ms_total": 0.0,
            "forensic_dependency_slack_ms_total": 0.0,
            "forensic_dependency_lead_window_ms_total": 0.0,
            "forensic_dependency_before_premat_start_count": 0,
            # ^^^ THOG
        }
        # vvv THOG per-family forensic totals identify whether one matrix family is poisoning overlap
        self._forensic_by_family: Dict[str, Dict[str, float | int]] = {
            family: self._new_forensic_family_row()
            for family in self._families()
        }
        # ^^^ THOG

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

    # vvv THOG first-tranche GPU forensics: six independent observables, all gated by the existing timing diagnostic
    @staticmethod
    def _new_forensic_family_row() -> Dict[str, float | int]:
        return {
            "launched": 0,
            "consumed": 0,
            "unused": 0,
            "duplicates": 0,
            "main_consume_count": 0,
            "main_consume_gpu_ms_total": 0.0,
            "premat_overlap_during_main_consume_ms_total": 0.0,
            "premat_gpu_ms_total": 0.0,
            "temporal_overlap_ms_total": 0.0,
            "wait_overlap_ms_total": 0.0,
            "useful_overlap_ms_total": 0.0,
            "outside_main_layer_ms_total": 0.0,
            "dependency_count": 0,
            "dependency_shortfall_count": 0,
            "dependency_shortfall_ms_total": 0.0,
            "dependency_slack_ms_total": 0.0,
            "dependency_lead_window_ms_total": 0.0,
            "dependency_before_premat_start_count": 0,
        }

    @staticmethod
    def _forensic_overlap_ms(
        left_start: float,
        left_end: float,
        right_start: float,
        right_end: float,
    ) -> float:
        return max(0.0, min(left_end, right_end) - max(left_start, right_start))

    def forensic_layer_start(self, layer_index: int) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        self._require_active()
        if self._device is None:
            raise RuntimeError("PREMAT forensic layer timing has no CUDA device")
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device=self._device))
        self._forensic_current_layers.append(
            _ForensicLayerInterval(
                pass_sequence=self._pass_sequence,
                layer_index=int(layer_index),
                start_event=event,
            )
        )

    def forensic_layer_end(self, layer_index: int) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        self._require_active()
        if self._device is None:
            raise RuntimeError("PREMAT forensic layer timing has no CUDA device")
        interval = next(
            (
                item
                for item in reversed(self._forensic_current_layers)
                if item.layer_index == int(layer_index) and item.end_event is None
            ),
            None,
        )
        if interval is None:
            raise RuntimeError(f"PREMAT forensic layer {layer_index} has no open interval")
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device=self._device))
        interval.end_event = event

    def forensic_main_work_start(self, family: str, layer_index: int) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        self._require_active()
        if self._device is None:
            raise RuntimeError("PREMAT forensic Main work timing has no CUDA device")
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device=self._device))
        self._forensic_current_main_work.append(
            _ForensicMainWorkInterval(
                pass_sequence=self._pass_sequence,
                layer_index=int(layer_index),
                family=str(family),
                start_event=event,
            )
        )

    def forensic_main_work_end(self, family: str, layer_index: int) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        self._require_active()
        if self._device is None:
            raise RuntimeError("PREMAT forensic Main work timing has no CUDA device")
        interval = next(
            (
                item
                for item in reversed(self._forensic_current_main_work)
                if item.layer_index == int(layer_index)
                and item.family == str(family)
                and item.end_event is None
            ),
            None,
        )
        if interval is None:
            raise RuntimeError(
                "PREMAT forensic Main work interval was not opened: "
                f"layer={layer_index}, family={family}"
            )
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device=self._device))
        interval.end_event = event

    def _record_forensic_dependency(
        self,
        candidate: _Candidate,
        current_stream,
        *,
        dependency_event: Optional[torch.cuda.Event] = None,
        wait_end_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        interval = candidate.forensic_interval
        if interval is None:
            return
        if interval.dependency_event is not None:
            raise RuntimeError(
                "PREMAT forensic candidate reached its Main Stream dependency twice: "
                f"layer={candidate.layer_index}, family={candidate.family}"
            )
        if dependency_event is None:
            dependency_event = torch.cuda.Event(enable_timing=True)
            dependency_event.record(current_stream)
        interval.dependency_event = dependency_event
        interval.wait_end_event = wait_end_event

    def _queue_forensic_pass(self) -> None:
        if not self._enable_gpu_timing_diagnostic:
            return
        origin = self._forensic_pass_origin_event
        if origin is None:
            raise RuntimeError("PREMAT forensic timing pass has no origin event")
        if any(interval.end_event is None for interval in self._forensic_current_layers):
            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream layer interval")
        if any(interval.end_event is None for interval in self._forensic_current_main_work):
            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream work interval")
        self._pending_forensic_passes.append(
            _PendingForensicPass(
                pass_sequence=self._pass_sequence,
                origin_event=origin,
                layer_intervals=tuple(self._forensic_current_layers),
                candidate_intervals=tuple(self._forensic_current_candidates),
                main_work_intervals=tuple(self._forensic_current_main_work),
            )
        )
        self._forensic_pass_origin_event = None
        self._forensic_current_layers = []
        self._forensic_current_candidates = []
        self._forensic_current_main_work = []

    @staticmethod
    def _forensic_derived_summary(values: Mapping[str, float | int]) -> Dict[str, object]:
        layer_count = int(values.get("main_layer_count", 0))
        premat_ms = float(values.get("premat_gpu_ms_total", 0.0))
        dependency_count = int(values.get("dependency_count", 0))
        launched = int(values.get("premat_launched", 0))
        main_consume_count = int(values.get("main_consume_count", 0))
        main_consume_ms = float(values.get("main_consume_gpu_ms_total", 0.0))
        return {
            **dict(values),
            "main_layer_gpu_ms_mean": (
                float(values.get("main_layer_gpu_ms_total", 0.0)) / layer_count
                if layer_count > 0
                else None
            ),
            "main_nonwait_gpu_ms_mean": (
                float(values.get("main_nonwait_gpu_ms_total", 0.0)) / layer_count
                if layer_count > 0
                else None
            ),
            "main_consume_gpu_ms_mean": (
                main_consume_ms / main_consume_count
                if main_consume_count > 0
                else None
            ),
            "main_consume_covered_by_premat_fraction": (
                min(
                    1.0,
                    float(values.get("premat_overlap_during_main_consume_ms_total", 0.0))
                    / main_consume_ms,
                )
                if main_consume_ms > 0.0
                else None
            ),
            "premat_useful_overlap_fraction": (
                min(1.0, float(values.get("premat_useful_overlap_ms_total", 0.0)) / premat_ms)
                if premat_ms > 0.0
                else None
            ),
            "premat_outside_main_layer_fraction": (
                min(1.0, float(values.get("premat_outside_main_layer_ms_total", 0.0)) / premat_ms)
                if premat_ms > 0.0
                else None
            ),
            "dependency_shortfall_rate": (
                int(values.get("dependency_shortfall_count", 0)) / dependency_count
                if dependency_count > 0
                else None
            ),
            "premat_consumption_fraction": (
                int(values.get("premat_consumed", 0)) / launched
                if launched > 0
                else None
            ),
        }

    def _forensic_report(self) -> Dict[str, object]:
        aggregate = self._aggregate
        values: Dict[str, float | int] = {
            "resolved_passes": int(aggregate["forensic_resolved_passes"]),
            "pending_passes": len(self._pending_forensic_passes),
            "premat_launched": int(aggregate["real_premat_materialisations_launched"]),
            "premat_consumed": int(aggregate["real_premat_materialisations_consumed"]),
            "premat_unused": int(aggregate["real_premat_materialisations_unused"]),
            "duplicate_main_materialisations": int(aggregate["duplicate_main_materialisations_after_premat"]),
            "main_layer_count": int(aggregate["forensic_main_layer_count"]),
            "main_layer_gpu_ms_total": float(aggregate["forensic_main_layer_gpu_ms_total"]),
            "main_wait_marker_ms_total": float(aggregate["forensic_main_wait_marker_ms_total"]),
            "main_nonwait_gpu_ms_total": float(aggregate["forensic_main_nonwait_gpu_ms_total"]),
            "main_consume_count": int(aggregate["forensic_main_consume_count"]),
            "main_consume_gpu_ms_total": float(aggregate["forensic_main_consume_gpu_ms_total"]),
            "premat_overlap_during_main_consume_ms_total": float(
                aggregate["forensic_premat_overlap_during_main_consume_ms_total"]
            ),
            "premat_gpu_ms_total": float(aggregate["forensic_premat_gpu_ms_total"]),
            "premat_temporal_overlap_ms_total": float(aggregate["forensic_premat_temporal_overlap_ms_total"]),
            "premat_wait_overlap_ms_total": float(aggregate["forensic_premat_wait_overlap_ms_total"]),
            "premat_useful_overlap_ms_total": float(aggregate["forensic_premat_useful_overlap_ms_total"]),
            "premat_outside_main_layer_ms_total": float(aggregate["forensic_premat_outside_main_layer_ms_total"]),
            "dependency_count": int(aggregate["forensic_dependency_count"]),
            "dependency_shortfall_count": int(aggregate["forensic_dependency_shortfall_count"]),
            "dependency_shortfall_ms_total": float(aggregate["forensic_dependency_shortfall_ms_total"]),
            "dependency_slack_ms_total": float(aggregate["forensic_dependency_slack_ms_total"]),
            "dependency_lead_window_ms_total": float(aggregate["forensic_dependency_lead_window_ms_total"]),
            "dependency_before_premat_start_count": int(aggregate["forensic_dependency_before_premat_start_count"]),
        }
        by_family: Dict[str, object] = {}
        for family, row in self._forensic_by_family.items():
            family_values = {
                "premat_launched": int(row["launched"]),
                "premat_consumed": int(row["consumed"]),
                "premat_unused": int(row["unused"]),
                "duplicate_main_materialisations": int(row["duplicates"]),
                "main_layer_count": 0,
                "main_layer_gpu_ms_total": 0.0,
                "main_wait_marker_ms_total": 0.0,
                "main_nonwait_gpu_ms_total": 0.0,
                "main_consume_count": int(row["main_consume_count"]),
                "main_consume_gpu_ms_total": float(row["main_consume_gpu_ms_total"]),
                "premat_overlap_during_main_consume_ms_total": float(
                    row["premat_overlap_during_main_consume_ms_total"]
                ),
                "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
                "premat_temporal_overlap_ms_total": float(row["temporal_overlap_ms_total"]),
                "premat_wait_overlap_ms_total": float(row["wait_overlap_ms_total"]),
                "premat_useful_overlap_ms_total": float(row["useful_overlap_ms_total"]),
                "premat_outside_main_layer_ms_total": float(row["outside_main_layer_ms_total"]),
                "dependency_count": int(row["dependency_count"]),
                "dependency_shortfall_count": int(row["dependency_shortfall_count"]),
                "dependency_shortfall_ms_total": float(row["dependency_shortfall_ms_total"]),
                "dependency_slack_ms_total": float(row["dependency_slack_ms_total"]),
                "dependency_lead_window_ms_total": float(row["dependency_lead_window_ms_total"]),
                "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
            }
            by_family[family] = self._forensic_derived_summary(family_values)
        result = self._forensic_derived_summary(values)
        result["enabled"] = self._enable_gpu_timing_diagnostic
        result["by_family"] = by_family
        return result

    def _update_pending_live_report_forensic(
        self,
        pass_sequence: int,
        aggregate_deltas: Mapping[str, float | int],
        pass_report: Mapping[str, object],
    ) -> None:
        snapshot = self._pending_completed_live_reports.get(int(pass_sequence))
        if snapshot is None:
            return
        aggregate = snapshot.get("aggregate")
        if isinstance(aggregate, dict):
            for name, delta in aggregate_deltas.items():
                aggregate[name] = aggregate.get(name, 0) + delta
        snapshot["forensic_pass"] = dict(pass_report)
        self._refresh_report_derived_aggregates(snapshot)

    def _resolve_pending_forensic_passes(self) -> None:
        if not self._pending_forensic_passes:
            return
        remaining: List[_PendingForensicPass] = []
        for pending in self._pending_forensic_passes:
            required_events: List[torch.cuda.Event] = [pending.origin_event]
            for layer in pending.layer_intervals:
                if layer.end_event is None:
                    raise RuntimeError("PREMAT forensic pass retained an open layer interval")
                required_events.extend((layer.start_event, layer.end_event))
            for candidate in pending.candidate_intervals:
                required_events.extend((candidate.start_event, candidate.end_event))
                if candidate.dependency_event is not None:
                    required_events.append(candidate.dependency_event)
                if candidate.wait_end_event is not None:
                    required_events.append(candidate.wait_end_event)
            for main_work in pending.main_work_intervals:
                if main_work.end_event is None:
                    raise RuntimeError("PREMAT forensic pass retained an open Main Stream work interval")
                required_events.extend((main_work.start_event, main_work.end_event))
            try:
                if not all(event.query() for event in required_events):
                    remaining.append(pending)
                    continue

                def at_ms(event: torch.cuda.Event) -> float:
                    return float(pending.origin_event.elapsed_time(event))

                layer_rows: List[Dict[str, object]] = []
                for layer in pending.layer_intervals:
                    assert layer.end_event is not None
                    start_ms = at_ms(layer.start_event)
                    end_ms = at_ms(layer.end_event)
                    layer_rows.append({
                        "layer_index": layer.layer_index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "gpu_ms": max(0.0, end_ms - start_ms),
                    })

                # vvv THOG exact foreground GEMM intervals are directly comparable between REAL and SHADOW PREMAT
                main_work_rows: List[Dict[str, object]] = []
                for main_work in pending.main_work_intervals:
                    assert main_work.end_event is not None
                    start_ms = at_ms(main_work.start_event)
                    end_ms = at_ms(main_work.end_event)
                    main_work_rows.append({
                        "layer_index": main_work.layer_index,
                        "family": main_work.family,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "gpu_ms": max(0.0, end_ms - start_ms),
                    })
                # ^^^ THOG

                wait_intervals: List[Tuple[float, float]] = []
                for candidate in pending.candidate_intervals:
                    if candidate.dependency_event is None or candidate.wait_end_event is None:
                        continue
                    wait_start_ms = at_ms(candidate.dependency_event)
                    wait_end_ms = at_ms(candidate.wait_end_event)
                    wait_intervals.append((wait_start_ms, max(wait_start_ms, wait_end_ms)))

                candidate_rows: List[Dict[str, object]] = []
                pass_family_rows: Dict[str, Dict[str, float | int]] = {}
                main_consume_ms_total = sum(float(row["gpu_ms"]) for row in main_work_rows)
                premat_overlap_during_main_consume_total = 0.0
                for main_work in main_work_rows:
                    family_row = pass_family_rows.setdefault(
                        str(main_work["family"]), self._new_forensic_family_row()
                    )
                    family_row["main_consume_count"] += 1
                    family_row["main_consume_gpu_ms_total"] += float(main_work["gpu_ms"])
                premat_ms_total = 0.0
                temporal_overlap_total = 0.0
                wait_overlap_total = 0.0
                useful_overlap_total = 0.0
                outside_main_total = 0.0
                dependency_count = 0
                shortfall_count = 0
                shortfall_ms_total = 0.0
                slack_ms_total = 0.0
                lead_window_ms_total = 0.0
                dependency_before_start_count = 0

                for candidate in pending.candidate_intervals:
                    start_ms = at_ms(candidate.start_event)
                    end_ms = at_ms(candidate.end_event)
                    materialisation_ms = max(0.0, end_ms - start_ms)
                    temporal_overlap_ms = min(
                        materialisation_ms,
                        sum(
                            self._forensic_overlap_ms(
                                start_ms, end_ms,
                                float(layer["start_ms"]), float(layer["end_ms"]),
                            )
                            for layer in layer_rows
                        ),
                    )
                    wait_overlap_ms = min(
                        temporal_overlap_ms,
                        sum(
                            self._forensic_overlap_ms(start_ms, end_ms, wait_start, wait_end)
                            for wait_start, wait_end in wait_intervals
                        ),
                    )
                    useful_overlap_ms = max(0.0, temporal_overlap_ms - wait_overlap_ms)
                    outside_main_ms = max(0.0, materialisation_ms - temporal_overlap_ms)
                    main_consume_overlap_ms = min(
                        materialisation_ms,
                        sum(
                            self._forensic_overlap_ms(
                                start_ms, end_ms,
                                float(main_work["start_ms"]), float(main_work["end_ms"]),
                            )
                            for main_work in main_work_rows
                        ),
                    )
                    premat_overlap_during_main_consume_total += main_consume_overlap_ms
                    dependency_ms = None
                    lead_window_ms = None
                    shortfall_ms = None
                    slack_ms = None
                    dependency_before_start = False
                    if candidate.dependency_event is not None:
                        dependency_count += 1
                        dependency_ms = at_ms(candidate.dependency_event)
                        lead_window_ms = max(0.0, dependency_ms - start_ms)
                        lead_window_ms_total += lead_window_ms
                        dependency_before_start = dependency_ms < start_ms
                        if dependency_before_start:
                            dependency_before_start_count += 1
                        completion_delta_ms = end_ms - dependency_ms
                        shortfall_ms = max(0.0, completion_delta_ms)
                        slack_ms = max(0.0, -completion_delta_ms)
                        shortfall_ms_total += shortfall_ms
                        slack_ms_total += slack_ms
                        if shortfall_ms > 0.0:
                            shortfall_count += 1

                    row = {
                        "layer_index": candidate.layer_index,
                        "family": candidate.family,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "materialisation_ms": materialisation_ms,
                        "dependency_ms": dependency_ms,
                        "lead_window_ms": lead_window_ms,
                        "shortfall_ms": shortfall_ms,
                        "slack_ms": slack_ms,
                        "dependency_before_premat_start": dependency_before_start,
                        "temporal_overlap_ms": temporal_overlap_ms,
                        "wait_overlap_ms": wait_overlap_ms,
                        "useful_overlap_ms": useful_overlap_ms,
                        "outside_main_layer_ms": outside_main_ms,
                        "main_consume_overlap_ms": main_consume_overlap_ms,
                        "consumed": bool(candidate.consumed),
                    }
                    candidate_rows.append(row)
                    family_row = pass_family_rows.setdefault(
                        candidate.family, self._new_forensic_family_row()
                    )
                    family_row["launched"] += 1
                    family_row["consumed"] += int(candidate.consumed)
                    family_row["unused"] += int(not candidate.consumed)
                    family_row["premat_gpu_ms_total"] += materialisation_ms
                    family_row["temporal_overlap_ms_total"] += temporal_overlap_ms
                    family_row["wait_overlap_ms_total"] += wait_overlap_ms
                    family_row["useful_overlap_ms_total"] += useful_overlap_ms
                    family_row["outside_main_layer_ms_total"] += outside_main_ms
                    family_row["premat_overlap_during_main_consume_ms_total"] += main_consume_overlap_ms
                    if candidate.dependency_event is not None:
                        family_row["dependency_count"] += 1
                        family_row["dependency_shortfall_count"] += int((shortfall_ms or 0.0) > 0.0)
                        family_row["dependency_shortfall_ms_total"] += float(shortfall_ms or 0.0)
                        family_row["dependency_slack_ms_total"] += float(slack_ms or 0.0)
                        family_row["dependency_lead_window_ms_total"] += float(lead_window_ms or 0.0)
                        family_row["dependency_before_premat_start_count"] += int(dependency_before_start)

                    premat_ms_total += materialisation_ms
                    temporal_overlap_total += temporal_overlap_ms
                    wait_overlap_total += wait_overlap_ms
                    useful_overlap_total += useful_overlap_ms
                    outside_main_total += outside_main_ms

                main_layer_ms_total = sum(float(row["gpu_ms"]) for row in layer_rows)
                main_wait_marker_ms_total = sum(max(0.0, end - start) for start, end in wait_intervals)
                main_nonwait_ms_total = max(0.0, main_layer_ms_total - main_wait_marker_ms_total)
                summary_values: Dict[str, float | int] = {
                    "premat_launched": len(pending.candidate_intervals),
                    "premat_consumed": sum(int(item.consumed) for item in pending.candidate_intervals),
                    "premat_unused": sum(int(not item.consumed) for item in pending.candidate_intervals),
                    "duplicate_main_materialisations": 0,
                    "main_layer_count": len(layer_rows),
                    "main_layer_gpu_ms_total": main_layer_ms_total,
                    "main_wait_marker_ms_total": main_wait_marker_ms_total,
                    "main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                    "main_consume_count": len(main_work_rows),
                    "main_consume_gpu_ms_total": main_consume_ms_total,
                    "premat_overlap_during_main_consume_ms_total": premat_overlap_during_main_consume_total,
                    "premat_gpu_ms_total": premat_ms_total,
                    "premat_temporal_overlap_ms_total": temporal_overlap_total,
                    "premat_wait_overlap_ms_total": wait_overlap_total,
                    "premat_useful_overlap_ms_total": useful_overlap_total,
                    "premat_outside_main_layer_ms_total": outside_main_total,
                    "dependency_count": dependency_count,
                    "dependency_shortfall_count": shortfall_count,
                    "dependency_shortfall_ms_total": shortfall_ms_total,
                    "dependency_slack_ms_total": slack_ms_total,
                    "dependency_lead_window_ms_total": lead_window_ms_total,
                    "dependency_before_premat_start_count": dependency_before_start_count,
                }
                summary = self._forensic_derived_summary(summary_values)
                pass_report: Dict[str, object] = {
                    "pass_sequence": pending.pass_sequence,
                    "summary": summary,
                    "layers": layer_rows,
                    "main_work": main_work_rows,
                    "candidates": candidate_rows,
                    "by_family": {
                        family: self._forensic_derived_summary({
                            "premat_launched": int(row["launched"]),
                            "premat_consumed": int(row["consumed"]),
                            "premat_unused": int(row["unused"]),
                            "duplicate_main_materialisations": int(row["duplicates"]),
                            "main_layer_count": 0,
                            "main_layer_gpu_ms_total": 0.0,
                            "main_wait_marker_ms_total": 0.0,
                            "main_nonwait_gpu_ms_total": 0.0,
                            "main_consume_count": int(row["main_consume_count"]),
                            "main_consume_gpu_ms_total": float(row["main_consume_gpu_ms_total"]),
                            "premat_overlap_during_main_consume_ms_total": float(
                                row["premat_overlap_during_main_consume_ms_total"]
                            ),
                            "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
                            "premat_temporal_overlap_ms_total": float(row["temporal_overlap_ms_total"]),
                            "premat_wait_overlap_ms_total": float(row["wait_overlap_ms_total"]),
                            "premat_useful_overlap_ms_total": float(row["useful_overlap_ms_total"]),
                            "premat_outside_main_layer_ms_total": float(row["outside_main_layer_ms_total"]),
                            "dependency_count": int(row["dependency_count"]),
                            "dependency_shortfall_count": int(row["dependency_shortfall_count"]),
                            "dependency_shortfall_ms_total": float(row["dependency_shortfall_ms_total"]),
                            "dependency_slack_ms_total": float(row["dependency_slack_ms_total"]),
                            "dependency_lead_window_ms_total": float(row["dependency_lead_window_ms_total"]),
                            "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
                        })
                        for family, row in pass_family_rows.items()
                    },
                }
            except RuntimeError:
                remaining.append(pending)
                continue

            aggregate_deltas: Dict[str, float | int] = {
                "forensic_resolved_passes": 1,
                "forensic_main_layer_count": len(layer_rows),
                "forensic_main_layer_gpu_ms_total": main_layer_ms_total,
                "forensic_main_wait_marker_ms_total": main_wait_marker_ms_total,
                "forensic_main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                "forensic_main_consume_count": len(main_work_rows),
                "forensic_main_consume_gpu_ms_total": main_consume_ms_total,
                "forensic_premat_overlap_during_main_consume_ms_total": premat_overlap_during_main_consume_total,
                "forensic_premat_gpu_ms_total": premat_ms_total,
                "forensic_premat_temporal_overlap_ms_total": temporal_overlap_total,
                "forensic_premat_wait_overlap_ms_total": wait_overlap_total,
                "forensic_premat_useful_overlap_ms_total": useful_overlap_total,
                "forensic_premat_outside_main_layer_ms_total": outside_main_total,
                "forensic_dependency_count": dependency_count,
                "forensic_dependency_shortfall_count": shortfall_count,
                "forensic_dependency_shortfall_ms_total": shortfall_ms_total,
                "forensic_dependency_slack_ms_total": slack_ms_total,
                "forensic_dependency_lead_window_ms_total": lead_window_ms_total,
                "forensic_dependency_before_premat_start_count": dependency_before_start_count,
            }
            for name, delta in aggregate_deltas.items():
                self._aggregate[name] += delta
            for family, row in pass_family_rows.items():
                cumulative = self._forensic_by_family.setdefault(
                    family, self._new_forensic_family_row()
                )
                for name in (
                    "main_consume_count",
                    "main_consume_gpu_ms_total",
                    "premat_overlap_during_main_consume_ms_total",
                    "premat_gpu_ms_total",
                    "temporal_overlap_ms_total",
                    "wait_overlap_ms_total",
                    "useful_overlap_ms_total",
                    "outside_main_layer_ms_total",
                    "dependency_count",
                    "dependency_shortfall_count",
                    "dependency_shortfall_ms_total",
                    "dependency_slack_ms_total",
                    "dependency_lead_window_ms_total",
                    "dependency_before_premat_start_count",
                ):
                    cumulative[name] += row[name]
            self._latest_forensic_pass = pass_report
            self._update_pending_live_report_forensic(
                pending.pass_sequence, aggregate_deltas, pass_report
            )
        self._pending_forensic_passes = remaining
        self._publish_completed_live_reports_if_ready()
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
        # A prior sampled pass may have had GPU wait timings still outstanding
        # when its host forward ended. Resolve/publish those non-blockingly before
        # replacing the current event window.
        self._resolve_pending_timings()
        self._resolve_pending_forensic_passes()                                                                          # <<< THOG resolve prior GPU-forensic pass without synchronising
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
        # vvv THOG one timing-enabled Main Stream origin makes cross-stream overlap arithmetic unambiguous
        self._forensic_current_layers = []
        self._forensic_current_candidates = []
        self._forensic_current_main_work = []
        self._forensic_pass_origin_event = None
        if self._enable_gpu_timing_diagnostic:
            self._forensic_pass_origin_event = torch.cuda.Event(enable_timing=True)
            self._forensic_pass_origin_event.record(
                torch.cuda.current_stream(device=self._device)
            )
        # ^^^ THOG
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
        self._resolve_pending_forensic_passes()                                                                          # <<< THOG opportunistically drain older forensic passes
        # vvv THOG classify real side materialisations that reached pass end without a consumption callback
        for candidate in tuple(self._candidates.values()):
            if candidate.real_premat_launched and not candidate.real_premat_consumed:
                self._aggregate["real_premat_materialisations_unused"] += 1
                self._forensic_by_family[candidate.family]["unused"] += 1
        # ^^^ THOG
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
        self._queue_forensic_pass()                                                                                       # <<< THOG defer GPU-forensic arithmetic until all timing events complete
        self._resolve_pending_forensic_passes()                                                                           # <<< THOG publish immediately when the GPU is already caught up
        self._capture_completed_live_report()
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
        lookahead_limit = 1 if self._target_layer == 10 else self._target_layer
        window_width = max(2, lookahead_limit + 1)
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
        # vvv THOG shadow PREMAT preserves scheduler/admission/event overhead but Main Stream still materialises the real weight
        if self._shadow_mode:
            return self._acquire_shadow(candidate, current_stream)
        # ^^^ THOG
        if (
            candidate.state
            in (CandidateState.AVAILABLE, CandidateState.MATERIALISING)
            and candidate.tensor is None
        ):
            raise RuntimeError(f"premat candidate {key} has no owned tensor")
        if candidate.state == CandidateState.AVAILABLE:
            self._record_forensic_dependency(candidate, current_stream)                                                    # <<< THOG mark actual Main Stream matrix-use dependency even when PREMAT is already complete
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
            # vvv THOG the Main Stream dependency is required; timestamp/classification events are diagnostic only
            if self._enable_gpu_timing_diagnostic:
                wait_start = torch.cuda.Event(enable_timing=True)
                wait_end = torch.cuda.Event(enable_timing=True)
                wait_start.record(current_stream)
                current_stream.wait_event(candidate.completion_event)
                wait_end.record(current_stream)
                self._record_forensic_dependency(
                    candidate, current_stream,
                    dependency_event=wait_start,
                    wait_end_event=wait_end,
                )                                                                                                         # <<< THOG retain the true dependency/wait interval for overlap forensics
                # Host submission has reached the dependency, but this does NOT tell
                # us whether the GPU Main Stream will actually wait there. Keep the
                # result provisional until CUDA has timestamped both streams.
                candidate.critical_path_miss = False
                candidate.final_outcome = "PENDING"
                wait_payload = self._transition(
                    candidate,
                    CandidateState.CONSUMING,
                    "critical_path_wait",
                    outcome="gpu_wait_pending",
                    reason="classification_pending_until_main_stream_dependency",
                    detail={
                        "gpu_wait_classification_pending": True,
                        "gpu_timing_diagnostic_enabled": True,
                    },
                )
                self._pending_timings.append(
                    _PendingCudaTiming(
                        kind="main_stream_wait",
                        start_event=wait_start,
                        end_event=wait_end,
                        event_payload=wait_payload,
                        pass_sequence=self._pass_sequence,
                        layer_index=candidate.layer_index,
                        family=candidate.family,
                        dependency_event=candidate.completion_event,
                        candidate=candidate,
                    )
                )
            else:
                current_stream.wait_event(candidate.completion_event)
                # Without GPU timestamps, MATERIALISING at host submission is only
                # a conservative host-observed partial classification.  This is
                # telemetry only and does not alter the dependency or materialisation.
                candidate.critical_path_miss = True
                candidate.final_outcome = "PARTIAL HIT"
                self._aggregate["waited_hits"] += 1
                self._transition(
                    candidate,
                    CandidateState.CONSUMING,
                    "critical_path_wait",
                    outcome="host_observed_materialising",
                    reason="gpu_timing_diagnostic_disabled",
                    detail={
                        "gpu_wait_classification_pending": False,
                        "gpu_timing_diagnostic_enabled": False,
                    },
                )
            # candidate.tensor remains strongly referenced through consumed().
            # ^^^ THOG
        elif candidate.state == CandidateState.UNAVAILABLE:
            # vvv THOG a Main fallback after a real side launch is a true duplicate and must be visible, not inferred
            if candidate.real_premat_launched:
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
            # ^^^ THOG
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
            # vvv THOG main-stream fallback timestamps are diagnostic; fallback materialisation itself is unchanged
            materialise_start = (
                torch.cuda.Event(enable_timing=True)
                if self._enable_gpu_timing_diagnostic
                else None
            )
            materialise_end = (
                torch.cuda.Event(enable_timing=True)
                if self._enable_gpu_timing_diagnostic
                else None
            )
            if materialise_start is not None:
                materialise_start.record(current_stream)
            # ^^^ THOG
            try:
                # The admission miss is the ordinary critical-path operation.
                # Keep its native autograd graph instead of routing it through
                # the no-grad Premat binding used only by Premat Stream hits.
                # vvv THOG preserve the ordinary Main materialisation statement; the active copy is only wrapped for profiling attribution
                # candidate.tensor = self._materialize(candidate.family, candidate.layer_index)
                with processing_operation_range("MAIN", "materialize", family=candidate.family, layer_index=candidate.layer_index):
                    candidate.tensor = self._materialize(candidate.family, candidate.layer_index)
                # ^^^ THOG
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
            # vvv THOG retain no CUDA timing objects when the diagnostic is disabled
            if materialise_start is not None and materialise_end is not None:
                materialise_end.record(current_stream)
                self._pending_timings.append(
                    _PendingCudaTiming(
                        kind="main_stream_materialisation",
                        start_event=materialise_start,
                        end_event=materialise_end,
                        event_payload=fallback_payload,
                        pass_sequence=self._pass_sequence,
                        layer_index=candidate.layer_index,
                        family=candidate.family,
                        candidate=candidate,
                    )
                )
            # ^^^ THOG
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


    # vvv THOG shadow-mode deadline path: no side-stream weight exists; ordinary differentiable materialisation remains on Main Stream
    def _acquire_shadow(self, candidate: _Candidate, current_stream) -> Tensor:
        if candidate.state == CandidateState.MATERIALISING:
            if candidate.completion_event is None:
                raise RuntimeError(
                    f"shadow premat candidate {(candidate.layer_index, candidate.family)} has no completion event"
                )
            # Retain the normal PREMAT dependency/event-management overhead even
            # though the side stream deliberately performed no weight work.
            current_stream.wait_event(candidate.completion_event)
        elif candidate.state not in (CandidateState.AVAILABLE, CandidateState.UNAVAILABLE):
            raise RuntimeError(
                "shadow premat candidate cannot be acquired from "
                f"{candidate.state.value}"
            )

        # The scheduler carried theoretical PREMAT memory/certificate charges up
        # to the deadline. Release those before the ordinary Main Stream weight
        # replaces the hypothetical side-stream resident tensor.
        if candidate.retained_counted or candidate.transient_counted:
            self._rollback_candidate_charge(candidate)
        self._release_allocator_certificate_candidate(candidate)
        candidate.owner = "main"
        candidate.critical_path_miss = False
        candidate.final_outcome = "SHADOW MAIN MATERIALISATION"
        candidate.launch_ns = candidate.launch_ns or time.perf_counter_ns()

        if candidate.state == CandidateState.UNAVAILABLE:
            self._transition(
                candidate,
                CandidateState.MATERIALISING,
                "shadow_materialising_on_critical_path",
                decision="shadow_main_claim",
                outcome="shadow_main_materialisation",
                reason="shadow_mode",
            )

        shadow_payload = self._record(
            "shadow_main_materialising",
            candidate=candidate,
            decision="shadow_main_claim",
            outcome="ordinary_deadline_materialisation",
            reason="shadow_mode",
            detail={"shadow_mode": True},
        )
        materialise_start = (
            torch.cuda.Event(enable_timing=True)
            if self._enable_gpu_timing_diagnostic
            else None
        )
        materialise_end = (
            torch.cuda.Event(enable_timing=True)
            if self._enable_gpu_timing_diagnostic
            else None
        )
        if materialise_start is not None:
            materialise_start.record(current_stream)
        try:
            candidate.tensor = self._materialize(
                candidate.family,
                candidate.layer_index,
            )
        except BaseException as error:
            self._record(
                "shadow_main_materialisation_failed",
                candidate=candidate,
                outcome="failure",
                reason=type(error).__name__,
                detail={"error": str(error)},
            )
            raise RuntimeError(
                "Shadow PREMAT Main Stream materialisation failed; "
                f"layer={candidate.layer_index}, family={candidate.family}, "
                f"mode={self._attention_mode}, memory={self._memory_summary()}"
            ) from error
        if materialise_start is not None and materialise_end is not None:
            materialise_end.record(current_stream)
            self._pending_timings.append(
                _PendingCudaTiming(
                    kind="main_stream_materialisation",
                    start_event=materialise_start,
                    end_event=materialise_end,
                    event_payload=shadow_payload,
                    pass_sequence=self._pass_sequence,
                    layer_index=candidate.layer_index,
                    family=candidate.family,
                    candidate=candidate,
                )
            )
        candidate.available_ns = time.perf_counter_ns()
        if candidate.state == CandidateState.MATERIALISING:
            self._transition(
                candidate,
                CandidateState.AVAILABLE,
                "shadow_main_available",
                outcome="ordinary_deadline_materialisation",
                reason="shadow_mode",
            )
        self._transition(
            candidate,
            CandidateState.CONSUMING,
            "shadow_main_consuming",
            outcome="shadow_main_materialisation",
            reason="shadow_mode",
        )
        self._aggregate["ordinary_deadline_materialisations"] += 1
        self._aggregate["shadow_main_materialisations"] += 1
        if candidate.tensor is None:
            raise RuntimeError("shadow PREMAT Main Stream materialisation produced no tensor")
        candidate.tensor.record_stream(current_stream)
        return candidate.tensor
    # ^^^ THOG

    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:
        # Checkpoint replay has no schedulable lookahead lifetime.  Recreate the
        # exact ordinary differentiable materialisation on its execution stream.
        # vvv THOG catch any unexpected in-pass bypass of an already-launched PREMAT candidate as duplicate work
        if self._active:
            candidate = self._candidates.get((int(layer_index), str(family)))
            if candidate is not None and candidate.real_premat_launched:
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
        # ^^^ THOG
        # vvv THOG preserve checkpoint/replay Main materialisation while adding profiling attribution
        # return self._materialize(family, layer_index)
        with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):
            return self._materialize(family, layer_index)
        # ^^^ THOG

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
        # vvv THOG distinguish useful real PREMAT work from launched-but-never-consumed work
        if candidate.real_premat_launched and not candidate.real_premat_consumed:
            candidate.real_premat_consumed = True
            self._aggregate["real_premat_materialisations_consumed"] += 1
            self._forensic_by_family[candidate.family]["consumed"] += 1
            if candidate.forensic_interval is not None:
                candidate.forensic_interval.consumed = True
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
        self._resolve_pending_forensic_passes()                                                                          # <<< THOG make synchronized progress/final reports carry resolved forensic evidence
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
        forensic_pass = (
            dict(self._latest_forensic_pass)
            if self._latest_forensic_pass is not None
            and int(self._latest_forensic_pass.get("pass_sequence", -1)) == int(self._pass_sequence)
            else None
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
            # vvv THOG expose whether expensive CUDA timestamp diagnostics are part of this run
            "enable_gpu_timing_diagnostic": self._enable_gpu_timing_diagnostic,
            "shadow_mode": self._shadow_mode,
            # ^^^ THOG
            "headroom_mode": (
                "stay_below_current_peak"
                if self._stay_below_current_peak
                else "stay_within_global_buffer"
            ),
            "current_layer_index": current_layer,
            "next_layer_index": next_layer,
            "lookahead_layer_limit": (1 if self._target_layer == 10 else self._target_layer),
            "target_scope": (
                "relative_layer_1_then_0"
                if self._target_layer == 10
                else f"relative_layer_{self._target_layer}"
            ),
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
            # vvv THOG cumulative and per-pass evidence map directly onto the six first-tranche hypotheses
            "forensic": self._forensic_report(),
            "forensic_pass": forensic_pass,
            # ^^^ THOG
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

    # vvv THOG shadow charges are theoretical scheduler state and must not be subtracted from real CUDA allocation counters
    def _physical_retained_bytes(self) -> int:
        return 0 if self._shadow_mode else self._retained_bytes
    # ^^^ THOG

    def _observe_memory(self) -> PrematMemoryObservation:
        if self._device is None:
            raise RuntimeError("premat CUDA device is not initialized")
        allocated = int(torch.cuda.memory_allocated(self._device))
        reserved = int(torch.cuda.memory_reserved(self._device))
        free, total = torch.cuda.mem_get_info(self._device)
        return PrematMemoryObservation(
            process_allocated_bytes=allocated,
            process_reserved_bytes=reserved,
            process_ordinary_peak_bytes=max(self._ordinary_peak_bytes, allocated - self._physical_retained_bytes()),
            device_free_bytes=int(free),
            device_total_bytes=int(total),
        )

    def _update_ordinary_peak(self) -> None:
        observation = self._observe_memory()
        ordinary_allocated = max(
            0,
            observation.process_allocated_bytes - self._physical_retained_bytes(),
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
        valid_target_layers = self._target_layer_indices()
        first_candidate = self._next_premat_candidate()
        target_layer_index = (
            first_candidate.layer_index if first_candidate is not None else None
        )
        target_offset = self._target_offset_for_candidate(first_candidate)
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
                "configured_target_layer": self._target_layer,
                "target_offset": target_offset,
                "target_layer_index": target_layer_index,
                "matrix_order": self._weight_matrix_target_order,
                "candidate_start_index": (
                    first_candidate.order_position
                    if first_candidate is not None
                    else None
                ),
            },
        )
        if first_candidate is None:
            self._record_advance_return(
                invocation=invocation,
                trigger=trigger,
                submitted=submitted,
                reason=("target_exhausted" if valid_target_layers else "target_out_of_range"),
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
            target_layer_index = candidate.layer_index
            target_offset = self._target_offset_for_candidate(candidate)
            if target_offset is None:
                raise RuntimeError("Premat selected a candidate outside its configured target sweep")
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
                "configured_target_layer": self._target_layer,
                "target_offset": target_offset,
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
                    # vvv THOG completion is required for dependency/query semantics; the start timestamp is diagnostic only
                    candidate.materialisation_start_event = (
                        torch.cuda.Event(enable_timing=True)
                        if self._enable_gpu_timing_diagnostic
                        else None
                    )
                    candidate.completion_event = torch.cuda.Event(
                        enable_timing=self._enable_gpu_timing_diagnostic
                    )
                    if candidate.materialisation_start_event is not None:
                        candidate.materialisation_start_event.record(self._stream)
                    # ^^^ THOG
                    # vvv THOG shadow mode records the normal side-stream completion event but deliberately launches no weight materialisation
                    if self._shadow_mode:
                        candidate.tensor = None
                        candidate.completion_event.record(self._stream)
                    else:
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
                                # vvv THOG preserve the original side-stream materialisation block while adding profiling attribution
                                # candidate.tensor = self._materialize(
                                #     candidate.family,
                                #     candidate.layer_index,
                                # )
                                with processing_operation_range("PREMAT", "materialize", family=candidate.family, layer_index=candidate.layer_index):
                                    candidate.tensor = self._materialize(
                                        candidate.family,
                                        candidate.layer_index,
                                    )
                                # ^^^ THOG
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
                    # ^^^ THOG
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
            # vvv THOG count real side work independently of admission and retain its immutable CUDA interval for overlap analysis
            if not self._shadow_mode:
                if candidate.real_premat_launched:
                    raise RuntimeError(
                        "PREMAT candidate launched real side materialisation twice: "
                        f"layer={candidate.layer_index}, family={candidate.family}"
                    )
                candidate.real_premat_launched = True
                self._aggregate["real_premat_materialisations_launched"] += 1
                self._forensic_by_family[candidate.family]["launched"] += 1
                if (
                    self._enable_gpu_timing_diagnostic
                    and candidate.materialisation_start_event is not None
                    and candidate.completion_event is not None
                ):
                    candidate.forensic_interval = _ForensicCandidateInterval(
                        pass_sequence=self._pass_sequence,
                        layer_index=candidate.layer_index,
                        family=candidate.family,
                        start_event=candidate.materialisation_start_event,
                        end_event=candidate.completion_event,
                    )
                    self._forensic_current_candidates.append(candidate.forensic_interval)
            # ^^^ THOG
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
                outcome=(
                    "submitted_to_shadow_premat_stream"
                    if self._shadow_mode
                    else "submitted_to_premat_stream"
                ),
                reason=decision.reason,
                detail=launch_detail,
            )
            # vvv THOG no materialisation timestamp bookkeeping exists in normal/default PREMAT execution
            if (
                self._enable_gpu_timing_diagnostic
                and candidate.materialisation_start_event is not None
                and candidate.completion_event is not None
            ):
                self._pending_timings.append(
                    _PendingCudaTiming(
                        kind="premat_materialisation",
                        start_event=candidate.materialisation_start_event,
                        end_event=candidate.completion_event,
                        event_payload=launch_payload,
                        pass_sequence=self._pass_sequence,
                        layer_index=candidate.layer_index,
                        family=candidate.family,
                        candidate=candidate,
                    )
                )
            # ^^^ THOG
            self._update_queue_aggregates()
            submitted.append(
                {
                    "layer_index": candidate.layer_index,
                    "family": candidate.family,
                    "order_position": candidate.order_position,
                }
            )

    def _target_offsets(self) -> Tuple[int, ...]:
        # Target 10 is a mnemonic for the ordered sweep +1 then +0, not ten
        # layers of lookahead.  It is deliberately right-to-left only.
        return (1, 0) if self._target_layer == 10 else (self._target_layer,)

    def _target_layer_indices(self) -> Tuple[int, ...]:
        if self._position < 0:
            return ()
        result: List[int] = []
        for offset in self._target_offsets():
            position = self._position + offset
            if 0 <= position < len(self._layer_indices):
                layer_index = int(self._layer_indices[position])
                if layer_index not in result:
                    result.append(layer_index)
        return tuple(result)

    def _target_offset_for_candidate(
        self,
        candidate: Optional[_Candidate],
    ) -> Optional[int]:
        if candidate is None or self._position < 0:
            return None
        for offset in self._target_offsets():
            position = self._position + offset
            if (
                0 <= position < len(self._layer_indices)
                and int(self._layer_indices[position]) == int(candidate.layer_index)
            ):
                return int(offset)
        return None

    def _next_premat_candidate(
        self,
        *,
        excluded_sequences: Optional[set[int]] = None,
    ) -> Optional[_Candidate]:
        excluded = excluded_sequences or set()
        ordered = sorted(
            self._candidates.values(),
            key=lambda candidate: candidate.sequence,
        )
        for target_layer_index in self._target_layer_indices():
            candidate = next(
                (
                    item
                    for item in ordered
                    if item.state == CandidateState.UNAVAILABLE
                    and item.layer_index == target_layer_index
                    and item.sequence not in excluded
                ),
                None,
            )
            if candidate is not None:
                return candidate
        return None

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
            observation.process_allocated_bytes - self._physical_retained_bytes(),
        )
        charged_process_bytes = max(
            observation.process_allocated_bytes,
            ordinary_process_bytes + self._cumulative_charged_bytes(),
        )
        ordinary_device_bytes = max(
            0,
            observation.device_used_bytes - self._physical_retained_bytes(),
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
        candidate_target_offset = self._target_offset_for_candidate(candidate)
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
            "target_offset": (
                self._target_layer
                if candidate_target_offset is None
                else candidate_target_offset
            ),
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
        # Completed sampled passes are published by _capture_completed_live_report
        # only after all CUDA timings for that pass have resolved.  _record stays
        # purely observational and never introduces a host/GPU synchronisation.
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

    def _capture_completed_live_report(self) -> None:
        reporter = self._live_reporter
        if reporter is None:
            return
        capture_enabled = self._live_capture_enabled
        if capture_enabled is not None and not capture_enabled(self._pass_sequence):
            return
        # report() is non-blocking; unresolved CUDA timings remain in
        # _pending_timings and this snapshot is held privately until they resolve.
        snapshot = self.report()
        self._pending_completed_live_reports[self._pass_sequence] = snapshot
        self._publish_completed_live_reports_if_ready()

    @staticmethod
    def _refresh_report_derived_aggregates(snapshot: Dict[str, object]) -> None:
        aggregate = snapshot.get("aggregate")
        if not isinstance(aggregate, dict):
            return
        materialisation_ms = float(aggregate.get("premat_materialisation_ms_total", 0.0))
        wait_ms = float(aggregate.get("main_stream_wait_ms_total", 0.0))
        aggregate["premat_hidden_ms_estimate"] = max(0.0, materialisation_ms - wait_ms)
        pass_ms = float(aggregate.get("captured_pass_host_ms_total", 0.0))
        aggregate["premat_stream_busy_fraction_estimate"] = (
            min(1.0, materialisation_ms / pass_ms) if pass_ms > 0.0 else None
        )

    def _update_pending_live_report(
        self,
        timing: _PendingCudaTiming,
        aggregate_deltas: Mapping[str, float | int],
    ) -> None:
        snapshot = self._pending_completed_live_reports.get(int(timing.pass_sequence))
        if snapshot is None:
            return
        aggregate = snapshot.get("aggregate")
        if isinstance(aggregate, dict):
            for name, delta in aggregate_deltas.items():
                aggregate[name] = aggregate.get(name, 0) + delta
        payload = timing.event_payload
        payload_sequence = (
            int(payload.get("sequence"))
            if isinstance(payload, Mapping) and payload.get("sequence") is not None
            else None
        )
        events = snapshot.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                if payload_sequence is not None and int(event.get("sequence", -1)) == payload_sequence:
                    event.update(dict(payload))
                if (
                    timing.kind == "main_stream_wait"
                    and int(event.get("layer_index", -1)) == int(timing.layer_index or -1)
                    and str(event.get("family", "")) == str(timing.family or "")
                    and str(event.get("event", "")) == "consumed"
                    and isinstance(payload, Mapping)
                ):
                    event["critical_path_miss"] = bool(payload.get("critical_path_miss", False))
                    event["final_outcome"] = payload.get("final_outcome")
        candidates = snapshot.get("candidates")
        if timing.kind == "main_stream_wait" and isinstance(candidates, list) and isinstance(payload, Mapping):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if (
                    int(candidate.get("layer_index", -1)) == int(timing.layer_index or -1)
                    and str(candidate.get("family", "")) == str(timing.family or "")
                ):
                    candidate["critical_path_miss"] = bool(payload.get("critical_path_miss", False))
                    candidate["final_outcome"] = payload.get("final_outcome")
        self._refresh_report_derived_aggregates(snapshot)

    def _publish_completed_live_reports_if_ready(self) -> None:
        reporter = self._live_reporter
        if reporter is None or not self._pending_completed_live_reports:
            return
        pending_passes = {
            int(timing.pass_sequence)
            for timing in self._pending_timings
            if int(timing.pass_sequence) > 0
        }
        pending_passes.update(                                                                                           # <<< THOG a sampled Instra pass waits for overlap forensics as well as legacy timing classification
            int(pending.pass_sequence)
            for pending in self._pending_forensic_passes
            if int(pending.pass_sequence) > 0
        )
        for pass_sequence in sorted(tuple(self._pending_completed_live_reports)):
            if pass_sequence in pending_passes:
                continue
            snapshot = self._pending_completed_live_reports.pop(pass_sequence)
            self._refresh_report_derived_aggregates(snapshot)
            try:
                reporter(snapshot)
            except Exception as error:  # pragma: no cover - sink failures are environment-specific
                self._live_publish_error = f"{type(error).__name__}: {error}"
                self._live_reporter = None
                self._pending_completed_live_reports.clear()
                return

    def _apply_gpu_wait_classification(
        self,
        timing: _PendingCudaTiming,
        *,
        dependency_delta_ms: float,
        marker_elapsed_ms: float,
    ) -> Dict[str, float | int]:
        payload = timing.event_payload
        # vvv THOG keep GPU hit classification operational when PREMAT telemetry is disabled
        # if payload is None:
        #     raise RuntimeError("main-stream wait timing lost its event payload")
        if payload is None:
            payload = {}
        # ^^^ THOG
        candidate = timing.candidate
        gpu_wait_required = dependency_delta_ms > 0.0
        effective_wait_ms = max(0.0, dependency_delta_ms)
        final_outcome = "PARTIAL HIT" if gpu_wait_required else "FULL HIT"
        payload.update(
            {
                "outcome": "waited_for_premat" if gpu_wait_required else "fully_hidden",
                "reason": (
                    "premat_incomplete_at_main_stream_dependency"
                    if gpu_wait_required
                    else "premat_complete_before_main_stream_dependency"
                ),
                "critical_path_miss": gpu_wait_required,
                "final_outcome": final_outcome,
                "gpu_wait_classification_pending": False,
                "gpu_wait_required": gpu_wait_required,
                "gpu_dependency_delta_ms": dependency_delta_ms,
                "premat_lead_ms_at_main_stream_dependency": max(0.0, -dependency_delta_ms),
                "wait_marker_elapsed_ms": marker_elapsed_ms,
                "wait_ms": effective_wait_ms,
                "cuda_elapsed_ms": effective_wait_ms,
            }
        )
        if candidate is not None:
            candidate.critical_path_miss = gpu_wait_required
            candidate.final_outcome = final_outcome
        # consumed() normally runs on the host before the GPU reaches this wait.
        # Correct that already-recorded event once the GPU timeline is known.
        for event in self._events:
            if (
                int(event.get("pass_sequence", -1)) == int(timing.pass_sequence)
                and int(event.get("layer_index", -1)) == int(timing.layer_index or -1)
                and str(event.get("family", "")) == str(timing.family or "")
                and str(event.get("event", "")) == "consumed"
            ):
                event["critical_path_miss"] = gpu_wait_required
                event["final_outcome"] = final_outcome
        deltas: Dict[str, float | int] = {
            "main_stream_wait_ms_total": effective_wait_ms,
        }
        if gpu_wait_required:
            deltas["waited_hits"] = 1
        else:
            # available_hits intentionally remains the stricter host-observed
            # counter. fully_hidden_hits is the corrected GPU-effective FULL count.
            deltas["fully_hidden_hits"] = 1
        return deltas

    def _resolve_pending_timings(self) -> None:
        remaining: List[_PendingCudaTiming] = []
        for timing in self._pending_timings:
            try:
                if not timing.end_event.query():
                    remaining.append(timing)
                    continue
                marker_elapsed_ms = max(
                    0.0,
                    float(timing.start_event.elapsed_time(timing.end_event)),
                )
                if timing.kind == "main_stream_wait":
                    if timing.dependency_event is None:
                        raise RuntimeError("main-stream wait timing has no Premat dependency event")
                    # GPU timestamp ordering answers the question that host-time
                    # completion_event.query() cannot: had PREMAT completed by the
                    # instant the Main Stream actually reached its dependency?
                    dependency_delta_ms = float(
                        timing.start_event.elapsed_time(timing.dependency_event)
                    )
            except RuntimeError:
                remaining.append(timing)
                continue

            aggregate_deltas: Dict[str, float | int]
            if timing.kind == "premat_materialisation":
                aggregate_deltas = {"premat_materialisation_ms_total": marker_elapsed_ms}
                if timing.event_payload is not None:
                    timing.event_payload["materialisation_ms"] = marker_elapsed_ms
                    timing.event_payload["cuda_elapsed_ms"] = marker_elapsed_ms
            elif timing.kind == "main_stream_wait":
                aggregate_deltas = self._apply_gpu_wait_classification(
                    timing,
                    dependency_delta_ms=dependency_delta_ms,
                    marker_elapsed_ms=marker_elapsed_ms,
                )
            elif timing.kind == "main_stream_materialisation":
                aggregate_deltas = {"main_stream_materialisation_ms_total": marker_elapsed_ms}
                if timing.event_payload is not None:
                    timing.event_payload["main_stream_materialisation_ms"] = marker_elapsed_ms
                    timing.event_payload["cuda_elapsed_ms"] = marker_elapsed_ms
            else:
                raise RuntimeError(f"unknown premat CUDA timing kind: {timing.kind}")
            for name, delta in aggregate_deltas.items():
                self._aggregate[name] += delta
            self._update_pending_live_report(timing, aggregate_deltas)
        self._pending_timings = remaining
        self._publish_completed_live_reports_if_ready()

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
