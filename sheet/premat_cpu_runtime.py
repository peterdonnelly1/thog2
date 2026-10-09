# vvv THOG CPU provider shares PREMAT targets, admission, matrix-use records and Main Stream binding
from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
import math
import queue
import threading
import time
import uuid
import weakref

import torch

from .depth_numerical_policy import PrematBindingContext, effective_depth_policy
from .premat import CandidateEnvelope, CandidateState, PrematRuntime, decide_candidate_admission
from .premat_cpu import CpuPreparationProvider, FAMILY_NAMES, GemmPredictor
from .premat_cpu_config import cpu_config_dict
from .premat_processing import processing_operation_range


@dataclass
class CpuUpload:
    identity: str
    metadata: dict
    nbytes: int
    tensor: object = None
    pinned: object = None
    completion: object = None
    retirement: object = None
    consumed: bool = False
    late: bool = False
    failed: bool = False
    certificate: object = None


class CpuStorageLease:
    def __init__(self, runtime, upload, tensor, stream):
        self.runtime = weakref.ref(runtime)
        self.identity, self.tensor, self.stream = upload.identity, tensor, stream

    def __del__(self):
        runtime = self.runtime()
        if runtime is not None:
            runtime._retire_lease(self.identity, self.tensor, self.stream)


class CpuReplayContext(AbstractContextManager):
    def __init__(self, runtime, indices, microstep):
        self.runtime, self.indices, self.microstep = runtime, indices, microstep

    def __enter__(self):
        self.context = self.runtime.replay_scope(self.indices, self.microstep)
        return self.context.__enter__()

    def __exit__(self, *args):
        return self.context.__exit__(*args)


@dataclass
class ReplayReference:
    device: object
    dtype: object
    shape: tuple

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype == torch.float32 else 2


class CpuPrematRuntime(PrematRuntime):
    def __init__(self, *, trajectory, cpu_configuration, **kwargs):
        self.cpu_configuration = cpu_config_dict(cpu_configuration)
        transfer = self.cpu_configuration["premat_cpu_transfer_timing"]
        kwargs["timing"] = "previous_gemm_leading_edge" if transfer == "previous_gemm_leading_edge" else "as_the_code_flies"
        super().__init__(**kwargs)
        self.trajectory = trajectory
        self.cpu_lock, self.event_lock = threading.RLock(), threading.Lock()
        self.cpu_events = deque(maxlen=max(4096, trajectory.config.n_layer * 256))
        self.cpu_uploads, self.eval_leases = {}, {}
        self.cpu_provider = self.cpu_policy = self.cpu_snapshot = None
        self.cpu_sources, self.source_event = deque(), None
        self.source_thread, self.source_paused, self.source_error = None, False, None
        self.upload_queue = queue.Queue(maxsize=max(16, trajectory.config.n_layer * 6))
        self.upload_thread, self.cpu_closed = None, False
        self.cpu_phase = "original_forward"
        self.optimizer_step = self.micro_step = self.replay_sequence = 0
        self.cpu_cap = self.cpu_peak = self.pin_peak = 0
        self.cpu_predictor = GemmPredictor()
        self.prediction_context, self.clock, self.origin_event = None, {}, None
        self.prediction_events = deque(maxlen=4096)
        self.prediction_timer = None
        self.leading_opportunities = {}
        self.pin_in_progress_bytes = self.pin_in_progress_peak = 0
        self.active_source_job = None
        self.last_memory_state = None
        self.control_counts, self.control_ns = {}, {}

    def _cpu_record(self, event):
        fields = {"schema_version": 5, "host_time_ns": time.perf_counter_ns(), "optimizer_step": self.optimizer_step, "micro_step": self.micro_step, "phase": self.cpu_phase, "replay_invocation": self.replay_sequence if self.cpu_phase == "checkpoint_recompute" else None, **event}
        with self.event_lock:
            self.cpu_events.append(fields)

    @contextmanager
    def _control(self, name):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.control_counts[name] = self.control_counts.get(name, 0) + 1
            self.control_ns[name] = self.control_ns.get(name, 0) + time.perf_counter_ns() - started

    def begin(self, layer_indices, *, reference):
        with self.cpu_lock, self._control("pass_begin"):
            super().begin(layer_indices, reference=reference)
            self.cpu_policy = effective_depth_policy(self.trajectory)
            self._dtype_bytes = 4 if self.cpu_policy.output_dtype == "float32" else 2
            for layer in self._layer_indices:
                self._ensure_layer(layer)
            selected = tuple(family for family in self._families() if self._target_families is None or family in self._target_families)
            requested = self.cpu_configuration["premat_cpu_staging_limit_mb"]
            self.cpu_cap = int(requested * 1024 ** 2) if requested else 2 * sum(self._candidate_envelope(family).retained_bytes for family in selected)
            structure = (self._batch_size, self._sequence_length, self.cpu_policy.policy_id)
            if self.prediction_context is not None and self.prediction_context != structure:
                self.cpu_predictor.reset("shape_or_numerical_policy_changed")
            self.prediction_context = structure
            if self._shadow_mode:
                return
            if self.cpu_provider is None:
                self.cpu_provider = CpuPreparationProvider(self.trajectory, families=selected, preparation=self.cpu_configuration["premat_cpu_preparation"], batch_size=self.cpu_configuration["premat_cpu_layer_batch_size"], workers=self.cpu_configuration["premat_cpu_workers"], threads=self.cpu_configuration["premat_cpu_threads_per_worker"], record=self._cpu_record, wake=self._wake)
                try:
                    self.cpu_provider.start()
                except Exception as error:
                    self.source_error = f"CPU worker startup failed: {error}"
                    self._cpu_record({"event": "cpu_worker_start_failed", "error": self.source_error})
                    self.cpu_provider.close()
                    self.cpu_provider = None
                    return
            snapshot, fresh = self.cpu_provider.begin_snapshot(self.cpu_policy)
            self.cpu_snapshot = snapshot
            self.source_paused = False
            if fresh:
                self.source_error = None
                self.cpu_sources.clear()
                names = tuple(dict.fromkeys(name for family in selected for name in FAMILY_NAMES[family]))
                sources = [("depth_basis", self.trajectory.depth_basis)] + [(name, self.trajectory.coefficients[name]) for name in names]
                gate = torch.cuda.Event()
                gate.record(torch.cuda.current_stream(device=self._device))
                for name, source in sources:
                    self.cpu_sources.append({"snapshot_id": snapshot.snapshot_id, "name": name, "source": source.detach(), "offset": 0, "host": None, "gate": gate})
            self._start_sources()
            if self.cpu_configuration["premat_cpu_transfer_timing"] == "predicted_gemm_start":
                self._start_clock()
            self._cpu_record({"event": "cpu_pass_begin", "snapshot_id": snapshot.snapshot_id, "snapshot_reused": not fresh, "pass_sequence": self._pass_sequence, "staging_limit_bytes": self.cpu_cap, "numerical_policy": vars(self.cpu_policy)})

    def _start_sources(self):
        if self.cpu_sources and (self.source_thread is None or not self.source_thread.is_alive()):
            self.source_thread = threading.Thread(target=self._download_sources, name="premat-snapshot-copy", daemon=True)
            self.source_thread.start()

    def _download_sources(self):
        try:
            while not self.cpu_closed:
                with self.cpu_lock:
                    if self.source_paused or not self.cpu_sources:
                        return
                    job = self.cpu_sources[0]
                    self.active_source_job = job
                if job["host"] is None:
                    job["host"] = torch.empty_like(job["source"], device="cpu", pin_memory=True)
                with self.cpu_lock:
                    if self.source_paused or not self.cpu_sources or self.cpu_sources[0] is not job:
                        return
                    source, host = job["source"].reshape(-1), job["host"].reshape(-1)
                    end = min(source.numel(), job["offset"] + max(1, 16 * 1024 ** 2 // source.element_size()))
                    with processing_operation_range(owner="COPY", operation="snapshot_d2h", snapshot_id=job["snapshot_id"], source_name=job["name"], source_offset=job["offset"]), torch.cuda.stream(self._stream), torch.no_grad():
                        self._stream.wait_event(job["gate"])
                        host[job["offset"]:end].copy_(source[job["offset"]:end], non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(self._stream)
                        self.source_event = event
                    self._cpu_record({"event": "snapshot_d2h_submitted", "snapshot_id": job["snapshot_id"], "source_name": job["name"], "source_offset": job["offset"], "bytes": (end - job["offset"]) * source.element_size(), "direction": "D2H"})
                # This wait belongs to the coordinator, never to the readiness check.
                event.synchronize()
                with self.cpu_lock:
                    job["offset"] = end
                    complete = end == source.numel()
                    if complete and self.cpu_sources and self.cpu_sources[0] is job:
                        self.cpu_sources.popleft()
                if complete:
                    self.cpu_provider.publish_source(job["snapshot_id"], job["name"], job["host"])
                    self._cpu_record({"event": "snapshot_source_ready", "snapshot_id": job["snapshot_id"], "source_name": job["name"]})
                self._wake()
        except Exception as error:
            self.source_error = f"{type(error).__name__}: {error}"
            self._cpu_record({"event": "snapshot_download_failed", "error": self.source_error})
        finally:
            with self.cpu_lock:
                self.active_source_job = None
                self.source_thread = None
                if not self.cpu_closed and not self.source_paused and self.cpu_sources and not self.source_error:
                    self._start_sources()

    def before_optimizer_step(self):
        with self.cpu_lock:
            self.source_paused = True
            if self.source_event is not None:
                torch.cuda.current_stream(device=self._device).wait_event(self.source_event)
            self._cpu_record({"event": "snapshot_mutation_guard", "snapshot_id": None if self.cpu_snapshot is None else self.cpu_snapshot.snapshot_id})

    def after_optimizer_step(self):
        self._cpu_record({"event": "optimizer_boundary", "snapshot_id": None if self.cpu_snapshot is None else self.cpu_snapshot.snapshot_id})

    def _candidate_envelope(self, family):
        original = super()._candidate_envelope(family)
        names = FAMILY_NAMES[family]
        size = sum(self.trajectory.coefficients[name].shape[0] * self.trajectory.coefficients[name].shape[1] for name in names) * self._dtype_bytes
        fallback = (2 if len(names) > 1 else 1) * size
        return CandidateEnvelope(size, size + fallback + original.foreground_overlap_bytes, original.foreground_overlap_bytes)

    def _physical_retained_bytes(self):
        return sum(upload.nbytes for upload in self.cpu_uploads.values() if upload.tensor is not None or upload.consumed)

    def _sync_charge(self):
        self._retained_bytes = sum(upload.nbytes for upload in self.cpu_uploads.values())
        self._transient_bytes = 0
        self.cpu_peak = max(self.cpu_peak, self._retained_bytes)
        self.pin_peak = max(self.pin_peak, sum(upload.nbytes for upload in self.cpu_uploads.values() if upload.pinned is not None))
        states = self._memory_states()
        signature = tuple(states.values())
        if signature != self.last_memory_state:
            self.last_memory_state = signature
            self._cpu_record({"event": "cpu_memory", **states, "staging_limit_bytes": self.cpu_cap, "staging_peak_bytes": self.cpu_peak})

    def _memory_states(self):
        states = {"pending_upload_bytes": 0, "copying_bytes": 0, "available_bytes": 0, "autograd_retained_bytes": 0}
        for upload in self.cpu_uploads.values():
            key = "autograd_retained_bytes" if upload.consumed else "pending_upload_bytes" if upload.completion is None else "available_bytes" if upload.completion.query() else "copying_bytes"
            states[key] += upload.nbytes
        states.update(staging_bytes=self._retained_bytes, pinned_upload_bytes=sum(item.nbytes for item in self.cpu_uploads.values() if item.pinned is not None), pin_preparation_in_progress_bytes=self.pin_in_progress_bytes)
        jobs = list(self.cpu_sources)
        if self.active_source_job is not None and all(job is not self.active_source_job for job in jobs):
            jobs.append(self.active_source_job)
        states["pinned_snapshot_bytes"] = sum(job["host"].numel()*job["host"].element_size() for job in jobs if job["host"] is not None)
        return states

    def _wake(self):
        with self.cpu_lock:
            if self._active and not self.cpu_closed:
                self._advance(trigger="cpu_notification")

    def layer_start(self, layer_index):
        with self.cpu_lock, self._control("layer_start"):
            self._require_active()
            position = self._layer_indices.index(int(layer_index))
            if position < self._position:
                raise RuntimeError("CPU PREMAT layer order changed within a pass")
            self._position = position
            self._update_ordinary_peak()
            self._record("layer_start", layer=layer_index, outcome="reconsider")
            self._advance(trigger="layer_start")

    def event(self, name, *, layer_index):
        with self.cpu_lock:
            self._record(name, layer=layer_index, outcome="reconsider")
            self._advance(trigger=name)

    def _advance(self, *, trigger, eligible_keys=None, gate_event=None):
        with self.cpu_lock, self._control("admission"):
            if not self._active or self._position < 0:
                return
            self._refresh_available()
            if self._shadow_mode:
                self._shadow_admission(trigger, eligible_keys)
                return
            if self.cpu_provider is None:
                return
            timing = self.cpu_configuration["premat_cpu_transfer_timing"]
            if timing == "previous_gemm_leading_edge":
                if eligible_keys is not None:
                    for key in eligible_keys:
                        self.leading_opportunities[tuple(key)] = gate_event
                else:
                    eligible_keys = tuple(key for key in self.leading_opportunities if key in self._candidates and self._candidates[key].state == CandidateState.UNAVAILABLE)
                    if not eligible_keys:
                        return
                    gate_event = self.leading_opportunities[eligible_keys[0]]
            if timing == "demand_driven" and not trigger.startswith("demand"):
                return
            keys = eligible_keys
            if timing in ("as_soon_as_ready", "predicted_gemm_start") and keys is None:
                keys = tuple((layer, family) for layer in self._layer_indices[self._position:] for family in self._families())
            predictive_fallback = False
            while True:
                candidate = self._next_premat_candidate(eligible_keys=keys)
                if candidate is None:
                    return
                if self.cpu_configuration["premat_cpu_preparation"] == "scheduled":
                    self.cpu_provider.request(candidate.family, candidate.layer_index)
                matrix, metadata = self.cpu_provider.lookup(candidate.family, candidate.layer_index, self.cpu_policy)
                candidate.cpu_metadata = {**metadata, "snapshot_id": self.cpu_snapshot.snapshot_id, "numerical_policy": self.cpu_policy.policy_id, "phase": self.cpu_phase, "replay_invocation": self.replay_sequence if self.cpu_phase == "checkpoint_recompute" else None}
                if matrix is None:
                    return
                if timing == "predicted_gemm_start":
                    prediction = self.cpu_predictor.predict(self._prediction_key(candidate.family, candidate.layer_index), self.clock.get("host_origin_ns", time.perf_counter_ns()), self.cpu_configuration["premat_cpu_transfer_lead_ms"], self.clock.get("uncertainty_ms"))
                    candidate.cpu_metadata.update(prediction)
                    due = prediction.get("intended_upload_submission_ns")
                    if due is not None and due > time.perf_counter_ns():
                        if self.prediction_timer is None:
                            self.prediction_timer = threading.Timer((due - time.perf_counter_ns()) / 1e9, self._prediction_wake)
                            self.prediction_timer.daemon = True
                            self.prediction_timer.start()
                        return
                    if not prediction["prediction_available"] and eligible_keys is None and not predictive_fallback:
                        predictive_fallback = True
                        keys = tuple((layer, family) for layer in self._target_layer_indices() for family in self._families())
                        continue
                if self._retained_bytes + candidate.envelope.retained_bytes > self.cpu_cap:
                    candidate.admission_reason = "cpu_staging_limit"
                    self._record("admission_considered", candidate=candidate, decision="defer", reason=candidate.admission_reason)
                    return
                observation = self._observation_with_cumulative_charge(self._observe_memory())
                decision = decide_candidate_admission(observation=observation, envelope=candidate.envelope, stay_below_current_peak=self._stay_below_current_peak, gpu_memory_buffer_bytes=self._buffer_bytes)
                rescued = None
                fallback_reserve = candidate.envelope.materialisation_peak_bytes - candidate.envelope.retained_bytes
                if not decision.admitted and decision.reason == "global_device_buffer" and self._allocator_aware_admission == "cautious" and observation.device_free_bytes - self._buffer_bytes >= fallback_reserve:
                    destination = CandidateEnvelope(candidate.envelope.retained_bytes, candidate.envelope.retained_bytes, 0)
                    decision, rescued = self._cautious_allocator_aware_rescue(observation=observation, envelope=destination, original_decision=decision)
                candidate.admission_reason = decision.reason
                self._record("admission_considered", candidate=candidate, decision="admit" if decision.admitted else "defer", reason=decision.reason)
                if not decision.admitted:
                    return
                if self._allocator_aware_admission == "cautious":
                    reservation = self._reserve_allocator_certificate(request_bytes=candidate.envelope.retained_bytes, required=rescued is not None)
                    if reservation.get("tracked"):
                        for key, value in (("allocator_certificate_generation", "generation"), ("allocator_certificate_pool", "pool"), ("allocator_certificate_block_id", "block_id"), ("allocator_certificate_charge_bytes", "charge_bytes")):
                            setattr(candidate, key, reservation[value])
                self._submit_upload(candidate, matrix, gate_event)


    def _shadow_admission(self, trigger, eligible_keys):
        shadow_charge = sum(item.envelope.retained_bytes for item in self._candidates.values() if item.owner == "shadow" and item.state == CandidateState.MATERIALISING)
        while True:
            candidate = self._next_premat_candidate(eligible_keys=eligible_keys)
            if candidate is None or shadow_charge + candidate.envelope.retained_bytes > self.cpu_cap:
                return
            observation = self._observe_memory()
            decision = decide_candidate_admission(observation=observation, envelope=candidate.envelope, stay_below_current_peak=self._stay_below_current_peak, gpu_memory_buffer_bytes=self._buffer_bytes)
            self._record("shadow_admission", candidate=candidate, decision="admit" if decision.admitted else "defer", reason=decision.reason)
            if not decision.admitted:
                return
            candidate.owner, candidate.state = "shadow", CandidateState.MATERIALISING
            shadow_charge += candidate.envelope.retained_bytes

    def _prediction_wake(self):
        with self.cpu_lock:
            self.prediction_timer = None
        self._wake()

    def _metadata(self, upload):
        return {key: value for key, value in upload.metadata.items() if not key.endswith("_event")}

    def _submit_upload(self, candidate, matrix, gate_event):
        metadata = {**candidate.cpu_metadata, "matrix_use_id": self._use_id(candidate), "optimizer_step": self.optimizer_step, "micro_step": self.micro_step, "phase": self.cpu_phase, "layer_index": candidate.layer_index, "family": candidate.family, "submission_ns": time.perf_counter_ns()}
        upload = CpuUpload(uuid.uuid4().hex, metadata, candidate.envelope.retained_bytes, certificate=candidate)
        upload.metadata["upload_id"] = upload.identity
        self.cpu_uploads[upload.identity] = upload
        candidate.cpu_upload_id = upload.identity
        candidate.cpu_metadata.update({"upload_id": upload.identity, "upload_pending": True})
        candidate.owner, candidate.state = "cpu_upload", CandidateState.MATERIALISING
        candidate.launch_ns = upload.metadata["submission_ns"]
        self._sync_charge()
        self._aggregate["admitted"] += 1
        self._record("premat_submission", candidate=candidate, outcome="copy_queued")
        self._cpu_record({**metadata, "event": "upload_queued", "bytes": upload.nbytes})
        try:
            self.upload_queue.put_nowait((upload, matrix, gate_event))
        except queue.Full:
            upload.failed = True
            candidate.cpu_metadata["fallback_reason"] = "cpu_transfer_queue_full"
            self._cpu_record({"event": "upload_failed", "upload_id": upload.identity, "fallback_reason": "cpu_transfer_queue_full"})
            self._refresh_available()
            return
        if self.upload_thread is None or not self.upload_thread.is_alive():
            self.upload_thread = threading.Thread(target=self._upload_coordinator, name="premat-upload-copy", daemon=True)
            self.upload_thread.start()

    def _upload_coordinator(self):
        while not self.cpu_closed:
            try:
                upload, matrix, gate = self.upload_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.perf_counter_ns()
            try:
                with self.cpu_lock:
                    self.pin_in_progress_bytes += upload.nbytes
                    self.pin_in_progress_peak = max(self.pin_in_progress_peak, self.pin_in_progress_bytes)
                pinned = matrix.pin_memory()
                with self.cpu_lock:
                    if upload.identity not in self.cpu_uploads or upload.late or upload.failed:
                        continue
                    upload.pinned = pinned
                    with processing_operation_range(owner="COPY", operation="matrix_h2d", family=upload.metadata["family"], layer_index=upload.metadata["layer_index"], snapshot_id=upload.metadata["snapshot_id"], cpu_task_id=upload.metadata.get("cpu_task_id"), cpu_matrix_id=upload.metadata.get("cpu_matrix_id"), upload_id=upload.identity, matrix_use_id=upload.metadata["matrix_use_id"]), torch.cuda.stream(self._stream), torch.no_grad():
                        if gate is not None:
                            self._stream.wait_event(gate)
                        upload.tensor = torch.empty_like(pinned, device=self._device)
                        timed = self.cpu_configuration["premat_cpu_transfer_timing"] == "predicted_gemm_start"
                        start = torch.cuda.Event(enable_timing=timed)
                        start.record(self._stream)
                        upload.tensor.copy_(pinned, non_blocking=True)
                        upload.completion = torch.cuda.Event(enable_timing=timed)
                        upload.completion.record(self._stream)
                    upload.metadata.update({"copy_submission_ns": time.perf_counter_ns(), "pin_preparation_ms": (time.perf_counter_ns() - started) / 1e6, "copy_start_event": start, "origin_event": self.origin_event, "origin_clock": self.clock})
                    self._cpu_record({**self._metadata(upload), "event": "upload_submitted", "direction": "H2D", "bytes": upload.nbytes})
                    self._sync_charge()
            except Exception as error:
                with self.cpu_lock:
                    upload.failed = True
                    upload.metadata["fallback_reason"] = "cpu_upload_failed"
                    self._cpu_record({"event": "upload_failed", "upload_id": upload.identity, "error": f"{type(error).__name__}: {error}"})
                    if upload.tensor is not None and upload.completion is None:
                        upload.completion = torch.cuda.Event()
                        upload.completion.record(self._stream)
                    self._refresh_available()
            finally:
                with self.cpu_lock:
                    self.pin_in_progress_bytes -= upload.nbytes
                self.upload_queue.task_done()
            self._wake()

    def _refresh_available(self):
        self._collect_predictions()
        for identity, upload in tuple(self.cpu_uploads.items()):
            complete = upload.completion is not None and upload.completion.query()
            if complete:
                upload.pinned = None
                if "copy_complete_observed_ns" not in upload.metadata:
                    upload.metadata["copy_complete_observed_ns"] = time.perf_counter_ns()
                    self._cpu_record({**self._metadata(upload), "event": "upload_complete_observed", "arrival_time_basis": "host_completion_observation_upper_bound"})
                    if self.cpu_configuration["premat_cpu_transfer_timing"] == "predicted_gemm_start":
                        self.cpu_predictor.uploads.append(upload.metadata["copy_start_event"].elapsed_time(upload.completion))
                        origin, clock = upload.metadata.get("origin_event"), upload.metadata.get("origin_clock", {})
                        if origin is not None and clock.get("uncertainty_ms") is not None and clock["uncertainty_ms"] <= 5 and origin.query():
                            arrival = clock["host_origin_ns"] + int(origin.elapsed_time(upload.completion) * 1e6)
                            upload.metadata.update(actual_gpu_available_ns=arrival, clock_uncertainty_ms=clock["uncertainty_ms"])
                            intended = upload.metadata.get("intended_gpu_available_ns")
                            self._cpu_record({**self._metadata(upload), "event": "upload_arrival_observed", "availability_error_ms": None if intended is None else (arrival-intended)/1e6, "arrival_time_basis": "qualified_cuda_event_clock"})
            safe_release = upload.retirement is not None and upload.retirement.query()
            if (upload.late or upload.failed) and (complete or upload.completion is None):
                safe_release = True
            if safe_release:
                self._cpu_record({**self._metadata(upload), "event": "gpu_storage_released", "released_bytes": upload.nbytes})
                if upload.certificate is not None:
                    self._release_allocator_certificate_candidate(upload.certificate)
                del self.cpu_uploads[identity]
                continue
            if complete and not upload.consumed and not upload.late and not upload.failed:
                candidate = next((item for item in self._candidates.values() if getattr(item, "cpu_upload_id", None) == identity), None)
                if candidate is not None and candidate.state == CandidateState.MATERIALISING:
                    candidate.tensor, candidate.completion_event = upload.tensor, upload.completion
                    candidate.available_ns = upload.metadata["copy_complete_observed_ns"]
                    candidate.cpu_metadata["upload_pending"] = False
                    self._transition(candidate, CandidateState.AVAILABLE, "available", outcome="gpu_available")
        self._sync_charge()

    def _use_id(self, candidate):
        return f"{self.optimizer_step}:{self.micro_step}:{self.cpu_phase}:{self.replay_sequence}:{self._pass_sequence}:{candidate.layer_index}:{candidate.family}"

    def acquire(self, family, layer_index):
        with self.cpu_lock, self._control("readiness"):
            self._require_active()
            candidate = self._candidates[(int(layer_index), str(family))]
            targeted = self._target_families is None or family in self._target_families
            candidate.deadline_ns = time.perf_counter_ns()
            if targeted and self.cpu_provider is not None and self.cpu_configuration["premat_cpu_preparation"] == "demand_driven":
                self.cpu_provider.request(family, layer_index)
            if self.cpu_configuration["premat_cpu_transfer_timing"] == "demand_driven":
                self._advance(trigger="demand_readiness", eligible_keys=((layer_index, family),))
            self._refresh_available()
            matrix, current = (None, {}) if self.cpu_provider is None else self.cpu_provider.lookup(family, layer_index, self.cpu_policy)
            upload = self.cpu_uploads.get(getattr(candidate, "cpu_upload_id", None))
            usable = targeted and matrix is not None and candidate.state == CandidateState.AVAILABLE and upload is not None and not upload.late and not upload.failed and upload.completion.query()
            metadata = {**getattr(candidate, "cpu_metadata", {}), "matrix_use_id": self._use_id(candidate), "snapshot_id": None if self.cpu_snapshot is None else self.cpu_snapshot.snapshot_id, "phase": self.cpu_phase, "readiness_check_ns": candidate.deadline_ns}
            stream = torch.cuda.current_stream(device=self._device)
            if usable:
                tensor = upload.tensor
                tensor.record_stream(stream)
                lease = CpuStorageLease(self, upload, tensor, stream)
                upload.consumed, upload.tensor, candidate.tensor = True, None, None
                differentiable = torch.is_grad_enabled() and (self.trajectory.depth_basis.requires_grad or any(self.trajectory.coefficients[name].requires_grad for name in FAMILY_NAMES[family]))
                if not differentiable:
                    self.eval_leases[candidate.sequence] = lease
                output = self._attach(family, layer_index, tensor, binding_context=PrematBindingContext(self.cpu_policy, lease))
                candidate.final_outcome, candidate.state = "FULL HIT", CandidateState.CONSUMING
                self._aggregate["available_hits"] += 1
                self._aggregate["fully_hidden_hits"] += 1
                metadata.update({"materialisation_device": "cpu", "final_outcome": "FULL HIT"})
            else:
                if upload is not None:
                    upload.late = True
                    self._cpu_record({**metadata, "upload_id": upload.identity, "event": "late_copy", "bytes": upload.nbytes})
                reason = "not_targeted" if not targeted else "shadow_mode" if self._shadow_mode else "cpu_snapshot_download_failed" if self.source_error else "cpu_upload_failed" if upload is not None and upload.failed else "upload_not_complete" if upload is not None else candidate.admission_reason if candidate.admission_reason not in ("admitted", "not_checked") else current.get("fallback_reason", "cpu_preparation_not_ready")
                metadata.update({"materialisation_device": "gpu", "final_outcome": "COMPLETE MISS" if targeted else "NOT TARGETED", "fallback_reason": reason})
                candidate.final_outcome = metadata["final_outcome"]
                candidate.owner, candidate.state, candidate.tensor = "main", CandidateState.CONSUMING, None
                self._aggregate["main_stream_misses"] += int(targeted)
                self._aggregate["ordinary_deadline_materialisations"] += 1
                output = self._materialize(family, layer_index)
            candidate.cpu_metadata = metadata
            self._record("matrix_acquired", candidate=candidate, outcome=candidate.final_outcome, reason=metadata.get("fallback_reason"))
            self._cpu_record({**metadata, "event": "matrix_use", "family": family, "layer_index": layer_index, "pass_sequence": self._pass_sequence, "targeted": targeted})
            self._refresh_available()
            return output

    def materialize_for_consumption(self, family, layer_index):
        output = super().materialize_for_consumption(family, layer_index)
        if self.cpu_phase == "checkpoint_recompute":
            targeted = self._target_families is None or family in self._target_families
            self._cpu_record({"event": "matrix_use", "matrix_use_id": f"{self.optimizer_step}:{self.micro_step}:replay:{self.replay_sequence}:{layer_index}:{family}", "snapshot_id": None if self.cpu_snapshot is None else self.cpu_snapshot.snapshot_id, "family": family, "layer_index": layer_index, "materialisation_device": "gpu", "final_outcome": "COMPLETE MISS" if targeted else "NOT TARGETED", "fallback_reason": "checkpoint_replay_disabled", "targeted": targeted})
        return output

    def consumed(self, family, layer_index):
        with self.cpu_lock:
            candidate = self._candidates[(int(layer_index), str(family))]
            candidate.consumed_ns, candidate.tensor = time.perf_counter_ns(), None
            candidate.state = CandidateState.CONSUMED
            self._record("consumed", candidate=candidate, outcome=candidate.final_outcome)
            self.eval_leases.pop(candidate.sequence, None)
            self._advance(trigger="consumed")

    def _retire_lease(self, identity, tensor, stream):
        with self.cpu_lock:
            upload = self.cpu_uploads.get(identity)
            if upload is not None:
                upload.tensor = tensor
                upload.retirement = torch.cuda.Event()
                upload.retirement.record(stream)
                upload.metadata["release_submitted_ns"] = time.perf_counter_ns()

    def _record(self, event, **kwargs):
        payload = super()._record(event, **kwargs)
        if payload is not None:
            payload.update({"schema_version": 5, "materialisation_device": "cpu_and_gpu", "phase": self.cpu_phase, "snapshot_id": None if self.cpu_snapshot is None else self.cpu_snapshot.snapshot_id})
            if kwargs.get("candidate") is not None:
                payload.update(getattr(kwargs["candidate"], "cpu_metadata", {}))
        return payload

    def _candidate_payload(self, candidate):
        return {**super()._candidate_payload(candidate), **getattr(candidate, "cpu_metadata", {})}

    def end(self):
        with self.cpu_lock:
            if self.prediction_timer is not None:
                self.prediction_timer.cancel()
                self.prediction_timer = None
            for candidate in self._candidates.values():
                upload = self.cpu_uploads.get(getattr(candidate, "cpu_upload_id", None))
                if upload is not None and not upload.consumed:
                    upload.late = True
                candidate.tensor = None
                candidate.retained_counted = candidate.transient_counted = False
                if upload is not None:
                    fields = ("allocator_certificate_generation", "allocator_certificate_pool", "allocator_certificate_block_id", "allocator_certificate_charge_bytes")
                    upload.certificate = type("Certificate", (), {key: getattr(candidate, key) for key in fields})()
                    for key in fields:
                        setattr(candidate, key, 0 if key.endswith("bytes") else None)
            super().end()
            self.leading_opportunities.clear()
            self._refresh_available()

    def checkpoint_context(self, indices, microstep=None):
        return nullcontext(), CpuReplayContext(self, tuple(indices), self.micro_step if microstep is None else microstep)

    @contextmanager
    def replay_scope(self, indices, microstep):
        old_phase, old_micro = self.cpu_phase, self.micro_step
        self.cpu_phase, self.micro_step = "checkpoint_recompute", microstep
        self.replay_sequence += 1
        self._cpu_record({"event": "checkpoint_segment_entry", "segment_layers": list(indices), "replay_policy": self.cpu_configuration["premat_cpu_checkpoint_replay"]})
        owned = False
        try:
            if self.cpu_configuration["premat_cpu_checkpoint_replay"] == "enabled":
                reference = ReplayReference(self._device, getattr(torch, self.cpu_policy.coefficient_dtype), (self._batch_size, self._sequence_length, self._n_embd))
                self.begin(indices, reference=reference)
                owned = True
            yield
        finally:
            if owned:
                self.end()
            self._cpu_record({"event": "checkpoint_segment_exit", "segment_layers": list(indices)})
            self.cpu_phase, self.micro_step = old_phase, old_micro

    def _prediction_key(self, family, layer):
        return (self.cpu_phase, int(layer), str(family), self._batch_size, self._sequence_length, self.cpu_policy.policy_id)

    def _start_clock(self):
        clock = {"uncertainty_ms": None}
        origin = torch.cuda.Event(enable_timing=True)
        lower = time.perf_counter_ns()
        origin.record(torch.cuda.current_stream(device=self._device))
        self.clock, self.origin_event = clock, origin
        def align():
            origin.synchronize()
            upper = time.perf_counter_ns()
            clock.update({"host_origin_ns": (lower + upper) // 2, "uncertainty_ms": (upper - lower) / 2e6})
            self._wake()
        threading.Thread(target=align, daemon=True, name="premat-clock-alignment").start()

    def forensic_main_work_start(self, family, layer_index):
        super().forensic_main_work_start(family, layer_index)
        if self._active and self.cpu_configuration["premat_cpu_transfer_timing"] == "predicted_gemm_start":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device=self._device))
            identity = {"optimizer_step": self.optimizer_step, "micro_step": self.micro_step, "phase": self.cpu_phase, "replay_invocation": self.replay_sequence if self.cpu_phase == "checkpoint_recompute" else None, "pass_sequence": self._pass_sequence}
            self.prediction_events.append((event, self.origin_event, self.clock, self._prediction_key(family, layer_index), identity))

    def _collect_predictions(self):
        for _ in range(len(self.prediction_events)):
            event, origin, clock, context, identity = self.prediction_events.popleft()
            if origin is None or not event.query() or clock.get("uncertainty_ms") is None:
                self.prediction_events.append((event, origin, clock, context, identity))
                continue
            offset = origin.elapsed_time(event)
            self.cpu_predictor.observe(context, offset, clock["uncertainty_ms"])
            self._cpu_record({**identity, "event": "gemm_start_observed", "phase": context[0], "family": context[2], "layer_index": context[1], "actual_gemm_start_ns": clock["host_origin_ns"] + int(offset * 1e6), "clock_uncertainty_ms": clock["uncertainty_ms"]})

    def report(self):
        with self.cpu_lock:
            report = super().report()
            self._refresh_available()
            provider = {} if self.cpu_provider is None else self.cpu_provider.report()
            with self.event_lock:
                events = list(self.cpu_events)
            states = {"pending_upload_bytes": 0, "copying_bytes": 0, "available_bytes": 0, "autograd_retained_bytes": 0}
            for upload in self.cpu_uploads.values():
                key = "autograd_retained_bytes" if upload.consumed else "pending_upload_bytes" if upload.completion is None else "available_bytes" if upload.completion.query() else "copying_bytes"
                states[key] += upload.nbytes
            runtime = {**provider, **states, **self._memory_states(), "pin_preparation_peak_bytes": self.pin_in_progress_peak, "staging_bytes": self._retained_bytes, "staging_peak_bytes": self.cpu_peak, "staging_limit_bytes": self.cpu_cap, "pinned_upload_bytes": sum(upload.nbytes for upload in self.cpu_uploads.values() if upload.pinned is not None), "pinned_upload_peak_bytes": self.pin_peak, "pinned_snapshot_bytes": sum(job["host"].numel() * job["host"].element_size() for job in self.cpu_sources if job["host"] is not None), "control_path_counts": dict(self.control_counts), "control_path_host_ms": {name: value / 1e6 for name, value in self.control_ns.items()}, "prediction_context_count": len(self.cpu_predictor.samples), "prediction_event_count": len(self.prediction_events), "prediction_timer_count": int(self.prediction_timer is not None), "snapshot_download_error": self.source_error, "gpu_co_residency": "N/A: CPU materialisation"}
            report.update({"version": 5, "schema_version": 5, "materialisation_device": "cpu_and_gpu", "cpu_configuration": self.cpu_configuration, "phase": self.cpu_phase, "cpu_runtime": runtime, "cpu_lifecycle": events})
            return report

    def close(self):
        with self.cpu_lock:
            self.cpu_closed, self.source_paused = True, True
            if self.prediction_timer is not None:
                self.prediction_timer.cancel()
            provider = self.cpu_provider
            for upload in self.cpu_uploads.values():
                if upload.completion is not None:
                    upload.completion.synchronize()
                if upload.retirement is not None:
                    upload.retirement.synchronize()
            self.cpu_uploads.clear()
            self.cpu_sources.clear()
        if provider is not None:
            provider.close()
# ^^^ THOG
