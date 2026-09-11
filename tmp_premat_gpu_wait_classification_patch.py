from __future__ import annotations

from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Runtime: classify FULL/PARTIAL on the GPU execution timeline, not host time.
# ---------------------------------------------------------------------------
p = Path("sheet/premat.py")
text = p.read_text()
text = replace_once(
    text,
    '''@dataclass
class _PendingCudaTiming:
    kind: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    event_payload: Optional[Dict[str, object]]
''',
    '''@dataclass
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
''',
    "pending timing metadata",
)
text = replace_once(
    text,
    '''        self._pending_timings: List[_PendingCudaTiming] = []
        self._pending_releases: List[_PendingRelease] = []
''',
    '''        self._pending_timings: List[_PendingCudaTiming] = []
        # Completed sampled passes are retained only until all CUDA timings for
        # that pass have actually resolved. This lets Instra receive a final
        # GPU-timeline classification without synchronising the training stream.
        self._pending_completed_live_reports: Dict[int, Dict[str, object]] = {}
        self._pending_releases: List[_PendingRelease] = []
''',
    "pending live reports storage",
)
text = replace_once(
    text,
    '''        self._layer_indices = resolved
        self._position = -1
        # One forward pass is one accumulation microstep's complete Premat
''',
    '''        self._layer_indices = resolved
        self._position = -1
        # A prior sampled pass may have had GPU wait timings still outstanding
        # when its host forward ended. Resolve/publish those non-blockingly before
        # replacing the current event window.
        self._resolve_pending_timings()
        # One forward pass is one accumulation microstep's complete Premat
''',
    "resolve old timings before event clear",
)
text = replace_once(
    text,
    '''        if self._pass_start_ns is not None:
            self._aggregate["captured_pass_host_ms_total"] += max(
                0.0,
                (time.perf_counter_ns() - self._pass_start_ns) / 1_000_000.0,
            )
        for candidate in tuple(self._candidates.values()):
''',
    '''        if self._pass_start_ns is not None:
            self._aggregate["captured_pass_host_ms_total"] += max(
                0.0,
                (time.perf_counter_ns() - self._pass_start_ns) / 1_000_000.0,
            )
        self._capture_completed_live_report()
        for candidate in tuple(self._candidates.values()):
''',
    "capture finalized live report at pass end",
)
old_wait_branch = '''        elif candidate.state == CandidateState.MATERIALISING:
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
'''
new_wait_branch = '''        elif candidate.state == CandidateState.MATERIALISING:
            if candidate.completion_event is None:
                raise RuntimeError(f"materialising candidate {key} has no completion event")
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_end = torch.cuda.Event(enable_timing=True)
            wait_start.record(current_stream)
            current_stream.wait_event(candidate.completion_event)
            wait_end.record(current_stream)
            # Host submission has reached the dependency, but this does NOT tell
            # us whether the GPU Main Stream will actually wait there. Keep the
            # result provisional until CUDA has timestamped both streams.
            candidate.critical_path_miss = False
            candidate.final_outcome = "PENDING"
            # candidate.tensor remains strongly referenced through consumed().
            wait_payload = self._transition(
                candidate,
                CandidateState.CONSUMING,
                "critical_path_wait",
                outcome="gpu_wait_pending",
                reason="classification_pending_until_main_stream_dependency",
                detail={"gpu_wait_classification_pending": True},
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
'''
text = replace_once(text, old_wait_branch, new_wait_branch, "provisional main-stream wait")
text = replace_once(
    text,
    '''                _PendingCudaTiming(
                    kind="main_stream_materialisation",
                    start_event=materialise_start,
                    end_event=materialise_end,
                    event_payload=fallback_payload,
                )
''',
    '''                _PendingCudaTiming(
                    kind="main_stream_materialisation",
                    start_event=materialise_start,
                    end_event=materialise_end,
                    event_payload=fallback_payload,
                    pass_sequence=self._pass_sequence,
                    layer_index=candidate.layer_index,
                    family=candidate.family,
                    candidate=candidate,
                )
''',
    "main materialisation timing metadata",
)
text = replace_once(
    text,
    '''                _PendingCudaTiming(
                    kind="premat_materialisation",
                    start_event=candidate.materialisation_start_event,
                    end_event=candidate.completion_event,
                    event_payload=launch_payload,
                )
''',
    '''                _PendingCudaTiming(
                    kind="premat_materialisation",
                    start_event=candidate.materialisation_start_event,
                    end_event=candidate.completion_event,
                    event_payload=launch_payload,
                    pass_sequence=self._pass_sequence,
                    layer_index=candidate.layer_index,
                    family=candidate.family,
                    candidate=candidate,
                )
''',
    "premat materialisation timing metadata",
)
old_reporter_block = '''        # vvv THOG publish only the completed sampled microstep.  Partial live
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
'''
new_reporter_block = '''        # Completed sampled passes are published by _capture_completed_live_report
        # only after all CUDA timings for that pass have resolved.  _record stays
        # purely observational and never introduces a host/GPU synchronisation.
        return payload
'''
text = replace_once(text, old_reporter_block, new_reporter_block, "defer live publication")

old_resolve = '''    def _resolve_pending_timings(self) -> None:
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
'''
new_resolve = '''    def _capture_completed_live_report(self) -> None:
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
        if payload is None:
            raise RuntimeError("main-stream wait timing lost its event payload")
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
'''
text = replace_once(text, old_resolve, new_resolve, "GPU-timeline pending timing resolver")
p.write_text(text)


# ---------------------------------------------------------------------------
# Compact telemetry must retain the fields needed for corrected classification.
# Preserve the optimizer update chosen when the sampled pass was selected even
# if final CUDA timing resolution happens during a later host microstep.
# ---------------------------------------------------------------------------
p = Path("sheet/wandb_telemetry.py")
text = p.read_text()
text = replace_once(
    text,
    '''    "wait_ms",
    "predicted_retained_bytes",
''',
    '''    "wait_ms",
    "wait_marker_elapsed_ms",
    "gpu_dependency_delta_ms",
    "premat_lead_ms_at_main_stream_dependency",
    "gpu_wait_required",
    "gpu_wait_classification_pending",
    "final_outcome",
    "predicted_retained_bytes",
''',
    "compact GPU wait fields",
)
text = replace_once(
    text,
    '''        captured_update: Optional[int] = None
        captured_pass_sequence: Optional[int] = None

        def publish_premat(snapshot: Mapping[str, Any]) -> None:
            live_sink.publish(
                max(1, int(trainer.state.completed_updates) + 1),
                snapshot,
            )

        def capture_premat_update(pass_sequence: int) -> bool:
            nonlocal captured_update, captured_pass_sequence
''',
    '''        captured_update: Optional[int] = None
        captured_pass_sequence: Optional[int] = None
        captured_pass_updates: Dict[int, int] = {}

        def publish_premat(snapshot: Mapping[str, Any]) -> None:
            pass_sequence = int(snapshot.get("pass_sequence", 0))
            update = captured_pass_updates.pop(
                pass_sequence,
                max(1, int(trainer.state.completed_updates) + 1),
            )
            live_sink.publish(update, snapshot)

        def capture_premat_update(pass_sequence: int) -> bool:
            nonlocal captured_update, captured_pass_sequence
''',
    "delay-safe optimizer update mapping",
)
text = replace_once(
    text,
    '''            if captured_update != update:
                captured_update = update
                captured_pass_sequence = int(pass_sequence)
            return int(pass_sequence) == captured_pass_sequence
''',
    '''            if captured_update != update:
                captured_update = update
                captured_pass_sequence = int(pass_sequence)
            selected = int(pass_sequence) == captured_pass_sequence
            if selected:
                captured_pass_updates[int(pass_sequence)] = update
            return selected
''',
    "remember sampled pass update",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Instra playback: a host-MATERIALISING candidate can still become a GPU FULL
# hit. Show the synthetic AVAILABLE state when the GPU timeline proves that the
# Premat completion preceded the Main Stream dependency marker.
# ---------------------------------------------------------------------------
p = Path("sheet/local_dashboard_assets/dashboard_premat.js")
text = p.read_text()
text = replace_once(
    text,
    '''    wait_ms: null,
    materialisation_ms: null,
''',
    '''    wait_ms: null,
    wait_marker_elapsed_ms: null,
    gpu_dependency_delta_ms: null,
    gpu_wait_required: null,
    materialisation_ms: null,
''',
    "Instra record wait diagnostics",
)
text = replace_once(
    text,
    '''  if (event.wait_ms !== null && event.wait_ms !== undefined && Number.isFinite(Number(event.wait_ms))) {
    record.wait_ms = Number(event.wait_ms);
  }
''',
    '''  if (event.wait_ms !== null && event.wait_ms !== undefined && Number.isFinite(Number(event.wait_ms))) {
    record.wait_ms = Number(event.wait_ms);
  }
  if (event.wait_marker_elapsed_ms !== null && event.wait_marker_elapsed_ms !== undefined
      && Number.isFinite(Number(event.wait_marker_elapsed_ms))) {
    record.wait_marker_elapsed_ms = Number(event.wait_marker_elapsed_ms);
  }
  if (event.gpu_dependency_delta_ms !== null && event.gpu_dependency_delta_ms !== undefined
      && Number.isFinite(Number(event.gpu_dependency_delta_ms))) {
    record.gpu_dependency_delta_ms = Number(event.gpu_dependency_delta_ms);
  }
  if (typeof event.gpu_wait_required === "boolean") {
    record.gpu_wait_required = event.gpu_wait_required;
  }
''',
    "Instra capture GPU wait diagnostics",
)
text = replace_once(
    text,
    '''    const waited = event.outcome === "waited_for_premat"
      || event.reason === "materialising_at_deadline";
''',
    '''    const waited = event.gpu_wait_required === true
      || event.outcome === "waited_for_premat"
      || event.reason === "materialising_at_deadline";
''',
    "Instra waited classification",
)
text = replace_once(
    text,
    '''      } else {
        record.path = "full";
        record.outcome = "FULL HIT";
        premat_append_trace(record, "CONSUMING - NO WAIT");
        add_frame({key, state: "consuming-full", outcome: record.outcome}, event, "CONSUMING - NO WAIT");
      }
''',
    '''      } else {
        record.path = "full";
        record.outcome = "FULL HIT";
        if (event_name === "critical_path_wait" && event.gpu_wait_required === false) {
          // The host reached acquire() while PREMAT was still running, but CUDA
          // proves PREMAT completed before the Main Stream reached the wait.
          // Render the effective AVAILABLE state so the recapitulation follows
          // the same Full Success path as a host-observed AVAILABLE candidate.
          premat_append_trace(record, "AVAILABLE");
          add_frame({key, state: "available", outcome: record.outcome}, event, "AVAILABLE");
        }
        premat_append_trace(record, "CONSUMING - NO WAIT");
        add_frame({key, state: "consuming-full", outcome: record.outcome}, event, "CONSUMING - NO WAIT");
      }
''',
    "Instra synthetic available for GPU-effective full",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Runtime regression tests.
# ---------------------------------------------------------------------------
p = Path("tests/test_premat.py")
text = p.read_text()
text = replace_once(
    text,
    '''    def __init__(self, cuda, *, enable_timing: bool) -> None:
        self.cuda = cuda
        self.enable_timing = enable_timing
        self.complete = False
''',
    '''    def __init__(self, cuda, *, enable_timing: bool) -> None:
        self.cuda = cuda
        self.enable_timing = enable_timing
        self.complete = False
        self.elapsed_time_override = None
''',
    "fake event timing override field",
)
text = replace_once(
    text,
    '''    def elapsed_time(self, _other) -> float:
        return 0.25
''',
    '''    def elapsed_time(self, other) -> float:
        if self.elapsed_time_override is not None:
            return float(self.elapsed_time_override(other))
        return 0.25
''',
    "fake event timing override",
)
insert_after = '''def test_materialising_deadline_waits_once_and_never_duplicates(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    first_target_calls = [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    assert calls == first_target_calls
    assert fake_cuda.stream_creations == 1
    runtime.layer_start(5)
    weight = runtime.acquire("DOWN", 5)
    assert calls.count(("DOWN", 5)) == 1
    assert calls[:4] == first_target_calls
    assert weight.recorded_streams[-1] is runtime._stream
    assert len(fake_cuda.main_stream.waited_events) == 1
    candidate = runtime._candidates[(5, "DOWN")]
    assert candidate.owner == "premat"
    assert candidate.state == CandidateState.CONSUMING
    assert candidate.critical_path_miss
    report = runtime.report()
    assert report["aggregate"]["waited_hits"] == 1
    assert report["aggregate"]["main_stream_wait_ms_total"] == pytest.approx(0.25)
'''
replacement = insert_after + '''\n\ndef test_materialising_at_host_acquire_can_resolve_to_gpu_full_hit(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    runtime.layer_start(5)
    runtime.acquire("DOWN", 5)
    timing = next(item for item in runtime._pending_timings if item.kind == "main_stream_wait")
    timing.start_event.elapsed_time_override = (
        lambda other: -0.125 if other is timing.dependency_event else 0.010
    )
    runtime._resolve_pending_timings()

    candidate = timing.candidate
    assert candidate is not None
    assert candidate.final_outcome == "FULL HIT"
    assert candidate.critical_path_miss is False
    report = runtime.report()
    assert report["aggregate"]["fully_hidden_hits"] == 1
    assert report["aggregate"]["waited_hits"] == 0
    assert report["aggregate"]["main_stream_wait_ms_total"] == pytest.approx(0.0)
    resolved = timing.event_payload
    assert resolved is not None
    assert resolved["gpu_wait_required"] is False
    assert resolved["gpu_dependency_delta_ms"] == pytest.approx(-0.125)
    assert resolved["wait_marker_elapsed_ms"] == pytest.approx(0.010)
    assert resolved["wait_ms"] == pytest.approx(0.0)
    assert calls.count(("DOWN", 5)) == 1


def test_sampled_live_report_waits_for_gpu_classification_without_sync(monkeypatch) -> None:
    runtime, fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    snapshots = []
    runtime.set_live_reporter(snapshots.append, lambda pass_sequence: pass_sequence == 1)
    runtime.layer_start(3)
    runtime.layer_start(5)
    fake_cuda.complete_main_on_record = False
    runtime.acquire("DOWN", 5)
    wait_timing = next(item for item in runtime._pending_timings if item.kind == "main_stream_wait")
    wait_timing.start_event.elapsed_time_override = (
        lambda other: -0.050 if other is wait_timing.dependency_event else 0.008
    )
    runtime.end()

    # Host pass completion must not publish a provisional PARTIAL/FULL result.
    assert snapshots == []
    assert 1 in runtime._pending_completed_live_reports

    wait_timing.end_event.complete = True
    runtime.begin((3, 5, 7), reference=_FakeTensor())

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["pass_sequence"] == 1
    resolved = next(
        event
        for event in snapshot["events"]
        if event.get("event") == "critical_path_wait"
        and event.get("family") == "DOWN"
    )
    assert resolved["final_outcome"] == "FULL HIT"
    assert resolved["gpu_wait_required"] is False
    assert resolved["wait_ms"] == pytest.approx(0.0)
    consumed = next(
        event
        for event in snapshot["events"]
        if event.get("event") == "consumed"
        and event.get("family") == "DOWN"
    )
    assert consumed["final_outcome"] == "FULL HIT"
    assert consumed["critical_path_miss"] is False
'''
text = replace_once(text, insert_after, replacement, "runtime GPU wait regression insertion")
p.write_text(text)


# ---------------------------------------------------------------------------
# Instra reducer regression: GPU-effective FULL follows the full-success path.
# ---------------------------------------------------------------------------
p = Path("tests/test_premat_instra.py")
text = p.read_text()
anchor = '''def test_history_reducer_covers_all_retained_steps_and_layers_and_exports_csv() -> None:
'''
new_test = '''def test_gpu_effective_full_hit_uses_full_success_recap_path() -> None:
    snapshot = _playback_snapshot()
    for event in snapshot["events"]:
        if event.get("event") == "critical_path_wait" and event.get("family") == "O":
            event.update({
                "outcome": "fully_hidden",
                "reason": "premat_complete_before_main_stream_dependency",
                "critical_path_miss": False,
                "gpu_wait_required": False,
                "gpu_dependency_delta_ms": -0.075,
                "wait_marker_elapsed_ms": 0.009,
                "wait_ms": 0.0,
                "final_outcome": "FULL HIT",
            })
        if event.get("event") == "consumed" and event.get("family") == "O":
            event["critical_path_miss"] = False
            event["final_outcome"] = "FULL HIT"
    rendered = _javascript_model(snapshot)
    records = {(record["layer_index"], record["family"]): record for record in rendered["records"]}
    assert records[(0, "O")]["outcome"] == "FULL HIT"
    assert records[(0, "O")]["trace"] == [
        "PRE-MATERIALISING", "AVAILABLE", "CONSUMING - NO WAIT", "FULL HIT"
    ]
    assert records[(0, "O")]["wait_ms"] == 0.0
    assert records[(0, "O")]["gpu_wait_required"] is False


''' + anchor
text = replace_once(text, anchor, new_test, "Instra GPU-effective full regression")
p.write_text(text)

print("patched PREMAT GPU-timeline hit classification and Instra accounting")
