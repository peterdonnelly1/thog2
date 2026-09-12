from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


premat_path = Path("sheet/premat.py")
source = premat_path.read_text()
source = replace_once(
    source,
    'PREMAT_TELEMETRY_VERSION = 3\n',
    'PREMAT_TELEMETRY_VERSION = 4\n',
    'telemetry version',
)

source = replace_once(
    source,
    '''def _conservative_certificate_charge_bytes(request_bytes: int) -> int:\n    """Upper-bound native allocator rounding for one cached-block request.\n\n    PyTorch's configurable native rounding cannot round a request beyond the\n    next power of two.  Using that bound intentionally depletes a cached\n    contiguous-space certificate faster than the normal allocator would.\n    """\n    size = max(0, int(request_bytes))\n    if size == 0:\n        return 0\n    rounded = 1 << (size - 1).bit_length()\n    return max(512, rounded)\n\n\n@dataclass\nclass _Candidate:\n''',
    '''def _conservative_certificate_charge_bytes(request_bytes: int) -> int:\n    """Upper-bound native allocator rounding for one cached-block request.\n\n    PyTorch's configurable native rounding cannot round a request beyond the\n    next power of two.  Using that bound intentionally depletes a cached\n    contiguous-space certificate faster than the normal allocator would.\n    """\n    size = max(0, int(request_bytes))\n    if size == 0:\n        return 0\n    rounded = 1 << (size - 1).bit_length()\n    return max(512, rounded)\n\n\n# vvv THOG GPU-forensic intervals are retained only while the explicit timing diagnostic is enabled\n@dataclass\nclass _ForensicLayerInterval:\n    pass_sequence: int\n    layer_index: int\n    start_event: torch.cuda.Event\n    end_event: Optional[torch.cuda.Event] = None\n\n\n@dataclass\nclass _ForensicCandidateInterval:\n    pass_sequence: int\n    layer_index: int\n    family: str\n    start_event: torch.cuda.Event\n    end_event: torch.cuda.Event\n    dependency_event: Optional[torch.cuda.Event] = None\n    wait_end_event: Optional[torch.cuda.Event] = None\n    consumed: bool = False\n\n\n@dataclass\nclass _PendingForensicPass:\n    pass_sequence: int\n    origin_event: torch.cuda.Event\n    layer_intervals: Tuple[_ForensicLayerInterval, ...]\n    candidate_intervals: Tuple[_ForensicCandidateInterval, ...]\n# ^^^ THOG\n\n\n@dataclass\nclass _Candidate:\n''',
    'forensic dataclasses',
)

source = replace_once(
    source,
    '''    allocator_certificate_block_id: Optional[int] = None\n    allocator_certificate_charge_bytes: int = 0\n\n\n@dataclass\nclass _PendingCudaTiming:\n''',
    '''    allocator_certificate_block_id: Optional[int] = None\n    allocator_certificate_charge_bytes: int = 0\n    # vvv THOG zero-cost host counters remain available even when CUDA forensic timing is disabled\n    real_premat_launched: bool = False\n    real_premat_consumed: bool = False\n    forensic_interval: Optional[_ForensicCandidateInterval] = None\n    # ^^^ THOG\n\n\n@dataclass\nclass _PendingCudaTiming:\n''',
    'candidate forensic fields',
)

source = replace_once(
    source,
    '''        self._pending_timings: List[_PendingCudaTiming] = []\n        # Completed sampled passes are retained only until all CUDA timings for\n''',
    '''        self._pending_timings: List[_PendingCudaTiming] = []\n        # vvv THOG explicit GPU timing diagnostic also captures Main/PREMAT overlap without synchronising training\n        self._forensic_pass_origin_event: Optional[torch.cuda.Event] = None\n        self._forensic_current_layers: List[_ForensicLayerInterval] = []\n        self._forensic_current_candidates: List[_ForensicCandidateInterval] = []\n        self._pending_forensic_passes: List[_PendingForensicPass] = []\n        self._latest_forensic_pass: Optional[Dict[str, object]] = None\n        # ^^^ THOG\n        # Completed sampled passes are retained only until all CUDA timings for\n''',
    'forensic runtime state',
)

source = replace_once(
    source,
    '''            "allocator_snapshot_count": 0,\n        }\n\n    @property\n''',
    '''            "allocator_snapshot_count": 0,\n            # vvv THOG forensic counters separate real side work, Main work, overlap and dependency shortfall\n            "real_premat_materialisations_launched": 0,\n            "real_premat_materialisations_consumed": 0,\n            "real_premat_materialisations_unused": 0,\n            "duplicate_main_materialisations_after_premat": 0,\n            "forensic_resolved_passes": 0,\n            "forensic_main_layer_count": 0,\n            "forensic_main_layer_gpu_ms_total": 0.0,\n            "forensic_main_wait_marker_ms_total": 0.0,\n            "forensic_main_nonwait_gpu_ms_total": 0.0,\n            "forensic_premat_gpu_ms_total": 0.0,\n            "forensic_premat_temporal_overlap_ms_total": 0.0,\n            "forensic_premat_wait_overlap_ms_total": 0.0,\n            "forensic_premat_useful_overlap_ms_total": 0.0,\n            "forensic_premat_outside_main_layer_ms_total": 0.0,\n            "forensic_dependency_count": 0,\n            "forensic_dependency_shortfall_count": 0,\n            "forensic_dependency_shortfall_ms_total": 0.0,\n            "forensic_dependency_slack_ms_total": 0.0,\n            "forensic_dependency_lead_window_ms_total": 0.0,\n            "forensic_dependency_before_premat_start_count": 0,\n            # ^^^ THOG\n        }\n        # vvv THOG per-family forensic totals identify whether one matrix family is poisoning overlap\n        self._forensic_by_family: Dict[str, Dict[str, float | int]] = {\n            family: self._new_forensic_family_row()\n            for family in self._families()\n        }\n        # ^^^ THOG\n\n    @property\n''',
    'forensic aggregate fields',
)

source = replace_once(
    source,
    '''    def set_live_reporter(\n        self,\n        reporter: Optional[PrematLiveReporter],\n        capture_enabled: Optional[PrematLiveCapturePredicate] = None,\n    ) -> None:\n        self._live_reporter = reporter\n        self._live_capture_enabled = capture_enabled\n        self._live_publish_error = None\n    # ^^^ THOG\n\n    def begin(\n''',
    '''    def set_live_reporter(\n        self,\n        reporter: Optional[PrematLiveReporter],\n        capture_enabled: Optional[PrematLiveCapturePredicate] = None,\n    ) -> None:\n        self._live_reporter = reporter\n        self._live_capture_enabled = capture_enabled\n        self._live_publish_error = None\n    # ^^^ THOG\n\n    # vvv THOG first-tranche GPU forensics: six independent observables, all gated by the existing timing diagnostic\n    @staticmethod\n    def _new_forensic_family_row() -> Dict[str, float | int]:\n        return {\n            "launched": 0,\n            "consumed": 0,\n            "unused": 0,\n            "duplicates": 0,\n            "premat_gpu_ms_total": 0.0,\n            "temporal_overlap_ms_total": 0.0,\n            "wait_overlap_ms_total": 0.0,\n            "useful_overlap_ms_total": 0.0,\n            "outside_main_layer_ms_total": 0.0,\n            "dependency_count": 0,\n            "dependency_shortfall_count": 0,\n            "dependency_shortfall_ms_total": 0.0,\n            "dependency_slack_ms_total": 0.0,\n            "dependency_lead_window_ms_total": 0.0,\n            "dependency_before_premat_start_count": 0,\n        }\n\n    @staticmethod\n    def _forensic_overlap_ms(\n        left_start: float,\n        left_end: float,\n        right_start: float,\n        right_end: float,\n    ) -> float:\n        return max(0.0, min(left_end, right_end) - max(left_start, right_start))\n\n    def forensic_layer_start(self, layer_index: int) -> None:\n        if not self._enable_gpu_timing_diagnostic:\n            return\n        self._require_active()\n        if self._device is None:\n            raise RuntimeError("PREMAT forensic layer timing has no CUDA device")\n        event = torch.cuda.Event(enable_timing=True)\n        event.record(torch.cuda.current_stream(device=self._device))\n        self._forensic_current_layers.append(\n            _ForensicLayerInterval(\n                pass_sequence=self._pass_sequence,\n                layer_index=int(layer_index),\n                start_event=event,\n            )\n        )\n\n    def forensic_layer_end(self, layer_index: int) -> None:\n        if not self._enable_gpu_timing_diagnostic:\n            return\n        self._require_active()\n        if self._device is None:\n            raise RuntimeError("PREMAT forensic layer timing has no CUDA device")\n        interval = next(\n            (\n                item\n                for item in reversed(self._forensic_current_layers)\n                if item.layer_index == int(layer_index) and item.end_event is None\n            ),\n            None,\n        )\n        if interval is None:\n            raise RuntimeError(f"PREMAT forensic layer {layer_index} has no open interval")\n        event = torch.cuda.Event(enable_timing=True)\n        event.record(torch.cuda.current_stream(device=self._device))\n        interval.end_event = event\n\n    def _record_forensic_dependency(\n        self,\n        candidate: _Candidate,\n        current_stream,\n        *,\n        dependency_event: Optional[torch.cuda.Event] = None,\n        wait_end_event: Optional[torch.cuda.Event] = None,\n    ) -> None:\n        if not self._enable_gpu_timing_diagnostic:\n            return\n        interval = candidate.forensic_interval\n        if interval is None:\n            return\n        if interval.dependency_event is not None:\n            raise RuntimeError(\n                "PREMAT forensic candidate reached its Main Stream dependency twice: "\n                f"layer={candidate.layer_index}, family={candidate.family}"\n            )\n        if dependency_event is None:\n            dependency_event = torch.cuda.Event(enable_timing=True)\n            dependency_event.record(current_stream)\n        interval.dependency_event = dependency_event\n        interval.wait_end_event = wait_end_event\n\n    def _queue_forensic_pass(self) -> None:\n        if not self._enable_gpu_timing_diagnostic:\n            return\n        origin = self._forensic_pass_origin_event\n        if origin is None:\n            raise RuntimeError("PREMAT forensic timing pass has no origin event")\n        if any(interval.end_event is None for interval in self._forensic_current_layers):\n            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream layer interval")\n        self._pending_forensic_passes.append(\n            _PendingForensicPass(\n                pass_sequence=self._pass_sequence,\n                origin_event=origin,\n                layer_intervals=tuple(self._forensic_current_layers),\n                candidate_intervals=tuple(self._forensic_current_candidates),\n            )\n        )\n        self._forensic_pass_origin_event = None\n        self._forensic_current_layers = []\n        self._forensic_current_candidates = []\n\n    @staticmethod\n    def _forensic_derived_summary(values: Mapping[str, float | int]) -> Dict[str, object]:\n        layer_count = int(values.get("main_layer_count", 0))\n        premat_ms = float(values.get("premat_gpu_ms_total", 0.0))\n        dependency_count = int(values.get("dependency_count", 0))\n        launched = int(values.get("premat_launched", 0))\n        return {\n            **dict(values),\n            "main_layer_gpu_ms_mean": (\n                float(values.get("main_layer_gpu_ms_total", 0.0)) / layer_count\n                if layer_count > 0\n                else None\n            ),\n            "main_nonwait_gpu_ms_mean": (\n                float(values.get("main_nonwait_gpu_ms_total", 0.0)) / layer_count\n                if layer_count > 0\n                else None\n            ),\n            "premat_useful_overlap_fraction": (\n                min(1.0, float(values.get("premat_useful_overlap_ms_total", 0.0)) / premat_ms)\n                if premat_ms > 0.0\n                else None\n            ),\n            "premat_outside_main_layer_fraction": (\n                min(1.0, float(values.get("premat_outside_main_layer_ms_total", 0.0)) / premat_ms)\n                if premat_ms > 0.0\n                else None\n            ),\n            "dependency_shortfall_rate": (\n                int(values.get("dependency_shortfall_count", 0)) / dependency_count\n                if dependency_count > 0\n                else None\n            ),\n            "premat_consumption_fraction": (\n                int(values.get("premat_consumed", 0)) / launched\n                if launched > 0\n                else None\n            ),\n        }\n\n    def _forensic_report(self) -> Dict[str, object]:\n        aggregate = self._aggregate\n        values: Dict[str, float | int] = {\n            "resolved_passes": int(aggregate["forensic_resolved_passes"]),\n            "pending_passes": len(self._pending_forensic_passes),\n            "premat_launched": int(aggregate["real_premat_materialisations_launched"]),\n            "premat_consumed": int(aggregate["real_premat_materialisations_consumed"]),\n            "premat_unused": int(aggregate["real_premat_materialisations_unused"]),\n            "duplicate_main_materialisations": int(aggregate["duplicate_main_materialisations_after_premat"]),\n            "main_layer_count": int(aggregate["forensic_main_layer_count"]),\n            "main_layer_gpu_ms_total": float(aggregate["forensic_main_layer_gpu_ms_total"]),\n            "main_wait_marker_ms_total": float(aggregate["forensic_main_wait_marker_ms_total"]),\n            "main_nonwait_gpu_ms_total": float(aggregate["forensic_main_nonwait_gpu_ms_total"]),\n            "premat_gpu_ms_total": float(aggregate["forensic_premat_gpu_ms_total"]),\n            "premat_temporal_overlap_ms_total": float(aggregate["forensic_premat_temporal_overlap_ms_total"]),\n            "premat_wait_overlap_ms_total": float(aggregate["forensic_premat_wait_overlap_ms_total"]),\n            "premat_useful_overlap_ms_total": float(aggregate["forensic_premat_useful_overlap_ms_total"]),\n            "premat_outside_main_layer_ms_total": float(aggregate["forensic_premat_outside_main_layer_ms_total"]),\n            "dependency_count": int(aggregate["forensic_dependency_count"]),\n            "dependency_shortfall_count": int(aggregate["forensic_dependency_shortfall_count"]),\n            "dependency_shortfall_ms_total": float(aggregate["forensic_dependency_shortfall_ms_total"]),\n            "dependency_slack_ms_total": float(aggregate["forensic_dependency_slack_ms_total"]),\n            "dependency_lead_window_ms_total": float(aggregate["forensic_dependency_lead_window_ms_total"]),\n            "dependency_before_premat_start_count": int(aggregate["forensic_dependency_before_premat_start_count"]),\n        }\n        by_family: Dict[str, object] = {}\n        for family, row in self._forensic_by_family.items():\n            family_values = {\n                "premat_launched": int(row["launched"]),\n                "premat_consumed": int(row["consumed"]),\n                "premat_unused": int(row["unused"]),\n                "duplicate_main_materialisations": int(row["duplicates"]),\n                "main_layer_count": 0,\n                "main_layer_gpu_ms_total": 0.0,\n                "main_wait_marker_ms_total": 0.0,\n                "main_nonwait_gpu_ms_total": 0.0,\n                "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),\n                "premat_temporal_overlap_ms_total": float(row["temporal_overlap_ms_total"]),\n                "premat_wait_overlap_ms_total": float(row["wait_overlap_ms_total"]),\n                "premat_useful_overlap_ms_total": float(row["useful_overlap_ms_total"]),\n                "premat_outside_main_layer_ms_total": float(row["outside_main_layer_ms_total"]),\n                "dependency_count": int(row["dependency_count"]),\n                "dependency_shortfall_count": int(row["dependency_shortfall_count"]),\n                "dependency_shortfall_ms_total": float(row["dependency_shortfall_ms_total"]),\n                "dependency_slack_ms_total": float(row["dependency_slack_ms_total"]),\n                "dependency_lead_window_ms_total": float(row["dependency_lead_window_ms_total"]),\n                "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),\n            }\n            by_family[family] = self._forensic_derived_summary(family_values)\n        result = self._forensic_derived_summary(values)\n        result["enabled"] = self._enable_gpu_timing_diagnostic\n        result["by_family"] = by_family\n        return result\n\n    def _update_pending_live_report_forensic(\n        self,\n        pass_sequence: int,\n        aggregate_deltas: Mapping[str, float | int],\n        pass_report: Mapping[str, object],\n    ) -> None:\n        snapshot = self._pending_completed_live_reports.get(int(pass_sequence))\n        if snapshot is None:\n            return\n        aggregate = snapshot.get("aggregate")\n        if isinstance(aggregate, dict):\n            for name, delta in aggregate_deltas.items():\n                aggregate[name] = aggregate.get(name, 0) + delta\n        snapshot["forensic_pass"] = dict(pass_report)\n        self._refresh_report_derived_aggregates(snapshot)\n\n    def _resolve_pending_forensic_passes(self) -> None:\n        if not self._pending_forensic_passes:\n            return\n        remaining: List[_PendingForensicPass] = []\n        for pending in self._pending_forensic_passes:\n            required_events: List[torch.cuda.Event] = [pending.origin_event]\n            for layer in pending.layer_intervals:\n                if layer.end_event is None:\n                    raise RuntimeError("PREMAT forensic pass retained an open layer interval")\n                required_events.extend((layer.start_event, layer.end_event))\n            for candidate in pending.candidate_intervals:\n                required_events.extend((candidate.start_event, candidate.end_event))\n                if candidate.dependency_event is not None:\n                    required_events.append(candidate.dependency_event)\n                if candidate.wait_end_event is not None:\n                    required_events.append(candidate.wait_end_event)\n            try:\n                if not all(event.query() for event in required_events):\n                    remaining.append(pending)\n                    continue\n\n                def at_ms(event: torch.cuda.Event) -> float:\n                    return float(pending.origin_event.elapsed_time(event))\n\n                layer_rows: List[Dict[str, object]] = []\n                for layer in pending.layer_intervals:\n                    assert layer.end_event is not None\n                    start_ms = at_ms(layer.start_event)\n                    end_ms = at_ms(layer.end_event)\n                    layer_rows.append({\n                        "layer_index": layer.layer_index,\n                        "start_ms": start_ms,\n                        "end_ms": end_ms,\n                        "gpu_ms": max(0.0, end_ms - start_ms),\n                    })\n\n                wait_intervals: List[Tuple[float, float]] = []\n                for candidate in pending.candidate_intervals:\n                    if candidate.dependency_event is None or candidate.wait_end_event is None:\n                        continue\n                    wait_start_ms = at_ms(candidate.dependency_event)\n                    wait_end_ms = at_ms(candidate.wait_end_event)\n                    wait_intervals.append((wait_start_ms, max(wait_start_ms, wait_end_ms)))\n\n                candidate_rows: List[Dict[str, object]] = []\n                pass_family_rows: Dict[str, Dict[str, float | int]] = {}\n                premat_ms_total = 0.0\n                temporal_overlap_total = 0.0\n                wait_overlap_total = 0.0\n                useful_overlap_total = 0.0\n                outside_main_total = 0.0\n                dependency_count = 0\n                shortfall_count = 0\n                shortfall_ms_total = 0.0\n                slack_ms_total = 0.0\n                lead_window_ms_total = 0.0\n                dependency_before_start_count = 0\n\n                for candidate in pending.candidate_intervals:\n                    start_ms = at_ms(candidate.start_event)\n                    end_ms = at_ms(candidate.end_event)\n                    materialisation_ms = max(0.0, end_ms - start_ms)\n                    temporal_overlap_ms = min(\n                        materialisation_ms,\n                        sum(\n                            self._forensic_overlap_ms(\n                                start_ms, end_ms,\n                                float(layer["start_ms"]), float(layer["end_ms"]),\n                            )\n                            for layer in layer_rows\n                        ),\n                    )\n                    wait_overlap_ms = min(\n                        temporal_overlap_ms,\n                        sum(\n                            self._forensic_overlap_ms(start_ms, end_ms, wait_start, wait_end)\n                            for wait_start, wait_end in wait_intervals\n                        ),\n                    )\n                    useful_overlap_ms = max(0.0, temporal_overlap_ms - wait_overlap_ms)\n                    outside_main_ms = max(0.0, materialisation_ms - temporal_overlap_ms)\n                    dependency_ms = None\n                    lead_window_ms = None\n                    shortfall_ms = None\n                    slack_ms = None\n                    dependency_before_start = False\n                    if candidate.dependency_event is not None:\n                        dependency_count += 1\n                        dependency_ms = at_ms(candidate.dependency_event)\n                        lead_window_ms = max(0.0, dependency_ms - start_ms)\n                        lead_window_ms_total += lead_window_ms\n                        dependency_before_start = dependency_ms < start_ms\n                        if dependency_before_start:\n                            dependency_before_start_count += 1\n                        completion_delta_ms = end_ms - dependency_ms\n                        shortfall_ms = max(0.0, completion_delta_ms)\n                        slack_ms = max(0.0, -completion_delta_ms)\n                        shortfall_ms_total += shortfall_ms\n                        slack_ms_total += slack_ms\n                        if shortfall_ms > 0.0:\n                            shortfall_count += 1\n\n                    row = {\n                        "layer_index": candidate.layer_index,\n                        "family": candidate.family,\n                        "start_ms": start_ms,\n                        "end_ms": end_ms,\n                        "materialisation_ms": materialisation_ms,\n                        "dependency_ms": dependency_ms,\n                        "lead_window_ms": lead_window_ms,\n                        "shortfall_ms": shortfall_ms,\n                        "slack_ms": slack_ms,\n                        "dependency_before_premat_start": dependency_before_start,\n                        "temporal_overlap_ms": temporal_overlap_ms,\n                        "wait_overlap_ms": wait_overlap_ms,\n                        "useful_overlap_ms": useful_overlap_ms,\n                        "outside_main_layer_ms": outside_main_ms,\n                        "consumed": bool(candidate.consumed),\n                    }\n                    candidate_rows.append(row)\n                    family_row = pass_family_rows.setdefault(\n                        candidate.family, self._new_forensic_family_row()\n                    )\n                    family_row["launched"] += 1\n                    family_row["consumed"] += int(candidate.consumed)\n                    family_row["unused"] += int(not candidate.consumed)\n                    family_row["premat_gpu_ms_total"] += materialisation_ms\n                    family_row["temporal_overlap_ms_total"] += temporal_overlap_ms\n                    family_row["wait_overlap_ms_total"] += wait_overlap_ms\n                    family_row["useful_overlap_ms_total"] += useful_overlap_ms\n                    family_row["outside_main_layer_ms_total"] += outside_main_ms\n                    if candidate.dependency_event is not None:\n                        family_row["dependency_count"] += 1\n                        family_row["dependency_shortfall_count"] += int((shortfall_ms or 0.0) > 0.0)\n                        family_row["dependency_shortfall_ms_total"] += float(shortfall_ms or 0.0)\n                        family_row["dependency_slack_ms_total"] += float(slack_ms or 0.0)\n                        family_row["dependency_lead_window_ms_total"] += float(lead_window_ms or 0.0)\n                        family_row["dependency_before_premat_start_count"] += int(dependency_before_start)\n\n                    premat_ms_total += materialisation_ms\n                    temporal_overlap_total += temporal_overlap_ms\n                    wait_overlap_total += wait_overlap_ms\n                    useful_overlap_total += useful_overlap_ms\n                    outside_main_total += outside_main_ms\n\n                main_layer_ms_total = sum(float(row["gpu_ms"]) for row in layer_rows)\n                main_wait_marker_ms_total = sum(max(0.0, end - start) for start, end in wait_intervals)\n                main_nonwait_ms_total = max(0.0, main_layer_ms_total - main_wait_marker_ms_total)\n                summary_values: Dict[str, float | int] = {\n                    "premat_launched": len(pending.candidate_intervals),\n                    "premat_consumed": sum(int(item.consumed) for item in pending.candidate_intervals),\n                    "premat_unused": sum(int(not item.consumed) for item in pending.candidate_intervals),\n                    "duplicate_main_materialisations": 0,\n                    "main_layer_count": len(layer_rows),\n                    "main_layer_gpu_ms_total": main_layer_ms_total,\n                    "main_wait_marker_ms_total": main_wait_marker_ms_total,\n                    "main_nonwait_gpu_ms_total": main_nonwait_ms_total,\n                    "premat_gpu_ms_total": premat_ms_total,\n                    "premat_temporal_overlap_ms_total": temporal_overlap_total,\n                    "premat_wait_overlap_ms_total": wait_overlap_total,\n                    "premat_useful_overlap_ms_total": useful_overlap_total,\n                    "premat_outside_main_layer_ms_total": outside_main_total,\n                    "dependency_count": dependency_count,\n                    "dependency_shortfall_count": shortfall_count,\n                    "dependency_shortfall_ms_total": shortfall_ms_total,\n                    "dependency_slack_ms_total": slack_ms_total,\n                    "dependency_lead_window_ms_total": lead_window_ms_total,\n                    "dependency_before_premat_start_count": dependency_before_start_count,\n                }\n                summary = self._forensic_derived_summary(summary_values)\n                pass_report: Dict[str, object] = {\n                    "pass_sequence": pending.pass_sequence,\n                    "summary": summary,\n                    "layers": layer_rows,\n                    "candidates": candidate_rows,\n                    "by_family": {\n                        family: self._forensic_derived_summary({\n                            "premat_launched": int(row["launched"]),\n                            "premat_consumed": int(row["consumed"]),\n                            "premat_unused": int(row["unused"]),\n                            "duplicate_main_materialisations": int(row["duplicates"]),\n                            "main_layer_count": 0,\n                            "main_layer_gpu_ms_total": 0.0,\n                            "main_wait_marker_ms_total": 0.0,\n                            "main_nonwait_gpu_ms_total": 0.0,\n                            "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),\n                            "premat_temporal_overlap_ms_total": float(row["temporal_overlap_ms_total"]),\n                            "premat_wait_overlap_ms_total": float(row["wait_overlap_ms_total"]),\n                            "premat_useful_overlap_ms_total": float(row["useful_overlap_ms_total"]),\n                            "premat_outside_main_layer_ms_total": float(row["outside_main_layer_ms_total"]),\n                            "dependency_count": int(row["dependency_count"]),\n                            "dependency_shortfall_count": int(row["dependency_shortfall_count"]),\n                            "dependency_shortfall_ms_total": float(row["dependency_shortfall_ms_total"]),\n                            "dependency_slack_ms_total": float(row["dependency_slack_ms_total"]),\n                            "dependency_lead_window_ms_total": float(row["dependency_lead_window_ms_total"]),\n                            "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),\n                        })\n                        for family, row in pass_family_rows.items()\n                    },\n                }\n            except RuntimeError:\n                remaining.append(pending)\n                continue\n\n            aggregate_deltas: Dict[str, float | int] = {\n                "forensic_resolved_passes": 1,\n                "forensic_main_layer_count": len(layer_rows),\n                "forensic_main_layer_gpu_ms_total": main_layer_ms_total,\n                "forensic_main_wait_marker_ms_total": main_wait_marker_ms_total,\n                "forensic_main_nonwait_gpu_ms_total": main_nonwait_ms_total,\n                "forensic_premat_gpu_ms_total": premat_ms_total,\n                "forensic_premat_temporal_overlap_ms_total": temporal_overlap_total,\n                "forensic_premat_wait_overlap_ms_total": wait_overlap_total,\n                "forensic_premat_useful_overlap_ms_total": useful_overlap_total,\n                "forensic_premat_outside_main_layer_ms_total": outside_main_total,\n                "forensic_dependency_count": dependency_count,\n                "forensic_dependency_shortfall_count": shortfall_count,\n                "forensic_dependency_shortfall_ms_total": shortfall_ms_total,\n                "forensic_dependency_slack_ms_total": slack_ms_total,\n                "forensic_dependency_lead_window_ms_total": lead_window_ms_total,\n                "forensic_dependency_before_premat_start_count": dependency_before_start_count,\n            }\n            for name, delta in aggregate_deltas.items():\n                self._aggregate[name] += delta\n            for family, row in pass_family_rows.items():\n                cumulative = self._forensic_by_family.setdefault(\n                    family, self._new_forensic_family_row()\n                )\n                for name in (\n                    "premat_gpu_ms_total",\n                    "temporal_overlap_ms_total",\n                    "wait_overlap_ms_total",\n                    "useful_overlap_ms_total",\n                    "outside_main_layer_ms_total",\n                    "dependency_count",\n                    "dependency_shortfall_count",\n                    "dependency_shortfall_ms_total",\n                    "dependency_slack_ms_total",\n                    "dependency_lead_window_ms_total",\n                    "dependency_before_premat_start_count",\n                ):\n                    cumulative[name] += row[name]\n            self._latest_forensic_pass = pass_report\n            self._update_pending_live_report_forensic(\n                pending.pass_sequence, aggregate_deltas, pass_report\n            )\n        self._pending_forensic_passes = remaining\n        self._publish_completed_live_reports_if_ready()\n    # ^^^ THOG\n\n    def begin(\n''',
    'forensic methods',
)

source = replace_once(
    source,
    '''        self._resolve_pending_timings()\n        # One forward pass is one accumulation microstep's complete Premat\n''',
    '''        self._resolve_pending_timings()\n        self._resolve_pending_forensic_passes()                                                                          # <<< THOG resolve prior GPU-forensic pass without synchronising\n        # One forward pass is one accumulation microstep's complete Premat\n''',
    'begin resolves forensic',
)

source = replace_once(
    source,
    '''        self._pass_start_ns = time.perf_counter_ns()\n        self._active = True\n        observation = self._observe_memory()\n''',
    '''        self._pass_start_ns = time.perf_counter_ns()\n        self._active = True\n        # vvv THOG one timing-enabled Main Stream origin makes cross-stream overlap arithmetic unambiguous\n        self._forensic_current_layers = []\n        self._forensic_current_candidates = []\n        self._forensic_pass_origin_event = None\n        if self._enable_gpu_timing_diagnostic:\n            self._forensic_pass_origin_event = torch.cuda.Event(enable_timing=True)\n            self._forensic_pass_origin_event.record(\n                torch.cuda.current_stream(device=self._device)\n            )\n        # ^^^ THOG\n        observation = self._observe_memory()\n''',
    'begin forensic origin',
)

source = replace_once(
    source,
    '''    def end(self) -> None:\n        if not self._active:\n            return\n        self._resolve_pending_timings()\n        for candidate in tuple(self._candidates.values()):\n''',
    '''    def end(self) -> None:\n        if not self._active:\n            return\n        self._resolve_pending_timings()\n        self._resolve_pending_forensic_passes()                                                                          # <<< THOG opportunistically drain older forensic passes\n        # vvv THOG classify real side materialisations that reached pass end without a consumption callback\n        for candidate in tuple(self._candidates.values()):\n            if candidate.real_premat_launched and not candidate.real_premat_consumed:\n                self._aggregate["real_premat_materialisations_unused"] += 1\n                self._forensic_by_family[candidate.family]["unused"] += 1\n        # ^^^ THOG\n        for candidate in tuple(self._candidates.values()):\n''',
    'end unused counters',
)

source = replace_once(
    source,
    '''        if self._pass_start_ns is not None:\n            self._aggregate["captured_pass_host_ms_total"] += max(\n                0.0,\n                (time.perf_counter_ns() - self._pass_start_ns) / 1_000_000.0,\n            )\n        self._capture_completed_live_report()\n''',
    '''        if self._pass_start_ns is not None:\n            self._aggregate["captured_pass_host_ms_total"] += max(\n                0.0,\n                (time.perf_counter_ns() - self._pass_start_ns) / 1_000_000.0,\n            )\n        self._queue_forensic_pass()                                                                                       # <<< THOG defer GPU-forensic arithmetic until all timing events complete\n        self._resolve_pending_forensic_passes()                                                                           # <<< THOG publish immediately when the GPU is already caught up\n        self._capture_completed_live_report()\n''',
    'end queue forensic',
)

source = replace_once(
    source,
    '''        current_stream = torch.cuda.current_stream(device=self._device)\n        # vvv THOG shadow PREMAT preserves scheduler/admission/event overhead but Main Stream still materialises the real weight\n''',
    '''        current_stream = torch.cuda.current_stream(device=self._device)\n        # vvv THOG shadow PREMAT preserves scheduler/admission/event overhead but Main Stream still materialises the real weight\n''',
    'acquire current stream anchor',
)

source = replace_once(
    source,
    '''        if candidate.state == CandidateState.AVAILABLE:\n            self._aggregate["available_hits"] += 1\n            self._aggregate["fully_hidden_hits"] += 1\n            candidate.final_outcome = "FULL HIT"\n''',
    '''        if candidate.state == CandidateState.AVAILABLE:\n            self._record_forensic_dependency(candidate, current_stream)                                                    # <<< THOG mark actual Main Stream matrix-use dependency even when PREMAT is already complete\n            self._aggregate["available_hits"] += 1\n            self._aggregate["fully_hidden_hits"] += 1\n            candidate.final_outcome = "FULL HIT"\n''',
    'available dependency marker',
)

source = replace_once(
    source,
    '''                current_stream.wait_event(candidate.completion_event)\n                wait_end.record(current_stream)\n                # Host submission has reached the dependency, but this does NOT tell\n''',
    '''                current_stream.wait_event(candidate.completion_event)\n                wait_end.record(current_stream)\n                self._record_forensic_dependency(\n                    candidate, current_stream,\n                    dependency_event=wait_start,\n                    wait_end_event=wait_end,\n                )                                                                                                         # <<< THOG retain the true dependency/wait interval for overlap forensics\n                # Host submission has reached the dependency, but this does NOT tell\n''',
    'materialising dependency marker',
)

source = replace_once(
    source,
    '''        elif candidate.state == CandidateState.UNAVAILABLE:\n            candidate.owner = "main"\n            candidate.critical_path_miss = True\n''',
    '''        elif candidate.state == CandidateState.UNAVAILABLE:\n            # vvv THOG a Main fallback after a real side launch is a true duplicate and must be visible, not inferred\n            if candidate.real_premat_launched:\n                self._aggregate["duplicate_main_materialisations_after_premat"] += 1\n                self._forensic_by_family[candidate.family]["duplicates"] += 1\n            # ^^^ THOG\n            candidate.owner = "main"\n            candidate.critical_path_miss = True\n''',
    'duplicate fallback counter',
)

source = replace_once(
    source,
    '''    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:\n        # Checkpoint replay has no schedulable lookahead lifetime.  Recreate the\n        # exact ordinary differentiable materialisation on its execution stream.\n        return self._materialize(family, layer_index)\n''',
    '''    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:\n        # Checkpoint replay has no schedulable lookahead lifetime.  Recreate the\n        # exact ordinary differentiable materialisation on its execution stream.\n        # vvv THOG catch any unexpected in-pass bypass of an already-launched PREMAT candidate as duplicate work\n        if self._active:\n            candidate = self._candidates.get((int(layer_index), str(family)))\n            if candidate is not None and candidate.real_premat_launched:\n                self._aggregate["duplicate_main_materialisations_after_premat"] += 1\n                self._forensic_by_family[candidate.family]["duplicates"] += 1\n        # ^^^ THOG\n        return self._materialize(family, layer_index)\n''',
    'duplicate replay guard',
)

source = replace_once(
    source,
    '''        candidate.consumed_ns = time.perf_counter_ns()\n        candidate.tensor = None\n''',
    '''        # vvv THOG distinguish useful real PREMAT work from launched-but-never-consumed work\n        if candidate.real_premat_launched and not candidate.real_premat_consumed:\n            candidate.real_premat_consumed = True\n            self._aggregate["real_premat_materialisations_consumed"] += 1\n            self._forensic_by_family[candidate.family]["consumed"] += 1\n            if candidate.forensic_interval is not None:\n                candidate.forensic_interval.consumed = True\n        # ^^^ THOG\n        candidate.consumed_ns = time.perf_counter_ns()\n        candidate.tensor = None\n''',
    'consumed counter',
)

source = replace_once(
    source,
    '''    def report(self) -> Dict[str, object]:\n        self._resolve_pending_timings()\n        self._resolve_pending_releases()\n''',
    '''    def report(self) -> Dict[str, object]:\n        self._resolve_pending_timings()\n        self._resolve_pending_forensic_passes()                                                                          # <<< THOG make synchronized progress/final reports carry resolved forensic evidence\n        self._resolve_pending_releases()\n''',
    'report resolves forensic',
)

source = replace_once(
    source,
    '''        pass_complete = bool(\n            self._events\n            and self._events[-1].get("event") == "pass_end"\n        )\n        return {\n''',
    '''        pass_complete = bool(\n            self._events\n            and self._events[-1].get("event") == "pass_end"\n        )\n        forensic_pass = (\n            dict(self._latest_forensic_pass)\n            if self._latest_forensic_pass is not None\n            and int(self._latest_forensic_pass.get("pass_sequence", -1)) == int(self._pass_sequence)\n            else None\n        )\n        return {\n''',
    'report forensic pass selection',
)

source = replace_once(
    source,
    '''            "aggregate": aggregate,\n            "live_publish_error": self._live_publish_error,\n''',
    '''            "aggregate": aggregate,\n            # vvv THOG cumulative and per-pass evidence map directly onto the six first-tranche hypotheses\n            "forensic": self._forensic_report(),\n            "forensic_pass": forensic_pass,\n            # ^^^ THOG\n            "live_publish_error": self._live_publish_error,\n''',
    'report forensic payload',
)

source = replace_once(
    source,
    '''                    # ^^^ THOG\n            except BaseException as error:\n''',
    '''                    # ^^^ THOG\n            except BaseException as error:\n''',
    'advance success anchor noop',
)

source = replace_once(
    source,
    '''            self._aggregate["admitted"] += 1\n            observed_admission_lag_ms = max(\n''',
    '''            # vvv THOG count real side work independently of admission and retain its immutable CUDA interval for overlap analysis\n            if not self._shadow_mode:\n                if candidate.real_premat_launched:\n                    raise RuntimeError(\n                        "PREMAT candidate launched real side materialisation twice: "\n                        f"layer={candidate.layer_index}, family={candidate.family}"\n                    )\n                candidate.real_premat_launched = True\n                self._aggregate["real_premat_materialisations_launched"] += 1\n                self._forensic_by_family[candidate.family]["launched"] += 1\n                if (\n                    self._enable_gpu_timing_diagnostic\n                    and candidate.materialisation_start_event is not None\n                    and candidate.completion_event is not None\n                ):\n                    candidate.forensic_interval = _ForensicCandidateInterval(\n                        pass_sequence=self._pass_sequence,\n                        layer_index=candidate.layer_index,\n                        family=candidate.family,\n                        start_event=candidate.materialisation_start_event,\n                        end_event=candidate.completion_event,\n                    )\n                    self._forensic_current_candidates.append(candidate.forensic_interval)\n            # ^^^ THOG\n            self._aggregate["admitted"] += 1\n            observed_admission_lag_ms = max(\n''',
    'advance real launch tracking',
)

source = replace_once(
    source,
    '''        pending_passes = {\n            int(timing.pass_sequence)\n            for timing in self._pending_timings\n            if int(timing.pass_sequence) > 0\n        }\n''',
    '''        pending_passes = {\n            int(timing.pass_sequence)\n            for timing in self._pending_timings\n            if int(timing.pass_sequence) > 0\n        }\n        pending_passes.update(                                                                                           # <<< THOG a sampled Instra pass waits for overlap forensics as well as legacy timing classification\n            int(pending.pass_sequence)\n            for pending in self._pending_forensic_passes\n            if int(pending.pass_sequence) > 0\n        )\n''',
    'live publish waits for forensic',
)

premat_path.write_text(source)


model_path = Path("sheet/model.py")
model = model_path.read_text()
model = replace_once(
    model,
    '''    def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:\n        if self._premat_runtime is not None and self._premat_runtime.active:\n            self._premat_runtime.layer_start(layer_index)\n        layer_materializations = None\n''',
    '''    def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:\n        if self._premat_runtime is not None and self._premat_runtime.active:\n            self._premat_runtime.layer_start(layer_index)\n            self._premat_runtime.forensic_layer_start(layer_index)                                                      # <<< THOG time Main Stream layer body only when explicit GPU diagnostic is enabled\n        layer_materializations = None\n''',
    'model forensic layer start',
)
model = replace_once(
    model,
    '''        if self.config.fast_discard:\n            del layer_materializations, hyperblock_mlp_factors\n        if self._premat_runtime is not None and self._premat_runtime.active:\n            self._premat_runtime.layer_complete(layer_index)\n        return inputs\n''',
    '''        if self.config.fast_discard:\n            del layer_materializations, hyperblock_mlp_factors\n        if self._premat_runtime is not None and self._premat_runtime.active:\n            self._premat_runtime.forensic_layer_end(layer_index)                                                        # <<< THOG close Main Stream layer interval before optional host-only delay\n            self._premat_runtime.layer_complete(layer_index)\n        return inputs\n''',
    'model forensic layer end',
)
model_path.write_text(model)


js_path = Path("sheet/local_dashboard_assets/dashboard_premat.js")
js = js_path.read_text()
old_summary = '''  by_id("premat_summary").innerHTML = [\n    ["mode", model.attention_mode],\n    ["target", Number(snapshot.target_layer ?? snapshot.target_offset ?? 1) === 10 ? "l+1 → l+0" : `l+${Number(snapshot.target_offset ?? snapshot.target_layer ?? 1)}`],\n    ["matrix order", String(snapshot.matrix_order ?? snapshot.weight_matrix_target_order ?? snapshot.target_order ?? "r_to_l")],\n    ["priority", snapshot.cuda_stream_priority || "normal"],\n    ["layer delay", `${Number(snapshot.diagnostic_layer_delay_ms || 0)} ms`],\n    ["buffer margin", margin_text],\n    ["headroom", premat_bytes(memory.premat_headroom_bytes)],\n    ["full hits", String(outcomes["FULL HIT"])],\n    ["partial hits", String(outcomes["PARTIAL HIT"])],\n    ["complete misses", String(outcomes["COMPLETE MISS"])],\n  ].map(([label, value]) => premat_summary_item(label, value)).join("");\n'''
new_summary = '''  // vvv THOG timing-diagnostic snapshots expose six forensic discriminators without requiring detailed history retention\n  const summary_items = [\n    ["mode", model.attention_mode],\n    ["target", Number(snapshot.target_layer ?? snapshot.target_offset ?? 1) === 10 ? "l+1 → l+0" : `l+${Number(snapshot.target_offset ?? snapshot.target_layer ?? 1)}`],\n    ["matrix order", String(snapshot.matrix_order ?? snapshot.weight_matrix_target_order ?? snapshot.target_order ?? "r_to_l")],\n    ["priority", snapshot.cuda_stream_priority || "normal"],\n    ["layer delay", `${Number(snapshot.diagnostic_layer_delay_ms || 0)} ms`],\n    ["buffer margin", margin_text],\n    ["headroom", premat_bytes(memory.premat_headroom_bytes)],\n    ["full hits", String(outcomes["FULL HIT"])],\n    ["partial hits", String(outcomes["PARTIAL HIT"])],\n    ["complete misses", String(outcomes["COMPLETE MISS"])],\n  ];\n  const forensic = snapshot?.forensic_pass?.summary || null;\n  if (snapshot?.enable_gpu_timing_diagnostic === true && forensic) {\n    const percent = value => Number.isFinite(Number(value)) ? `${(100 * Number(value)).toFixed(1)}%` : "—";\n    summary_items.push(\n      ["MAIN GPU / layer", premat_ms(forensic.main_layer_gpu_ms_mean)],\n      ["MAIN non-wait / layer", premat_ms(forensic.main_nonwait_gpu_ms_mean)],\n      ["useful PREMAT overlap", percent(forensic.premat_useful_overlap_fraction)],\n      ["PREMAT outside MAIN", percent(forensic.premat_outside_main_layer_fraction)],\n      ["dependency shortfall", `${premat_ms(forensic.dependency_shortfall_ms_total)} · ${Number(forensic.dependency_shortfall_count || 0)}/${Number(forensic.dependency_count || 0)}`],\n      ["PREMAT launched / used / unused", `${Number(forensic.premat_launched || 0)} / ${Number(forensic.premat_consumed || 0)} / ${Number(forensic.premat_unused || 0)}`],\n      ["duplicate materialisations", String(Number(forensic.duplicate_main_materialisations || 0))],\n      ["dependency before PREMAT start", String(Number(forensic.dependency_before_premat_start_count || 0))],\n    );\n  }\n  by_id("premat_summary").innerHTML = summary_items\n    .map(([label, value]) => premat_summary_item(label, value)).join("");\n  // ^^^ THOG\n'''
js = replace_once(js, old_summary, new_summary, 'Instra forensic summary')
js_path.write_text(js)


test_path = Path("tests/test_premat.py")
tests = test_path.read_text()
tests = tests.replace('assert canonical["premat_schema_version"] == 3', 'assert canonical["premat_schema_version"] == 4', 1)

insert_anchor = '''# vvv THOG shadow PREMAT regression coverage\n'''
new_tests = r'''# vvv THOG first-tranche GPU forensic regression coverage

def test_real_premat_launch_use_and_unused_counters_are_unambiguous(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        enable_gpu_timing_diagnostic=False,
    )
    runtime.layer_start(3)
    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    runtime.layer_start(5)
    runtime.acquire("DOWN", 5)
    runtime.consumed("DOWN", 5)
    runtime.end()
    aggregate = runtime.report()["aggregate"]
    assert aggregate["real_premat_materialisations_launched"] == 4
    assert aggregate["real_premat_materialisations_consumed"] == 1
    assert aggregate["real_premat_materialisations_unused"] == 3
    assert aggregate["duplicate_main_materialisations_after_premat"] == 0


def test_forensic_pass_separates_useful_overlap_wait_and_slack(monkeypatch) -> None:
    from sheet.premat import (
        _ForensicCandidateInterval,
        _ForensicLayerInterval,
        _PendingForensicPass,
    )

    runtime, fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        enable_gpu_timing_diagnostic=True,
    )
    origin = _FakeEvent(fake_cuda, enable_timing=True)
    origin.complete = True

    def event(timestamp_ms: float):
        item = _FakeEvent(fake_cuda, enable_timing=True)
        item.complete = True
        item.forensic_timestamp_ms = timestamp_ms
        return item

    origin.elapsed_time_override = lambda other: float(other.forensic_timestamp_ms)
    layer = _ForensicLayerInterval(
        pass_sequence=1,
        layer_index=3,
        start_event=event(1.0),
        end_event=event(11.0),
    )
    candidate = _ForensicCandidateInterval(
        pass_sequence=1,
        layer_index=5,
        family="DOWN",
        start_event=event(2.0),
        end_event=event(9.0),
        dependency_event=event(8.0),
        wait_end_event=event(9.0),
        consumed=True,
    )
    runtime._pending_forensic_passes = [
        _PendingForensicPass(
            pass_sequence=1,
            origin_event=origin,
            layer_intervals=(layer,),
            candidate_intervals=(candidate,),
        )
    ]
    runtime._resolve_pending_forensic_passes()
    report = runtime.report()
    forensic = report["forensic"]
    assert forensic["resolved_passes"] == 1
    assert forensic["main_layer_gpu_ms_total"] == pytest.approx(10.0)
    assert forensic["main_wait_marker_ms_total"] == pytest.approx(1.0)
    assert forensic["main_nonwait_gpu_ms_total"] == pytest.approx(9.0)
    assert forensic["premat_gpu_ms_total"] == pytest.approx(7.0)
    assert forensic["premat_temporal_overlap_ms_total"] == pytest.approx(7.0)
    assert forensic["premat_wait_overlap_ms_total"] == pytest.approx(1.0)
    assert forensic["premat_useful_overlap_ms_total"] == pytest.approx(6.0)
    assert forensic["premat_outside_main_layer_ms_total"] == pytest.approx(0.0)
    assert forensic["dependency_count"] == 1
    assert forensic["dependency_shortfall_count"] == 1
    assert forensic["dependency_shortfall_ms_total"] == pytest.approx(1.0)
    assert forensic["dependency_lead_window_ms_total"] == pytest.approx(6.0)
    assert forensic["dependency_before_premat_start_count"] == 0
    latest = report["forensic_pass"]["summary"]
    assert latest["premat_useful_overlap_fraction"] == pytest.approx(6.0 / 7.0)
    assert latest["dependency_shortfall_rate"] == pytest.approx(1.0)


def test_compact_storage_preserves_forensic_pass_without_detailed_history() -> None:
    from sheet.wandb_telemetry import _premat_snapshot_for_storage

    forensic_pass = {
        "pass_sequence": 7,
        "summary": {"premat_useful_overlap_fraction": 0.75},
        "candidates": [{"family": "DOWN", "useful_overlap_ms": 2.5}],
    }
    stored = _premat_snapshot_for_storage(
        {
            "pass_complete": True,
            "events": [{"sequence": 1, "event": "pass_end"}],
            "candidates": [{"large": "ordinary detailed candidate"}],
            "forensic_pass": forensic_pass,
        },
        retain_detailed_history=False,
    )
    assert stored["forensic_pass"] == forensic_pass
    assert "candidates" not in stored


def test_premat_instra_surfaces_first_tranche_forensic_discriminators() -> None:
    javascript = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")
    assert "MAIN non-wait / layer" in javascript
    assert "useful PREMAT overlap" in javascript
    assert "PREMAT outside MAIN" in javascript
    assert "dependency shortfall" in javascript
    assert "PREMAT launched / used / unused" in javascript
    assert "duplicate materialisations" in javascript
# ^^^ THOG


'''
if insert_anchor not in tests:
    raise SystemExit('missing test insertion anchor')
tests = tests.replace(insert_anchor, new_tests + insert_anchor, 1)
test_path.write_text(tests)
