from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


premat_path = Path("sheet/premat.py")
source = premat_path.read_text()

source = replace_once(
    source,
    '''@dataclass
class _PendingForensicPass:
    pass_sequence: int
    origin_event: torch.cuda.Event
    layer_intervals: Tuple[_ForensicLayerInterval, ...]
    candidate_intervals: Tuple[_ForensicCandidateInterval, ...]
# ^^^ THOG
''',
    '''@dataclass
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
''',
    "main work forensic dataclass",
)

source = replace_once(
    source,
    '''        self._forensic_current_layers: List[_ForensicLayerInterval] = []
        self._forensic_current_candidates: List[_ForensicCandidateInterval] = []
        self._pending_forensic_passes: List[_PendingForensicPass] = []
''',
    '''        self._forensic_current_layers: List[_ForensicLayerInterval] = []
        self._forensic_current_candidates: List[_ForensicCandidateInterval] = []
        self._forensic_current_main_work: List[_ForensicMainWorkInterval] = []
        self._pending_forensic_passes: List[_PendingForensicPass] = []
''',
    "main work runtime state",
)

source = replace_once(
    source,
    '''            "forensic_main_nonwait_gpu_ms_total": 0.0,
            "forensic_premat_gpu_ms_total": 0.0,
''',
    '''            "forensic_main_nonwait_gpu_ms_total": 0.0,
            "forensic_main_consume_count": 0,
            "forensic_main_consume_gpu_ms_total": 0.0,
            "forensic_premat_overlap_during_main_consume_ms_total": 0.0,
            "forensic_premat_gpu_ms_total": 0.0,
''',
    "main work aggregate fields",
)

source = replace_once(
    source,
    '''            "duplicates": 0,
            "premat_gpu_ms_total": 0.0,
''',
    '''            "duplicates": 0,
            "main_consume_count": 0,
            "main_consume_gpu_ms_total": 0.0,
            "premat_overlap_during_main_consume_ms_total": 0.0,
            "premat_gpu_ms_total": 0.0,
''',
    "family main work fields",
)

source = replace_once(
    source,
    '''        interval.end_event = event

    def _record_forensic_dependency(
''',
    '''        interval.end_event = event

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
''',
    "main work marker methods",
)

source = replace_once(
    source,
    '''        if any(interval.end_event is None for interval in self._forensic_current_layers):
            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream layer interval")
        self._pending_forensic_passes.append(
''',
    '''        if any(interval.end_event is None for interval in self._forensic_current_layers):
            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream layer interval")
        if any(interval.end_event is None for interval in self._forensic_current_main_work):
            raise RuntimeError("PREMAT forensic timing pass ended with an open Main Stream work interval")
        self._pending_forensic_passes.append(
''',
    "main work queue validation",
)

source = replace_once(
    source,
    '''                layer_intervals=tuple(self._forensic_current_layers),
                candidate_intervals=tuple(self._forensic_current_candidates),
            )
        )
        self._forensic_pass_origin_event = None
        self._forensic_current_layers = []
        self._forensic_current_candidates = []
''',
    '''                layer_intervals=tuple(self._forensic_current_layers),
                candidate_intervals=tuple(self._forensic_current_candidates),
                main_work_intervals=tuple(self._forensic_current_main_work),
            )
        )
        self._forensic_pass_origin_event = None
        self._forensic_current_layers = []
        self._forensic_current_candidates = []
        self._forensic_current_main_work = []
''',
    "main work queue payload",
)

source = replace_once(
    source,
    '''        dependency_count = int(values.get("dependency_count", 0))
        launched = int(values.get("premat_launched", 0))
        return {
''',
    '''        dependency_count = int(values.get("dependency_count", 0))
        launched = int(values.get("premat_launched", 0))
        main_consume_count = int(values.get("main_consume_count", 0))
        main_consume_ms = float(values.get("main_consume_gpu_ms_total", 0.0))
        return {
''',
    "derived main work locals",
)

source = replace_once(
    source,
    '''            "main_nonwait_gpu_ms_mean": (
                float(values.get("main_nonwait_gpu_ms_total", 0.0)) / layer_count
                if layer_count > 0
                else None
            ),
            "premat_useful_overlap_fraction": (
''',
    '''            "main_nonwait_gpu_ms_mean": (
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
''',
    "derived main work summary",
)

source = replace_once(
    source,
    '''            "main_nonwait_gpu_ms_total": float(aggregate["forensic_main_nonwait_gpu_ms_total"]),
            "premat_gpu_ms_total": float(aggregate["forensic_premat_gpu_ms_total"]),
''',
    '''            "main_nonwait_gpu_ms_total": float(aggregate["forensic_main_nonwait_gpu_ms_total"]),
            "main_consume_count": int(aggregate["forensic_main_consume_count"]),
            "main_consume_gpu_ms_total": float(aggregate["forensic_main_consume_gpu_ms_total"]),
            "premat_overlap_during_main_consume_ms_total": float(
                aggregate["forensic_premat_overlap_during_main_consume_ms_total"]
            ),
            "premat_gpu_ms_total": float(aggregate["forensic_premat_gpu_ms_total"]),
''',
    "cumulative report main work",
)

source = replace_once(
    source,
    '''                "main_nonwait_gpu_ms_total": 0.0,
                "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
''',
    '''                "main_nonwait_gpu_ms_total": 0.0,
                "main_consume_count": int(row["main_consume_count"]),
                "main_consume_gpu_ms_total": float(row["main_consume_gpu_ms_total"]),
                "premat_overlap_during_main_consume_ms_total": float(
                    row["premat_overlap_during_main_consume_ms_total"]
                ),
                "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
''',
    "family report main work",
)

source = replace_once(
    source,
    '''            for candidate in pending.candidate_intervals:
                required_events.extend((candidate.start_event, candidate.end_event))
                if candidate.dependency_event is not None:
''',
    '''            for candidate in pending.candidate_intervals:
                required_events.extend((candidate.start_event, candidate.end_event))
                if candidate.dependency_event is not None:
''',
    "candidate required events noop",
)
source = replace_once(
    source,
    '''                if candidate.wait_end_event is not None:
                    required_events.append(candidate.wait_end_event)
            try:
''',
    '''                if candidate.wait_end_event is not None:
                    required_events.append(candidate.wait_end_event)
            for main_work in pending.main_work_intervals:
                if main_work.end_event is None:
                    raise RuntimeError("PREMAT forensic pass retained an open Main Stream work interval")
                required_events.extend((main_work.start_event, main_work.end_event))
            try:
''',
    "main work required events",
)

source = replace_once(
    source,
    '''                wait_intervals: List[Tuple[float, float]] = []
                for candidate in pending.candidate_intervals:
''',
    '''                # vvv THOG exact foreground GEMM intervals are directly comparable between REAL and SHADOW PREMAT
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
''',
    "main work rows",
)

source = replace_once(
    source,
    '''                candidate_rows: List[Dict[str, object]] = []
                pass_family_rows: Dict[str, Dict[str, float | int]] = {}
                premat_ms_total = 0.0
''',
    '''                candidate_rows: List[Dict[str, object]] = []
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
''',
    "main work pass totals",
)

source = replace_once(
    source,
    '''                    useful_overlap_ms = max(0.0, temporal_overlap_ms - wait_overlap_ms)
                    outside_main_ms = max(0.0, materialisation_ms - temporal_overlap_ms)
                    dependency_ms = None
''',
    '''                    useful_overlap_ms = max(0.0, temporal_overlap_ms - wait_overlap_ms)
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
''',
    "candidate main work overlap",
)

source = replace_once(
    source,
    '''                        "outside_main_layer_ms": outside_main_ms,
                        "consumed": bool(candidate.consumed),
''',
    '''                        "outside_main_layer_ms": outside_main_ms,
                        "main_consume_overlap_ms": main_consume_overlap_ms,
                        "consumed": bool(candidate.consumed),
''',
    "candidate row main work overlap",
)

source = replace_once(
    source,
    '''                    family_row["outside_main_layer_ms_total"] += outside_main_ms
                    if candidate.dependency_event is not None:
''',
    '''                    family_row["outside_main_layer_ms_total"] += outside_main_ms
                    family_row["premat_overlap_during_main_consume_ms_total"] += main_consume_overlap_ms
                    if candidate.dependency_event is not None:
''',
    "family side main work overlap",
)

source = replace_once(
    source,
    '''                    "main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                    "premat_gpu_ms_total": premat_ms_total,
''',
    '''                    "main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                    "main_consume_count": len(main_work_rows),
                    "main_consume_gpu_ms_total": main_consume_ms_total,
                    "premat_overlap_during_main_consume_ms_total": premat_overlap_during_main_consume_total,
                    "premat_gpu_ms_total": premat_ms_total,
''',
    "pass summary main work",
)

source = replace_once(
    source,
    '''                    "layers": layer_rows,
                    "candidates": candidate_rows,
                    "by_family": {
''',
    '''                    "layers": layer_rows,
                    "main_work": main_work_rows,
                    "candidates": candidate_rows,
                    "by_family": {
''',
    "pass report main work rows",
)

source = replace_once(
    source,
    '''                            "main_nonwait_gpu_ms_total": 0.0,
                            "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
''',
    '''                            "main_nonwait_gpu_ms_total": 0.0,
                            "main_consume_count": int(row["main_consume_count"]),
                            "main_consume_gpu_ms_total": float(row["main_consume_gpu_ms_total"]),
                            "premat_overlap_during_main_consume_ms_total": float(
                                row["premat_overlap_during_main_consume_ms_total"]
                            ),
                            "premat_gpu_ms_total": float(row["premat_gpu_ms_total"]),
''',
    "pass family summary main work",
)

source = replace_once(
    source,
    '''                "forensic_main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                "forensic_premat_gpu_ms_total": premat_ms_total,
''',
    '''                "forensic_main_nonwait_gpu_ms_total": main_nonwait_ms_total,
                "forensic_main_consume_count": len(main_work_rows),
                "forensic_main_consume_gpu_ms_total": main_consume_ms_total,
                "forensic_premat_overlap_during_main_consume_ms_total": premat_overlap_during_main_consume_total,
                "forensic_premat_gpu_ms_total": premat_ms_total,
''',
    "aggregate deltas main work",
)

source = replace_once(
    source,
    '''                    "premat_gpu_ms_total",
                    "temporal_overlap_ms_total",
''',
    '''                    "main_consume_count",
                    "main_consume_gpu_ms_total",
                    "premat_overlap_during_main_consume_ms_total",
                    "premat_gpu_ms_total",
                    "temporal_overlap_ms_total",
''',
    "family cumulative main work fields",
)

source = replace_once(
    source,
    '''        self._forensic_current_layers = []
        self._forensic_current_candidates = []
        self._forensic_pass_origin_event = None
''',
    '''        self._forensic_current_layers = []
        self._forensic_current_candidates = []
        self._forensic_current_main_work = []
        self._forensic_pass_origin_event = None
''',
    "begin resets main work",
)

premat_path.write_text(source)


model_path = Path("sheet/model.py")
model = model_path.read_text()

model = replace_once(
    model,
    '''    def _premat_event(self, name: str, layer_index: int) -> None:
        if self._premat_runtime is not None and self._premat_runtime.active:
            self._premat_runtime.event(name, layer_index=layer_index)
    # ^^^ THOG
''',
    '''    def _premat_event(self, name: str, layer_index: int) -> None:
        if self._premat_runtime is not None and self._premat_runtime.active:
            self._premat_runtime.event(name, layer_index=layer_index)

    # vvv THOG bracket identical foreground matrix-use GEMMs so REAL/SHADOW comparison isolates side-stream contention
    def _premat_forensic_main_work_start(self, family: str, layer_index: int) -> None:
        runtime = self._premat_runtime
        marker = getattr(runtime, "forensic_main_work_start", None)
        if runtime is not None and runtime.active and callable(marker):
            marker(family, layer_index)

    def _premat_forensic_main_work_end(self, family: str, layer_index: int) -> None:
        runtime = self._premat_runtime
        marker = getattr(runtime, "forensic_main_work_end", None)
        if runtime is not None and runtime.active and callable(marker):
            marker(family, layer_index)
    # ^^^ THOG
    # ^^^ THOG
''',
    "model main work helpers",
)

# Unfused QK and V foreground GEMMs.
model = replace_once(
    model,
    '''            query, key = F.linear(inputs, qk_weight, qk_bias).split(self.config.n_embd, dim=2)
''',
    '''            self._premat_forensic_main_work_start("QK", layer_index)
            query, key = F.linear(inputs, qk_weight, qk_bias).split(self.config.n_embd, dim=2)
            self._premat_forensic_main_work_end("QK", layer_index)
''',
    "QK foreground timing",
)
model = replace_once(
    model,
    '''            value = F.linear(inputs, value_weight, value_bias)
''',
    '''            self._premat_forensic_main_work_start("V", layer_index)
            value = F.linear(inputs, value_weight, value_bias)
            self._premat_forensic_main_work_end("V", layer_index)
''',
    "V foreground timing",
)
# Fused QKV foreground GEMM.
model = replace_once(
    model,
    '''            query, key, value = F.linear(inputs, attention_weight, attention_bias).split(self.config.n_embd, dim=2)
''',
    '''            self._premat_forensic_main_work_start("QKV", layer_index)
            query, key, value = F.linear(inputs, attention_weight, attention_bias).split(self.config.n_embd, dim=2)
            self._premat_forensic_main_work_end("QKV", layer_index)
''',
    "QKV foreground timing",
)
# Attention output foreground GEMM.
model = replace_once(
    model,
    '''        projected = F.linear(attended, output_weight, output_bias)
''',
    '''        self._premat_forensic_main_work_start("O", layer_index)
        projected = F.linear(attended, output_weight, output_bias)
        self._premat_forensic_main_work_end("O", layer_index)
''',
    "O foreground timing",
)
# DEPTH MLP foreground GEMMs; direct factorised/HYPERBLOCK paths remain untouched.
model = replace_once(
    model,
    '''            hidden = F.linear(inputs, expansion_weight, expansion_bias)
''',
    '''            self._premat_forensic_main_work_start("UP", layer_index)
            hidden = F.linear(inputs, expansion_weight, expansion_bias)
            self._premat_forensic_main_work_end("UP", layer_index)
''',
    "UP foreground timing",
)
model = replace_once(
    model,
    '''            output = F.linear(hidden, contraction_weight, contraction_bias)
''',
    '''            self._premat_forensic_main_work_start("DOWN", layer_index)
            output = F.linear(hidden, contraction_weight, contraction_bias)
            self._premat_forensic_main_work_end("DOWN", layer_index)
''',
    "DOWN foreground timing",
)
model_path.write_text(model)


js_path = Path("sheet/local_dashboard_assets/dashboard_premat.js")
js = js_path.read_text()
js = replace_once(
    js,
    '''      ["MAIN non-wait / layer", premat_ms(forensic.main_nonwait_gpu_ms_mean)],
      ["useful PREMAT overlap", percent(forensic.premat_useful_overlap_fraction)],
''',
    '''      ["MAIN non-wait / layer", premat_ms(forensic.main_nonwait_gpu_ms_mean)],
      ["MAIN consume GEMM mean", premat_ms(forensic.main_consume_gpu_ms_mean)],
      ["MAIN GEMM covered by PREMAT", percent(forensic.main_consume_covered_by_premat_fraction)],
      ["useful PREMAT overlap", percent(forensic.premat_useful_overlap_fraction)],
''',
    "Instra main work summary",
)
js = replace_once(
    js,
    '''      ["dependency before PREMAT start", String(Number(forensic.dependency_before_premat_start_count || 0))],
    );
''',
    '''      ["dependency before PREMAT start", String(Number(forensic.dependency_before_premat_start_count || 0))],
    );
    const family_order = model.attention_mode === "unfused" ? ["QK", "V", "O", "UP", "DOWN"] : ["QKV", "O", "UP", "DOWN"];
    const family_text = family_order.map(family => {
      const row = snapshot?.forensic_pass?.by_family?.[family];
      return `${family}:${row && Number.isFinite(Number(row.main_consume_gpu_ms_mean)) ? Number(row.main_consume_gpu_ms_mean).toFixed(3) : "—"}`;
    }).join(" · ");
    summary_items.push(["MAIN GEMM ms by family", family_text]);
''',
    "Instra family work summary",
)
js_path.write_text(js)


test_path = Path("tests/test_premat.py")
tests = test_path.read_text()
# The CPU checkpoint runtime is deliberately minimal; production model helpers tolerate its missing diagnostic methods.
# Add an explicit source assertion so that compatibility stays intentional.
tests = replace_once(
    tests,
    '''    assert "MAIN non-wait / layer" in javascript
    assert "useful PREMAT overlap" in javascript
''',
    '''    assert "MAIN non-wait / layer" in javascript
    assert "MAIN consume GEMM mean" in javascript
    assert "MAIN GEMM covered by PREMAT" in javascript
    assert "MAIN GEMM ms by family" in javascript
    assert "useful PREMAT overlap" in javascript
''',
    "Instra test main work labels",
)
# Extend the synthetic forensic interval test with one identical foreground GEMM interval.
tests = replace_once(
    tests,
    '''    from sheet.premat import (
        _ForensicCandidateInterval,
        _ForensicLayerInterval,
        _PendingForensicPass,
    )
''',
    '''    from sheet.premat import (
        _ForensicCandidateInterval,
        _ForensicLayerInterval,
        _ForensicMainWorkInterval,
        _PendingForensicPass,
    )
''',
    "test imports main work interval",
)
tests = replace_once(
    tests,
    '''            origin_event=origin,
            layer_intervals=(layer,),
            candidate_intervals=(candidate,),
        )
''',
    '''            origin_event=origin,
            layer_intervals=(layer,),
            candidate_intervals=(candidate,),
            main_work_intervals=(
                _ForensicMainWorkInterval(
                    pass_sequence=1,
                    layer_index=3,
                    family="QKV",
                    start_event=event(3.0),
                    end_event=event(7.0),
                ),
            ),
        )
''',
    "synthetic main work interval",
)
tests = replace_once(
    tests,
    '''    assert forensic["main_nonwait_gpu_ms_total"] == pytest.approx(9.0)
    assert forensic["premat_gpu_ms_total"] == pytest.approx(7.0)
''',
    '''    assert forensic["main_nonwait_gpu_ms_total"] == pytest.approx(9.0)
    assert forensic["main_consume_count"] == 1
    assert forensic["main_consume_gpu_ms_total"] == pytest.approx(4.0)
    assert forensic["premat_overlap_during_main_consume_ms_total"] == pytest.approx(4.0)
    assert forensic["premat_gpu_ms_total"] == pytest.approx(7.0)
''',
    "synthetic main work assertions",
)
tests = replace_once(
    tests,
    '''    assert latest["premat_useful_overlap_fraction"] == pytest.approx(6.0 / 7.0)
    assert latest["dependency_shortfall_rate"] == pytest.approx(1.0)
''',
    '''    assert latest["main_consume_gpu_ms_mean"] == pytest.approx(4.0)
    assert latest["main_consume_covered_by_premat_fraction"] == pytest.approx(1.0)
    assert latest["premat_useful_overlap_fraction"] == pytest.approx(6.0 / 7.0)
    assert latest["dependency_shortfall_rate"] == pytest.approx(1.0)
''',
    "synthetic derived main work assertions",
)
test_path.write_text(tests)
