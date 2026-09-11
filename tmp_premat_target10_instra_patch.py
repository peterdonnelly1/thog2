from __future__ import annotations

from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


# ---------------------------------------------------------------------------
# PREMAT runtime: target 10 == target +1 then target +0.
# ---------------------------------------------------------------------------
p = Path("sheet/premat.py")
text = p.read_text()
text = replace_once(
    text,
    'PREMAT_TARGET_LAYERS = (0, 1, 2)\n',
    'PREMAT_TARGET_LAYERS = (0, 1, 2, 10)\n',
    "target layer choices",
)
text = replace_once(
    text,
    '        window_width = max(2, self._target_layer + 1)\n',
    '        lookahead_limit = 1 if self._target_layer == 10 else self._target_layer\n        window_width = max(2, lookahead_limit + 1)\n',
    "target10 window",
)
text = replace_once(
    text,
    '            "lookahead_layer_limit": self._target_layer,\n            "target_scope": f"relative_layer_{self._target_layer}",\n',
    '            "lookahead_layer_limit": (1 if self._target_layer == 10 else self._target_layer),\n            "target_scope": (\n                "relative_layer_1_then_0"\n                if self._target_layer == 10\n                else f"relative_layer_{self._target_layer}"\n            ),\n',
    "target10 report provenance",
)
old_advance_head = '''        target_position = self._position + self._target_layer
        target_layer_index = (
            self._layer_indices[target_position]
            if 0 <= target_position < len(self._layer_indices)
            else None
        )
        first_candidate = self._next_premat_candidate()
'''
new_advance_head = '''        valid_target_layers = self._target_layer_indices()
        first_candidate = self._next_premat_candidate()
        target_layer_index = (
            first_candidate.layer_index if first_candidate is not None else None
        )
        target_offset = self._target_offset_for_candidate(first_candidate)
'''
text = replace_once(text, old_advance_head, new_advance_head, "advance target head")
text = replace_once(
    text,
    '''                "target_offset": self._target_layer,
                "target_layer_index": target_layer_index,
''',
    '''                "configured_target_layer": self._target_layer,
                "target_offset": target_offset,
                "target_layer_index": target_layer_index,
''',
    "advance begin target detail",
)
text = replace_once(
    text,
    '''        if target_layer_index is None:
            self._record_advance_return(
                invocation=invocation,
                trigger=trigger,
                submitted=submitted,
                reason="target_out_of_range",
            )
            return
''',
    '''        if first_candidate is None:
            self._record_advance_return(
                invocation=invocation,
                trigger=trigger,
                submitted=submitted,
                reason=("target_exhausted" if valid_target_layers else "target_out_of_range"),
            )
            return
''',
    "advance initial empty target",
)
text = replace_once(
    text,
    '''                )
                return
            considered_ns = time.perf_counter_ns()
''',
    '''                )
                return
            target_layer_index = candidate.layer_index
            target_offset = self._target_offset_for_candidate(candidate)
            if target_offset is None:
                raise RuntimeError("Premat selected a candidate outside its configured target sweep")
            considered_ns = time.perf_counter_ns()
''',
    "advance per-candidate target",
)
text = replace_once(
    text,
    '''                "target_offset": self._target_layer,
                "target_layer_index": target_layer_index,
                "matrix_order": self._weight_matrix_target_order,
                "order_position": candidate.order_position,
''',
    '''                "configured_target_layer": self._target_layer,
                "target_offset": target_offset,
                "target_layer_index": target_layer_index,
                "matrix_order": self._weight_matrix_target_order,
                "order_position": candidate.order_position,
''',
    "decision actual target offset",
)
old_next = '''    def _next_premat_candidate(
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
'''
new_next = '''    def _target_offsets(self) -> Tuple[int, ...]:
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
'''
text = replace_once(text, old_next, new_next, "target10 candidate sweep")
p.write_text(text)


# ---------------------------------------------------------------------------
# Run/training configuration and CLI.
# ---------------------------------------------------------------------------
p = Path("sheet/run_config.py")
text = p.read_text()
text = replace_once(
    text,
    '    premat_instra: str = "disabled"\n',
    '    premat_instra: str = "disabled"\n    premat_retain_detailed_premat_history: bool = False\n',
    "run config retain field",
)
text = replace_once(
    text,
    '''        validate_premat_configuration(
            premat=self.premat,
''',
    '''        if self.premat_target_layer == 10:
            object.__setattr__(self, "premat_weight_matrix_target_order", "r_to_l")
        if not isinstance(self.premat_retain_detailed_premat_history, bool):
            raise ValueError("premat_retain_detailed_premat_history must be bool")
        validate_premat_configuration(
            premat=self.premat,
''',
    "run config target10 canonicalization",
)
text = replace_once(
    text,
    '            premat_instra=self.premat_instra,\n            plastic__layer_count__cuda_allocator_reserve_gib=',
    '            premat_instra=self.premat_instra,\n            premat_retain_detailed_premat_history=self.premat_retain_detailed_premat_history,\n            plastic__layer_count__cuda_allocator_reserve_gib=',
    "run to training retain pass",
)
text = replace_once(
    text,
    '''        if not self.plastic__enabled:
            for name in PLASTIC_RUN_CONFIG_FIELDS:
                values.pop(name, None)
        return values
''',
    '''        if not self.plastic__enabled:
            for name in PLASTIC_RUN_CONFIG_FIELDS:
                values.pop(name, None)
        if not self.premat_retain_detailed_premat_history:
            values.pop("premat_retain_detailed_premat_history", None)
        return values
''',
    "run persistent default retention omission",
)
text = replace_once(
    text,
    '''            values["premat_lookahead_layer_limit"] = self.premat_target_layer
            values["premat_target_scope"] = f"relative_layer_{self.premat_target_layer}"
''',
    '''            values["premat_lookahead_layer_limit"] = (
                1 if self.premat_target_layer == 10 else self.premat_target_layer
            )
            values["premat_target_scope"] = (
                "relative_layer_1_then_0"
                if self.premat_target_layer == 10
                else f"relative_layer_{self.premat_target_layer}"
            )
''',
    "run canonical target10 scope",
)
p.write_text(text)

p = Path("sheet/training_config.py")
text = p.read_text()
text = replace_once(
    text,
    '    premat_instra: str = "disabled"\n',
    '    premat_instra: str = "disabled"\n    premat_retain_detailed_premat_history: bool = False\n',
    "training config retain field",
)
text = replace_once(
    text,
    '''        validate_premat_configuration(
            premat=self.premat,
''',
    '''        if self.premat_target_layer == 10:
            self.premat_weight_matrix_target_order = "r_to_l"
        if not isinstance(self.premat_retain_detailed_premat_history, bool):
            raise ValueError("premat_retain_detailed_premat_history must be bool")
        validate_premat_configuration(
            premat=self.premat,
''',
    "training target10 canonicalization",
)
text = replace_once(
    text,
    '''        if not self.plastic__enabled:
            for name in PLASTIC_TRAINING_CONFIG_FIELDS:
                values.pop(name, None)
        return values
''',
    '''        if not self.plastic__enabled:
            for name in PLASTIC_TRAINING_CONFIG_FIELDS:
                values.pop(name, None)
        if not self.premat_retain_detailed_premat_history:
            values.pop("premat_retain_detailed_premat_history", None)
        return values
''',
    "training persistent default retention omission",
)
p.write_text(text)

p = Path("sheet/model.py")
text = p.read_text()
text = replace_once(
    text,
    '''        # vvv THOG validate the public premat controls and force pass-local materialisation lifetimes when enabled
        validate_premat_configuration(
''',
    '''        # vvv THOG validate the public premat controls and force pass-local materialisation lifetimes when enabled
        if self.premat_target_layer == 10:
            self.premat_weight_matrix_target_order = "r_to_l"
        validate_premat_configuration(
''',
    "model target10 canonicalization",
)
p.write_text(text)

p = Path("run_thog2_owt_core.py")
text = p.read_text()
text = replace_once(
    text,
    '''def build_parser() -> argparse.ArgumentParser:
''',
    '''def _true_false(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
''',
    "true false parser",
)
text = replace_once(
    text,
    '    parser.add_argument("--premat_target_layer", type=int, choices=(0, 1, 2), default=1)\n',
    '    parser.add_argument("--premat_target_layer", type=int, choices=(0, 1, 2, 10), default=1, help="PREMAT relative target: 10 means ordered +1 then +0 sweep")\n',
    "target10 CLI choice",
)
text = replace_once(
    text,
    '    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\n',
    '    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\n    parser.add_argument("--premat_retain_detailed_premat_history", type=_true_false, default=False, metavar="true|false")\n',
    "retain history CLI",
)
text = replace_once(
    text,
    '''def config_from_arguments(arguments: argparse.Namespace, *, geometry_plan=None) -> OwtRunConfig:
    geometry_plan = geometry_plan if geometry_plan is not None else geometry_plan_from_arguments(arguments)
''',
    '''def config_from_arguments(arguments: argparse.Namespace, *, geometry_plan=None) -> OwtRunConfig:
    if (
        int(arguments.premat_target_layer) == 10
        and arguments.premat_weight_matrix_target_order == "l_to_r"
    ):
        print(
            "\\033[1;38;5;208mTHOG2 WARNING: --premat_target_layer 10 means +1 then +0 "
            "and requires right-to-left matrix targeting; overriding "
            "--premat_weight_matrix_target_order l_to_r -> r_to_l.\\033[0m",
            flush=True,
        )
        arguments.premat_weight_matrix_target_order = "r_to_l"
    geometry_plan = geometry_plan if geometry_plan is not None else geometry_plan_from_arguments(arguments)
''',
    "CLI target10 warning",
)
text = replace_once(
    text,
    '        premat_instra=arguments.premat_instra,\n        plastic__layer_count__cuda_allocator_reserve_gib=',
    '        premat_instra=arguments.premat_instra,\n        premat_retain_detailed_premat_history=arguments.premat_retain_detailed_premat_history,\n        plastic__layer_count__cuda_allocator_reserve_gib=',
    "CLI retain pass",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Detailed-history retention: full or recap-only compact snapshots.
# ---------------------------------------------------------------------------
p = Path("sheet/wandb_telemetry.py")
text = p.read_text()
insert = '''
_PREMAT_RECAP_EVENT_NAMES = frozenset((
    "materialising",
    "available",
    "critical_path_wait",
    "materialising_on_critical_path",
    "available_on_critical_path",
    "consuming",
    "consumed",
    "pass_end_release",
    "pass_end",
))
_PREMAT_RECAP_EVENT_FIELDS = frozenset((
    "sequence",
    "event",
    "layer_index",
    "family",
    "current_layer_index",
    "owner",
    "decision",
    "outcome",
    "reason",
    "admission_reason",
    "new_state",
    "state",
    "critical_path_miss",
    "elapsed_ms",
    "materialisation_ms",
    "main_stream_materialisation_ms",
    "wait_ms",
    "predicted_retained_bytes",
    "target_offset",
    "target_order",
    "target_order_position",
))


def _premat_snapshot_for_storage(
    snapshot: Mapping[str, Any],
    *,
    retain_detailed_history: bool,
) -> Dict[str, Any]:
    payload = dict(snapshot)
    payload["detailed_history_retained"] = bool(retain_detailed_history)
    payload["history_detail_level"] = (
        "detailed" if retain_detailed_history else "recap_only"
    )
    if retain_detailed_history:
        return payload
    payload.pop("candidates", None)
    compact_events = []
    for event in snapshot.get("events", ()):
        if not isinstance(event, Mapping):
            continue
        event_name = str(event.get("event", ""))
        if event_name not in _PREMAT_RECAP_EVENT_NAMES:
            continue
        compact_events.append({
            key: event[key]
            for key in _PREMAT_RECAP_EVENT_FIELDS
            if key in event
        })
    payload["events"] = compact_events
    payload["event_count"] = len(compact_events)
    return payload

'''
text = replace_once(
    text,
    '# vvv THOG latest-only asynchronous live feed: a bounded queue prevents Instra\n',
    insert + '# vvv THOG latest-only asynchronous live feed: a bounded queue prevents Instra\n',
    "compact snapshot helper",
)
text = replace_once(
    text,
    '''    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)
''',
    '''    def __init__(
        self,
        database_path: Path,
        *,
        retain_detailed_history: bool = False,
    ) -> None:
        self._database_path = Path(database_path)
        self._retain_detailed_history = bool(retain_detailed_history)
''',
    "live sink retention ctor",
)
text = replace_once(
    text,
    '''        item = (int(optimizer_update), snapshot)
''',
    '''        stored_snapshot = _premat_snapshot_for_storage(
            snapshot,
            retain_detailed_history=self._retain_detailed_history,
        )
        item = (int(optimizer_update), stored_snapshot)
''',
    "live sink compact publish",
)
text = replace_once(
    text,
    '''        live_sink = _PrematLiveSink(local_store.path)
''',
    '''        live_sink = _PrematLiveSink(
            local_store.path,
            retain_detailed_history=bool(
                getattr(trainer.config, "premat_retain_detailed_premat_history", False)
            ),
        )
''',
    "attach retention sink",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Instra HTML/key/controls.
# ---------------------------------------------------------------------------
p = Path("sheet/local_dashboard_assets/index.html")
text = p.read_text()
text = replace_once(
    text,
    '<input id="premat_state_duration" type="range" min="10" max="2000" step="10" value="250">\n                <output id="premat_state_duration_label" for="premat_state_duration">0.25 s</output>',
    '<input id="premat_state_duration" type="range" min="1" max="1000" step="1" value="100">\n                <output id="premat_state_duration_label" for="premat_state_duration">0.10 s</output>',
    "faster playback slider",
)
old_key = '''            <div class="premat-key" aria-label="Premat pipeline key">
              <div class="premat-key-section">
                <strong>PROCESSING</strong>
                <div class="premat-key-items">
                  <span class="premat-key-item"><span>OUT OF SCOPE</span><span class="premat-key-swatch premat-neutral"></span></span>
                  <span class="premat-key-item"><span>NOT YET REACHED</span><span class="premat-key-swatch premat-pending"></span></span>
                  <span class="premat-key-item"><span>PRE-MATERIALISING</span><span class="premat-key-swatch premat-state-materialising"></span></span>
                  <span class="premat-key-item"><span>AVAILABLE</span><span class="premat-key-swatch premat-state-available"></span></span>
                  <span class="premat-key-item"><span>CONSUMING -<br>NO WAITING</span><span class="premat-key-swatch premat-state-consuming-full"></span></span>
                  <span class="premat-key-item"><span>WAITING FOR<br>PRE-MATERIALISATION</span><span class="premat-key-swatch premat-state-waiting"></span></span>
                  <span class="premat-key-item"><span>CONSUMING AFTER WAIT</span><span class="premat-key-swatch premat-state-consuming-waited"></span></span>
                  <span class="premat-key-item"><span>PREMAT NOT STARTED -<br>MAIN CODE MATERIALISING</span><span class="premat-key-swatch premat-state-main-materialising"></span></span>
                  <span class="premat-key-item"><span>MAIN CODE CONSUMING</span><span class="premat-key-swatch premat-state-main-consuming"></span></span>
                </div>
              </div>
              <div class="premat-key-section">
                <strong>OUTCOMES</strong>
                <div class="premat-key-items premat-key-outcome-items">
                  <span class="premat-key-item"><span>FULL HIT</span><span class="premat-key-swatch premat-state-consumed-full"></span></span>
                  <span class="premat-key-item"><span>PARTIAL HIT</span><span class="premat-key-swatch premat-state-consumed-waited"></span></span>
                  <span class="premat-key-item"><span>COMPLETE MISS</span><span class="premat-key-swatch premat-state-consumed-main"></span></span>
                </div>
              </div>
            </div>
'''
new_key = '''            <div class="premat-key" aria-label="Premat pipeline key">
              <div class="premat-key-flow"><strong>FULL SUCCESS</strong><span class="premat-key-flow-step"><span>PRE-MATERIALISING</span><span class="premat-key-swatch premat-state-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>AVAILABLE</span><span class="premat-key-swatch premat-state-available"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>CONSUMING - NO WAIT</span><span class="premat-key-swatch premat-state-consuming-full"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>FULL HIT</span><span class="premat-key-swatch premat-state-consumed-full"></span></span></div>
              <div class="premat-key-flow"><strong>PART SUCCESS</strong><span class="premat-key-flow-step"><span>PRE-MATERIALISING</span><span class="premat-key-swatch premat-state-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>WAITING FOR PRE-MATERIALISATION</span><span class="premat-key-swatch premat-state-waiting"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>CONSUMING AFTER WAIT</span><span class="premat-key-swatch premat-state-consuming-waited"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>PARTIAL HIT</span><span class="premat-key-swatch premat-state-consumed-waited"></span></span></div>
              <div class="premat-key-flow"><strong>MISS</strong><span class="premat-key-flow-step"><span>PREMAT NOT STARTED - MAIN STREAM MATERIALISING</span><span class="premat-key-swatch premat-state-main-materialising"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>MAIN STREAM CONSUMING</span><span class="premat-key-swatch premat-state-main-consuming"></span></span><span class="premat-key-arrow">→</span><span class="premat-key-flow-step"><span>COMPLETE MISS</span><span class="premat-key-swatch premat-state-consumed-main"></span></span></div>
            </div>
'''
text = replace_once(text, old_key, new_key, "key flow restructure")
text = replace_once(
    text,
    '              <span class="premat-mode" id="premat_mode">waiting</span>\n',
    '              <span class="premat-mode" id="premat_mode">waiting</span>\n              <span class="premat-step-totals" id="premat_step_totals">F — · P — · M —</span>\n',
    "recap title totals",
)
text = replace_once(
    text,
    '          <div class="premat-recap-view" id="premat_recap_view">\n            <div class="premat-summary" id="premat_summary"></div>\n',
    '          <div class="premat-recap-view" id="premat_recap_view">\n            <div class="premat-selected-step-summary" id="premat_selected_step_summary" hidden></div>\n            <div class="premat-summary" id="premat_summary"></div>\n',
    "selected history step summary",
)
text = replace_once(
    text,
    '<div class="premat-panel-heading"><strong>Premat history</strong><span id="premat_history_detail">Loading retained steps…</span></div>\n',
    '<div class="premat-panel-heading premat-history-heading"><strong>Premat history</strong><span id="premat_history_detail">Loading retained steps…</span><span class="premat-history-grand-totals" id="premat_history_grand_totals">F — · P — · M —</span></div>\n',
    "history title totals",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Instra CSS.
# ---------------------------------------------------------------------------
p = Path("sheet/local_dashboard_assets/dashboard_premat.css")
text = p.read_text()
text = replace_once(
    text,
    '.premat-header-title { min-width: 0; }\n',
    '.premat-header-title { min-width: 0; display: flex; align-items: center; }\n.premat-step-totals { min-width: 150px; margin-left: 24px; color: #3d4856; font-size: 11px; font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; }\n',
    "anchored recap totals",
)
text = replace_once(
    text,
    '''.premat-key-section {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr);
  align-items: center;
  gap: 12px;
}
.premat-key-section > strong {
  color: var(--text, #343942);
  text-align: left;
}
.premat-key-items {
  display: grid;
  grid-template-columns: repeat(9, minmax(100px, 1fr));
  align-items: stretch;
  gap: 10px;
}
.premat-key-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 42px;
  align-items: center;
  gap: 6px;
  min-width: 0;
  line-height: 1.15;
}
.premat-key-item > span:first-child { color: #4b5563; text-align: right; }
''',
    '''.premat-key-flow {
  display: grid;
  grid-template-columns: 92px minmax(150px, 1fr) 34px minmax(150px, 1fr) 34px minmax(150px, 1fr) 34px minmax(150px, 1fr);
  align-items: center;
  gap: 8px;
}
.premat-key-flow > strong { color: var(--text, #343942); }
.premat-key-flow-step {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 42px;
  align-items: center;
  gap: 7px;
  min-width: 0;
  line-height: 1.15;
}
.premat-key-flow-step > span:first-child { color: #4b5563; text-align: right; }
.premat-key-arrow { color: #68717e; font-size: 25px; font-weight: 700; text-align: center; }
''',
    "key flow CSS",
)
text += '''
/* THOG Premat view isolation and compact-history summaries. */
body.premat-tab-active [data-chart-group]:not(#premat_chart_group) { display: none !important; }
.premat-selected-step-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 5px 14px;
  margin-bottom: 7px;
  padding: 7px 9px;
  border: 1px solid #d9e0e6;
  border-radius: 6px;
  background: #f6f8fa;
  color: #45505e;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}
.premat-selected-step-summary[hidden] { display: none !important; }
.premat-history-heading { display: grid; grid-template-columns: 150px minmax(260px, 1fr) 190px; align-items: center; column-gap: 12px; }
.premat-history-heading > span { margin-left: 0 !important; }
.premat-history-grand-totals { justify-self: end; min-width: 180px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 650; }
.premat-history-subtotals th { top: 0 !important; background: #e9eef3 !important; font-weight: 650; }
.premat-history-headings th { top: 31px !important; }
'''
p.write_text(text)


# ---------------------------------------------------------------------------
# Instra JS.
# ---------------------------------------------------------------------------
p = Path("sheet/local_dashboard_assets/dashboard_premat.js")
text = p.read_text()
text = replace_once(text, 'const PREMAT_FINAL_HOLD_MS = 1000;\n', 'const PREMAT_FINAL_HOLD_MS = 250;\n', "faster final hold")
text = text.replace('PREMAT NOT STARTED - MAIN CODE MATERIALISING', 'PREMAT NOT STARTED - MAIN STREAM MATERIALISING')
text = text.replace('MAIN CODE CONSUMING', 'MAIN STREAM CONSUMING')
text = text.replace('CONSUMING - NO WAITING', 'CONSUMING - NO WAIT')
text = replace_once(
    text,
    '''function premat_history_model(snapshots) {
''',
    '''function premat_count_totals(outcomes, layers) {
  const totals = {full_hits: 0, partial_hits: 0, complete_misses: 0};
  for (const layer_index of layers) {
    const counts = outcomes[layer_index] || {};
    totals.full_hits += Number(counts.full_hits || 0);
    totals.partial_hits += Number(counts.partial_hits || 0);
    totals.complete_misses += Number(counts.complete_misses || 0);
  }
  return totals;
}

function premat_totals_text(totals) {
  return `F ${Number(totals?.full_hits || 0)} · P ${Number(totals?.partial_hits || 0)} · M ${Number(totals?.complete_misses || 0)}`;
}

function premat_detailed_history_retained(snapshot) {
  return snapshot?.detailed_history_retained !== false;
}

function premat_history_model(snapshots) {
''',
    "history total helpers",
)
text = replace_once(
    text,
    '''    return {
      optimizer_update: Number(snapshot.optimizer_update ?? 0),
      buffer_margin_bytes,
      headroom_bytes: Number.isFinite(headroom) ? headroom : null,
      outcomes,
    };
  });
  return {layers, rows};
}
''',
    '''    return {
      optimizer_update: Number(snapshot.optimizer_update ?? 0),
      buffer_margin_bytes,
      headroom_bytes: Number.isFinite(headroom) ? headroom : null,
      outcomes,
      totals: premat_count_totals(outcomes, layers),
      detailed_history_retained: premat_detailed_history_retained(snapshot),
    };
  });
  const layer_totals = {};
  for (const layer_index of layers) {
    layer_totals[layer_index] = {full_hits: 0, partial_hits: 0, complete_misses: 0};
    for (const row of rows) {
      const counts = row.outcomes[layer_index] || {};
      layer_totals[layer_index].full_hits += Number(counts.full_hits || 0);
      layer_totals[layer_index].partial_hits += Number(counts.partial_hits || 0);
      layer_totals[layer_index].complete_misses += Number(counts.complete_misses || 0);
    }
  }
  return {layers, rows, layer_totals, grand_totals: premat_count_totals(layer_totals, layers)};
}
''',
    "history model totals",
)
text = replace_once(
    text,
    '''function premat_render_summary(snapshot, model) {
  const outcomes = {"FULL HIT": 0, "PARTIAL HIT": 0, "COMPLETE MISS": 0};
  for (const record of model.records.values()) outcomes[record.outcome] = (outcomes[record.outcome] || 0) + 1;
''',
    '''function premat_render_summary(snapshot, model) {
  const outcomes = {"FULL HIT": 0, "PARTIAL HIT": 0, "COMPLETE MISS": 0};
  for (const record of model.records.values()) outcomes[record.outcome] = (outcomes[record.outcome] || 0) + 1;
  const step_totals = by_id("premat_step_totals");
  if (step_totals) step_totals.textContent = `F ${outcomes["FULL HIT"]} · P ${outcomes["PARTIAL HIT"]} · M ${outcomes["COMPLETE MISS"]}`;
''',
    "recap title totals update",
)
text = replace_once(
    text,
    '["target", `l+${Number(snapshot.target_offset ?? snapshot.target_layer ?? 1)}`],\n',
    '["target", Number(snapshot.target_layer ?? snapshot.target_offset ?? 1) === 10 ? "l+1 → l+0" : `l+${Number(snapshot.target_offset ?? snapshot.target_layer ?? 1)}`],\n',
    "target10 summary label",
)
text = replace_once(
    text,
    '''function premat_frame_delay(frame) {
  if (frame?.final) return PREMAT_FINAL_HOLD_MS;
  return premat_view.state_duration_ms * Number(frame?.duration_multiplier || 1);
}
''',
    '''function premat_frame_delay(frame) {
  if (frame?.final) return Math.max(10, Math.min(PREMAT_FINAL_HOLD_MS, premat_view.state_duration_ms * 2));
  return premat_view.state_duration_ms * Number(frame?.duration_multiplier || 1);
}
''',
    "scaled final frame delay",
)
old_open = '''function premat_open_history_step(step) {
  const snapshot = premat_find_history_snapshot(step);
  if (!snapshot) return;
  premat_view.inspector_return_panel = "history";
  premat_render_inspector(snapshot, premat_build_model(snapshot));
  premat_show_panel("inspector");
}
'''
new_open = '''function premat_render_selected_step_summary(snapshot, model) {
  const container = by_id("premat_selected_step_summary");
  if (!container) return;
  const pieces = [];
  const outcomes = {};
  for (const layer_index of model.layers) {
    const counts = {full_hits: 0, partial_hits: 0, complete_misses: 0};
    for (const family of model.families) {
      const outcome = model.records.get(premat_candidate_key(layer_index, family))?.outcome;
      if (outcome === "FULL HIT") counts.full_hits += 1;
      else if (outcome === "PARTIAL HIT") counts.partial_hits += 1;
      else if (outcome === "COMPLETE MISS") counts.complete_misses += 1;
    }
    outcomes[layer_index] = counts;
    pieces.push(`<span><strong>Layer ${layer_index + 1}</strong> ${premat_escape(premat_totals_text(counts))}</span>`);
  }
  const totals = premat_count_totals(outcomes, model.layers);
  pieces.unshift(`<span><strong>Step ${Number(snapshot.optimizer_update ?? 0)}</strong> ${premat_escape(premat_totals_text(totals))}</span>`);
  container.innerHTML = pieces.join("");
  container.hidden = false;
}

function premat_open_history_step(step) {
  const snapshot = premat_find_history_snapshot(step);
  if (!snapshot) return;
  const model = premat_build_model(snapshot);
  premat_view.inspector_return_panel = "history";
  if (!premat_detailed_history_retained(snapshot)) {
    premat_start_snapshot(snapshot);
    premat_render_selected_step_summary(snapshot, model);
    premat_show_panel("recap");
    return;
  }
  premat_render_inspector(snapshot, model);
  premat_show_panel("inspector");
}
'''
text = replace_once(text, old_open, new_open, "compact history opens recap")
text = replace_regex(
    text,
    r'''function premat_render_history\(snapshots\) \{.*?\n\}\nfunction premat_artifact_name\(\)''',
    '''function premat_render_history(snapshots) {
  const model = premat_history_model(snapshots);
  premat_view.history_model = model;
  const subtotal = [
    '<tr class="premat-history-subtotals"><th>Σ retained</th><th></th><th></th>',
    ...model.layers.map(layer_index => `<th>${premat_escape(premat_totals_text(model.layer_totals[layer_index]))}</th>`),
    `<th>${premat_escape(premat_totals_text(model.grand_totals))}</th><th></th></tr>`,
  ].join("");
  const headings = [
    '<tr class="premat-history-headings"><th>Step</th><th>Buffer margin</th><th>Headroom</th>',
    ...model.layers.map(layer_index => `<th>Layer ${layer_index + 1}<br>F / P / M</th>`),
    '<th>Step total</th><th>Download</th></tr>',
  ].join("");
  const download_icon = '<svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M5 21h14"></path></svg>';
  const rows = model.rows.map(row => {
    const step = Number(row.optimizer_update);
    const layer_cells = model.layers.map(layer_index => {
      const counts = row.outcomes[layer_index] || {};
      const text = premat_totals_text(counts);
      return `<td class="premat-history-outcomes" title="Layer ${layer_index + 1}: ${premat_escape(text)}">${premat_escape(text)}</td>`;
    }).join("");
    const download = `<td class="premat-history-download-cell"><button class="premat-history-download-button" type="button" data-premat-history-download="${step}" aria-label="Download Premat summary/data for step ${step}" title="Download step ${step}">${download_icon}</button></td>`;
    return `<tr class="premat-history-row" data-premat-history-step="${step}" tabindex="0" role="button" aria-label="Open Premat data for step ${step}"><td><strong>${step}</strong></td><td>${premat_escape(premat_bytes(row.buffer_margin_bytes))}</td><td>${premat_escape(premat_bytes(row.headroom_bytes))}</td>${layer_cells}<td class="premat-history-outcomes"><strong>${premat_escape(premat_totals_text(row.totals))}</strong></td>${download}</tr>`;
  });
  by_id("premat_history_head").innerHTML = subtotal + headings;
  const body = by_id("premat_history_body");
  body.innerHTML = rows.length
    ? rows.join("")
    : `<tr><td class="premat-history-empty" colspan="${Math.max(5, model.layers.length + 5)}">No retained Premat history for this run.</td></tr>`;
  body.querySelectorAll("[data-premat-history-step]").forEach(element => {
    const open = () => premat_open_history_step(element.dataset.prematHistoryStep);
    element.addEventListener("click", event => {
      if (event.target.closest("[data-premat-history-download]")) return;
      open();
    });
    element.addEventListener("keydown", event => {
      if (event.target.closest("[data-premat-history-download]")) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
  body.querySelectorAll("[data-premat-history-download]").forEach(button => {
    button.addEventListener("click", event => {
      event.stopPropagation();
      const snapshot = premat_find_history_snapshot(button.dataset.prematHistoryDownload);
      if (snapshot) premat_download_step(snapshot);
    });
  });
  const retained_detailed = model.rows.some(row => row.detailed_history_retained);
  by_id("premat_history_detail").textContent = `${model.rows.length} retained steps · ${model.layers.length} layers · select a row for ${retained_detailed ? "matrix detail" : "recapitulation"}`;
  const grand = by_id("premat_history_grand_totals");
  if (grand) grand.textContent = premat_totals_text(model.grand_totals);
  by_id("premat_history_download").disabled = model.rows.length === 0;
  const raw_button = by_id("premat_history_raw_download");
  raw_button.disabled = !retained_detailed;
  raw_button.hidden = !retained_detailed;
}
function premat_artifact_name()''',
    "history rendering totals and retention",
)
text = replace_once(
    text,
    '''function premat_download_csv(csv, filename) {
  const blob = new Blob(["\\ufeff", csv], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
''',
    '''async function premat_download_csv(csv, filename) {
  const blob = new Blob(["\\ufeff", csv], {type: "text/csv;charset=utf-8"});
  if (typeof window.showSaveFilePicker === "function") {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{description: "CSV file", accept: {"text/csv": [".csv"]}}],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return;
    } catch (error) {
      if (error?.name === "AbortError") return;
      console.warn("Premat native save picker unavailable; falling back to browser download", error);
    }
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
''',
    "native save picker",
)
text = replace_once(
    text,
    '''function premat_start_snapshot(snapshot) {
  if (!premat_snapshot_complete(snapshot)) return;
''',
    '''function premat_start_snapshot(snapshot) {
  if (!premat_snapshot_complete(snapshot)) return;
  const selected_summary = by_id("premat_selected_step_summary");
  if (selected_summary) selected_summary.hidden = true;
''',
    "clear selected step summary",
)
text = replace_once(
    text,
    '''  by_id("premat_step").textContent = "—";
''',
    '''  by_id("premat_step").textContent = "—";
  by_id("premat_step_totals").textContent = "F — · P — · M —";
  by_id("premat_selected_step_summary").hidden = true;
  by_id("premat_history_grand_totals").textContent = "F — · P — · M —";
''',
    "reset summary totals",
)
text = replace_once(
    text,
    '''  by_id("premat_chart_group").hidden = !premat_selected;
  by_id("depth_chart_group").hidden = premat_selected;
''',
    '''  by_id("premat_chart_group").hidden = !premat_selected;
  document.body.classList.toggle("premat-tab-active", Boolean(premat_selected));
  document.querySelectorAll('[data-chart-group]:not(#premat_chart_group)').forEach(group => {
    group.hidden = Boolean(premat_selected);
  });
''',
    "hide all non-premat chart groups",
)
text = replace_once(
    text,
    '''    premat_view.state_duration_ms = Math.max(10, Number(duration?.value || 250));
    if (duration_label) duration_label.textContent = `${(premat_view.state_duration_ms / 1000).toFixed(2)} s`;
''',
    '''    premat_view.state_duration_ms = Math.max(1, Number(duration?.value || 100));
    if (duration_label) duration_label.textContent = premat_view.state_duration_ms < 100
      ? `${Math.round(premat_view.state_duration_ms)} ms`
      : `${(premat_view.state_duration_ms / 1000).toFixed(2)} s`;
''',
    "fast slider JS",
)
text = replace_once(
    text,
    '''  by_id("premat_inspect_button")?.addEventListener("click", () => {
    if (!premat_view.active_snapshot || !premat_view.active_model) return;
    premat_view.inspector_return_panel = "recap";
    premat_render_inspector(premat_view.active_snapshot, premat_view.active_model);
    premat_show_panel("inspector");
  });
''',
    '''  by_id("premat_inspect_button")?.addEventListener("click", () => {
    if (!premat_view.active_snapshot || !premat_view.active_model) return;
    if (!premat_detailed_history_retained(premat_view.active_snapshot)) {
      premat_render_selected_step_summary(premat_view.active_snapshot, premat_view.active_model);
      premat_show_panel("recap");
      return;
    }
    premat_view.inspector_return_panel = "recap";
    premat_render_inspector(premat_view.active_snapshot, premat_view.active_model);
    premat_show_panel("inspector");
  });
''',
    "compact inspector behavior",
)
text = replace_once(
    text,
    '''  by_id("premat_history_raw_download")?.addEventListener("click", premat_download_raw_event_history);
  update_duration();
''',
    '''  by_id("premat_history_raw_download")?.addEventListener("click", premat_download_raw_event_history);
  window.addEventListener("keydown", event => {
    if (event.key === "Escape" && premat_view.active_panel === "inspector") {
      event.preventDefault();
      premat_close_inspector();
    }
  });
  update_duration();
''',
    "escape inspector",
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
p = Path("tests/test_premat.py")
text = p.read_text()
text = replace_once(
    text,
    'from run_thog2_owt_core import build_parser\n',
    'from run_thog2_owt_core import build_parser, config_from_arguments\n',
    "test config import",
)
text = replace_once(
    text,
    'def test_public_cli_exposes_exactly_the_eleven_premat_options() -> None:\n',
    'def test_public_cli_exposes_exactly_the_twelve_premat_options() -> None:\n',
    "CLI option test rename",
)
text = replace_once(
    text,
    '        "--premat_instra",\n    }\n',
    '        "--premat_instra",\n        "--premat_retain_detailed_premat_history",\n    }\n',
    "CLI option set retain",
)
append = '''

# vvv THOG target 10 and recap-only Instra history regression coverage

def test_target_layer_10_sweeps_next_then_current_right_to_left(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        target_layer=10,
        weight_matrix_target_order="r_to_l",
    )
    runtime.layer_start(3)
    assert calls == [
        ("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5),
        ("DOWN", 3), ("UP", 3), ("O", 3), ("QKV", 3),
    ]
    report = runtime.report()
    assert report["target_layer"] == 10
    assert report["lookahead_layer_limit"] == 1
    assert report["target_scope"] == "relative_layer_1_then_0"


def test_target_layer_10_forces_r_to_l_in_run_config() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        premat="enabled",
        premat_target_layer=10,
        premat_weight_matrix_target_order="l_to_r",
        device="cuda",
    )
    assert config.premat_weight_matrix_target_order == "r_to_l"
    canonical = config.canonical_dict(world_size=1)
    assert canonical["premat_lookahead_layer_limit"] == 1
    assert canonical["premat_target_scope"] == "relative_layer_1_then_0"


def test_target_layer_10_cli_warns_in_bright_orange_and_overrides(capsys) -> None:
    arguments = build_parser().parse_args([
        "--model-type", "sheet",
        "--premat", "enabled",
        "--premat_target_layer", "10",
        "--premat_weight_matrix_target_order", "l_to_r",
    ])
    config = config_from_arguments(arguments)
    captured = capsys.readouterr().out
    assert "\\x1b[1;38;5;208m" in captured
    assert "overriding" in captured
    assert config.premat_weight_matrix_target_order == "r_to_l"


def test_premat_retain_detailed_history_cli_defaults_false_and_accepts_true() -> None:
    parser = build_parser()
    assert parser.parse_args([]).premat_retain_detailed_premat_history is False
    assert parser.parse_args([
        "--premat_retain_detailed_premat_history", "true"
    ]).premat_retain_detailed_premat_history is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--premat_retain_detailed_premat_history", "maybe"])


def test_compact_premat_storage_retains_recap_transitions_only() -> None:
    from sheet.wandb_telemetry import _premat_snapshot_for_storage

    snapshot = {
        "pass_complete": True,
        "events": [
            {"sequence": 1, "event": "admission_considered", "family": "DOWN", "layer_index": 1, "detail": {"huge": "payload"}},
            {"sequence": 2, "event": "materialising", "family": "DOWN", "layer_index": 1, "owner": "premat", "new_state": "MATERIALISING", "detail": {"huge": "payload"}},
            {"sequence": 3, "event": "consuming", "family": "DOWN", "layer_index": 1, "owner": "premat", "new_state": "CONSUMING"},
            {"sequence": 4, "event": "consumed", "family": "DOWN", "layer_index": 1, "owner": "premat", "new_state": "CONSUMED"},
            {"sequence": 5, "event": "pass_end"},
        ],
        "candidates": [{"large": "candidate payload"}],
        "memory": {"device_free_bytes": 1},
    }
    compact = _premat_snapshot_for_storage(snapshot, retain_detailed_history=False)
    assert compact["detailed_history_retained"] is False
    assert compact["history_detail_level"] == "recap_only"
    assert "candidates" not in compact
    assert [event["event"] for event in compact["events"]] == [
        "materialising", "consuming", "consumed", "pass_end"
    ]
    assert all("detail" not in event for event in compact["events"])
    detailed = _premat_snapshot_for_storage(snapshot, retain_detailed_history=True)
    assert detailed["detailed_history_retained"] is True
    assert len(detailed["events"]) == 5


def test_premat_instra_source_contains_requested_interactions() -> None:
    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")
    javascript = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")
    assert 'min="1" max="1000" step="1"' in html
    assert "FULL SUCCESS" in html and "PART SUCCESS" in html
    assert "MAIN STREAM CONSUMING" in html
    assert "MAIN STREAM MATERIALISING" in html
    assert "CONSUMING - NO WAIT" in html
    assert "OUT OF SCOPE" not in html
    assert "NOT YET REACHED</span><span class=\"premat-key-swatch" not in html
    assert "showSaveFilePicker" in javascript
    assert 'event.key === "Escape"' in javascript
    assert 'premat-tab-active' in javascript
    assert 'detailed_history_retained' in javascript
# ^^^ THOG
'''
text += append
p.write_text(text)

print("patched PREMAT target10, history retention, and Instra UX")
