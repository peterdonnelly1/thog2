from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# sheet/premat.py: diagnostic CUDA timing is explicit and default-off.
# ---------------------------------------------------------------------------
p = Path("sheet/premat.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    diagnostic_layer_delay_ms: float,\n    logging: str,\n    instra: str,\n) -> None:\n''',
    '''    diagnostic_layer_delay_ms: float,\n    enable_gpu_timing_diagnostic: bool = False,\n    logging: str,\n    instra: str,\n) -> None:\n''',
    "premat validation signature",
)
text = replace_once(
    text,
    '''    if (\n        isinstance(diagnostic_layer_delay_ms, bool)\n        or not isinstance(diagnostic_layer_delay_ms, (int, float))\n        or not math.isfinite(float(diagnostic_layer_delay_ms))\n        or float(diagnostic_layer_delay_ms) < 0.0\n    ):\n        raise ValueError(\n            "premat_diagnostic_layer_delay_ms must be finite and non-negative"\n        )\n''',
    '''    if (\n        isinstance(diagnostic_layer_delay_ms, bool)\n        or not isinstance(diagnostic_layer_delay_ms, (int, float))\n        or not math.isfinite(float(diagnostic_layer_delay_ms))\n        or float(diagnostic_layer_delay_ms) < 0.0\n    ):\n        raise ValueError(\n            "premat_diagnostic_layer_delay_ms must be finite and non-negative"\n        )\n    # vvv THOG PREMAT CUDA timestamping is an explicit diagnostic, never a default execution cost\n    if not isinstance(enable_gpu_timing_diagnostic, bool):\n        raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n    # ^^^ THOG\n''',
    "premat diagnostic timing validation",
)
text = replace_once(
    text,
    '''        cuda_stream_priority: str,\n        diagnostic_layer_delay_ms: float,\n        logging_enabled: bool,\n    ) -> None:\n''',
    '''        cuda_stream_priority: str,\n        diagnostic_layer_delay_ms: float,\n        enable_gpu_timing_diagnostic: bool = False,\n        logging_enabled: bool,\n    ) -> None:\n''',
    "runtime timing diagnostic argument",
)
text = replace_once(
    text,
    '''        self._cuda_stream_priority = cuda_stream_priority\n        self._diagnostic_layer_delay_ms = float(diagnostic_layer_delay_ms)\n        self._logging_enabled = bool(logging_enabled)\n''',
    '''        self._cuda_stream_priority = cuda_stream_priority\n        self._diagnostic_layer_delay_ms = float(diagnostic_layer_delay_ms)\n        # vvv THOG keep correctness/synchronisation events but make timestamp diagnostics opt-in\n        if not isinstance(enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n        self._enable_gpu_timing_diagnostic = enable_gpu_timing_diagnostic\n        # ^^^ THOG\n        self._logging_enabled = bool(logging_enabled)\n''',
    "runtime timing diagnostic storage",
)

old_wait = '''        elif candidate.state == CandidateState.MATERIALISING:\n            if candidate.completion_event is None:\n                raise RuntimeError(f"materialising candidate {key} has no completion event")\n            wait_start = torch.cuda.Event(enable_timing=True)\n            wait_end = torch.cuda.Event(enable_timing=True)\n            wait_start.record(current_stream)\n            current_stream.wait_event(candidate.completion_event)\n            wait_end.record(current_stream)\n            # Host submission has reached the dependency, but this does NOT tell\n            # us whether the GPU Main Stream will actually wait there. Keep the\n            # result provisional until CUDA has timestamped both streams.\n            candidate.critical_path_miss = False\n            candidate.final_outcome = "PENDING"\n            # candidate.tensor remains strongly referenced through consumed().\n            wait_payload = self._transition(\n                candidate,\n                CandidateState.CONSUMING,\n                "critical_path_wait",\n                outcome="gpu_wait_pending",\n                reason="classification_pending_until_main_stream_dependency",\n                detail={"gpu_wait_classification_pending": True},\n            )\n            self._pending_timings.append(\n                _PendingCudaTiming(\n                    kind="main_stream_wait",\n                    start_event=wait_start,\n                    end_event=wait_end,\n                    event_payload=wait_payload,\n                    pass_sequence=self._pass_sequence,\n                    layer_index=candidate.layer_index,\n                    family=candidate.family,\n                    dependency_event=candidate.completion_event,\n                    candidate=candidate,\n                )\n            )\n'''
new_wait = '''        elif candidate.state == CandidateState.MATERIALISING:\n            if candidate.completion_event is None:\n                raise RuntimeError(f"materialising candidate {key} has no completion event")\n            # vvv THOG the Main Stream dependency is required; timestamp/classification events are diagnostic only\n            if self._enable_gpu_timing_diagnostic:\n                wait_start = torch.cuda.Event(enable_timing=True)\n                wait_end = torch.cuda.Event(enable_timing=True)\n                wait_start.record(current_stream)\n                current_stream.wait_event(candidate.completion_event)\n                wait_end.record(current_stream)\n                # Host submission has reached the dependency, but this does NOT tell\n                # us whether the GPU Main Stream will actually wait there. Keep the\n                # result provisional until CUDA has timestamped both streams.\n                candidate.critical_path_miss = False\n                candidate.final_outcome = "PENDING"\n                wait_payload = self._transition(\n                    candidate,\n                    CandidateState.CONSUMING,\n                    "critical_path_wait",\n                    outcome="gpu_wait_pending",\n                    reason="classification_pending_until_main_stream_dependency",\n                    detail={\n                        "gpu_wait_classification_pending": True,\n                        "gpu_timing_diagnostic_enabled": True,\n                    },\n                )\n                self._pending_timings.append(\n                    _PendingCudaTiming(\n                        kind="main_stream_wait",\n                        start_event=wait_start,\n                        end_event=wait_end,\n                        event_payload=wait_payload,\n                        pass_sequence=self._pass_sequence,\n                        layer_index=candidate.layer_index,\n                        family=candidate.family,\n                        dependency_event=candidate.completion_event,\n                        candidate=candidate,\n                    )\n                )\n            else:\n                current_stream.wait_event(candidate.completion_event)\n                # Without GPU timestamps, MATERIALISING at host submission is only\n                # a conservative host-observed partial classification.  This is\n                # telemetry only and does not alter the dependency or materialisation.\n                candidate.critical_path_miss = True\n                candidate.final_outcome = "PARTIAL HIT"\n                self._aggregate["waited_hits"] += 1\n                self._transition(\n                    candidate,\n                    CandidateState.CONSUMING,\n                    "critical_path_wait",\n                    outcome="host_observed_materialising",\n                    reason="gpu_timing_diagnostic_disabled",\n                    detail={\n                        "gpu_wait_classification_pending": False,\n                        "gpu_timing_diagnostic_enabled": False,\n                    },\n                )\n            # candidate.tensor remains strongly referenced through consumed().\n            # ^^^ THOG\n'''
text = replace_once(text, old_wait, new_wait, "diagnostic-gated main-stream wait timing")

old_fallback_timing = '''            materialise_start = torch.cuda.Event(enable_timing=True)\n            materialise_end = torch.cuda.Event(enable_timing=True)\n            materialise_start.record(current_stream)\n            try:\n'''
new_fallback_timing = '''            # vvv THOG main-stream fallback timestamps are diagnostic; fallback materialisation itself is unchanged\n            materialise_start = (\n                torch.cuda.Event(enable_timing=True)\n                if self._enable_gpu_timing_diagnostic\n                else None\n            )\n            materialise_end = (\n                torch.cuda.Event(enable_timing=True)\n                if self._enable_gpu_timing_diagnostic\n                else None\n            )\n            if materialise_start is not None:\n                materialise_start.record(current_stream)\n            # ^^^ THOG\n            try:\n'''
text = replace_once(text, old_fallback_timing, new_fallback_timing, "fallback timing start")
text = replace_once(
    text,
    '''            materialise_end.record(current_stream)\n            self._pending_timings.append(\n                _PendingCudaTiming(\n                    kind="main_stream_materialisation",\n                    start_event=materialise_start,\n                    end_event=materialise_end,\n                    event_payload=fallback_payload,\n                    pass_sequence=self._pass_sequence,\n                    layer_index=candidate.layer_index,\n                    family=candidate.family,\n                    candidate=candidate,\n                )\n            )\n''',
    '''            # vvv THOG retain no CUDA timing objects when the diagnostic is disabled\n            if materialise_start is not None and materialise_end is not None:\n                materialise_end.record(current_stream)\n                self._pending_timings.append(\n                    _PendingCudaTiming(\n                        kind="main_stream_materialisation",\n                        start_event=materialise_start,\n                        end_event=materialise_end,\n                        event_payload=fallback_payload,\n                        pass_sequence=self._pass_sequence,\n                        layer_index=candidate.layer_index,\n                        family=candidate.family,\n                        candidate=candidate,\n                    )\n                )\n            # ^^^ THOG\n''',
    "fallback timing append",
)

old_submit_events = '''                    candidate.materialisation_start_event = torch.cuda.Event(\n                        enable_timing=True\n                    )\n                    candidate.completion_event = torch.cuda.Event(enable_timing=True)\n                    candidate.materialisation_start_event.record(self._stream)\n'''
new_submit_events = '''                    # vvv THOG completion is required for dependency/query semantics; the start timestamp is diagnostic only\n                    candidate.materialisation_start_event = (\n                        torch.cuda.Event(enable_timing=True)\n                        if self._enable_gpu_timing_diagnostic\n                        else None\n                    )\n                    candidate.completion_event = torch.cuda.Event(\n                        enable_timing=self._enable_gpu_timing_diagnostic\n                    )\n                    if candidate.materialisation_start_event is not None:\n                        candidate.materialisation_start_event.record(self._stream)\n                    # ^^^ THOG\n'''
text = replace_once(text, old_submit_events, new_submit_events, "premat submission timing events")
text = replace_once(
    text,
    '''            self._pending_timings.append(\n                _PendingCudaTiming(\n                    kind="premat_materialisation",\n                    start_event=candidate.materialisation_start_event,\n                    end_event=candidate.completion_event,\n                    event_payload=launch_payload,\n                    pass_sequence=self._pass_sequence,\n                    layer_index=candidate.layer_index,\n                    family=candidate.family,\n                    candidate=candidate,\n                )\n            )\n''',
    '''            # vvv THOG no materialisation timestamp bookkeeping exists in normal/default PREMAT execution\n            if (\n                self._enable_gpu_timing_diagnostic\n                and candidate.materialisation_start_event is not None\n                and candidate.completion_event is not None\n            ):\n                self._pending_timings.append(\n                    _PendingCudaTiming(\n                        kind="premat_materialisation",\n                        start_event=candidate.materialisation_start_event,\n                        end_event=candidate.completion_event,\n                        event_payload=launch_payload,\n                        pass_sequence=self._pass_sequence,\n                        layer_index=candidate.layer_index,\n                        family=candidate.family,\n                        candidate=candidate,\n                    )\n                )\n            # ^^^ THOG\n''',
    "premat timing append",
)
text = replace_once(
    text,
    '''            "diagnostic_layer_delay_ms": self._diagnostic_layer_delay_ms,\n            "headroom_mode": (\n''',
    '''            "diagnostic_layer_delay_ms": self._diagnostic_layer_delay_ms,\n            # vvv THOG expose whether expensive CUDA timestamp diagnostics are part of this run\n            "enable_gpu_timing_diagnostic": self._enable_gpu_timing_diagnostic,\n            # ^^^ THOG\n            "headroom_mode": (\n''',
    "premat report diagnostic flag",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# sheet/model.py: model config and runtime propagation.
# ---------------------------------------------------------------------------
p = Path("sheet/model.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    premat_logging: str = "disabled"\n''',
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    # vvv THOG expensive GPU timestamp diagnostics are opt-in and separate from PREMAT correctness events\n    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n    premat_logging: str = "disabled"\n''',
    "SheetGPTConfig diagnostic field",
)
text = replace_once(
    text,
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            logging=self.premat_logging,\n''',
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            logging=self.premat_logging,\n''',
    "SheetGPTConfig premat validation propagation",
)
text = replace_once(
    text,
    '''                diagnostic_layer_delay_ms=config.premat_diagnostic_layer_delay_ms,\n                logging_enabled=(\n''',
    '''                diagnostic_layer_delay_ms=config.premat_diagnostic_layer_delay_ms,\n                enable_gpu_timing_diagnostic=config.premat_enable_gpu_timing_diagnostic,\n                logging_enabled=(\n''',
    "PrematRuntime model propagation",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# sheet/training_config.py: persistent execution diagnostic + model arguments.
# ---------------------------------------------------------------------------
p = Path("sheet/training_config.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips"}\n''',
    '''EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic"}\n''',
    "execution override field",
)
text = replace_once(
    text,
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    premat_logging: str = "disabled"\n''',
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    # vvv THOG diagnostic timestamps default off so ordinary PREMAT carries no measurement cost\n    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n    premat_logging: str = "disabled"\n''',
    "TrainingConfig diagnostic field",
)
text = replace_once(
    text,
    '''        if not isinstance(self.premat_retain_detailed_premat_history, bool):\n            raise ValueError("premat_retain_detailed_premat_history must be bool")\n        validate_premat_configuration(\n''',
    '''        if not isinstance(self.premat_retain_detailed_premat_history, bool):\n            raise ValueError("premat_retain_detailed_premat_history must be bool")\n        if not isinstance(self.premat_enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n        validate_premat_configuration(\n''',
    "TrainingConfig diagnostic validation",
)
text = replace_once(
    text,
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            logging=self.premat_logging,\n''',
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            logging=self.premat_logging,\n''',
    "TrainingConfig premat validation propagation",
)
text = replace_once(
    text,
    '''        if not self.premat_retain_detailed_premat_history:\n            values.pop("premat_retain_detailed_premat_history", None)\n        return values\n''',
    '''        if not self.premat_retain_detailed_premat_history:\n            values.pop("premat_retain_detailed_premat_history", None)\n        # vvv THOG default-off diagnostic does not perturb established checkpoint identity\n        if not self.premat_enable_gpu_timing_diagnostic:\n            values.pop("premat_enable_gpu_timing_diagnostic", None)\n        # ^^^ THOG\n        return values\n''',
    "TrainingConfig default omission",
)
text = replace_once(
    text,
    '''                    "premat_diagnostic_layer_delay_ms": float(self.premat_diagnostic_layer_delay_ms),\n                    "premat_logging": self.premat_logging,\n''',
    '''                    "premat_diagnostic_layer_delay_ms": float(self.premat_diagnostic_layer_delay_ms),\n                    "premat_enable_gpu_timing_diagnostic": self.premat_enable_gpu_timing_diagnostic,\n                    "premat_logging": self.premat_logging,\n''',
    "TrainingConfig model arguments",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# sheet/run_config.py: public run config, provenance, naming, trainer propagation.
# ---------------------------------------------------------------------------
p = Path("sheet/run_config.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    premat_logging: str = "disabled"\n''',
    '''    premat_diagnostic_layer_delay_ms: float = 0.0\n    # vvv THOG CUDA timing/classification is a conspicuous opt-in diagnostic\n    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n    premat_logging: str = "disabled"\n''',
    "OwtRunConfig diagnostic field",
)
text = replace_once(
    text,
    '''        if not isinstance(self.premat_retain_detailed_premat_history, bool):\n            raise ValueError("premat_retain_detailed_premat_history must be bool")\n        validate_premat_configuration(\n''',
    '''        if not isinstance(self.premat_retain_detailed_premat_history, bool):\n            raise ValueError("premat_retain_detailed_premat_history must be bool")\n        if not isinstance(self.premat_enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n        validate_premat_configuration(\n''',
    "OwtRunConfig diagnostic validation",
)
text = replace_once(
    text,
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            logging=self.premat_logging,\n''',
    '''            diagnostic_layer_delay_ms=self.premat_diagnostic_layer_delay_ms,\n            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            logging=self.premat_logging,\n''',
    "OwtRunConfig premat validation propagation",
)
text = replace_once(
    text,
    '''            or float(self.premat_diagnostic_layer_delay_ms) != 0.0\n            or self.premat_logging != "disabled"\n''',
    '''            or float(self.premat_diagnostic_layer_delay_ms) != 0.0\n            or self.premat_enable_gpu_timing_diagnostic\n            or self.premat_logging != "disabled"\n''',
    "artifact diagnostic condition",
)
text = replace_once(
    text,
    '''            premat_fragment = (\n                "PM__"\n''',
    '''            timing_diagnostic_fragment = (\n                "GTD_" if self.premat_enable_gpu_timing_diagnostic else ""\n            )\n            premat_fragment = (\n                "PM__"\n''',
    "artifact diagnostic fragment setup",
)
text = replace_once(
    text,
    '''                f"{delay_fragment}"\n                f"L{self.premat_logging[0].upper()}_"\n''',
    '''                f"{delay_fragment}"\n                f"{timing_diagnostic_fragment}"\n                f"L{self.premat_logging[0].upper()}_"\n''',
    "artifact diagnostic fragment insertion",
)
text = replace_once(
    text,
    '''            premat_diagnostic_layer_delay_ms=float(self.premat_diagnostic_layer_delay_ms),\n            premat_logging=self.premat_logging,\n''',
    '''            premat_diagnostic_layer_delay_ms=float(self.premat_diagnostic_layer_delay_ms),\n            premat_enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            premat_logging=self.premat_logging,\n''',
    "training config diagnostic propagation",
)
text = replace_once(
    text,
    '''        if not self.premat_retain_detailed_premat_history:\n            values.pop("premat_retain_detailed_premat_history", None)\n        return values\n''',
    '''        if not self.premat_retain_detailed_premat_history:\n            values.pop("premat_retain_detailed_premat_history", None)\n        # vvv THOG default-off diagnostic does not perturb established run identity\n        if not self.premat_enable_gpu_timing_diagnostic:\n            values.pop("premat_enable_gpu_timing_diagnostic", None)\n        # ^^^ THOG\n        return values\n''',
    "run config default omission",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# run_thog2_owt_core.py: public CLI, resume override, config, startup report.
# ---------------------------------------------------------------------------
p = Path("run_thog2_owt_core.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    parser.add_argument("--premat_diagnostic_layer_delay_ms", type=float, default=0.0)\n    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\n''',
    '''    parser.add_argument("--premat_diagnostic_layer_delay_ms", type=float, default=0.0)\n    # vvv THOG expensive CUDA timestamp/classification diagnostic is explicit and default-off\n    parser.add_argument(\n        "--premat_enable_gpu_timing_diagnostic",\n        type=_true_false,\n        default=False,\n        metavar="true|false",\n        help="enable PREMAT CUDA timestamp/classification diagnostics; default false",\n    )\n    # ^^^ THOG\n    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\n''',
    "CLI diagnostic argument",
)
text = replace_once(
    text,
    '''        premat_diagnostic_layer_delay_ms=arguments.premat_diagnostic_layer_delay_ms,\n        premat_logging=arguments.premat_logging,\n''',
    '''        premat_diagnostic_layer_delay_ms=arguments.premat_diagnostic_layer_delay_ms,\n        premat_enable_gpu_timing_diagnostic=arguments.premat_enable_gpu_timing_diagnostic,\n        premat_logging=arguments.premat_logging,\n''',
    "CLI config propagation",
)
text = replace_once(
    text,
    '''            f"premat_diagnostic_layer_delay_ms={config.premat_diagnostic_layer_delay_ms:g} "\n            f"premat_logging={config.premat_logging} "\n''',
    '''            f"premat_diagnostic_layer_delay_ms={config.premat_diagnostic_layer_delay_ms:g} "\n            f"premat_enable_gpu_timing_diagnostic={str(config.premat_enable_gpu_timing_diagnostic).lower()} "\n            f"premat_logging={config.premat_logging} "\n''',
    "startup report diagnostic field",
)
# The resume override dict is intentionally one line in this file.
text = replace_once(
    text,
    '''"max_nonfinite_update_skips": training_config.max_nonfinite_update_skips})\n''',
    '''"max_nonfinite_update_skips": training_config.max_nonfinite_update_skips, "premat_enable_gpu_timing_diagnostic": training_config.premat_enable_gpu_timing_diagnostic})\n''',
    "resume diagnostic override",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# train_OWT_core.sh: wrapper user surface and forwarding.
# ---------------------------------------------------------------------------
p = Path("train_OWT_core.sh")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="0"\nPREMAT_LOGGING="disabled"\n''',
    '''PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="0"\n# vvv THOG CUDA timestamp diagnostics are deliberately conspicuous and default-off\nPREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=false\n# ^^^ THOG\nPREMAT_LOGGING="disabled"\n''',
    "wrapper diagnostic default",
)
text = replace_once(
    text,
    '''  --premat_diagnostic_layer_delay_ms VALUE=${PREMAT_DIAGNOSTIC_LAYER_DELAY_MS}  diagnostic host-dispatch delay after each non-final layer\n  --premat_logging enabled|disabled=${PREMAT_LOGGING}\n''',
    '''  --premat_diagnostic_layer_delay_ms VALUE=${PREMAT_DIAGNOSTIC_LAYER_DELAY_MS}  diagnostic host-dispatch delay after each non-final layer\n  --premat_enable_gpu_timing_diagnostic true|false=${PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC}  CUDA timestamp/classification diagnostic; default false\n  --premat_logging enabled|disabled=${PREMAT_LOGGING}\n''',
    "wrapper diagnostic help",
)
text = replace_once(
    text,
    '''    --premat|--premat_attention_mode|--premat_target_layer|--premat_weight_matrix_target_order|--premat_gpu_memory_buffer_gb|--premat_cuda_stream_priority|--premat_diagnostic_layer_delay_ms|--premat_logging|--premat_instra)\n''',
    '''    --premat|--premat_attention_mode|--premat_target_layer|--premat_weight_matrix_target_order|--premat_gpu_memory_buffer_gb|--premat_cuda_stream_priority|--premat_diagnostic_layer_delay_ms|--premat_enable_gpu_timing_diagnostic|--premat_logging|--premat_instra)\n''',
    "wrapper diagnostic split argument matcher",
)
text = replace_once(
    text,
    '''        --premat_diagnostic_layer_delay_ms) PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="$2" ;;\n        --premat_logging) PREMAT_LOGGING="$2" ;;\n''',
    '''        --premat_diagnostic_layer_delay_ms) PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="$2" ;;\n        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC="$2" ;;\n        --premat_logging) PREMAT_LOGGING="$2" ;;\n''',
    "wrapper diagnostic split assignment",
)
text = replace_once(
    text,
    '''    --premat=*|--premat_attention_mode=*|--premat_target_layer=*|--premat_weight_matrix_target_order=*|--premat_gpu_memory_buffer_gb=*|--premat_cuda_stream_priority=*|--premat_diagnostic_layer_delay_ms=*|--premat_logging=*|--premat_instra=*)\n''',
    '''    --premat=*|--premat_attention_mode=*|--premat_target_layer=*|--premat_weight_matrix_target_order=*|--premat_gpu_memory_buffer_gb=*|--premat_cuda_stream_priority=*|--premat_diagnostic_layer_delay_ms=*|--premat_enable_gpu_timing_diagnostic=*|--premat_logging=*|--premat_instra=*)\n''',
    "wrapper diagnostic equals matcher",
)
text = replace_once(
    text,
    '''        --premat_diagnostic_layer_delay_ms) PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="$premat_value" ;;\n        --premat_logging) PREMAT_LOGGING="$premat_value" ;;\n''',
    '''        --premat_diagnostic_layer_delay_ms) PREMAT_DIAGNOSTIC_LAYER_DELAY_MS="$premat_value" ;;\n        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC="$premat_value" ;;\n        --premat_logging) PREMAT_LOGGING="$premat_value" ;;\n''',
    "wrapper diagnostic equals assignment",
)
text = replace_once(
    text,
    '''case "$PREMAT_TARGET_LAYER" in 0|1|2) ;; *) echo "PREMAT_TARGET_LAYER must be 0, 1 or 2." >&2; exit 2 ;; esac\n''',
    '''case "$PREMAT_TARGET_LAYER" in 0|1|2|10) ;; *) echo "PREMAT_TARGET_LAYER must be 0, 1, 2 or 10 (10 means +1 then +0)." >&2; exit 2 ;; esac\n''',
    "wrapper target10 validation refresh",
)
text = replace_once(
    text,
    '''validate_nonnegative_number "$PREMAT_DIAGNOSTIC_LAYER_DELAY_MS" "PREMAT_DIAGNOSTIC_LAYER_DELAY_MS"\nif [[ "$PREMAT" == enabled ]]; then\n''',
    '''validate_nonnegative_number "$PREMAT_DIAGNOSTIC_LAYER_DELAY_MS" "PREMAT_DIAGNOSTIC_LAYER_DELAY_MS"\nvalidate_true_false "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC" "PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC"\nif [[ "$PREMAT" == enabled ]]; then\n''',
    "wrapper diagnostic validation",
)
text = replace_once(
    text,
    '''  optional_args+=(--premat_diagnostic_layer_delay_ms "$PREMAT_DIAGNOSTIC_LAYER_DELAY_MS")\n  optional_args+=(--premat_logging "$PREMAT_LOGGING")\n''',
    '''  optional_args+=(--premat_diagnostic_layer_delay_ms "$PREMAT_DIAGNOSTIC_LAYER_DELAY_MS")\n  optional_args+=(--premat_enable_gpu_timing_diagnostic "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC")\n  optional_args+=(--premat_logging "$PREMAT_LOGGING")\n''',
    "wrapper diagnostic forwarding",
)
text = text.replace(
    '--premat_target_layer 0|1|2=${PREMAT_TARGET_LAYER}',
    '--premat_target_layer 0|1|2|10=${PREMAT_TARGET_LAYER}  10 means ordered +1 then +0',
    1,
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# tests/test_premat.py: preserve timing tests explicitly; add default-off ablation.
# ---------------------------------------------------------------------------
p = Path("tests/test_premat.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    reference_element_bytes: int = 2,\n    attach=None,\n):\n''',
    '''    reference_element_bytes: int = 2,\n    attach=None,\n    enable_gpu_timing_diagnostic: bool = True,\n):\n''',
    "test runtime helper diagnostic argument",
)
text = replace_once(
    text,
    '''        diagnostic_layer_delay_ms=diagnostic_layer_delay_ms,\n        logging_enabled=True,\n''',
    '''        diagnostic_layer_delay_ms=diagnostic_layer_delay_ms,\n        enable_gpu_timing_diagnostic=enable_gpu_timing_diagnostic,\n        logging_enabled=True,\n''',
    "test runtime helper propagation",
)
text = replace_once(
    text,
    '''def test_public_cli_exposes_exactly_the_thirteen_premat_options() -> None:\n''',
    '''def test_public_cli_exposes_exactly_the_fourteen_premat_options() -> None:\n''',
    "premat option count test name",
)
text = replace_once(
    text,
    '''        "--premat_diagnostic_layer_delay_ms",\n        "--premat_logging",\n''',
    '''        "--premat_diagnostic_layer_delay_ms",\n        "--premat_enable_gpu_timing_diagnostic",\n        "--premat_logging",\n''',
    "premat option set diagnostic flag",
)
anchor = '''def test_sampled_live_report_waits_for_gpu_classification_without_sync(monkeypatch) -> None:\n'''
new_tests = '''# vvv THOG PREMAT GPU timing is a default-off diagnostic rather than ordinary scheduler work\ndef test_gpu_timing_diagnostic_cli_defaults_false_and_accepts_explicit_true() -> None:\n    parser = build_parser()\n    default_arguments = parser.parse_args(["--model-type", "sheet"])\n    assert default_arguments.premat_enable_gpu_timing_diagnostic is False\n\n    enabled_arguments = parser.parse_args([\n        "--model-type",\n        "sheet",\n        "--premat_enable_gpu_timing_diagnostic",\n        "true",\n    ])\n    enabled_config = config_from_arguments(enabled_arguments)\n    assert enabled_config.premat_enable_gpu_timing_diagnostic is True\n    assert enabled_config.to_training_config(\n        vocab_size=32,\n        world_size=1,\n        out_dir=Path("out-test"),\n    ).premat_enable_gpu_timing_diagnostic is True\n\n\ndef test_gpu_timing_diagnostic_disabled_keeps_dependency_without_timestamp_events(monkeypatch) -> None:\n    runtime, fake_cuda, calls = _runtime(\n        monkeypatch,\n        stay_below_current_peak=False,\n        enable_gpu_timing_diagnostic=False,\n    )\n    runtime.layer_start(3)\n    first_candidate = runtime._candidates[(5, "DOWN")]\n    assert first_candidate.materialisation_start_event is None\n    assert first_candidate.completion_event is not None\n    assert first_candidate.completion_event.enable_timing is False\n    assert runtime._pending_timings == []\n\n    runtime.layer_start(5)\n    runtime.acquire("DOWN", 5)\n    candidate = runtime._candidates[(5, "DOWN")]\n    assert len(fake_cuda.main_stream.waited_events) == 1\n    assert runtime._pending_timings == []\n    assert candidate.final_outcome == "PARTIAL HIT"\n    assert runtime._aggregate["waited_hits"] == 1\n    assert runtime._aggregate["main_stream_wait_ms_total"] == pytest.approx(0.0)\n    assert runtime.report()["enable_gpu_timing_diagnostic"] is False\n    assert calls.count(("DOWN", 5)) == 1\n# ^^^ THOG\n\n\n'''
text = replace_once(text, anchor, new_tests + anchor, "diagnostic ablation regression tests")
p.write_text(text, encoding="utf-8")
