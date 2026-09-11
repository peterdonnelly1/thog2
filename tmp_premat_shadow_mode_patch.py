from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# sheet/premat.py
# ---------------------------------------------------------------------------
p = Path("sheet/premat.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    enable_gpu_timing_diagnostic: bool = False,\n    logging: str,\n",
    "    enable_gpu_timing_diagnostic: bool = False,\n    shadow_mode: bool = False,\n    logging: str,\n",
    "premat validation shadow signature",
)
text = replace_once(
    text,
    '    if not isinstance(enable_gpu_timing_diagnostic, bool):\n        raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n    # ^^^ THOG\n',
    '    if not isinstance(enable_gpu_timing_diagnostic, bool):\n        raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n    # ^^^ THOG\n    # vvv THOG shadow PREMAT is an explicit diagnostic and never a default execution path\n    if not isinstance(shadow_mode, bool):\n        raise ValueError("premat_enable_shadow_mode must be bool")\n    if shadow_mode and premat != "enabled":\n        raise ValueError("premat_enable_shadow_mode requires premat enabled")\n    # ^^^ THOG\n',
    "premat validation shadow body",
)
text = replace_once(
    text,
    "        enable_gpu_timing_diagnostic: bool = False,\n        logging_enabled: bool,\n",
    "        enable_gpu_timing_diagnostic: bool = False,\n        shadow_mode: bool = False,\n        logging_enabled: bool,\n",
    "runtime shadow signature",
)
text = replace_once(
    text,
    '        self._enable_gpu_timing_diagnostic = enable_gpu_timing_diagnostic\n        # ^^^ THOG\n        self._logging_enabled = bool(logging_enabled)\n',
    '        self._enable_gpu_timing_diagnostic = enable_gpu_timing_diagnostic\n        # ^^^ THOG\n        # vvv THOG shadow mode runs admission/scheduling but suppresses side-stream weight materialisation\n        if not isinstance(shadow_mode, bool):\n            raise ValueError("premat_enable_shadow_mode must be bool")\n        self._shadow_mode = shadow_mode\n        # ^^^ THOG\n        self._logging_enabled = bool(logging_enabled)\n',
    "runtime shadow field",
)
text = replace_once(
    text,
    '            "ordinary_deadline_materialisations": 0,\n            "premat_materialisation_ms_total": 0.0,\n',
    '            "ordinary_deadline_materialisations": 0,\n            "shadow_main_materialisations": 0,\n            "premat_materialisation_ms_total": 0.0,\n',
    "shadow aggregate",
)
text = replace_once(
    text,
    '        current_stream = torch.cuda.current_stream(device=self._device)\n        if (\n            candidate.state\n',
    '        current_stream = torch.cuda.current_stream(device=self._device)\n        # vvv THOG shadow PREMAT preserves scheduler/admission/event overhead but Main Stream still materialises the real weight\n        if self._shadow_mode:\n            return self._acquire_shadow(candidate, current_stream)\n        # ^^^ THOG\n        if (\n            candidate.state\n',
    "shadow acquire dispatch",
)
shadow_helper = '''\n    # vvv THOG shadow-mode deadline path: no side-stream weight exists; ordinary differentiable materialisation remains on Main Stream\n    def _acquire_shadow(self, candidate: _Candidate, current_stream) -> Tensor:\n        if candidate.state == CandidateState.MATERIALISING:\n            if candidate.completion_event is None:\n                raise RuntimeError(\n                    f"shadow premat candidate {(candidate.layer_index, candidate.family)} has no completion event"\n                )\n            # Retain the normal PREMAT dependency/event-management overhead even\n            # though the side stream deliberately performed no weight work.\n            current_stream.wait_event(candidate.completion_event)\n        elif candidate.state not in (CandidateState.AVAILABLE, CandidateState.UNAVAILABLE):\n            raise RuntimeError(\n                "shadow premat candidate cannot be acquired from "\n                f"{candidate.state.value}"\n            )\n\n        # The scheduler carried theoretical PREMAT memory/certificate charges up\n        # to the deadline. Release those before the ordinary Main Stream weight\n        # replaces the hypothetical side-stream resident tensor.\n        if candidate.retained_counted or candidate.transient_counted:\n            self._rollback_candidate_charge(candidate)\n        self._release_allocator_certificate_candidate(candidate)\n        candidate.owner = "main"\n        candidate.critical_path_miss = False\n        candidate.final_outcome = "SHADOW MAIN MATERIALISATION"\n        candidate.launch_ns = candidate.launch_ns or time.perf_counter_ns()\n\n        if candidate.state == CandidateState.UNAVAILABLE:\n            self._transition(\n                candidate,\n                CandidateState.MATERIALISING,\n                "shadow_materialising_on_critical_path",\n                decision="shadow_main_claim",\n                outcome="shadow_main_materialisation",\n                reason="shadow_mode",\n            )\n\n        shadow_payload = self._record(\n            "shadow_main_materialising",\n            candidate=candidate,\n            decision="shadow_main_claim",\n            outcome="ordinary_deadline_materialisation",\n            reason="shadow_mode",\n            detail={"shadow_mode": True},\n        )\n        materialise_start = (\n            torch.cuda.Event(enable_timing=True)\n            if self._enable_gpu_timing_diagnostic\n            else None\n        )\n        materialise_end = (\n            torch.cuda.Event(enable_timing=True)\n            if self._enable_gpu_timing_diagnostic\n            else None\n        )\n        if materialise_start is not None:\n            materialise_start.record(current_stream)\n        try:\n            candidate.tensor = self._materialize(\n                candidate.family,\n                candidate.layer_index,\n            )\n        except BaseException as error:\n            self._record(\n                "shadow_main_materialisation_failed",\n                candidate=candidate,\n                outcome="failure",\n                reason=type(error).__name__,\n                detail={"error": str(error)},\n            )\n            raise RuntimeError(\n                "Shadow PREMAT Main Stream materialisation failed; "\n                f"layer={candidate.layer_index}, family={candidate.family}, "\n                f"mode={self._attention_mode}, memory={self._memory_summary()}"\n            ) from error\n        if materialise_start is not None and materialise_end is not None:\n            materialise_end.record(current_stream)\n            self._pending_timings.append(\n                _PendingCudaTiming(\n                    kind="main_stream_materialisation",\n                    start_event=materialise_start,\n                    end_event=materialise_end,\n                    event_payload=shadow_payload,\n                    pass_sequence=self._pass_sequence,\n                    layer_index=candidate.layer_index,\n                    family=candidate.family,\n                    candidate=candidate,\n                )\n            )\n        candidate.available_ns = time.perf_counter_ns()\n        if candidate.state == CandidateState.MATERIALISING:\n            self._transition(\n                candidate,\n                CandidateState.AVAILABLE,\n                "shadow_main_available",\n                outcome="ordinary_deadline_materialisation",\n                reason="shadow_mode",\n            )\n        self._transition(\n            candidate,\n            CandidateState.CONSUMING,\n            "shadow_main_consuming",\n            outcome="shadow_main_materialisation",\n            reason="shadow_mode",\n        )\n        self._aggregate["ordinary_deadline_materialisations"] += 1\n        self._aggregate["shadow_main_materialisations"] += 1\n        if candidate.tensor is None:\n            raise RuntimeError("shadow PREMAT Main Stream materialisation produced no tensor")\n        candidate.tensor.record_stream(current_stream)\n        return candidate.tensor\n    # ^^^ THOG\n\n'''
text = replace_once(
    text,
    "    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:\n",
    shadow_helper + "    def materialize_for_consumption(self, family: str, layer_index: int) -> Tensor:\n",
    "insert shadow acquire helper",
)
text = replace_once(
    text,
    '''                    # Autocast caches lower-precision casts of FP32 parameter\n                    # leaves for the enclosing forward.  A cast produced here\n                    # belongs to the Premat Stream and, under PyTorch 2.8\n                    # no_grad(), is detached.  Publishing it through the shared\n                    # autocast cache lets ordinary Main Stream materialisation\n                    # reuse storage with neither a dependency nor a gradient\n                    # edge.  Keep autocast's dtype policy, but make Premat casts\n                    # private to this submission.\n                    autocast_cache_enabled = torch.is_autocast_cache_enabled()\n                    torch.set_autocast_cache_enabled(False)\n                    try:\n                        with torch.no_grad():\n                            candidate.tensor = self._materialize(\n                                candidate.family,\n                                candidate.layer_index,\n                            )\n                    finally:\n                        torch.set_autocast_cache_enabled(autocast_cache_enabled)\n                    actual_retained_bytes = int(\n                        candidate.tensor.numel() * candidate.tensor.element_size()\n                    )\n                    if actual_retained_bytes > candidate.envelope.retained_bytes:\n                        raise RuntimeError(\n                            "materialised tensor exceeds its pre-admission retained estimate: "\n                            f"actual={actual_retained_bytes}, "\n                            f"predicted={candidate.envelope.retained_bytes}"\n                        )\n                    candidate.tensor.record_stream(self._stream)\n                    candidate.completion_event.record(self._stream)\n''',
    '''                    # vvv THOG shadow mode records the normal side-stream completion event but deliberately launches no weight materialisation\n                    if self._shadow_mode:\n                        candidate.tensor = None\n                        candidate.completion_event.record(self._stream)\n                    else:\n                        # Autocast caches lower-precision casts of FP32 parameter\n                        # leaves for the enclosing forward.  A cast produced here\n                        # belongs to the Premat Stream and, under PyTorch 2.8\n                        # no_grad(), is detached.  Publishing it through the shared\n                        # autocast cache lets ordinary Main Stream materialisation\n                        # reuse storage with neither a dependency nor a gradient\n                        # edge.  Keep autocast's dtype policy, but make Premat casts\n                        # private to this submission.\n                        autocast_cache_enabled = torch.is_autocast_cache_enabled()\n                        torch.set_autocast_cache_enabled(False)\n                        try:\n                            with torch.no_grad():\n                                candidate.tensor = self._materialize(\n                                    candidate.family,\n                                    candidate.layer_index,\n                                )\n                        finally:\n                            torch.set_autocast_cache_enabled(autocast_cache_enabled)\n                        actual_retained_bytes = int(\n                            candidate.tensor.numel() * candidate.tensor.element_size()\n                        )\n                        if actual_retained_bytes > candidate.envelope.retained_bytes:\n                            raise RuntimeError(\n                                "materialised tensor exceeds its pre-admission retained estimate: "\n                                f"actual={actual_retained_bytes}, "\n                                f"predicted={candidate.envelope.retained_bytes}"\n                            )\n                        candidate.tensor.record_stream(self._stream)\n                        candidate.completion_event.record(self._stream)\n                    # ^^^ THOG\n''',
    "shadow suppress side materialisation",
)
text = replace_once(
    text,
    '                outcome="submitted_to_premat_stream",\n                reason=decision.reason,\n',
    '                outcome=(\n                    "submitted_to_shadow_premat_stream"\n                    if self._shadow_mode\n                    else "submitted_to_premat_stream"\n                ),\n                reason=decision.reason,\n',
    "shadow launch outcome",
)
text = replace_once(
    text,
    '    def _observe_memory(self) -> PrematMemoryObservation:\n        if self._device is None:\n',
    '    # vvv THOG shadow charges are theoretical scheduler state and must not be subtracted from real CUDA allocation counters\n    def _physical_retained_bytes(self) -> int:\n        return 0 if self._shadow_mode else self._retained_bytes\n    # ^^^ THOG\n\n    def _observe_memory(self) -> PrematMemoryObservation:\n        if self._device is None:\n',
    "physical retained helper",
)
text = text.replace("allocated - self._retained_bytes", "allocated - self._physical_retained_bytes()", 2)
text = replace_once(
    text,
    "            observation.process_allocated_bytes - self._retained_bytes,\n",
    "            observation.process_allocated_bytes - self._physical_retained_bytes(),\n",
    "charged process physical retained",
)
text = replace_once(
    text,
    "            observation.device_used_bytes - self._retained_bytes,\n",
    "            observation.device_used_bytes - self._physical_retained_bytes(),\n",
    "charged device physical retained",
)
text = replace_once(
    text,
    '            "enable_gpu_timing_diagnostic": self._enable_gpu_timing_diagnostic,\n            # ^^^ THOG\n',
    '            "enable_gpu_timing_diagnostic": self._enable_gpu_timing_diagnostic,\n            "shadow_mode": self._shadow_mode,\n            # ^^^ THOG\n',
    "shadow report flag",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# sheet/model.py
# ---------------------------------------------------------------------------
p = Path("sheet/model.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n",
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    premat_enable_shadow_mode: bool = False\n    # ^^^ THOG\n",
    "model config shadow field",
)
text = replace_once(
    text,
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            logging=self.premat_logging,\n",
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            shadow_mode=self.premat_enable_shadow_mode,\n            logging=self.premat_logging,\n",
    "model validation shadow",
)
text = replace_once(
    text,
    "                enable_gpu_timing_diagnostic=config.premat_enable_gpu_timing_diagnostic,\n                logging_enabled=(\n",
    "                enable_gpu_timing_diagnostic=config.premat_enable_gpu_timing_diagnostic,\n                shadow_mode=config.premat_enable_shadow_mode,\n                logging_enabled=(\n",
    "runtime construction shadow",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# sheet/training_config.py
# ---------------------------------------------------------------------------
p = Path("sheet/training_config.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n",
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    premat_enable_shadow_mode: bool = False\n    # ^^^ THOG\n",
    "training config shadow field",
)
text = replace_once(
    text,
    '        if not isinstance(self.premat_enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n        validate_premat_configuration(\n',
    '        if not isinstance(self.premat_enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n        if not isinstance(self.premat_enable_shadow_mode, bool):\n            raise ValueError("premat_enable_shadow_mode must be bool")\n        validate_premat_configuration(\n',
    "training config shadow validation",
)
text = replace_once(
    text,
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            logging=self.premat_logging,\n",
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            shadow_mode=self.premat_enable_shadow_mode,\n            logging=self.premat_logging,\n",
    "training validate shadow arg",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# sheet/run_config.py
# ---------------------------------------------------------------------------
p = Path("sheet/run_config.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    # ^^^ THOG\n",
    "    premat_enable_gpu_timing_diagnostic: bool = False\n    premat_enable_shadow_mode: bool = False\n    # ^^^ THOG\n",
    "run config shadow field",
)
# validate bool near the existing timing validation if present; otherwise validation is delegated downstream.
needle = '        if not isinstance(self.premat_enable_gpu_timing_diagnostic, bool):\n            raise ValueError("premat_enable_gpu_timing_diagnostic must be bool")\n'
if needle in text:
    text = replace_once(
        text,
        needle,
        needle + '        if not isinstance(self.premat_enable_shadow_mode, bool):\n            raise ValueError("premat_enable_shadow_mode must be bool")\n',
        "run config shadow validation",
    )
# pass through to TrainingConfig
text = replace_once(
    text,
    "            premat_enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            premat_logging=self.premat_logging,\n",
    "            premat_enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n            premat_enable_shadow_mode=self.premat_enable_shadow_mode,\n            premat_logging=self.premat_logging,\n",
    "run to training shadow",
)
text = replace_once(
    text,
    "            or self.premat_enable_gpu_timing_diagnostic\n            or self.premat_logging != \"disabled\"\n",
    "            or self.premat_enable_gpu_timing_diagnostic\n            or self.premat_enable_shadow_mode\n            or self.premat_logging != \"disabled\"\n",
    "artifact shadow condition",
)
text = replace_once(
    text,
    '            timing_diagnostic_fragment = (\n                "GTD_" if self.premat_enable_gpu_timing_diagnostic else ""\n            )\n            premat_fragment = (\n',
    '            timing_diagnostic_fragment = (\n                "GTD_" if self.premat_enable_gpu_timing_diagnostic else ""\n            )\n            shadow_fragment = "SHADOW_" if self.premat_enable_shadow_mode else ""\n            premat_fragment = (\n',
    "artifact shadow fragment",
)
text = replace_once(
    text,
    '                f"{timing_diagnostic_fragment}"\n                f"L{self.premat_logging[0].upper()}_"\n',
    '                f"{timing_diagnostic_fragment}"\n                f"{shadow_fragment}"\n                f"L{self.premat_logging[0].upper()}_"\n',
    "artifact shadow insertion",
)
text = replace_once(
    text,
    '        if not self.premat_enable_gpu_timing_diagnostic:\n            values.pop("premat_enable_gpu_timing_diagnostic", None)\n        # ^^^ THOG\n',
    '        if not self.premat_enable_gpu_timing_diagnostic:\n            values.pop("premat_enable_gpu_timing_diagnostic", None)\n        if not self.premat_enable_shadow_mode:\n            values.pop("premat_enable_shadow_mode", None)\n        # ^^^ THOG\n',
    "persistent shadow default",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# run_thog2_owt_core.py
# ---------------------------------------------------------------------------
p = Path("run_thog2_owt_core.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    parser.add_argument(\n        "--premat_enable_gpu_timing_diagnostic",\n        type=_true_false,\n        default=False,\n        metavar="true|false",\n        help="enable PREMAT CUDA timestamp/classification diagnostics; default false",\n    )\n    # ^^^ THOG\n''',
    '''    parser.add_argument(\n        "--premat_enable_gpu_timing_diagnostic",\n        type=_true_false,\n        default=False,\n        metavar="true|false",\n        help="enable PREMAT CUDA timestamp/classification diagnostics; default false",\n    )\n    parser.add_argument(\n        "--premat_enable_shadow_mode",\n        type=_true_false,\n        default=False,\n        metavar="true|false",\n        help="run PREMAT scheduling/admission without side-stream weight materialisation; default false",\n    )\n    # ^^^ THOG\n''',
    "parser shadow option",
)
text = replace_once(
    text,
    "        premat_enable_gpu_timing_diagnostic=arguments.premat_enable_gpu_timing_diagnostic,\n        premat_logging=arguments.premat_logging,\n",
    "        premat_enable_gpu_timing_diagnostic=arguments.premat_enable_gpu_timing_diagnostic,\n        premat_enable_shadow_mode=arguments.premat_enable_shadow_mode,\n        premat_logging=arguments.premat_logging,\n",
    "config shadow mapping",
)
text = replace_once(
    text,
    '            f"premat_enable_gpu_timing_diagnostic={str(config.premat_enable_gpu_timing_diagnostic).lower()} "\n            f"premat_logging={config.premat_logging} "\n',
    '            f"premat_enable_gpu_timing_diagnostic={str(config.premat_enable_gpu_timing_diagnostic).lower()} "\n            f"premat_enable_shadow_mode={str(config.premat_enable_shadow_mode).lower()} "\n            f"premat_logging={config.premat_logging} "\n',
    "startup report shadow",
)
text = replace_once(
    text,
    '    if arguments.print_resolved_json or arguments.dry_run:\n        print(json.dumps(payload, indent=2, sort_keys=True))\n        return 0\n    # vvv THOG an actual premat run fails before model construction/first forward when CUDA is unavailable\n',
    '    if arguments.print_resolved_json or arguments.dry_run:\n        print(json.dumps(payload, indent=2, sort_keys=True))\n        return 0\n    # vvv THOG diagnostic execution switches print one conspicuous warning each before CUDA/model/data startup\n    if config.premat_enable_gpu_timing_diagnostic:\n        print(\n            "\\033[1;91mWARNING: PREMAT GPU TIMING DIAGNOSTIC ENABLED -- extra CUDA timing/classification instrumentation is active.\\033[0m",\n            file=sys.stderr,\n            flush=True,\n        )\n    if config.premat_enable_shadow_mode:\n        print(\n            "\\033[1;91mWARNING: PREMAT SHADOW MODE ENABLED -- side-stream weight materialisation is suppressed; Main Stream materialises normally.\\033[0m",\n            file=sys.stderr,\n            flush=True,\n        )\n    # ^^^ THOG\n    # vvv THOG an actual premat run fails before model construction/first forward when CUDA is unavailable\n',
    "early warning block",
)
text = replace_once(
    text,
    '"premat_enable_gpu_timing_diagnostic": training_config.premat_enable_gpu_timing_diagnostic})\n',
    '"premat_enable_gpu_timing_diagnostic": training_config.premat_enable_gpu_timing_diagnostic, "premat_enable_shadow_mode": training_config.premat_enable_shadow_mode})\n',
    "resume shadow override",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# train_OWT_core.sh
# ---------------------------------------------------------------------------
p = Path("train_OWT_core.sh")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=false\n# ^^^ THOG\n",
    "PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=false\nPREMAT_ENABLE_SHADOW_MODE=false\n# ^^^ THOG\n",
    "wrapper shadow default",
)
text = replace_once(
    text,
    "  --premat_enable_gpu_timing_diagnostic true|false=${PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC}  CUDA timestamp/classification diagnostic; default false\n",
    "  --premat_enable_gpu_timing_diagnostic true|false=${PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC}  CUDA timestamp/classification diagnostic; default false\n  --premat_enable_shadow_mode true|false=${PREMAT_ENABLE_SHADOW_MODE}  schedule/admit normally but suppress side-stream weight materialisation; default false\n",
    "wrapper shadow help",
)
text = text.replace(
    "--premat_diagnostic_layer_delay_ms|--premat_enable_gpu_timing_diagnostic|--premat_logging",
    "--premat_diagnostic_layer_delay_ms|--premat_enable_gpu_timing_diagnostic|--premat_enable_shadow_mode|--premat_logging",
    1,
)
text = replace_once(
    text,
    "        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=\"$2\" ;;\n        --premat_logging) PREMAT_LOGGING=\"$2\" ;;\n",
    "        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=\"$2\" ;;\n        --premat_enable_shadow_mode) PREMAT_ENABLE_SHADOW_MODE=\"$2\" ;;\n        --premat_logging) PREMAT_LOGGING=\"$2\" ;;\n",
    "wrapper shadow positional parse",
)
text = text.replace(
    "--premat_diagnostic_layer_delay_ms=*|--premat_enable_gpu_timing_diagnostic=*|--premat_logging=*",
    "--premat_diagnostic_layer_delay_ms=*|--premat_enable_gpu_timing_diagnostic=*|--premat_enable_shadow_mode=*|--premat_logging=*",
    1,
)
text = replace_once(
    text,
    "        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=\"$premat_value\" ;;\n        --premat_logging) PREMAT_LOGGING=\"$premat_value\" ;;\n",
    "        --premat_enable_gpu_timing_diagnostic) PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC=\"$premat_value\" ;;\n        --premat_enable_shadow_mode) PREMAT_ENABLE_SHADOW_MODE=\"$premat_value\" ;;\n        --premat_logging) PREMAT_LOGGING=\"$premat_value\" ;;\n",
    "wrapper shadow equals parse",
)
text = replace_once(
    text,
    'validate_true_false "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC" "PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC"\n',
    'validate_true_false "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC" "PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC"\nvalidate_true_false "$PREMAT_ENABLE_SHADOW_MODE" "PREMAT_ENABLE_SHADOW_MODE"\n',
    "wrapper shadow validation",
)
text = replace_once(
    text,
    '  optional_args+=(--premat_enable_gpu_timing_diagnostic "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC")\n  optional_args+=(--premat_logging "$PREMAT_LOGGING")\n',
    '  optional_args+=(--premat_enable_gpu_timing_diagnostic "$PREMAT_ENABLE_GPU_TIMING_DIAGNOSTIC")\n  optional_args+=(--premat_enable_shadow_mode "$PREMAT_ENABLE_SHADOW_MODE")\n  optional_args+=(--premat_logging "$PREMAT_LOGGING")\n',
    "wrapper shadow forward",
)
write(str(p), text)


# ---------------------------------------------------------------------------
# tests/test_premat.py
# ---------------------------------------------------------------------------
p = Path("tests/test_premat.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    enable_gpu_timing_diagnostic: bool = True,\n):\n",
    "    enable_gpu_timing_diagnostic: bool = True,\n    shadow_mode: bool = False,\n):\n",
    "test runtime shadow arg",
)
text = replace_once(
    text,
    "        enable_gpu_timing_diagnostic=enable_gpu_timing_diagnostic,\n        logging_enabled=True,\n",
    "        enable_gpu_timing_diagnostic=enable_gpu_timing_diagnostic,\n        shadow_mode=shadow_mode,\n        logging_enabled=True,\n",
    "test runtime shadow pass",
)
append = '''\n\n# vvv THOG shadow PREMAT regression coverage\ndef test_shadow_mode_schedules_without_side_stream_weight_materialisation(monkeypatch) -> None:\n    runtime, fake_cuda, calls = _runtime(\n        monkeypatch,\n        stay_below_current_peak=False,\n        enable_gpu_timing_diagnostic=False,\n        shadow_mode=True,\n    )\n    runtime.layer_start(3)\n    # Scheduling/admission ran, but the shadow side stream launched no materialiser.\n    assert calls == []\n    report = runtime.report()\n    assert report["shadow_mode"] is True\n    assert report["aggregate"]["admitted"] > 0\n\n    tensor = runtime.acquire("DOWN", 5)\n    assert isinstance(tensor, _FakeTensor)\n    assert calls == [("DOWN", 5)]\n    assert fake_cuda.main_stream.waited_events\n    report = runtime.report()\n    assert report["aggregate"]["shadow_main_materialisations"] == 1\n    assert report["aggregate"]["ordinary_deadline_materialisations"] == 1\n    runtime.consumed("DOWN", 5)\n    runtime.end()\n\n\ndef test_shadow_mode_cli_defaults_false_and_accepts_true() -> None:\n    parser = build_parser()\n    assert parser.parse_args([]).premat_enable_shadow_mode is False\n    assert parser.parse_args(["--premat_enable_shadow_mode", "true"]).premat_enable_shadow_mode is True\n    with pytest.raises(SystemExit):\n        parser.parse_args(["--premat_enable_shadow_mode", "maybe"])\n\n\ndef test_shadow_mode_requires_premat_enabled() -> None:\n    with pytest.raises(ValueError, match="requires premat enabled"):\n        validate_premat_configuration(\n            premat="disabled",\n            attention_mode="fused",\n            target_layer=1,\n            weight_matrix_target_order="r_to_l",\n            stay_below_current_peak=True,\n            stay_within_global_buffer=False,\n            gpu_memory_buffer_gb=0.0,\n            allocator_aware_admission="disabled",\n            cuda_stream_priority="normal",\n            diagnostic_layer_delay_ms=0.0,\n            enable_gpu_timing_diagnostic=False,\n            shadow_mode=True,\n            logging="disabled",\n            instra="disabled",\n        )\n# ^^^ THOG\n'''
if "test_shadow_mode_schedules_without_side_stream_weight_materialisation" in text:
    raise RuntimeError("shadow tests already present")
text += append
write(str(p), text)

print("PREMAT shadow-mode patch applied")
