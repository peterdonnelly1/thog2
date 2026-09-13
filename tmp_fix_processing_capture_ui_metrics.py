# vvv THOG temporary exact-source transformer for Processing diagnostic repairs
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if addition in text:
        raise RuntimeError(f"{path}: addition already present")
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(f"{path}: expected one append marker, found {count}")
    target.write_text(text.replace(marker, marker + addition, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# sheet/premat_processing.py: explicit capture update, exact kernel overlap,
# AD10x warp-slot alias, and metric self-description.
# ---------------------------------------------------------------------------
replace_once(
    "sheet/premat_processing.py",
    'PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ = 10_000\nPROCESSING_MIN_CAPTURE_FREQUENCY_HZ = 10\n',
    'PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ = 10_000\nPROCESSING_DEFAULT_CAPTURE_UPDATE = 1                                                                                                                       # <<< THOG preserve update-one capture unless the user requests a settled update\nPROCESSING_MIN_CAPTURE_FREQUENCY_HZ = 10\n',
)

replace_once(
    "sheet/premat_processing.py",
    '''def validate_processing_configuration(logging: str, capture_frequency_hz: int, device: str) -> None:\n    if logging not in PROCESSING_SWITCHES:\n        raise ValueError(f"premat_processing_logging must be enabled or disabled; got {logging!r}")\n    if isinstance(capture_frequency_hz, bool) or not isinstance(capture_frequency_hz, int):\n        raise ValueError("premat_processing_logging_capture_frequency_hz must be an integer")\n    if not PROCESSING_MIN_CAPTURE_FREQUENCY_HZ <= capture_frequency_hz <= PROCESSING_MAX_CAPTURE_FREQUENCY_HZ:\n        raise ValueError(\n            "premat_processing_logging_capture_frequency_hz must lie in "\n            f"[{PROCESSING_MIN_CAPTURE_FREQUENCY_HZ}, {PROCESSING_MAX_CAPTURE_FREQUENCY_HZ}]"\n        )\n    if logging == "enabled" and not str(device).startswith("cuda"):\n        raise ValueError("--premat_processing_logging enabled requires a CUDA device")\n''',
    '''# vvv THOG Processing capture-update is diagnostic execution state and independent of ordinary log cadence\ndef validate_processing_configuration(\n    logging: str,\n    capture_frequency_hz: int,\n    device: str,\n    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,\n    max_updates: Optional[int] = None,\n) -> None:\n    if logging not in PROCESSING_SWITCHES:\n        raise ValueError(f"premat_processing_logging must be enabled or disabled; got {logging!r}")\n    if isinstance(capture_frequency_hz, bool) or not isinstance(capture_frequency_hz, int):\n        raise ValueError("premat_processing_logging_capture_frequency_hz must be an integer")\n    if not PROCESSING_MIN_CAPTURE_FREQUENCY_HZ <= capture_frequency_hz <= PROCESSING_MAX_CAPTURE_FREQUENCY_HZ:\n        raise ValueError(\n            "premat_processing_logging_capture_frequency_hz must lie in "\n            f"[{PROCESSING_MIN_CAPTURE_FREQUENCY_HZ}, {PROCESSING_MAX_CAPTURE_FREQUENCY_HZ}]"\n        )\n    if isinstance(capture_update, bool) or not isinstance(capture_update, int) or capture_update < 1:\n        raise ValueError("premat_processing_logging_capture_update must be a positive integer")\n    if (\n        logging == "enabled"\n        and max_updates is not None\n        and capture_update > int(max_updates)\n    ):\n        raise ValueError(\n            "premat_processing_logging_capture_update must not exceed max_updates; "\n            f"got capture_update={capture_update}, max_updates={max_updates}"\n        )\n    if logging == "enabled" and not str(device).startswith("cuda"):\n        raise ValueError("--premat_processing_logging enabled requires a CUDA device")\n# ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''def processing_requested_from_argv(arguments: Sequence[str]) -> tuple[bool, int]:\n    logging = str(_argv_value(arguments, "--premat_processing_logging", "disabled"))\n    raw_frequency = _argv_value(\n        arguments,\n        "--premat_processing_logging_capture_frequency_hz",\n        str(PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ),\n    )\n    try:\n        frequency = int(str(raw_frequency))\n    except ValueError as error:\n        raise ValueError(\n            "--premat_processing_logging_capture_frequency_hz must be an integer"\n        ) from error\n    validate_processing_configuration(logging, frequency, "cuda")\n    return logging == "enabled", frequency\n''',
    '''# vvv THOG parse the wrapper-only capture update before torch/CUDA imports exactly like the existing Processing controls\ndef processing_requested_from_argv(arguments: Sequence[str]) -> tuple[bool, int, int]:\n    logging = str(_argv_value(arguments, "--premat_processing_logging", "disabled"))\n    raw_frequency = _argv_value(\n        arguments,\n        "--premat_processing_logging_capture_frequency_hz",\n        str(PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ),\n    )\n    raw_capture_update = _argv_value(\n        arguments,\n        "--premat_processing_logging_capture_update",\n        str(PROCESSING_DEFAULT_CAPTURE_UPDATE),\n    )\n    try:\n        frequency = int(str(raw_frequency))\n    except ValueError as error:\n        raise ValueError(\n            "--premat_processing_logging_capture_frequency_hz must be an integer"\n        ) from error\n    try:\n        capture_update = int(str(raw_capture_update))\n    except ValueError as error:\n        raise ValueError(\n            "--premat_processing_logging_capture_update must be an integer"\n        ) from error\n    validate_processing_configuration(logging, frequency, "cuda", capture_update)\n    return logging == "enabled", frequency, capture_update\n# ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    replacements = {\n        "--premat_processing_logging": "--processing_logging_internal",\n        "--premat_processing_logging_capture_frequency_hz": "--processing_logging_capture_frequency_hz_internal",\n    }\n''',
    '''    replacements = {\n        "--premat_processing_logging": "--processing_logging_internal",\n        "--premat_processing_logging_capture_frequency_hz": "--processing_logging_capture_frequency_hz_internal",\n        "--premat_processing_logging_capture_update": "--processing_logging_capture_update_internal",                                                    # <<< THOG wrapper-only exact optimizer update for the one bounded capture\n    }\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''            "premat_processing_logging_capture_frequency_hz": config.get(\n                "premat_processing_logging_capture_frequency_hz"\n            ),\n''',
    '''            "premat_processing_logging_capture_frequency_hz": config.get(\n                "premat_processing_logging_capture_frequency_hz"\n            ),\n            "premat_processing_logging_capture_update": config.get(\n                "premat_processing_logging_capture_update"\n            ),\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''def _capture_update_target(max_updates: int, log_interval: int) -> int:\n    return min(max(1, int(max_updates)), max(1, int(log_interval)))\n\n\ndef should_capture_processing_forward(\n    *,\n    enabled: bool,\n    completed_updates: int,\n    max_updates: int,\n    log_interval: int,\n    micro_step: int,\n) -> bool:\n    global _capture_done\n    if not enabled or _capture_done or int(micro_step) != 0:\n        return False\n    prospective_update = int(completed_updates) + 1\n    return prospective_update >= _capture_update_target(max_updates, log_interval)\n''',
    '''# vvv THOG capture exactly the requested optimizer update; ordinary log_interval no longer controls Nsight timing\ndef should_capture_processing_forward(\n    *,\n    enabled: bool,\n    completed_updates: int,\n    max_updates: int,\n    log_interval: int,\n    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,\n    micro_step: int,\n) -> bool:\n    global _capture_done\n    del log_interval\n    if not enabled or _capture_done or int(micro_step) != 0:\n        return False\n    if int(capture_update) > int(max_updates):\n        return False\n    prospective_update = int(completed_updates) + 1\n    return prospective_update == int(capture_update)\n# ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    log_interval: int,\n    micro_step: int,\n    device: Any,\n) -> Iterator[None]:\n''',
    '''    log_interval: int,\n    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,\n    micro_step: int,\n    device: Any,\n) -> Iterator[None]:\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''        max_updates=max_updates,\n        log_interval=log_interval,\n        micro_step=micro_step,\n''',
    '''        max_updates=max_updates,\n        log_interval=log_interval,\n        capture_update=capture_update,\n        micro_step=micro_step,\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''            "log_interval": int(log_interval),\n            "max_updates": int(max_updates),\n''',
    '''            "log_interval": int(log_interval),\n            "requested_capture_update": int(capture_update),                                                                                              # <<< THOG make the requested settled capture point explicit in bundle metadata\n            "max_updates": int(max_updates),\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''        "active_sm_unused_warp_slots_pct": ("Active SM Unused Warp Slots", "tpc__warps_inactive_sm_active"),\n''',
    '''        "active_sm_unused_warp_slots_pct": (\n            "Active SM Unused Warp Slots",\n            "Unallocated Warps in Active SM",                                                                                                             # <<< THOG AD10x General Metrics display name for unused active-SM warp slots\n            "tpc__warps_inactive_sm_active",\n        ),\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''def _metric_rows(\n    connection: sqlite3.Connection,\n    tables: set[str],\n    capture_start: int,\n    capture_end: int,\n) -> tuple[list[Dict[str, Any]], Dict[str, str]]:\n    required = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}\n    if not required.issubset(tables):\n        return [], {}\n''',
    '''def _metric_rows(\n    connection: sqlite3.Connection,\n    tables: set[str],\n    capture_start: int,\n    capture_end: int,\n) -> tuple[list[Dict[str, Any]], Dict[str, str], list[str]]:\n    required = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}\n    if not required.issubset(tables):\n        return [], {}, []\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''    if None in (metric_id, metric_name, gpu_metric_id, timestamp, value):\n        return [], {}\n''',
    '''    if None in (metric_id, metric_name, gpu_metric_id, timestamp, value):\n        return [], {}, []\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''    if not selected:\n        return [], mapping\n''',
    '''    available_metric_names = sorted(set(names.values()))                                                                                             # <<< THOG retain profiler-advertised metric names so missing aliases are diagnosable from the bundle\n    if not selected:\n        return [], mapping, available_metric_names\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''    return list(samples_by_time.values()), mapping\n''',
    '''    return list(samples_by_time.values()), mapping, available_metric_names\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''def _union_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:\n''',
    '''# vvv THOG Main/PREMAT overlap is measured only where actual CUDA kernels execute, never across semantic-span gaps\ndef _merged_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:\n    merged: list[tuple[float, float]] = []\n    for left, right in sorted((float(left), float(right)) for left, right in intervals if right > left):\n        if not merged or left > merged[-1][1]:\n            merged.append((left, right))\n        else:\n            merged[-1] = (merged[-1][0], max(merged[-1][1], right))\n    return merged\n\n\ndef _interval_duration(intervals: Iterable[tuple[float, float]]) -> float:\n    return sum(right - left for left, right in _merged_intervals(intervals))\n\n\ndef _union_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:\n''',
)
# Close the block after _union_overlap before metric mean helper.
replace_once(
    "sheet/premat_processing.py",
    '''    return total\n\n\n# vvv THOG Nsight GPU_METRICS rows are sparse by timestamp; ignore absent/non-numeric metric cells rather than coercing empty placeholders\ndef _mean_metric(samples: Sequence[Mapping[str, Any]], key: str, start_us: float, end_us: float) -> Optional[float]:\n''',
    '''    return total\n\n\ndef _kernel_union_overlap(\n    main_intervals: Iterable[tuple[float, float]],\n    premat_intervals: Iterable[tuple[float, float]],\n) -> float:\n    return sum(\n        _union_overlap(premat_intervals, left, right)\n        for left, right in _merged_intervals(main_intervals)\n    )\n# ^^^ THOG\n\n\n# vvv THOG Nsight GPU_METRICS rows are sparse by timestamp; ignore absent/non-numeric metric cells rather than coercing empty placeholders\ndef _mean_metric(samples: Sequence[Mapping[str, Any]], key: str, start_us: float, end_us: float) -> Optional[float]:\n''',
)
append_once(
    "sheet/premat_processing.py",
    '''    return sum(values) / len(values) if values else None\n# ^^^ THOG\n''',
    '''\n\n# vvv THOG summary counters sample only timestamps at which the labelled Main operation has a CUDA kernel in flight\ndef _mean_metric_intervals(\n    samples: Sequence[Mapping[str, Any]],\n    key: str,\n    intervals: Iterable[tuple[float, float]],\n) -> Optional[float]:\n    merged = _merged_intervals(intervals)\n    values: list[float] = []\n    for row in samples:\n        timestamp = float(row["time_us"])\n        if not any(left <= timestamp <= right for left, right in merged):\n            continue\n        raw_value = row.get(key)\n        if raw_value in (None, ""):\n            continue\n        try:\n            value = float(raw_value)\n        except (TypeError, ValueError):\n            continue\n        if math.isfinite(value):\n            values.append(value)\n    return sum(values) / len(values) if values else None\n# ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''        raw_samples, metric_mapping = _metric_rows(connection, tables, capture_start, capture_end)\n''',
    '''        raw_samples, metric_mapping, available_metric_names = _metric_rows(connection, tables, capture_start, capture_end)\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''        if not sample_rows:\n            warnings.append("No requested GPU Metrics samples were found in the Nsight export")\n''',
    '''        if not sample_rows:\n            warnings.append("No requested GPU Metrics samples were found in the Nsight export")\n        missing_metrics = [\n            key\n            for key in (\n                "sm_active_pct",\n                "sm_issue_pct",\n                "tensor_active_pct",\n                "active_sm_unused_warp_slots_pct",\n            )\n            if key not in metric_mapping\n        ]\n        if missing_metrics:\n            warnings.append(\n                "Requested GPU metrics not mapped: " + ", ".join(missing_metrics)\n            )\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''        start_us = min(left for left, _ in kernel_intervals)\n        end_us = max(right for _, right in kernel_intervals)\n        duration_us = end_us - start_us\n        overlap_us = _union_overlap(premat_intervals, start_us, end_us)\n        summary_rows.append({\n''',
    '''        merged_main_intervals = _merged_intervals(kernel_intervals)                                                                                   # <<< THOG remove internal Main-kernel gaps from duration and PREMAT-overlap accounting\n        start_us = min(left for left, _ in merged_main_intervals)\n        end_us = max(right for _, right in merged_main_intervals)\n        duration_us = _interval_duration(merged_main_intervals)\n        overlap_us = _kernel_union_overlap(merged_main_intervals, premat_intervals)\n        summary_rows.append({\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''            "sm_active_pct_mean": _mean_metric(sample_rows, "sm_active_pct", start_us, end_us),\n            "sm_issue_pct_mean": _mean_metric(sample_rows, "sm_issue_pct", start_us, end_us),\n            "tensor_active_pct_mean": _mean_metric(sample_rows, "tensor_active_pct", start_us, end_us),\n            "active_sm_unused_warp_slots_pct_mean": _mean_metric(\n                sample_rows, "active_sm_unused_warp_slots_pct", start_us, end_us\n            ),\n''',
    '''            "sm_active_pct_mean": _mean_metric_intervals(sample_rows, "sm_active_pct", merged_main_intervals),\n            "sm_issue_pct_mean": _mean_metric_intervals(sample_rows, "sm_issue_pct", merged_main_intervals),\n            "tensor_active_pct_mean": _mean_metric_intervals(sample_rows, "tensor_active_pct", merged_main_intervals),\n            "active_sm_unused_warp_slots_pct_mean": _mean_metric_intervals(\n                sample_rows, "active_sm_unused_warp_slots_pct", merged_main_intervals\n            ),\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''        "metric_mapping": metric_mapping,\n        "warnings": warnings,\n''',
    '''        "metric_mapping": metric_mapping,\n        "available_gpu_metric_names": available_metric_names,                                                                                             # <<< THOG expose Nsight metric vocabulary used for alias diagnosis\n        "warnings": warnings,\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    requested, frequency = processing_requested_from_argv(arguments)\n''',
    '''    requested, frequency, capture_update = processing_requested_from_argv(arguments)\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''        f"THOG2 PREMAT processing capture: Nsight Systems @ {frequency} Hz; "\n        "capturing one forward microstep",\n''',
    '''        f"THOG2 PREMAT processing capture: Nsight Systems @ {frequency} Hz; "\n        f"capturing update {capture_update}, first forward microstep",\n''',
)
replace_once(
    "sheet/premat_processing.py",
    '''    "PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ",\n''',
    '''    "PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ",\n    "PROCESSING_DEFAULT_CAPTURE_UPDATE",                                                                                                                  # <<< THOG public default for explicit capture-update diagnostics\n''',
)

# ---------------------------------------------------------------------------
# Core/run/training config plumbing for the wrapper-only capture update.
# ---------------------------------------------------------------------------
replace_once(
    "run_thog2_owt_core.py",
    '    parser.epilog = "Processing diagnostics: --premat_processing_logging enabled|disabled; --premat_processing_logging_capture_frequency_hz HZ (default 10000)"\n',
    '    parser.epilog = "Processing diagnostics: --premat_processing_logging enabled|disabled; --premat_processing_logging_capture_frequency_hz HZ (default 10000); --premat_processing_logging_capture_update UPDATE (default 1)"\n',
)
replace_once(
    "run_thog2_owt_core.py",
    '''    # vvv THOG public processing controls are consumed by the wrapper before core parsing; hidden aliases preserve the 15-option PREMAT scheduler surface\n    parser.add_argument("--processing_logging_internal", dest="premat_processing_logging", choices=("enabled", "disabled"), default="disabled", help=argparse.SUPPRESS)\n    parser.add_argument("--processing_logging_capture_frequency_hz_internal", dest="premat_processing_logging_capture_frequency_hz", type=int, default=10000, help=argparse.SUPPRESS)\n    # ^^^ THOG\n''',
    '''    # vvv THOG public processing controls are consumed by the wrapper before core parsing; hidden aliases keep them out of the PREMAT scheduler surface\n    parser.add_argument("--processing_logging_internal", dest="premat_processing_logging", choices=("enabled", "disabled"), default="disabled", help=argparse.SUPPRESS)\n    parser.add_argument("--processing_logging_capture_frequency_hz_internal", dest="premat_processing_logging_capture_frequency_hz", type=int, default=10000, help=argparse.SUPPRESS)\n    parser.add_argument("--processing_logging_capture_update_internal", dest="premat_processing_logging_capture_update", type=int, default=1, help=argparse.SUPPRESS)                    # <<< THOG exact optimizer update for bounded Nsight capture\n    # ^^^ THOG\n''',
)
replace_once(
    "run_thog2_owt_core.py",
    '''        premat_processing_logging=arguments.premat_processing_logging,\n        premat_processing_logging_capture_frequency_hz=arguments.premat_processing_logging_capture_frequency_hz,\n''',
    '''        premat_processing_logging=arguments.premat_processing_logging,\n        premat_processing_logging_capture_frequency_hz=arguments.premat_processing_logging_capture_frequency_hz,\n        premat_processing_logging_capture_update=arguments.premat_processing_logging_capture_update,                                                       # <<< THOG pass explicit Processing capture update into resolved run configuration\n''',
)

replace_once(
    "sheet/run_config.py",
    '''    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n''',
    '''    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n    premat_processing_logging_capture_update: int = 1                                                                                                      # <<< THOG exact optimizer update selected for bounded Processing capture\n''',
)
replace_once(
    "sheet/run_config.py",
    '''        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n        )\n''',
    '''        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n            self.premat_processing_logging_capture_update,\n            self.max_iters,\n        )\n''',
)
replace_once(
    "sheet/run_config.py",
    '''            processing_fragment = (\n                f"_PROC{self.premat_processing_logging_capture_frequency_hz}"\n                if self.premat_processing_logging == "enabled"\n                else ""\n            )\n''',
    '''            processing_fragment = (\n                f"_PROC{self.premat_processing_logging_capture_frequency_hz}"\n                + (f"U{self.premat_processing_logging_capture_update}" if self.premat_processing_logging_capture_update != 1 else "")\n                if self.premat_processing_logging == "enabled"\n                else ""\n            )\n''',
)
replace_once(
    "sheet/run_config.py",
    '''            premat_processing_logging=self.premat_processing_logging,\n            premat_processing_logging_capture_frequency_hz=self.premat_processing_logging_capture_frequency_hz,\n''',
    '''            premat_processing_logging=self.premat_processing_logging,\n            premat_processing_logging_capture_frequency_hz=self.premat_processing_logging_capture_frequency_hz,\n            premat_processing_logging_capture_update=self.premat_processing_logging_capture_update,                                                       # <<< THOG propagate requested Processing capture update into TrainingConfig\n''',
)

replace_once(
    "sheet/training_config.py",
    '''EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic", "premat_processing_logging", "premat_processing_logging_capture_frequency_hz"}                         # <<< THOG processing capture is diagnostic execution state, never checkpoint model identity\n''',
    '''EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic", "premat_processing_logging", "premat_processing_logging_capture_frequency_hz", "premat_processing_logging_capture_update"}                         # <<< THOG processing capture is diagnostic execution state, never checkpoint model identity\n''',
)
replace_once(
    "sheet/training_config.py",
    '''    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n''',
    '''    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n    premat_processing_logging_capture_update: int = 1                                                                                                      # <<< THOG exact optimizer update selected for bounded Processing capture\n''',
)
replace_once(
    "sheet/training_config.py",
    '''        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n        )\n''',
    '''        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n            self.premat_processing_logging_capture_update,\n            self.max_updates,\n        )\n''',
)

replace_once(
    "sheet/trainer_step.py",
    '''                        max_updates=self.config.max_updates,\n                        log_interval=self.config.log_interval,\n                        micro_step=micro_step,\n''',
    '''                        max_updates=self.config.max_updates,\n                        log_interval=self.config.log_interval,\n                        capture_update=self.config.premat_processing_logging_capture_update,                                                               # <<< THOG decouple bounded Nsight capture timing from ordinary log cadence\n                        micro_step=micro_step,\n''',
)

# ---------------------------------------------------------------------------
# Local dashboard: live tok/s is available during the run; Nsight cards appear
# only after normalization has completed.
# ---------------------------------------------------------------------------
replace_once(
    "run_thog2_local_dashboard.py",
    '''    # vvv THOG Nsight-normalized processing data lives beside charts.sqlite3 and is immutable after capture\n    def processing(self) -> Dict[str, Any]:\n        path = self.database_path.parent / "processing" / "processing_data.json"\n        if not path.is_file():\n            return {"available": False, "revision": None, "data": None}\n        stat_result = path.stat()\n        # vvv THOG combine immutable Nsight evidence with the lightweight optimizer-progress throughput history\n        throughput = self.reader.processing_throughput()\n        data = json.loads(path.read_text())\n        data["throughput"] = list(throughput)\n        throughput_tail = throughput[-1] if throughput else None\n        throughput_revision = (\n            "0"\n            if throughput_tail is None\n            else f"{len(throughput)}:{throughput_tail['optimizer_update']}:{throughput_tail['tokens_per_second']:.12g}"\n        )\n        # ^^^ THOG\n        return {\n            "available": True,\n            "revision": f"{stat_result.st_mtime_ns}:{stat_result.st_size}:{throughput_revision}",\n            "data": data,\n        }\n    # ^^^ THOG\n''',
    '''    # vvv THOG live Processing exposes tok/s immediately; immutable Nsight evidence joins only after parent normalization completes\n    def processing(self) -> Dict[str, Any]:\n        path = self.database_path.parent / "processing" / "processing_data.json"\n        throughput = self.reader.processing_throughput()\n        throughput_tail = throughput[-1] if throughput else None\n        throughput_revision = (\n            "0"\n            if throughput_tail is None\n            else f"{len(throughput)}:{throughput_tail['optimizer_update']}:{throughput_tail['tokens_per_second']:.12g}"\n        )\n        trace_available = path.is_file()\n        if not trace_available and not throughput:\n            return {"available": False, "trace_available": False, "revision": None, "data": None}\n        if trace_available:\n            stat_result = path.stat()\n            data = json.loads(path.read_text())\n            trace_revision = f"{stat_result.st_mtime_ns}:{stat_result.st_size}"\n        else:\n            data = {"metadata": None, "samples": [], "intervals": [], "summary": []}\n            trace_revision = "pending"\n        data["throughput"] = list(throughput)\n        return {\n            "available": True,\n            "trace_available": trace_available,\n            "revision": f"{trace_revision}:{throughput_revision}",\n            "data": data,\n        }\n    # ^^^ THOG\n''',
)

# Standard plot mounts/shells.
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '<div class="processing-plot" id="processing_timeline_plot"></div>',
    '<div class="plot-shell processing-plot-shell"><div class="plot-mount processing-plot" id="processing_timeline_plot"></div></div>',
)
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '<div class="processing-plot" id="processing_throughput_plot"></div>',
    '<div class="plot-shell processing-plot-shell"><div class="plot-mount processing-plot" id="processing_throughput_plot"></div></div>',
)
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '<div class="processing-plot processing-contention-plot" id="processing_contention_plot"></div>',
    '<div class="plot-shell processing-plot-shell"><div class="plot-mount processing-plot processing-contention-plot" id="processing_contention_plot"></div></div>',
)
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '<article class="processing-card chart-card processing-timeline-card" data-chart="processing_timeline">',
    '<article class="processing-card chart-card processing-timeline-card" id="processing_timeline_card" data-chart="processing_timeline" hidden>',
)
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '<article class="processing-card chart-card processing-contention-card" data-chart="processing_contention">',
    '<article class="processing-card chart-card processing-contention-card" id="processing_contention_card" data-chart="processing_contention" hidden>',
)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.css",
    '.processing-plot { width: 100%; min-width: 0; min-height: 140px; height: auto; flex: 1 1 auto; overflow: hidden; }\n',
    '.processing-plot-shell { min-width: 0; min-height: 140px; flex: 1 1 auto; }                                                      /* THOG standard INSTRA plot shell for Processing */\n.processing-plot { width: 100%; min-width: 0; min-height: 140px; height: 100%; overflow: hidden; }\n',
)

# Premat tab no longer owns Processing visibility.
replace_once(
    "sheet/local_dashboard_assets/dashboard_premat.js",
    '''  document.querySelectorAll('[data-chart-group]:not(#premat_chart_group)').forEach(group => {\n    group.hidden = Boolean(premat_selected);\n  });\n''',
    '''  document.querySelectorAll('[data-chart-group]:not(#premat_chart_group):not(#processing_chart_group)').forEach(group => {\n    group.hidden = Boolean(premat_selected);\n  });\n  if (typeof window.processing_apply_detail_tab === "function") {\n    window.processing_apply_detail_tab(!premat_selected);                                                                                                  // THOG Processing alone owns its availability while Premat owns only tab selection\n  } else if (premat_selected) {\n    by_id("processing_chart_group").hidden = true;\n  }\n''',
)

# Processing JS: local standard first-render lifecycle + split live/trace visibility.
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''const processing_view = {\n  run_id: null,\n  revision: null,\n  timer: null,\n};\n''',
    '''const processing_view = {\n  run_id: null,\n  revision: null,\n  timer: null,\n  available: false,\n  trace_available: false,\n  charts_tab_visible: true,\n};\n\n// vvv THOG Processing cards use the ordinary INSTRA plot-mount readiness contract instead of direct Plotly.react on empty divs\nasync function processing_plot(mount_id, traces, layout) {\n  const mount = by_id(mount_id);\n  if (!mount) return;\n  const resolved_layout = {...layout, autosize: true};\n  if (mount.dataset.plotReady === "true") {\n    await Plotly.react(mount, traces, resolved_layout, plot_config);\n  } else {\n    mount.replaceChildren();\n    await Plotly.newPlot(mount, traces, resolved_layout, plot_config);\n    mount.dataset.plotReady = "true";\n  }\n}\n\nfunction processing_sync_visibility() {\n  const group = by_id("processing_chart_group");\n  if (!group) return;\n  group.hidden = !(processing_view.charts_tab_visible && processing_view.available);\n  const timeline = by_id("processing_timeline_card");\n  const contention = by_id("processing_contention_card");\n  if (timeline) timeline.hidden = !processing_view.trace_available;\n  if (contention) contention.hidden = !processing_view.trace_available;\n}\n\nwindow.processing_apply_detail_tab = charts_selected => {\n  processing_view.charts_tab_visible = Boolean(charts_selected);\n  processing_sync_visibility();\n};\n// ^^^ THOG\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    'function processing_render_timeline(payload) {',
    'async function processing_render_timeline(payload) {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '  Plotly.react("processing_timeline_plot", traces, {',
    '  await processing_plot("processing_timeline_plot", traces, {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    'function processing_render_throughput(payload) {',
    'async function processing_render_throughput(payload) {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '  Plotly.react("processing_throughput_plot", traces, {',
    '  await processing_plot("processing_throughput_plot", traces, {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    'function processing_render_contention(payload) {',
    'async function processing_render_contention(payload) {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '  Plotly.react("processing_contention_plot", traces, {',
    '  await processing_plot("processing_contention_plot", traces, {',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''    xaxis: {title: "Main operation overlapped by PREMAT (%)", range: [0, 100]},\n    yaxis: {title: "Main operation duration (ms)"},\n''',
    '''    xaxis: {title: "Main GPU kernel time overlapped by PREMAT (%)", range: [0, 100]},\n    yaxis: {title: "Main GPU kernel duration (ms)"},\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''      hovertemplate: `${family} %{text}<br>PREMAT overlap %{x:.1f}%<br>Main duration %{y:.4f} ms<extra></extra>`,\n''',
    '''      hovertemplate: `${family} %{text}<br>PREMAT overlap of Main GPU kernels %{x:.1f}%<br>Main GPU kernel duration %{y:.4f} ms<extra></extra>`,\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''function processing_render(payload) {\n  const group = by_id("processing_chart_group");\n  group.hidden = false;\n  // vvv THOG static Processing cards participate in the same saved-size contract as ordinary INSTRA charts\n  if (typeof apply_saved_panel_sizes === "function") apply_saved_panel_sizes();\n  // ^^^ THOG\n  const capture = payload.metadata?.capture || {};\n  by_id("processing_step").textContent = String(capture.optimizer_update ?? "—");\n  const warning_count = (payload.metadata?.warnings || []).length;\n  by_id("processing_status").textContent = `${Number(payload.metadata?.capture_frequency_hz || 0).toLocaleString()} Hz · ${Number(payload.metadata?.capture_duration_ms || 0).toFixed(2)} ms capture${warning_count ? ` · ${warning_count} warning${warning_count === 1 ? "" : "s"}` : ""}`;\n  processing_set_downloads(payload.metadata);\n  processing_render_timeline(payload);\n  processing_render_throughput(payload);\n  processing_render_contention(payload);\n  processing_render_summary(payload);\n  // vvv THOG Plotly must re-measure after the hidden Processing group becomes visible and after any restored panel geometry is applied\n  requestAnimationFrame(() => {\n    for (const chart_name of ["processing_timeline", "processing_throughput", "processing_contention"]) {\n      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);\n      if (card && typeof resize_plot_in_card === "function") resize_plot_in_card(card);\n    }\n  });\n  // ^^^ THOG\n}\n''',
    '''// vvv THOG tok/s is live; Nsight cards are not instantiated until normalized trace data exists after the profiled child completes\nasync function processing_render(payload, trace_available) {\n  processing_view.available = true;\n  processing_view.trace_available = Boolean(trace_available);\n  processing_sync_visibility();\n  if (typeof apply_saved_panel_sizes === "function") apply_saved_panel_sizes();\n  const capture = payload.metadata?.capture || {};\n  const throughput = payload.throughput || [];\n  const latest_throughput = throughput.length ? throughput[throughput.length - 1] : null;\n  by_id("processing_step").textContent = String(capture.optimizer_update ?? latest_throughput?.optimizer_update ?? "—");\n  if (trace_available) {\n    const warning_count = (payload.metadata?.warnings || []).length;\n    by_id("processing_status").textContent = `${Number(payload.metadata?.capture_frequency_hz || 0).toLocaleString()} Hz · ${Number(payload.metadata?.capture_duration_ms || 0).toFixed(2)} ms capture${warning_count ? ` · ${warning_count} warning${warning_count === 1 ? "" : "s"}` : ""}`;\n    processing_set_downloads(payload.metadata);\n    await processing_render_timeline(payload);\n    await processing_render_contention(payload);\n    processing_render_summary(payload);\n  } else {\n    by_id("processing_status").textContent = "Live throughput · Nsight charts appear after run completion";\n    processing_set_downloads(null);\n  }\n  await processing_render_throughput(payload);\n  processing_sync_visibility();\n  requestAnimationFrame(() => {\n    const chart_names = trace_available\n      ? ["processing_timeline", "processing_throughput", "processing_contention"]\n      : ["processing_throughput"];\n    for (const chart_name of chart_names) {\n      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);\n      if (card && typeof resize_plot_in_card === "function") resize_plot_in_card(card);\n    }\n  });\n}\n// ^^^ THOG\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''  if (!run_id) {\n    by_id("processing_chart_group").hidden = true;\n    processing_view.run_id = null;\n    processing_view.revision = null;\n    return;\n  }\n''',
    '''  if (!run_id) {\n    processing_view.available = false;\n    processing_view.trace_available = false;\n    processing_sync_visibility();\n    processing_view.run_id = null;\n    processing_view.revision = null;\n    return;\n  }\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''    if (!payload.available) {\n      by_id("processing_chart_group").hidden = true;\n      processing_view.run_id = run_id;\n      processing_view.revision = null;\n      return;\n    }\n''',
    '''    if (!payload.available) {\n      processing_view.available = false;\n      processing_view.trace_available = false;\n      processing_sync_visibility();\n      processing_view.run_id = run_id;\n      processing_view.revision = null;\n      return;\n    }\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''    processing_view.run_id = run_id;\n    processing_view.revision = payload.revision;\n    processing_render(payload.data);\n''',
    '''    processing_view.run_id = run_id;\n    processing_view.revision = payload.revision;\n    await processing_render(payload.data, payload.trace_available === true);\n''',
)

# ---------------------------------------------------------------------------
# Tests: public capture update, AD10x alias, true kernel overlap, plot mounts,
# and ownership of Processing visibility.
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_premat_processing.py",
    '''    processing_requested_from_argv,\n    register_processing_handoff,\n''',
    '''    processing_requested_from_argv,\n    register_processing_handoff,\n    should_capture_processing_forward,\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''    enabled, frequency = processing_requested_from_argv([\n        "--premat_processing_logging", "enabled",\n        "--premat_processing_logging_capture_frequency_hz", "12345",\n    ])\n    assert enabled is True\n    assert frequency == 12345\n''',
    '''    enabled, frequency, capture_update = processing_requested_from_argv([\n        "--premat_processing_logging", "enabled",\n        "--premat_processing_logging_capture_frequency_hz", "12345",\n        "--premat_processing_logging_capture_update", "5",\n    ])\n    assert enabled is True\n    assert frequency == 12345\n    assert capture_update == 5\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''        "--premat_processing_logging_capture_frequency_hz=12345",\n        "--model-type", "sheet",\n    ]) == [\n        "--processing_logging_internal", "enabled",\n        "--processing_logging_capture_frequency_hz_internal=12345",\n        "--model-type", "sheet",\n''',
    '''        "--premat_processing_logging_capture_frequency_hz=12345",\n        "--premat_processing_logging_capture_update", "5",\n        "--model-type", "sheet",\n    ]) == [\n        "--processing_logging_internal", "enabled",\n        "--processing_logging_capture_frequency_hz_internal=12345",\n        "--processing_logging_capture_update_internal", "5",\n        "--model-type", "sheet",\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''    with pytest.raises(ValueError, match="capture_frequency_hz"):\n        validate_processing_configuration("enabled", 9, "cuda")\n''',
    '''    with pytest.raises(ValueError, match="capture_frequency_hz"):\n        validate_processing_configuration("enabled", 9, "cuda")\n    with pytest.raises(ValueError, match="capture_update"):\n        validate_processing_configuration("enabled", 10000, "cuda", 0)\n    with pytest.raises(ValueError, match="must not exceed"):\n        validate_processing_configuration("enabled", 10000, "cuda", 6, 5)\n''',
)
append_once(
    "tests/test_premat_processing.py",
    '''def test_processing_handoff_records_target_matrix(tmp_path: Path, monkeypatch) -> None:\n''',
    '''\n\ndef test_processing_capture_update_is_exact_and_log_interval_independent(monkeypatch) -> None:\n    import sheet.premat_processing as processing\n    monkeypatch.setattr(processing, "_capture_done", False)\n    assert not should_capture_processing_forward(\n        enabled=True, completed_updates=3, max_updates=5, log_interval=1, capture_update=5, micro_step=0\n    )\n    assert should_capture_processing_forward(\n        enabled=True, completed_updates=4, max_updates=5, log_interval=99, capture_update=5, micro_step=0\n    )\n    assert not should_capture_processing_forward(\n        enabled=True, completed_updates=4, max_updates=5, log_interval=1, capture_update=5, micro_step=1\n    )\n\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''        assert f'data-maximize="{chart_name}"' in html\n    assert html.count('class="panel-resizer panel-resizer-corner"') >= 3\n''',
    '''        assert f'data-maximize="{chart_name}"' in html\n        assert f'id="{chart_name}_plot"' in html\n    assert html.count('class="plot-mount processing-plot') >= 3\n    assert html.count('class="panel-resizer panel-resizer-corner"') >= 3\n''',
)
append_once(
    "tests/test_premat_processing.py",
    '''def test_processing_throughput_round_trips_through_local_store(tmp_path: Path) -> None:\n''',
    '''\n\ndef test_processing_visibility_is_owned_by_processing_view() -> None:\n    premat_js = Path("sheet/local_dashboard_assets/dashboard_premat.js").read_text(encoding="utf-8")\n    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")\n    assert ":not(#processing_chart_group)" in premat_js\n    assert "processing_apply_detail_tab" in premat_js\n    assert "trace_available" in processing_js\n    assert "Nsight charts appear after run completion" in processing_js\n    assert "Plotly.newPlot" in processing_js\n    assert 'dataset.plotReady = "true"' in processing_js\n\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''                (4, "Active SM Unused Warp Slots %"),\n''',
    '''                (4, "Unallocated Warps in Active SMs [Throughput %]"),\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''    assert payload["samples"][0]["sm_active_pct"] == 80.0\n''',
    '''    assert payload["samples"][0]["sm_active_pct"] == 80.0\n    assert payload["samples"][0]["active_sm_unused_warp_slots_pct"] == 20.0\n    assert "active_sm_unused_warp_slots_pct" in payload["metadata"]["metric_mapping"]\n    assert "Unallocated Warps in Active SMs [Throughput %]" in payload["metadata"]["available_gpu_metric_names"]\n''',
)

# Add a focused gap-overlap regression that would fail under the old semantic-span calculation.
append_once(
    "tests/test_premat_processing.py",
    '# ^^^ THOG\n',
    '''\n\n# vvv THOG PREMAT in a gap between Main kernels is not simultaneous Main/PREMAT execution\ndef test_processing_overlap_excludes_internal_main_kernel_gaps(tmp_path: Path) -> None:\n    database = tmp_path / "gap.sqlite"\n    output = tmp_path / "processing_gap"\n    with sqlite3.connect(database) as connection:\n        connection.executescript(\n            """\n            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);\n            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);\n            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);\n            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER, correlationId INTEGER, shortName INTEGER);\n            CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);\n            CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL);\n            """\n        )\n        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "main_a"), (2, "main_b"), (3, "premat_gap")])\n        connection.executemany(\n            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",\n            [\n                (1000, 10000, 7, "THOG2_PREMAT_PROCESSING_CAPTURE"),\n                (1500, 7500, 7, "THOG2_PROCESSING|owner=MAIN|operation=consume|family=QKV|layer=0"),\n                (3500, 5500, 7, "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=O|layer=1"),\n            ],\n        )\n        connection.executemany(\n            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",\n            [(1600, 1650, 7, 11), (5600, 5650, 7, 12), (3600, 3650, 7, 13)],\n        )\n        connection.executemany(\n            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",\n            [\n                (2000, 3000, 3, 11, 1),\n                (6000, 7000, 3, 12, 2),\n                (4000, 5000, 9, 13, 3),\n            ],\n        )\n        connection.commit()\n    payload = normalize_nsys_sqlite(database, output, capture_frequency_hz=10000)\n    summary = payload["summary"][0]\n    assert summary["duration_ms"] == pytest.approx(0.002)\n    assert summary["premat_overlap_ms"] == pytest.approx(0.0)\n    assert summary["premat_overlap_pct"] == pytest.approx(0.0)\n# ^^^ THOG\n''',
)

# Implementation log entry.
append_once(
    "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md",
    '\n',
    '- Processing diagnostics now have an explicit wrapper-only `--premat_processing_logging_capture_update` (default 1), decoupling the one bounded Nsight forward capture from ordinary logging cadence. INSTRA shows live tok/s while a run is active and defers the two Nsight-dependent cards until normalized trace data exists; Processing cards now use standard plot mounts/first-render lifecycle. The normalizer recognizes the AD10x `Unallocated Warps in Active SMs` metric alias, records available metric names, and computes Main/PREMAT overlap only across actual Main CUDA-kernel unions rather than semantic-span gaps.\n',
)

print("Processing capture/UI/metric repairs applied")
# ^^^ THOG
