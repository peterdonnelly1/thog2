# vvv THOG CPU preparation controls, capability validation and dormant-default identity
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Mapping

CPU_DEFAULTS = {
    "premat_materialisation_device": "gpu",
    "premat_cpu_preparation": "eager",
    "premat_cpu_layer_batch_size": "single_layer",
    "premat_cpu_workers": 1,
    "premat_cpu_threads_per_worker": 0,
    "premat_cpu_transfer_timing": "as_the_code_flies",
    "premat_cpu_transfer_lead_ms": 0.0,
    "premat_cpu_staging_limit_mb": 0.0,
    "premat_cpu_checkpoint_replay": "disabled",
}
CPU_CHOICES = {
    "premat_materialisation_device": ("gpu", "cpu_and_gpu"),
    "premat_cpu_preparation": ("eager", "scheduled", "demand_driven"),
    "premat_cpu_transfer_timing": ("as_the_code_flies", "previous_gemm_leading_edge", "as_soon_as_ready", "demand_driven", "predicted_gemm_start"),
    "premat_cpu_checkpoint_replay": ("enabled", "disabled"),
}


def cpu_config_dict(config) -> dict:
    values = config if isinstance(config, Mapping) else vars(config)
    return {name: values.get(name, default) for name, default in CPU_DEFAULTS.items()}


def normalize_cpu_batch(value):
    if value in ("single_layer", "all_layers"):
        return value
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("premat_cpu_layer_batch_size requires single_layer, all_layers or integer >= 2")
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError("premat_cpu_layer_batch_size requires single_layer, all_layers or integer >= 2") from error
    if number < 2 or str(number) != str(value).strip():
        raise ValueError("premat_cpu_layer_batch_size requires single_layer, all_layers or integer >= 2")
    return str(number)


def resolve_cpu_threads(workers: int, requested: int) -> tuple[int, int]:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    budget = max(1, available - 1)
    if workers > budget or (requested and workers * requested > budget):
        raise ValueError(f"CPU worker/thread request exceeds affinity budget {budget}; reserve a core for training")
    return max(1, budget // workers) if requested == 0 else requested, budget


def validate_cpu_configuration(config) -> None:
    values = config if isinstance(config, Mapping) else vars(config)
    controls = cpu_config_dict(values)
    for name, choices in CPU_CHOICES.items():
        if controls[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    normalize_cpu_batch(controls["premat_cpu_layer_batch_size"])
    for name, minimum in (("premat_cpu_workers", 1), ("premat_cpu_threads_per_worker", 0)):
        number = controls[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise ValueError(f"{name} requires integer >= {minimum}")
    for name in ("premat_cpu_transfer_lead_ms", "premat_cpu_staging_limit_mb"):
        number = controls[name]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} requires a finite non-negative number")
    if controls["premat_materialisation_device"] == "gpu":
        return
    if values.get("premat", "disabled") != "enabled":
        raise ValueError("cpu_and_gpu requires premat enabled")
    if values.get("model_type", "sheet") not in ("sheet", "thog2_sheet") or values.get("geometry_preset", "DEPTH") != "DEPTH":
        raise ValueError("cpu_and_gpu requires standalone DEPTH geometry")
    if values.get("plastic__enabled", False) or values.get("hyperblock__enabled", False) or values.get("layer_dropout", 0) or values.get("layer_dropout_probability", 0):
        raise ValueError("cpu_and_gpu does not support PLASTIC, HYPERBLOCK or sparse layer dropout")
    if not str(values.get("device", "cuda")).startswith("cuda") or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("cpu_and_gpu requires a single CUDA device and single training process")
    if values.get("dtype", "float32") not in ("float32", "float16", "bfloat16"):
        raise ValueError("cpu_and_gpu supports FP32, FP16 and BF16 only")
    if values.get("compile", False) or os.environ.get("THOG2_TORCH_COMPILE", "off").lower() not in ("off", "0", "false", "disabled", "no"):
        raise ValueError("cpu_and_gpu requires eager, uncompiled DEPTH execution")
    timing = controls["premat_cpu_transfer_timing"]
    if timing == "previous_gemm_leading_edge" and values.get("premat_attention_mode", "fused") != "fused":
        raise ValueError("previous_gemm_leading_edge requires fused attention")
    if timing != "predicted_gemm_start" and controls["premat_cpu_transfer_lead_ms"] != 0:
        raise ValueError("premat_cpu_transfer_lead_ms requires predicted_gemm_start")
    resolve_cpu_threads(controls["premat_cpu_workers"], controls["premat_cpu_threads_per_worker"])


def add_cpu_arguments(parser) -> None:
    for name, default in CPU_DEFAULTS.items():
        kwargs = {"default": default}
        if name in CPU_CHOICES:
            kwargs["choices"] = CPU_CHOICES[name]
        elif isinstance(default, int):
            kwargs["type"] = int
        elif isinstance(default, float):
            kwargs["type"] = float
        else:
            kwargs["type"] = normalize_cpu_batch
        parser.add_argument("--" + name, **kwargs)


def validate_cpu_arguments(args, argv) -> None:
    explicit = {arg.split("=", 1)[0][2:] for arg in argv if arg.startswith("--")}
    controls = cpu_config_dict(args)
    if controls["premat_materialisation_device"] == "gpu" and any(name in explicit for name in CPU_DEFAULTS if name.startswith("premat_cpu_")):
        raise ValueError("gpu mode rejects explicitly supplied CPU-only controls")
    if "premat_cpu_transfer_lead_ms" in explicit and controls["premat_cpu_transfer_timing"] != "predicted_gemm_start":
        raise ValueError("explicit premat_cpu_transfer_lead_ms requires predicted_gemm_start")
    if controls["premat_materialisation_device"] == "cpu_and_gpu" and "premat_timing" in explicit and args.premat_timing != controls["premat_cpu_transfer_timing"]:
        raise ValueError("premat_timing is inactive and conflicts with premat_cpu_transfer_timing")
    validate_cpu_configuration(args)


def strip_inactive_cpu(values: dict) -> dict:
    if values.get("premat_materialisation_device", "gpu") == "gpu":
        for name in CPU_DEFAULTS:
            values.pop(name, None)
    return values


def cpu_identity(config) -> str:
    controls = cpu_config_dict(config)
    if controls["premat_materialisation_device"] == "gpu":
        return ""
    if controls["premat_cpu_transfer_timing"] != "predicted_gemm_start":
        controls.pop("premat_cpu_transfer_lead_ms")
    return "cpu_" + hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()[:12]
# ^^^ THOG
