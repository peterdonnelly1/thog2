# vvv THOG
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .training_config import TrainingConfig


STAGE6_PROTOCOL_VERSION = "stage6_pilot_v1"
STAGE6_RUN_ORDER = ("dense", "q64", "q128", "q256")
STAGE6_CLASSIFICATIONS = (
    "viable_for_further_study",
    "viable_only_at_weak_compression",
    "inconclusive",
    "not_viable_under_tested_design",
)


@dataclass(frozen=True)
class PilotBudget:
    n_layer: int = 72
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 256
    depth_order: int = 16
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_updates: int = 250
    eval_interval: int = 25
    eval_batches: int = 20
    checkpoint_interval: int = 0
    checkpoint_segment_size: int = 4
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    warmup_updates: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    model_seed: int = 6101
    data_seed: int = 6102
    log_interval: int = 25

    @property
    def tokens_per_update(self) -> int:
        return (
            self.batch_size
            * self.gradient_accumulation_steps
            * self.block_size
        )

    @property
    def consumed_tokens(self) -> int:
        return self.tokens_per_update * self.max_updates


@dataclass(frozen=True)
class PilotRunSpec:
    run_id: str
    model_type: str
    base_row_order: int
    checkpoint_segment_size: int

    @property
    def row_order_label(self) -> str:
        return "dense" if self.model_type == "dense" else f"q{self.base_row_order}"


def stage6_run_specs(budget: PilotBudget) -> Tuple[PilotRunSpec, ...]:
    common = f"l{budget.n_layer}_h{budget.n_head}_d{budget.n_embd}_c{budget.block_size}"
    return (
        PilotRunSpec(f"dense_{common}", "dense", 1, 0),
        PilotRunSpec(f"sheet_q64_{common}", "thog2_sheet", 64, budget.checkpoint_segment_size),
        PilotRunSpec(f"sheet_q128_{common}", "thog2_sheet", 128, budget.checkpoint_segment_size),
        PilotRunSpec(f"sheet_q256_{common}", "thog2_sheet", 256, budget.checkpoint_segment_size),
    )


def dense_parameter_count(
    *,
    n_layer: int,
    n_embd: int,
    block_size: int,
    vocab_size: int,
    bias: bool = True,
) -> int:
    repeated_matrices = 12 * n_embd * n_embd
    repeated_vectors = (13 if bias else 2) * n_embd
    final_layer_norm = (2 if bias else 1) * n_embd
    return (
        vocab_size * n_embd
        + block_size * n_embd
        + n_layer * (repeated_matrices + repeated_vectors)
        + final_layer_norm
    )


def adamw_fp32_minimum_training_bytes(parameter_count: int) -> int:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return parameter_count * 16


def principal_dense_feasibility(
    *,
    vocab_size: int,
    device_total_bytes: Optional[int],
    reserve_fraction: float = 0.20,
) -> Dict[str, Any]:
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")
    parameter_count = dense_parameter_count(
        n_layer=144,
        n_embd=768,
        block_size=256,
        vocab_size=vocab_size,
    )
    minimum_bytes = adamw_fp32_minimum_training_bytes(parameter_count)
    usable_bytes = None
    feasible = None
    if device_total_bytes is not None:
        usable_bytes = int(device_total_bytes * (1.0 - reserve_fraction))
        feasible = minimum_bytes <= usable_bytes
    return {
        "logical_geometry": {
            "n_layer": 144,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 256,
        },
        "dense_parameters": parameter_count,
        "minimum_training_state_bytes": minimum_bytes,
        "minimum_training_state_gib": minimum_bytes / (1024 ** 3),
        "device_total_bytes": device_total_bytes,
        "usable_bytes_after_reserve": usable_bytes,
        "reserve_fraction": reserve_fraction,
        "feasible_under_current_fp32_adamw_path": feasible,
        "excludes": (
            "activations",
            "temporary tensors",
            "CUDA context",
            "allocator fragmentation",
            "evaluation logits",
        ),
    }


def build_training_config(
    spec: PilotRunSpec,
    budget: PilotBudget,
    *,
    vocab_size: int,
    device: str,
    dtype: str,
    out_dir: Path,
) -> TrainingConfig:
    return TrainingConfig(
        model_type=spec.model_type,
        block_size=budget.block_size,
        vocab_size=vocab_size,
        n_layer=budget.n_layer,
        n_head=budget.n_head,
        n_embd=budget.n_embd,
        dropout=0.0,
        bias=True,
        depth_order=budget.depth_order,
        base_row_order=spec.base_row_order,
        checkpoint_segment_size=spec.checkpoint_segment_size,
        batch_size=budget.batch_size,
        gradient_accumulation_steps=budget.gradient_accumulation_steps,
        max_updates=budget.max_updates,
        learning_rate=budget.learning_rate,
        min_learning_rate=budget.min_learning_rate,
        warmup_updates=budget.warmup_updates,
        decay_updates=budget.max_updates,
        decay_learning_rate=True,
        weight_decay=budget.weight_decay,
        beta1=budget.beta1,
        beta2=budget.beta2,
        grad_clip=budget.grad_clip,
        eval_interval=budget.eval_interval,
        eval_batches=budget.eval_batches,
        checkpoint_interval=budget.checkpoint_interval,
        log_interval=budget.log_interval,
        model_seed=budget.model_seed,
        data_seed=budget.data_seed,
        device=device,
        dtype=dtype,
        out_dir=str(out_dir),
    )


def logical_control_signature(config: TrainingConfig) -> Dict[str, Any]:
    return {
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "bias": config.bias,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_updates": config.max_updates,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "warmup_updates": config.warmup_updates,
        "decay_updates": config.decay_updates,
        "decay_learning_rate": config.decay_learning_rate,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "grad_clip": config.grad_clip,
        "eval_interval": config.eval_interval,
        "eval_batches": config.eval_batches,
        "model_seed": config.model_seed,
        "data_seed": config.data_seed,
        "device": config.device,
        "dtype": config.dtype,
    }


def protocol_manifest(
    *,
    budget: PilotBudget,
    vocab_size: int,
    device: str,
    dtype: str,
    output_root: Path,
    dataset: Dict[str, Any],
    device_total_bytes: Optional[int],
) -> Dict[str, Any]:
    specs = stage6_run_specs(budget)
    run_rows = []
    control_signatures = []
    for spec in specs:
        run_dir = output_root / spec.run_id
        config = build_training_config(
            spec,
            budget,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            out_dir=run_dir,
        )
        signature = logical_control_signature(config)
        control_signatures.append(signature)
        run_rows.append(
            {
                "run_id": spec.run_id,
                "model_type": spec.model_type,
                "base_row_order": spec.base_row_order,
                "row_order_4d": 4 * spec.base_row_order if spec.model_type == "thog2_sheet" else None,
                "checkpoint_segment_size": spec.checkpoint_segment_size,
                "out_dir": str(run_dir),
                "training_config": asdict(config),
                "logical_control_signature": signature,
            }
        )
    if any(signature != control_signatures[0] for signature in control_signatures[1:]):
        raise RuntimeError("stage6 run controls are not matched")
    manifest = {
        "protocol_version": STAGE6_PROTOCOL_VERSION,
        "status": "locked_before_training",
        "budget": asdict(budget),
        "tokens_per_update": budget.tokens_per_update,
        "consumed_tokens_per_run": budget.consumed_tokens,
        "dataset": dataset,
        "device": device,
        "dtype": dtype,
        "output_root": str(output_root),
        "runs": run_rows,
        "principal_dense_feasibility": principal_dense_feasibility(
            vocab_size=vocab_size,
            device_total_bytes=device_total_bytes,
        ),
        "scientific_scope": {
            "matched_geometry": f"L{budget.n_layer}/H{budget.n_head}/D{budget.n_embd}/C{budget.block_size}",
            "principal_l144_matched_comparison": "not claimed unless separately executed",
            "row_capacity_bracket": [64, 128, 256],
            "classification_options": list(STAGE6_CLASSIFICATIONS),
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def selected_run_rows(
    manifest: Dict[str, Any],
    selected: Iterable[str],
) -> Tuple[Dict[str, Any], ...]:
    requested = tuple(selected)
    unknown = sorted(set(requested) - set(STAGE6_RUN_ORDER))
    if unknown:
        raise ValueError(f"unknown stage6 run selectors: {unknown}")
    by_selector = {
        "dense": manifest["runs"][0],
        "q64": manifest["runs"][1],
        "q128": manifest["runs"][2],
        "q256": manifest["runs"][3],
    }
    return tuple(by_selector[name] for name in requested)


__all__ = [
    "PilotBudget",
    "PilotRunSpec",
    "STAGE6_CLASSIFICATIONS",
    "STAGE6_PROTOCOL_VERSION",
    "STAGE6_RUN_ORDER",
    "adamw_fp32_minimum_training_bytes",
    "build_training_config",
    "dense_parameter_count",
    "logical_control_signature",
    "principal_dense_feasibility",
    "protocol_manifest",
    "selected_run_rows",
    "stage6_run_specs",
]
# ^^^ THOG
