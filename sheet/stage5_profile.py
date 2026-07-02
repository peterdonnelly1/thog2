# vvv THOG
from __future__ import annotations

import gc
import resource
import time
from contextlib import nullcontext
from typing import Dict, Iterable, List

import torch
from torch import Tensor

from .basis import build_stabilized_basis, estimated_peak_tensor_bytes
from .stage4_trainer import Stage4Trainer
from .stage5_target import reduced_stage5_config, synthetic_token_splits
from .training_model import TrainingSheetGPT


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function):
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - started


def current_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss * 1024


def profile_basis_construction(
    device: torch.device,
    *,
    include_q1024: bool = False,
) -> List[Dict[str, object]]:
    geometries = [
        ("depth_p16", 144, 16),
        ("row_q128", 768, 128),
        ("row_4d_q512", 3072, 512),
    ]
    if include_q1024:
        geometries.append(("row_4d_q1024", 3072, 1024))
    rows: List[Dict[str, object]] = []
    for name, sample_count, order in geometries:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        rss_before = current_rss_bytes()
        basis, elapsed = timed_call(
            device,
            lambda: build_stabilized_basis(
                sample_count,
                order,
                runtime_dtype=torch.float32,
                device=device,
            ),
        )
        rss_after = current_rss_bytes()
        rows.append(
            {
                "name": name,
                "sample_count": sample_count,
                "order": order,
                "shape": list(basis.shape),
                "finite": bool(torch.isfinite(basis).all().item()),
                "elapsed_seconds": elapsed,
                "estimated_peak_tensor_bytes": estimated_peak_tensor_bytes(
                    sample_count,
                    order,
                    runtime_dtype=torch.float32,
                ),
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
            }
        )
        del basis
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def profile_materialisation(
    model: TrainingSheetGPT,
    *,
    dtype_name: str,
    layer_indices: Iterable[int],
) -> List[Dict[str, object]]:
    device = next(model.parameters()).device
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for layer_index in layer_indices:
            for metadata in model.trajectory.metadata:
                name = metadata.name
                coefficient = model.trajectory.coefficients[name]
                depth_row = model.trajectory.depth_basis[layer_index].to(coefficient)
                row_basis = model.trajectory.row_basis(name).to(coefficient)

                with autocast_context(device, dtype_name):
                    mixed, depth_seconds = timed_call(
                        device,
                        lambda: torch.einsum("p,rpq->rq", depth_row, coefficient),
                    )
                    generated, row_seconds = timed_call(
                        device,
                        lambda: mixed @ row_basis.transpose(0, 1),
                    )
                rows.append(
                    {
                        "family": name,
                        "layer_index": layer_index,
                        "generated_shape": list(generated.shape),
                        "depth_mix_seconds": depth_seconds,
                        "row_materialisation_seconds": row_seconds,
                        "total_seconds": depth_seconds + row_seconds,
                        "finite": bool(torch.isfinite(generated).all().item()),
                    }
                )
    return rows


def profile_block_components(
    model: TrainingSheetGPT,
    *,
    dtype_name: str,
    batch_size: int = 1,
    sequence_length: int = 256,
    layer_index: int = 0,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    hidden = torch.randn(
        batch_size,
        sequence_length,
        model.config.n_embd,
        device=device,
    )
    with torch.no_grad(), autocast_context(device, dtype_name):
        normalized, layer_norm_seconds = timed_call(
            device,
            lambda: model._sheet_layer_norm(
                hidden,
                "ln_1_weight",
                "ln_1_bias",
                layer_index,
            ),
        )
        attention, attention_seconds = timed_call(
            device,
            lambda: model._attention(normalized, layer_index),
        )
        mlp_input = hidden + attention
        normalized_mlp, second_layer_norm_seconds = timed_call(
            device,
            lambda: model._sheet_layer_norm(
                mlp_input,
                "ln_2_weight",
                "ln_2_bias",
                layer_index,
            ),
        )
        mlp, mlp_seconds = timed_call(
            device,
            lambda: model._mlp(normalized_mlp, layer_index),
        )
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "layer_index": layer_index,
        "first_layer_norm_seconds": layer_norm_seconds,
        "attention_seconds": attention_seconds,
        "second_layer_norm_seconds": second_layer_norm_seconds,
        "mlp_seconds": mlp_seconds,
        "total_seconds": (
            layer_norm_seconds
            + attention_seconds
            + second_layer_norm_seconds
            + mlp_seconds
        ),
        "finite": bool(torch.isfinite(mlp).all().item()),
    }


def profile_checkpoint_recomputation(
    *,
    device: str,
    dtype: str,
) -> Dict[str, object]:
    train_tokens, validation_tokens = synthetic_token_splits(512, length=8192)
    cases: Dict[str, Dict[str, object]] = {}
    for name, segment_size in (("reference", 0), ("checkpointed", 2)):
        config = reduced_stage5_config(
            device=device,
            dtype=dtype,
            checkpoint_segment_size=segment_size,
        )
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        try:
            metrics, elapsed = timed_call(
                trainer.device,
                trainer.train_one_update,
            )
            execution = trainer.raw_model.last_execution_report
            cases[name] = {
                "elapsed_seconds": elapsed,
                "training_loss": metrics["training_loss"],
                "gradient_norm": metrics["gradient_norm"],
                "checkpointing_used": execution.checkpointing_used,
                "checkpoint_segments": execution.checkpoint_segments,
                "memory": trainer.memory_telemetry.report(),
            }
        finally:
            trainer.close()
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    reference_seconds = float(cases["reference"]["elapsed_seconds"])
    checkpointed_seconds = float(cases["checkpointed"]["elapsed_seconds"])
    return {
        "reference": cases["reference"],
        "checkpointed": cases["checkpointed"],
        "checkpoint_to_reference_time_ratio": (
            checkpointed_seconds / reference_seconds
        ),
        "loss_delta": abs(
            float(cases["reference"]["training_loss"])
            - float(cases["checkpointed"]["training_loss"])
        ),
    }


__all__ = [
    "profile_basis_construction",
    "profile_block_components",
    "profile_checkpoint_recomputation",
    "profile_materialisation",
    "synchronize",
    "timed_call",
]
# ^^^ THOG
