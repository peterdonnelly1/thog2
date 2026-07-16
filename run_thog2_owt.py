# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sheet.basis import BASIS_VERSION
from sheet.checkpoints import load_payload
from sheet.compact_identity import ATTENTION_GEOMETRIES, BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESETS, MLP_GEOMETRIES
from sheet.residual_init import DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE, DEFAULT_RESIDUAL_INIT_DEPTH_VALUE, DEFAULT_RESIDUAL_INIT_POLICY, RESIDUAL_INIT_DEPTH_SOURCES, RESIDUAL_INIT_POLICIES
from sheet.run_config import (
    DEFAULT_EXPERIMENT_PREFIX,
    DEFAULT_O_ATTN_D_MODEL,
    DEFAULT_O_ATTN_OUT_PER_CHANNEL,
    DEFAULT_O_ATTN_QKV_PER_CHANNEL,
    DEFAULT_O_MLP_D_MODEL,
    DEFAULT_O_MLP_HIDDEN,
    OwtRunConfig,
)
from sheet.run_naming import compact_log_timestamp
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig
from sheet.wandb_telemetry import WandbTelemetry, attach_telemetry

REPOSITORY_ROOT = Path(__file__).resolve().parent


# vvv THOG
_CONSOLE_INTEGER_WIDTHS = {
    "completed_updates": 6,
    "max_updates": 6,
    "max_wall_minutes": 6,
    "consumed_tokens": 14,
    "tokens_per_update": 12,
    "checkpoint_bytes": 14,
}
_CONSOLE_FIXED_FLOATS = {
    "cumulative_training_seconds": (6, 0),
    "training_seconds": (6, 0),
    "cumulative_wall_seconds": (6, 0),
    "wall_seconds": (6, 0),
    "evaluation_seconds": (6, 0),
    "checkpoint_seconds": (6, 0),
    "gradient_norm": (8, 3),
    "training_loss": (9, 4),
    "validation_loss": (9, 4),
    "final_validation_loss": (9, 4),
    "tok/s": (12, 0),
}
_CONSOLE_SCIENTIFIC_FLOATS = {"learning_rate": (10, 3)}


# vvv THOG resumed throughput uses only tokens processed by the current process session
# Lifetime consumed_tokens remains available for progress and accounting.
def add_console_tokens_per_second(payload: Dict[str, Any]) -> Dict[str, Any]:
    values = dict(payload)
    elapsed = values.get("cumulative_training_seconds", values.get("training_seconds"))
    throughput_tokens = values.pop("session_consumed_tokens", None)
    if throughput_tokens is None:
        throughput_tokens = values.get("consumed_tokens")
    if elapsed is None or throughput_tokens is None:
        return values
    elapsed_value = float(elapsed)
    if elapsed_value <= 0.0:
        return values
    values["tok/s"] = float(throughput_tokens) / elapsed_value
    return values
# ^^^ THOG


def format_console_progress_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in _CONSOLE_INTEGER_WIDTHS:
            formatted[key] = f"{int(value):{_CONSOLE_INTEGER_WIDTHS[key]}d}"
        elif key in _CONSOLE_FIXED_FLOATS:
            width, precision = _CONSOLE_FIXED_FLOATS[key]
            formatted[key] = f"{float(value):{width}.{precision}f}"
        elif key in _CONSOLE_SCIENTIFIC_FLOATS:
            width, precision = _CONSOLE_SCIENTIFIC_FLOATS[key]
            formatted[key] = f"{float(value):{width}.{precision}e}"
        else:
            formatted[key] = value
    return formatted
# ^^^ THOG


class OwtTrainer(Stage6Trainer):
    """Stage 6 lifecycle with THOG-compatible global accumulation accounting."""

    @property
    def telemetry_token_multiplier(self) -> int:
        return int(self.distributed.world_size)

    def _print_progress(self, run_id: str, event: str, **payload: Any) -> None:
        values = dict(payload)
        if "consumed_tokens" in values:
            values["consumed_tokens"] = int(values["consumed_tokens"]) * int(self.distributed.world_size)
        # vvv THOG apply the same global-token multiplier to session throughput accounting
        if "session_consumed_tokens" in values:
            values["session_consumed_tokens"] = int(values["session_consumed_tokens"]) * int(self.distributed.world_size)
        # ^^^ THOG
        super()._print_progress(run_id, event, **format_console_progress_payload(add_console_tokens_per_second(values)))  # <<< THOG console progress now includes right-aligned tok/s and stable numeric widths.

    def run_pilot(self, **arguments: Any) -> Dict[str, Any]:
        result = super().run_pilot(**arguments)
        multiplier = int(self.distributed.world_size)
        if multiplier == 1:
            return result
        result["budget"]["tokens_per_update"] *= multiplier
        result["budget"]["consumed_tokens"] *= multiplier
        # vvv THOG preserve global-token accounting for resumed-session throughput fields
        result["budget"]["session_consumed_tokens"] *= multiplier
        # ^^^ THOG
        for row in result["updates"]:
            row["consumed_tokens"] *= multiplier
            # vvv THOG resumed-session token counts are global under DDP
            row["session_consumed_tokens"] *= multiplier
            # ^^^ THOG
        for row in result["evaluations"]:
            row["consumed_tokens"] *= multiplier
            # vvv THOG resumed-session token counts are global under DDP
            row["session_consumed_tokens"] *= multiplier
            # ^^^ THOG
        result["timing"]["tokens_per_training_second"] *= multiplier
        target = Path(arguments["result_path"])
        if self.distributed.is_primary:
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.distributed.barrier()
        return result


def load_vocab_size(dataset_dir: Path) -> int:
    metadata_path = dataset_dir / "meta.pkl"
    if not metadata_path.exists():
        return 50304
    with metadata_path.open("rb") as handle:
        return int(pickle.load(handle)["vocab_size"])


def load_tokens(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def source_identity() -> Dict[str, Optional[str]]:
    def git_output(*arguments: str) -> Optional[str]:
        completed = subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {"commit": git_output("rev-parse", "HEAD"), "branch": git_output("branch", "--show-current"), "tracked_status": git_output("status", "--porcelain", "--untracked-files=no")}


def run_digest(config: OwtRunConfig, dataset: Dict[str, Any], world_size: int) -> str:
    payload = {"config": config.canonical_dict(world_size=world_size), "dataset": dataset}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(dataset_dir: Path, block_size: int) -> Dict[str, Any]:
    train_path = dataset_dir / "train.bin"
    validation_path = dataset_dir / "val.bin"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"dataset must contain train.bin and val.bin: {dataset_dir}")
    token_bytes = np.dtype(np.uint16).itemsize
    if train_path.stat().st_size % token_bytes != 0:
        raise ValueError("train.bin size is not divisible by uint16 size")
    if validation_path.stat().st_size % token_bytes != 0:
        raise ValueError("val.bin size is not divisible by uint16 size")
    train_tokens = train_path.stat().st_size // token_bytes
    validation_tokens = validation_path.stat().st_size // token_bytes
    if train_tokens <= block_size or validation_tokens <= block_size:
        raise ValueError("dataset splits must be longer than block_size")
    return {"path": str(dataset_dir.resolve()), "format": "uint16_token_ids", "vocab_size": load_vocab_size(dataset_dir), "train_tokens": train_tokens, "validation_tokens": validation_tokens}


def validate_resume_controls(checkpoint_path: Path, expected: TrainingConfig) -> None:
    payload = load_payload(checkpoint_path)
    if "trainer_config" not in payload:
        return
    stored = TrainingConfig(**payload["trainer_config"])
    control_fields = ("batch_size", "gradient_accumulation_steps", "learning_rate", "min_learning_rate", "warmup_updates", "weight_decay", "beta1", "beta2", "grad_clip", "nonfinite_update_policy", "max_nonfinite_update_skips", "model_seed", "data_seed")
    mismatches = [f"{name}: checkpoint={getattr(stored, name)!r}, requested={getattr(expected, name)!r}" for name in control_fields if getattr(stored, name) != getattr(expected, name)]
    if mismatches:
        raise ValueError("resume control mismatch: " + "; ".join(mismatches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or resume one canonical THOG2 OpenWebText run")
    parser.add_argument("--model-type", choices=("dense", "sheet"), required=True)
    parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--host-label", default=socket.gethostname().split(".")[0])
    parser.add_argument("--run-name", default="AKAROA")
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--data-dir", default="data/openwebtext")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--result-root", default="results")
    parser.add_argument("--wandb-root", default="wandb")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--max-wall-minutes", type=int, default=0)                                                                                   # <<< THOG soft wall-clock budget; zero preserves existing update-count runs
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=160)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=72)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--o-depth", type=int, default=16)
    parser.add_argument("--o-attn-d-model", type=int, default=DEFAULT_O_ATTN_D_MODEL)
    parser.add_argument("--o-attn-qkv-per-channel", type=int, default=DEFAULT_O_ATTN_QKV_PER_CHANNEL)
    parser.add_argument("--o-attn-out-per-channel", type=int, default=DEFAULT_O_ATTN_OUT_PER_CHANNEL)
    parser.add_argument("--o-mlp-d-model", type=int, default=DEFAULT_O_MLP_D_MODEL)
    parser.add_argument("--o-mlp-hidden", type=int, default=DEFAULT_O_MLP_HIDDEN)
    parser.add_argument("--geometry-preset", choices=GEOMETRY_PRESETS, default=GEOMETRY_PRESET_DEPTH)
    parser.add_argument("--attention-geometry", choices=ATTENTION_GEOMETRIES)
    parser.add_argument("--mlp-geometry", choices=MLP_GEOMETRIES)
    parser.add_argument("--basis-family", choices=(BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT), default=BASIS_FAMILY_CHEBYSHEV)
    parser.add_argument("--basis-version", default="auto")
    parser.add_argument("--attention-backend", choices=("auto", "flash2", "sdpa", "math"), default="auto")
    parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    parser.add_argument("--run-start-label")
    parser.add_argument("--residual-init-policy", choices=RESIDUAL_INIT_POLICIES, default=DEFAULT_RESIDUAL_INIT_POLICY)
    parser.add_argument("--residual-init-depth-source", choices=RESIDUAL_INIT_DEPTH_SOURCES, default=DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE)
    parser.add_argument("--residual-init-depth-value", type=int, default=DEFAULT_RESIDUAL_INIT_DEPTH_VALUE)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-segment-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=6.0e-4)
    parser.add_argument("--min-lr", type=float, default=6.0e-5)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--nonfinite-update-policy", choices=("raise", "skip"), default="skip")                                                    # <<< THOG bounded recovery policy
    parser.add_argument("--max-nonfinite-update-skips", type=int, default=10)                                                                         # <<< THOG bounded recovery limit
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--data-seed", type=int, default=7331)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="thog")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--artifact-suffix")
    parser.add_argument("--artifact-name-limit", type=int, default=240)
    parser.add_argument("--log-timestamp")
    parser.add_argument("--print-artifact-name", action="store_true")
    parser.add_argument("--print-resolved-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_start_label_from_arguments(arguments: argparse.Namespace) -> Optional[str]:
    if arguments.run_start_label:
        return arguments.run_start_label
    if arguments.log_timestamp:
        return compact_log_timestamp(arguments.log_timestamp).replace("_", "-")
    return None


def configure_attention_backend(attention_backend: str) -> None:
    if attention_backend == "auto" or not torch.cuda.is_available():
        return
    if attention_backend == "flash2":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return
    if attention_backend == "sdpa":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return
    raise ValueError(f"unsupported attention backend: {attention_backend}")


def config_from_arguments(arguments: argparse.Namespace) -> OwtRunConfig:
    basis_version = BASIS_VERSION if arguments.basis_version == "auto" else arguments.basis_version
    config = OwtRunConfig(
        model_type=arguments.model_type,
        run_mode=arguments.run_mode,
        host_label=arguments.host_label,
        run_name=arguments.run_name,
        dataset=arguments.dataset,
        data_dir=arguments.data_dir,
        checkpoint_root=arguments.checkpoint_root,
        log_root=arguments.log_root,
        result_root=arguments.result_root,
        wandb_root=arguments.wandb_root,
        max_iters=arguments.max_iters,
        max_wall_minutes=arguments.max_wall_minutes,
        eval_interval=arguments.eval_interval,
        eval_iters=arguments.eval_iters,
        log_interval=arguments.log_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        block_size=arguments.block_size,
        n_layer=arguments.n_layer,
        n_head=arguments.n_head,
        n_embd=arguments.n_embd,
        o_depth=arguments.o_depth,
        o_attn_d_model=arguments.o_attn_d_model,
        o_attn_qkv_per_channel=arguments.o_attn_qkv_per_channel,
        o_attn_out_per_channel=arguments.o_attn_out_per_channel,
        o_mlp_d_model=arguments.o_mlp_d_model,
        o_mlp_hidden=arguments.o_mlp_hidden,
        geometry_preset=arguments.geometry_preset,
        attention_geometry=arguments.attention_geometry,
        mlp_geometry=arguments.mlp_geometry,
        basis_family=arguments.basis_family,
        basis_version=basis_version,
        attention_backend=arguments.attention_backend,
        experiment_prefix=arguments.experiment_prefix,
        run_start_label=run_start_label_from_arguments(arguments),
        residual_init_policy=arguments.residual_init_policy,
        residual_init_depth_source=arguments.residual_init_depth_source,
        residual_init_depth_value=arguments.residual_init_depth_value,
        activation_checkpointing=arguments.activation_checkpointing,
        checkpoint_segment_size=arguments.checkpoint_segment_size,
        learning_rate=arguments.learning_rate,
        min_lr=arguments.min_lr,
        warmup_iters=arguments.warmup_iters,
        weight_decay=arguments.weight_decay,
        beta1=arguments.beta1,
        beta2=arguments.beta2,
        grad_clip=arguments.grad_clip,
        nonfinite_update_policy=arguments.nonfinite_update_policy,
        max_nonfinite_update_skips=arguments.max_nonfinite_update_skips,
        dropout=arguments.dropout,
        bias=arguments.bias,
        model_seed=arguments.model_seed,
        data_seed=arguments.data_seed,
        device=arguments.device,
        dtype=arguments.dtype,
        wandb_enabled=arguments.wandb,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_mode=arguments.wandb_mode,
        artifact_suffix=arguments.artifact_suffix,
        artifact_name_limit=arguments.artifact_name_limit,
    )
    configure_attention_backend(config.attention_backend)
    return config


def resolved_payload(config: OwtRunConfig, *, world_size: int, log_timestamp: Optional[str]) -> Dict[str, Any]:
    paths = config.paths(log_timestamp=log_timestamp)
    return {"artifact_name": config.artifact_name, "artifact_prefix": config.artifact_prefix, "model_type": config.model_type, "run_mode": config.run_mode, "world_size": world_size, "tokens_per_iter": config.tokens_per_iter(), "canonical_config": config.canonical_dict(world_size=world_size), "paths": {name: str(path) for name, path in paths.items()}}


# vvv THOG print resolved model parameters and execution options immediately before training
def print_model_parameters_and_options(config: OwtRunConfig, trainer: OwtTrainer) -> None:
    report = trainer.parameter_report
    persistent = int(report["persistent_parameters"])
    dense_equivalent = int(report["dense_equivalent_total_parameters"])
    sheet_coefficients = int(report["sheet_coefficients"])
    compression = (dense_equivalent / persistent) if persistent else 0.0
    print("model parameters and options", flush=True)
    print(f"  parameters: persistent={persistent:,}  sheet coefficients={sheet_coefficients:,}  dense equivalent={dense_equivalent:,}  dense/persistent={compression:.2f}x", flush=True)
    print(f"  optimiser:  lr={config.learning_rate:.3e}  min_lr={config.min_lr:.3e}  warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}", flush=True)
    print(f"  wall stop:  max_wall_minutes={config.max_wall_minutes}", flush=True)                                                                           # <<< THOG show soft wall-clock budget for equal-time grids
    print(f"  non-finite: policy={config.nonfinite_update_policy}  max_skips={config.max_nonfinite_update_skips}", flush=True)                                # <<< THOG show bounded recovery policy before the run
    print(f"  batches:    micro={config.batch_size}  accumulation={config.gradient_accumulation_steps}  tokens/update={config.tokens_per_iter():,}", flush=True)
    if config.model_type == "sheet":
        model_config = trainer.raw_model.config
        print(f"  execution:  semantic_qkv_bypass={model_config.bypass_semantic_qkv_adapter}  vectorise_per_head={model_config.vectorise_per_head_materialisation}  direct_factorised_mlp={model_config.direct_factorised_mlp}  activation_checkpointing={config.activation_checkpointing}", flush=True)
    print(flush=True)
# ^^^ THOG


def main() -> int:
    arguments = build_parser().parse_args()
    config = config_from_arguments(arguments)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    payload = resolved_payload(config, world_size=world_size, log_timestamp=arguments.log_timestamp)
    if arguments.print_artifact_name:
        print(config.artifact_name)
        return 0
    if arguments.print_resolved_json or arguments.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paths = config.paths(log_timestamp=arguments.log_timestamp)
    checkpoint_path = paths["checkpoint_path"]
    result_path = paths["result_path"]
    if config.run_mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError(f"fresh run refuses to overwrite {checkpoint_path}")
    if config.run_mode == "resume" and not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint_path}")
    dataset_dir = Path(config.data_dir)
    dataset = validate_dataset(dataset_dir, config.block_size)
    training_config = config.to_training_config(vocab_size=int(dataset["vocab_size"]), world_size=world_size, out_dir=paths["checkpoint_dir"])
    if config.run_mode == "resume":
        validate_resume_controls(checkpoint_path, training_config)
    for key in ("checkpoint_dir", "log_dir", "result_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    train_tokens = load_tokens(dataset_dir / "train.bin")
    validation_tokens = load_tokens(dataset_dir / "val.bin")
    if config.run_mode == "resume":
        trainer = OwtTrainer.from_checkpoint(checkpoint_path, train_tokens, validation_tokens, expected_config=training_config, overrides={"device": training_config.device, "dtype": training_config.dtype, "max_updates": training_config.max_updates, "max_wall_minutes": training_config.max_wall_minutes, "eval_interval": training_config.eval_interval, "eval_batches": training_config.eval_batches, "checkpoint_interval": training_config.checkpoint_interval, "checkpoint_segment_size": training_config.checkpoint_segment_size, "out_dir": training_config.out_dir, "log_interval": training_config.log_interval, "nonfinite_update_policy": training_config.nonfinite_update_policy, "max_nonfinite_update_skips": training_config.max_nonfinite_update_skips})
    else:
        trainer = OwtTrainer(training_config, train_tokens, validation_tokens)
    canonical = config.canonical_dict(world_size=world_size)
    source = source_identity()
    telemetry = WandbTelemetry(enabled=(config.wandb_enabled and trainer.distributed.is_primary), project=config.wandb_project, entity=config.wandb_entity, mode=config.wandb_mode, root=Path(config.wandb_root), name=config.artifact_name, group=config.experiment_prefix, job_type="dense2" if config.model_type == "dense" else "sheet", config={**canonical, "source_commit": source["commit"], "source_branch": source["branch"], "dataset_record": dataset, "parameter_report": trainer.parameter_report})
    try:
        if trainer.distributed.is_primary:
            telemetry.start()
            telemetry.add_initial_summary(trainer.parameter_report)
        attach_telemetry(trainer, telemetry)
        if trainer.distributed.is_primary:
            print_model_parameters_and_options(config, trainer)                                                                                         # <<< THOG show the complete effective training setup before the first update
        result = trainer.run_pilot(run_id=config.artifact_name, protocol_sha256=run_digest(config, dataset, world_size), dataset=dataset, result_path=result_path)
        result["artifact"] = {"name": config.artifact_name, "prefix": config.artifact_prefix, "paths": {name: str(path) for name, path in paths.items()}}
        result["canonical_config"] = canonical
        result["source"] = source
        if trainer.distributed.is_primary:
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            telemetry.add_final_result(result)
            print(json.dumps({"artifact_name": config.artifact_name, "checkpoint": str(checkpoint_path), "result": str(result_path), "completed_updates": result["budget"]["completed_updates"], "consumed_tokens": result["budget"]["consumed_tokens"], "final_validation_loss": (result["evaluations"][-1]["val"] if result["evaluations"] else None)}, indent=2, sort_keys=True))
        return 0
    finally:
        if rank == 0:
            telemetry.finish()
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG