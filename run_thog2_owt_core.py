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
from sheet.bases import BASIS_FAMILIES
from sheet.bases.lapped_cosine import DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION, DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                          # <<< THOG lapped CLI defaults
# from sheet.checkpoints import load_payload
# vvv THOG resume-control preflight uses the same PLASTIC geometry-format guard as full trainer resume
from sheet.checkpoints import load_payload, validate_plastic_depth_checkpoint_format
# ^^^ THOG
from sheet.compact_identity import ATTENTION_GEOMETRIES, BASIS_FAMILY_CHEBYSHEV, DEFAULT_MLP_HIDDEN_COMPRESSOR, DEFAULT_MLP_HIDDEN_GROUP_SIZE, GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESETS, MLP_GEOMETRIES
# vvv THOG sampling-only chaos bump resume-control identity
from sheet.chaos_bump_sampling import CHAOS_BUMP_SAMPLING_CONFIG_FIELDS
# ^^^ THOG
from sheet.geometry_registry import AXIS_MLP_HIDDEN, format_geometry_plan, resolve_geometry_plan
# vvv THOG v0 exposes HYPERBLOCK as a Boolean mode while retaining explicit topology identity internally
from sheet.hyperblock import HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
# ^^^ THOG
# vvv THOG PLASTIC DEPTH categorical CLI choices
from sheet.plastic_depth import (
    PLASTIC_LAYER_COUNT_OBJECTIVES,
    PLASTIC_LAYER_SAMPLING_INITIALISATIONS,
)
# ^^^ THOG
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
from sheet.training_config import TrainingConfig, normalize_plastic_v0541_config_fields
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
        super()._print_progress(run_id, event, **format_console_progress_payload(add_console_tokens_per_second(values)))                                   # <<< THOG console progress now includes right-aligned tok/s and stable numeric widths.

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
            row["session_consumed_tokens"] *= multiplier                                                                                                   # <<< THOG resumed-session token counts are global under DDP
        for row in result["evaluations"]:
            row["consumed_tokens"] *= multiplier
            row["session_consumed_tokens"] *= multiplier                                                                                                   # <<< THOG resumed-session token counts are global under DDP
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
    # vvv THOG reject v0.1 or ambiguous PLASTIC geometry before reconstructing stored controls
    validate_plastic_depth_checkpoint_format(payload)
    # ^^^ THOG
    if "trainer_config" not in payload:
        return
    stored = TrainingConfig(**normalize_plastic_v0541_config_fields(payload["trainer_config"]))
    # vvv THOG layer-dropout execution policy must not silently change across normal resume
    control_fields = (
        "batch_size", "gradient_accumulation_steps", "learning_rate", "min_learning_rate", "warmup_updates", "weight_decay", "beta1", "beta2", "grad_clip",
        "nonfinite_update_policy", "max_nonfinite_update_skips", "model_seed", "data_seed",
        "layer_dropout_stratum_size", "layer_dropout_active_per_stratum", "layer_dropout_resample_steps",
        "plastic__layer_count_probe__number_of_sampled_valid_tokens",
        # vvv THOG a resumed bump schedule must retain every material control
        *CHAOS_BUMP_SAMPLING_CONFIG_FIELDS,
        # ^^^ THOG
    )
    # ^^^ THOG
    mismatches = [f"{name}: checkpoint={getattr(stored, name)!r}, requested={getattr(expected, name)!r}" for name in control_fields if getattr(stored, name) != getattr(expected, name)]
    if mismatches:
        raise ValueError("resume control mismatch: " + "; ".join(mismatches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or resume one canonical THOG2 OpenWebText run")
    parser.add_argument("--model-type", choices=("dense", "sheet"))
    parser.add_argument("--select-depth", action="store_true", help="select the universal registered DEPTH curve")
    parser.add_argument("--select-element", action="append", default=[], metavar="SELECTOR", help="select one registered permitted geometry; repeat for multiple elements")
    parser.add_argument("--option", dest="geometry_options", action="append", default=[], metavar="TARGET.PROPERTY=VALUE", help="assign an element- or axis-scoped systematic geometry option")
    parser.add_argument("--explain-geometry", action="store_true", help="resolve and report systematic geometry, then exit before model construction")
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
    parser.add_argument("--max-wall-minutes", type=int, default=0)                                                                                         # <<< THOG soft wall-clock budget; zero preserves existing update-count runs
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=160)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=72)
    # vvv THOG public layer-dropout controls; omitted stratum/cardinality means all layers active
    parser.add_argument("--layer-dropout-stratum-size", type=int)
    parser.add_argument("--layer-dropout-active-per-stratum", type=int)
    parser.add_argument("--layer-dropout-resample-steps", type=int, default=1)
    # ^^^ THOG
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--o-depth", type=int, default=16)
    parser.add_argument("--o-attn-d-model", type=int, default=DEFAULT_O_ATTN_D_MODEL)
    parser.add_argument("--o-attn-qkv-per-channel", type=int, default=DEFAULT_O_ATTN_QKV_PER_CHANNEL)
    parser.add_argument("--o-attn-out-per-channel", type=int, default=DEFAULT_O_ATTN_OUT_PER_CHANNEL)
    parser.add_argument("--o-mlp-d-model", type=int, default=DEFAULT_O_MLP_D_MODEL)
    parser.add_argument("--o-mlp-hidden", type=int, default=DEFAULT_O_MLP_HIDDEN)
    parser.add_argument("--mlp-hidden-group-size", type=int, default=DEFAULT_MLP_HIDDEN_GROUP_SIZE)
    parser.add_argument("--mlp-hidden-compressor", choices=BASIS_FAMILIES, default=DEFAULT_MLP_HIDDEN_COMPRESSOR)
    parser.add_argument("--depth-compress-layer-norm-and-bias", action=argparse.BooleanOptionalAction, default=False)                                      # <<< THOG DEPTH-only vector participation control
    parser.add_argument("--geometry-preset", choices=GEOMETRY_PRESETS, default=GEOMETRY_PRESET_DEPTH)
    parser.add_argument("--attention-geometry", choices=ATTENTION_GEOMETRIES)
    parser.add_argument("--mlp-geometry", choices=MLP_GEOMETRIES)
    parser.add_argument("--basis-family", choices=BASIS_FAMILIES, default=BASIS_FAMILY_CHEBYSHEV)
    parser.add_argument("--basis-version", default="auto")
    parser.add_argument(
        "--save-dense-initialisation-snapshot",
        action="store_true",
        help="save immutable A Normal DENSE step-zero model parameters and continue",
    )
    parser.add_argument(
        "--initialise-from-dense-snapshot",
        metavar="FILE",
        help="initialise B Compressor-baselined DENSE or C Compact Run from FILE",
    )
    # vvv THOG coupled field machine HYPERBLOCK controls; the topology is implicit while it is the sole implementation
    parser.add_argument("--hyperblock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hyperblock-compressor", choices=BASIS_FAMILIES, default=BASIS_FAMILY_CHEBYSHEV)
    parser.add_argument("--hyperblock-compressor-version", default="auto")
    parser.add_argument("--hyperblock-common-family-order", type=int, default=6)
    parser.add_argument("--hyperblock-attention-family-order", type=int, default=4)
    parser.add_argument("--hyperblock-mlp-family-order", type=int, default=2)
    parser.add_argument("--hyperblock-depth-order", type=int, default=16)
    parser.add_argument("--hyperblock-d-model-order", type=int, default=16)
    parser.add_argument("--hyperblock-mlp-hidden-order", type=int, default=16)
    parser.add_argument("--hyperblock-attention-head-order", type=int, default=16)
    parser.add_argument("--hyperblock-attention-head-channel-order", type=int, default=16)
    parser.add_argument("--hyperblock-mlp-hidden-multiplier", type=int, default=4)
    parser.add_argument("--hyperblock-loop-count", type=int, default=1)
    parser.add_argument("--hyperblock-loop-decay", type=float, default=1.0)
    # ^^^ THOG
    # vvv THOG PLASTIC DEPTH public controls map directly to the canonical double-underscore configuration fields
    parser.add_argument("--plastic__enabled", dest="plastic__enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plastic__coarse_phase", dest="plastic__coarse_phase", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--plastic__coarse_phase_roll_through", dest="plastic__coarse_phase_roll_through", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plastic__log_interval_coarse", dest="plastic__log_interval_coarse", type=int, default=10)
    parser.add_argument("--plastic__phase_1_n_steps", dest="plastic__phase_1_n_steps", type=int)
    parser.add_argument("--plastic__phase_1_starting_layer_count", dest="plastic__phase_1_starting_layer_count", type=int)
    parser.add_argument("--plastic__phase_1__number_of_trials", dest="plastic__phase_1__number_of_trials", type=int)
    parser.add_argument("--plastic__phase_1_evaluation_steps_count", dest="plastic__phase_1_evaluation_steps_count", type=int)
    parser.add_argument("--plastic__layers_to_sample", dest="plastic__layers_to_sample", type=int)
    parser.add_argument("--plastic__do_learn_layer_count", dest="plastic__do_learn_layer_count", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plastic__initial_layer_count", dest="plastic__initial_layer_count", type=int)
    parser.add_argument("--plastic__max_permitted_layers", dest="plastic__max_permitted_layers", type=int)
    parser.add_argument("--plastic__layer_sampling_initialisation", dest="plastic__layer_sampling_initialisation", choices=PLASTIC_LAYER_SAMPLING_INITIALISATIONS, default="equidistant")
    parser.add_argument("--plastic__layer_count_objective", dest="plastic__layer_count_objective", choices=PLASTIC_LAYER_COUNT_OBJECTIVES, default="lowest_loss")
    parser.add_argument("--plastic__layer_count_update_brake", dest="plastic__layer_count_update_brake", type=int, default=5)
    parser.add_argument("--plastic__layer_count_probe__probe_every_n_steps", dest="plastic__layer_count_probe__probe_every_n_steps", type=int)
    parser.add_argument("--plastic__layer_count_probe__number_of_sampled_valid_tokens", dest="plastic__layer_count_probe__number_of_sampled_valid_tokens", type=int, default=1024)
    parser.add_argument("--plastic__layer_count_probe_radius", dest="plastic__layer_count_probe_radius", type=int, default=1)
    parser.add_argument("--plastic__layer_count__max_allowable_layer_change", dest="plastic__layer_count__max_allowable_layer_change", type=int, default=1)
    parser.add_argument("--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence", dest="plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence", type=float, default=0.8)
    parser.add_argument("--plastic__layer_count_probe__window_size_as_number_of_probes", dest="plastic__layer_count_probe__window_size_as_number_of_probes", type=int, default=50)
    parser.add_argument("--plastic__layer_count_probe_noise_lambda", dest="plastic__layer_count_probe_noise_lambda", type=float, default=3.0)
    parser.add_argument("--plastic__wall_time_equivalent_time_gain_discount", dest="plastic__wall_time_equivalent_time_gain_discount", type=float, default=0.9)
    parser.add_argument("--plastic__wall_time_equivalent_time_gain_loss_rate_window", dest="plastic__wall_time_equivalent_time_gain_loss_rate_window", type=int, default=64)
    parser.add_argument("--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations", dest="plastic__wall_time_equivalent_time_gain_loss_rate_min_observations", type=int, default=16)
    parser.add_argument("--plastic__layer_count_cost_weight", dest="plastic__layer_count_cost_weight", type=float, default=0.0)
    parser.add_argument("--plastic__layer_count__memory_budget_gib", dest="plastic__layer_count__memory_budget_gib", type=float)
    parser.add_argument("--plastic__layer_count__cuda_allocator_reserve_gib", dest="plastic__layer_count__cuda_allocator_reserve_gib", type=float, default=0.5)
    parser.add_argument("--plastic__geometry_learning_rate_multiplier", dest="plastic__geometry_learning_rate_multiplier", type=float, default=0.1)
    parser.add_argument("--plastic__freeze_geometry_during_warmup", dest="plastic__freeze_geometry_during_warmup", action=argparse.BooleanOptionalAction, default=True)
    # ^^^ THOG
    # vvv THOG v1.3 sampling-only chaos bump public namespace
    parser.add_argument("--chaos_bump__sampling__enabled", dest="chaos_bump__sampling__enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chaos_bump__sampling__initial_lockout__steps", dest="chaos_bump__sampling__initial_lockout__steps", type=int, default=16)
    parser.add_argument("--chaos_bump__sampling__maximum_bumps", dest="chaos_bump__sampling__maximum_bumps", type=int, default=1)
    parser.add_argument("--chaos_bump__sampling__interlude__min_steps", dest="chaos_bump__sampling__interlude__min_steps", type=int, default=128)
    parser.add_argument("--chaos_bump__sampling__interlude__max_steps", dest="chaos_bump__sampling__interlude__max_steps", type=int, default=256)
    parser.add_argument("--chaos_bump__sampling__duration__min_steps", dest="chaos_bump__sampling__duration__min_steps", type=int, default=16)
    parser.add_argument("--chaos_bump__sampling__duration__max_steps", dest="chaos_bump__sampling__duration__max_steps", type=int, default=256)
    parser.add_argument("--chaos_bump__sampling__duration__max_fraction_of_elapsed_steps", dest="chaos_bump__sampling__duration__max_fraction_of_elapsed_steps", type=float, default=0.05)
    parser.add_argument("--chaos_bump__sampling__max_movement_fraction_of_local_gap", dest="chaos_bump__sampling__max_movement_fraction_of_local_gap", type=float, default=0.10)
    # ^^^ THOG
    # vvv THOG explicit lapped cosine controls
    parser.add_argument("--lapped-cosine-window-length", type=int, default=DEFAULT_LAPPED_COSINE_WINDOW_LENGTH)
    parser.add_argument("--lapped-cosine-overlap-fraction", type=float, default=DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION)
    # ^^^ THOG
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
    parser.add_argument("--nonfinite-update-policy", choices=("raise", "skip"), default="skip")                                                            # <<< THOG bounded recovery policy
    parser.add_argument("--max-nonfinite-update-skips", type=int, default=99999)                                                                           # <<< THOG bounded recovery limit
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
    # vvv THOG fixed-scale sparse/per-probe DEPTH response heatmap controls are accepted in every PLASTIC mode and route independently of scalar telemetry
    parser.add_argument(
        "--instrumentation__delta_loss_v_layer_heatmap",
        dest="instrumentation__delta_loss_v_layer_heatmap",
        choices=("log", "linear"),
        default=None,
    )
    parser.add_argument(
        "--instrumentation__delta_loss_v_layer_heatmap__destination",
        dest="instrumentation__delta_loss_v_layer_heatmap__destination",
        choices=("wandb", "local", "none"),
        default="local",
    )
    parser.add_argument(
        "--instrumentation__delta_loss_v_layer_heatmap_abs_limit",
        dest="instrumentation__delta_loss_v_layer_heatmap_abs_limit",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
        dest="instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
        type=int,
        default=250,
    )
    # ^^^ THOG
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


def systematic_geometry_requested(arguments: argparse.Namespace) -> bool:
    return bool(arguments.select_depth or arguments.select_element or arguments.geometry_options or arguments.explain_geometry)


def geometry_plan_from_arguments(arguments: argparse.Namespace):
    if arguments.hyperblock and systematic_geometry_requested(arguments):
        raise ValueError("--hyperblock cannot be combined with selector-based geometry controls")
    if not systematic_geometry_requested(arguments):
        return None
    if arguments.model_type == "dense" and arguments.initialise_from_dense_snapshot is None:
        raise ValueError("the systematic geometry UI requires model_type='sheet' or an omitted --model-type")
    if arguments.geometry_preset not in (None, GEOMETRY_PRESET_DEPTH):
        raise ValueError("the systematic geometry UI cannot be mixed with a non-default --geometry-preset")
    if arguments.attention_geometry is not None or arguments.mlp_geometry is not None:
        raise ValueError("the systematic geometry UI cannot be mixed with --attention-geometry or --mlp-geometry")
    if arguments.basis_family not in (None, BASIS_FAMILY_CHEBYSHEV):
        raise ValueError("the systematic geometry UI cannot be mixed with a non-default --basis-family; use --option DEPTH.compressor=...")
    if arguments.basis_version not in ("auto", BASIS_VERSION):
        raise ValueError("the systematic geometry UI cannot be mixed with --basis-version; assign compressor versions through --option")
    return resolve_geometry_plan(
        select_depth=arguments.select_depth,
        selected_elements=arguments.select_element,
        option_assignments=arguments.geometry_options,
        legacy_orders={
            "o_depth": arguments.o_depth,
            "o_attn_d_model": arguments.o_attn_d_model,
            "o_attn_qkv_per_channel": arguments.o_attn_qkv_per_channel,
            "o_attn_out_per_channel": arguments.o_attn_out_per_channel,
            "o_mlp_d_model": arguments.o_mlp_d_model,
            "o_mlp_hidden": arguments.o_mlp_hidden,
        },
        default_mlp_hidden_group_size=arguments.mlp_hidden_group_size,
    )


def _dense_snapshot_mapping_from_plan(arguments: argparse.Namespace, geometry_plan):
    if arguments.initialise_from_dense_snapshot is None or arguments.model_type != "dense":
        return None, None
    if geometry_plan is None:
        raise ValueError(
            "B Compressor-baselined DENSE requires --select-depth and "
            "--option DEPTH.order=P"
        )
    if not geometry_plan.depth_enabled or geometry_plan.selections:
        raise ValueError("DENSE snapshot v1 supports only pure DEPTH geometry")
    if geometry_plan.depth_compressor != BASIS_FAMILY_CHEBYSHEV:
        raise ValueError(
            "DENSE snapshot v1 supports only the Chebyshev compressor; "
            f"got {geometry_plan.depth_compressor!r}"
        )
    if geometry_plan.depth_compressor_version != BASIS_VERSION:
        raise ValueError(
            "DENSE snapshot v1 requires the current QR-stabilised Chebyshev version; "
            f"expected={BASIS_VERSION!r}, got={geometry_plan.depth_compressor_version!r}"
        )
    return int(geometry_plan.depth_order), str(geometry_plan.depth_compressor_version)


def validate_dense_snapshot_cli(
    arguments: argparse.Namespace,
    *,
    explicit=(),
    resolved_mode: Optional[str] = None,
) -> None:
    save_requested = bool(arguments.save_dense_initialisation_snapshot)
    initialise_requested = arguments.initialise_from_dense_snapshot is not None
    if save_requested and initialise_requested:
        raise ValueError(
            "--save-dense-initialisation-snapshot and "
            "--initialise-from-dense-snapshot are mutually exclusive"
        )
    mode = str(resolved_mode or arguments.run_mode or "fresh")
    if (save_requested or initialise_requested) and mode != "fresh":
        selected = (
            "--save-dense-initialisation-snapshot"
            if save_requested
            else "--initialise-from-dense-snapshot"
        )
        raise ValueError(f"{selected} may not be combined with resume or fork")
    residual_fields = {
        "residual_init_policy": "--residual-init-policy/-r",
        "residual_init_depth_source": "--residual-init-depth-source/-z",
        "residual_init_depth_value": "--residual-init-depth-value/-Z",
    }
    conflicts = [option for name, option in residual_fields.items() if name in explicit]
    if initialise_requested and conflicts:
        raise ValueError(
            "--initialise-from-dense-snapshot may not be combined with explicit "
            "residual-initialisation options: " + ", ".join(conflicts)
        )


def config_from_arguments(arguments: argparse.Namespace, *, geometry_plan=None) -> OwtRunConfig:
    geometry_plan = geometry_plan if geometry_plan is not None else geometry_plan_from_arguments(arguments)
    validate_dense_snapshot_cli(arguments)
    snapshot_order, snapshot_version = _dense_snapshot_mapping_from_plan(arguments, geometry_plan)
    if geometry_plan is not None and not geometry_plan.materializer.implemented:
        raise ValueError(geometry_plan.materializer.message)
    # vvv THOG preserve the exact pre-HYPERBLOCK config derivation lines for source history
    # model_type = arguments.model_type or ("sheet" if geometry_plan is not None else None)
    # geometry_preset=arguments.geometry_preset if adapter is None else adapter.legacy_geometry_preset,
    # attention_geometry=arguments.attention_geometry if adapter is None else None,
    # mlp_geometry=arguments.mlp_geometry if adapter is None else None,
    # basis_family=arguments.basis_family if adapter is None else adapter.legacy_basis_family,
    # ^^^ THOG
    model_type = arguments.model_type or ("sheet" if geometry_plan is not None or arguments.hyperblock else None)
    if model_type is None:
        raise ValueError("--model-type is required for legacy runs; systematic geometry selections imply model_type='sheet'")
    if arguments.hyperblock and model_type != "sheet":
        raise ValueError("--hyperblock requires --model-type sheet or an omitted --model-type")
    dense_snapshot_target = arguments.model_type == "dense" and arguments.initialise_from_dense_snapshot is not None
    adapter = None if geometry_plan is None or dense_snapshot_target else geometry_plan.materializer
    basis_version = arguments.basis_version if adapter is None else str(adapter.legacy_basis_version)
    selected_mlp_hidden_order = arguments.o_mlp_hidden
    if geometry_plan is not None:
        for selection in geometry_plan.selections:
            if AXIS_MLP_HIDDEN in selection.orders:
                selected_mlp_hidden_order = selection.orders[AXIS_MLP_HIDDEN]
    config = OwtRunConfig(
        model_type=model_type,
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
        layer_dropout_stratum_size=arguments.layer_dropout_stratum_size,                                                                                   # <<< THOG CLI stratification interval
        layer_dropout_active_per_stratum=arguments.layer_dropout_active_per_stratum,                                                                       # <<< THOG CLI exact active count per stratum
        layer_dropout_resample_steps=arguments.layer_dropout_resample_steps,                                                                               # <<< THOG CLI selection lifetime
        o_depth=arguments.o_depth if geometry_plan is None or geometry_plan.depth_order is None else geometry_plan.depth_order,
        o_attn_d_model=arguments.o_attn_d_model,
        o_attn_qkv_per_channel=arguments.o_attn_qkv_per_channel,
        o_attn_out_per_channel=arguments.o_attn_out_per_channel,
        o_mlp_d_model=arguments.o_mlp_d_model,
        o_mlp_hidden=selected_mlp_hidden_order,
        mlp_hidden_group_size=arguments.mlp_hidden_group_size if adapter is None or adapter.legacy_mlp_hidden_group_size is None else adapter.legacy_mlp_hidden_group_size,
        mlp_hidden_compressor=arguments.mlp_hidden_compressor if adapter is None or adapter.legacy_mlp_hidden_compressor is None else adapter.legacy_mlp_hidden_compressor,
        depth_compress_layer_norm_and_bias=arguments.depth_compress_layer_norm_and_bias,                                                                   # <<< THOG CLI vector mode
        geometry_preset=None if arguments.hyperblock else (arguments.geometry_preset if adapter is None else adapter.legacy_geometry_preset),
        attention_geometry=None if arguments.hyperblock else (arguments.attention_geometry if adapter is None else None),
        mlp_geometry=None if arguments.hyperblock else (arguments.mlp_geometry if adapter is None else None),
        basis_family=None if arguments.hyperblock else (arguments.basis_family if adapter is None else adapter.legacy_basis_family),
        basis_version=basis_version,
        resolved_geometry_plan=(
            None
            if geometry_plan is None or dense_snapshot_target
            else geometry_plan.to_dict()
        ),
        save_dense_initialisation_snapshot=arguments.save_dense_initialisation_snapshot,
        initialise_from_dense_snapshot=arguments.initialise_from_dense_snapshot,
        dense_snapshot_chebyshev_order=snapshot_order,
        dense_snapshot_chebyshev_version=snapshot_version,
        # vvv THOG pass the sole v0 topology explicitly into resolved run identity
        hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE if arguments.hyperblock else None,
        hyperblock_compressor=arguments.hyperblock_compressor,
        hyperblock_compressor_version=arguments.hyperblock_compressor_version,
        hyperblock_common_family_order=arguments.hyperblock_common_family_order,
        hyperblock_attention_family_order=arguments.hyperblock_attention_family_order,
        hyperblock_mlp_family_order=arguments.hyperblock_mlp_family_order,
        hyperblock_depth_order=arguments.hyperblock_depth_order,
        hyperblock_d_model_order=arguments.hyperblock_d_model_order,
        hyperblock_mlp_hidden_order=arguments.hyperblock_mlp_hidden_order,
        hyperblock_attention_head_order=arguments.hyperblock_attention_head_order,
        hyperblock_attention_head_channel_order=arguments.hyperblock_attention_head_channel_order,
        hyperblock_mlp_hidden_multiplier=arguments.hyperblock_mlp_hidden_multiplier,
        hyperblock_loop_count=arguments.hyperblock_loop_count,
        hyperblock_loop_decay=arguments.hyperblock_loop_decay,
        # ^^^ THOG
        # vvv THOG pass PLASTIC DEPTH CLI controls into resolved run identity
        plastic__enabled=arguments.plastic__enabled,
        plastic__coarse_phase=arguments.plastic__coarse_phase,
        plastic__coarse_phase_roll_through=arguments.plastic__coarse_phase_roll_through,
        plastic__log_interval_coarse=arguments.plastic__log_interval_coarse,
        plastic__phase_1_n_steps=arguments.plastic__phase_1_n_steps,
        plastic__phase_1_starting_layer_count=arguments.plastic__phase_1_starting_layer_count,
        plastic__phase_1__number_of_trials=arguments.plastic__phase_1__number_of_trials,
        plastic__phase_1_evaluation_steps_count=arguments.plastic__phase_1_evaluation_steps_count,
        plastic__layers_to_sample=arguments.plastic__layers_to_sample,
        plastic__do_learn_layer_count=arguments.plastic__do_learn_layer_count,
        plastic__initial_layer_count=arguments.plastic__initial_layer_count,
        plastic__max_permitted_layers=arguments.plastic__max_permitted_layers,
        plastic__layer_sampling_initialisation=arguments.plastic__layer_sampling_initialisation,
        plastic__layer_count_objective=arguments.plastic__layer_count_objective,
        plastic__layer_count_update_brake=arguments.plastic__layer_count_update_brake,
        plastic__layer_count_probe__probe_every_n_steps=arguments.plastic__layer_count_probe__probe_every_n_steps,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=arguments.plastic__layer_count_probe__number_of_sampled_valid_tokens,
        plastic__layer_count_probe_radius=arguments.plastic__layer_count_probe_radius,
        plastic__layer_count__max_allowable_layer_change=arguments.plastic__layer_count__max_allowable_layer_change,
        plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence=arguments.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence,
        plastic__layer_count_probe__window_size_as_number_of_probes=arguments.plastic__layer_count_probe__window_size_as_number_of_probes,
        plastic__layer_count_probe_noise_lambda=arguments.plastic__layer_count_probe_noise_lambda,
        plastic__wall_time_equivalent_time_gain_discount=arguments.plastic__wall_time_equivalent_time_gain_discount,
        plastic__wall_time_equivalent_time_gain_loss_rate_window=arguments.plastic__wall_time_equivalent_time_gain_loss_rate_window,
        plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=arguments.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations,
        plastic__layer_count_cost_weight=arguments.plastic__layer_count_cost_weight,
        plastic__layer_count__memory_budget_gib=arguments.plastic__layer_count__memory_budget_gib,
        plastic__layer_count__cuda_allocator_reserve_gib=arguments.plastic__layer_count__cuda_allocator_reserve_gib,
        plastic__geometry_learning_rate_multiplier=arguments.plastic__geometry_learning_rate_multiplier,
        plastic__freeze_geometry_during_warmup=arguments.plastic__freeze_geometry_during_warmup,
        # ^^^ THOG
        # vvv THOG pass sampling-only chaos bump controls into resolved run identity
        chaos_bump__sampling__enabled=arguments.chaos_bump__sampling__enabled,
        chaos_bump__sampling__initial_lockout__steps=arguments.chaos_bump__sampling__initial_lockout__steps,
        chaos_bump__sampling__maximum_bumps=arguments.chaos_bump__sampling__maximum_bumps,
        chaos_bump__sampling__interlude__min_steps=arguments.chaos_bump__sampling__interlude__min_steps,
        chaos_bump__sampling__interlude__max_steps=arguments.chaos_bump__sampling__interlude__max_steps,
        chaos_bump__sampling__duration__min_steps=arguments.chaos_bump__sampling__duration__min_steps,
        chaos_bump__sampling__duration__max_steps=arguments.chaos_bump__sampling__duration__max_steps,
        chaos_bump__sampling__duration__max_fraction_of_elapsed_steps=arguments.chaos_bump__sampling__duration__max_fraction_of_elapsed_steps,
        chaos_bump__sampling__max_movement_fraction_of_local_gap=arguments.chaos_bump__sampling__max_movement_fraction_of_local_gap,
        # ^^^ THOG
        lapped_cosine_window_length=arguments.lapped_cosine_window_length,                                                                                 # <<< THOG CLI locality control
        lapped_cosine_overlap_fraction=arguments.lapped_cosine_overlap_fraction,                                                                           # <<< THOG CLI overlap control
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
        instrumentation__delta_loss_v_layer_heatmap=arguments.instrumentation__delta_loss_v_layer_heatmap,
        instrumentation__delta_loss_v_layer_heatmap__destination=arguments.instrumentation__delta_loss_v_layer_heatmap__destination,
        # Heatmap history length is an Instra viewer choice, not a capture limit.
        instrumentation__delta_loss_v_layer_heatmap_linear=None,
        instrumentation__delta_loss_v_layer_heatmap_abs_limit=arguments.instrumentation__delta_loss_v_layer_heatmap_abs_limit,
        instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes=arguments.instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes,
        artifact_suffix=arguments.artifact_suffix,
        artifact_name_limit=arguments.artifact_name_limit,
    )
    configure_attention_backend(config.attention_backend)
    return config


def resolved_payload(config: OwtRunConfig, *, world_size: int, log_timestamp: Optional[str]) -> Dict[str, Any]:
    paths = config.paths(log_timestamp=log_timestamp)
    return {"artifact_name": config.artifact_name, "artifact_prefix": config.artifact_prefix, "model_type": config.model_type, "run_mode": config.run_mode, "world_size": world_size, "tokens_per_iter": config.tokens_per_iter(), "canonical_config": config.canonical_dict(world_size=world_size), "paths": {name: str(path) for name, path in paths.items()}}


# vvv THOG print resolved model parameters and execution options immediately before training
def _print_model_option(label: str, value: str) -> None:
    print(f"  {label:<24} {value}", flush=True)


def print_model_parameters_and_options(config: OwtRunConfig, trainer: OwtTrainer) -> None:
    report = trainer.parameter_report
    persistent = int(report["persistent_parameters"])
    dense_equivalent = int(report["dense_equivalent_total_parameters"])
    sheet_coefficients = int(report["sheet_coefficients"])
    compression = (dense_equivalent / persistent) if persistent else 0.0
    print("model parameters and options", flush=True)
    _print_model_option("parameters:", f"persistent={persistent:,}  sheet coefficients={sheet_coefficients:,}  dense equivalent={dense_equivalent:,}  dense/persistent={compression:.2f}x")
    _print_model_option("optimiser:", f"lr={config.learning_rate:.3e}  min_lr={config.min_lr:.3e}  warmup={config.warmup_iters}  weight_decay={config.weight_decay:g}  grad_clip={config.grad_clip:g}")
    _print_model_option("wall stop:", f"max_wall_minutes={config.max_wall_minutes}")
    _print_model_option("non-finite:", f"policy={config.nonfinite_update_policy}  max_skips={config.max_nonfinite_update_skips}")
    _print_model_option("batches:", f"micro={config.batch_size}  accumulation={config.gradient_accumulation_steps}  tokens/update={config.tokens_per_iter():,}")
    _print_model_option("layer dropout:", f"strata={config.layer_dropout_n_strata}  stratum_size={config.layer_dropout_stratum_size}  active/stratum={config.layer_dropout_active_per_stratum}  active_layers={config.n_active_layers}/{config.n_layer}  resample_steps={config.layer_dropout_resample_steps}")
    # vvv THOG PLASTIC DEPTH startup report makes resolved count authority and public sample coordinates explicit
    if config.plastic__enabled:
        plastic_report = report.get("plastic_depth", {})
        count_mode = "learned" if config.plastic__do_learn_layer_count else "fixed"
        _print_model_option("plastic depth:", "enabled")
        _print_model_option(
            "plastic layer count:",
            f"{count_mode}  initial={config.plastic__initial_active_layers}  current={plastic_report.get('active_layers')}  max={config.n_layer}",
        )
        _print_model_option("plastic sampling:", config.plastic__layer_sampling_initialisation)
        _print_model_option(
            "plastic objective:",
            f"{config.plastic__layer_count_objective}  update_brake={config.plastic__layer_count_update_brake}  "
            f"noise_window={config.plastic__layer_count_probe__window_size_as_number_of_probes}  "
            f"probe_tokens={config.plastic__layer_count_probe__number_of_sampled_valid_tokens}  "
            f"lambda={float(config.plastic__layer_count_probe_noise_lambda):g}",
        )
        public_coordinates = plastic_report.get("active_public_coordinates", ())
        _print_model_option(
            "plastic samples:",
            ", ".join(f"{float(value):.3f}" for value in public_coordinates),
        )
        full_coordinates = plastic_report.get("public_coordinates", ())
        if tuple(full_coordinates) != tuple(public_coordinates):
            _print_model_option(
                "plastic full lattice:",
                ", ".join(f"{float(value):.3f}" for value in full_coordinates),
            )
    # ^^^ THOG
    if config.model_type == "sheet":
        model_config = trainer.raw_model.config
        # vvv THOG preserve the pre-HYPERBLOCK-direct execution row exactly for source history
        # _print_model_option("execution:", f"semantic_qkv_bypass={model_config.bypass_semantic_qkv_adapter}  vectorise_per_head={model_config.vectorise_per_head_materialisation}  direct_factorised_mlp={model_config.direct_factorised_mlp}  activation_checkpointing={config.activation_checkpointing}  depth_compress_layer_norm_and_bias={model_config.depth_compress_layer_norm_and_bias}")
        _print_model_option("execution:", f"semantic_qkv_bypass={model_config.bypass_semantic_qkv_adapter}  vectorise_per_head={model_config.vectorise_per_head_materialisation}  direct_factorised_mlp={model_config.direct_factorised_mlp}  direct_factorised_hyperblock_mlp={model_config.direct_factorised_hyperblock_mlp}  activation_checkpointing={config.activation_checkpointing}  depth_compress_layer_norm_and_bias={model_config.depth_compress_layer_norm_and_bias}")
        # ^^^ THOG
        # vvv THOG HYPERBLOCK field identity and coefficient budget are first-class console diagnostics
        hyperblock = report.get("hyperblock")
        if isinstance(hyperblock, dict):
            plan = hyperblock["plan"]
            _print_model_option(
                "HYPERBLOCK:",
                f"topology={plan['topology']}  compressor={plan['compressor_family']}@{plan['compressor_version']}  coefficients={plan['coefficient_counts']['total']:,}  matrix_dense/coefficient={hyperblock['compression_ratio']:.2f}x  loops={hyperblock['loop_count']}  loop_decay={hyperblock['loop_decay']:.6g}",
            )
        # ^^^ THOG
    print(flush=True)
# ^^^ THOG


def main() -> int:
    arguments = build_parser().parse_args()
    geometry_plan = geometry_plan_from_arguments(arguments)
    if arguments.explain_geometry:
        if geometry_plan is None:
            raise ValueError("--explain-geometry requires systematic geometry selections")
        print(format_geometry_plan(geometry_plan, detailed=True))
        return 0
    config = config_from_arguments(arguments, geometry_plan=geometry_plan)
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
    # vvv THOG preserve shell interrupt status while allowing W&B to record a clean intentional stop
    telemetry_exit_code: Optional[int] = None
    telemetry_final_state = "stopped"
    # ^^^ THOG
    try:
        if trainer.distributed.is_primary:
            telemetry.start()
            telemetry.add_initial_summary(trainer.parameter_report)
        attach_telemetry(trainer, telemetry)
        if trainer.distributed.is_primary:
            if geometry_plan is not None:
                print(format_geometry_plan(geometry_plan), flush=True)
                print(flush=True)
            print_model_parameters_and_options(config, trainer)                                                                                            # <<< THOG show the complete effective training setup before the first update
        result = trainer.run_pilot(run_id=config.artifact_name, protocol_sha256=run_digest(config, dataset, world_size), dataset=dataset, result_path=result_path)
        result["artifact"] = {"name": config.artifact_name, "prefix": config.artifact_prefix, "paths": {name: str(path) for name, path in paths.items()}}
        result["canonical_config"] = canonical
        result["source"] = source
        if trainer.distributed.is_primary:
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            telemetry.add_final_result(result)
            print(json.dumps({"artifact_name": config.artifact_name, "checkpoint": str(checkpoint_path), "result": str(result_path), "completed_updates": result["budget"]["completed_updates"], "consumed_tokens": result["budget"]["consumed_tokens"], "final_validation_loss": (result["evaluations"][-1]["val"] if result["evaluations"] else None)}, indent=2, sort_keys=True))
        telemetry_final_state = "finished"
        return 0
    # vvv THOG convert Ctrl-C into a clean telemetry finish while retaining conventional process status 130
    except KeyboardInterrupt:
        telemetry_exit_code = 0
        telemetry_final_state = "stopped"
        if trainer.distributed.is_primary:
            print("interrupted by Ctrl-C; finishing telemetry cleanly", flush=True)
        return 130
    finally:
        if rank == 0:
            telemetry.finish(
                exit_code=telemetry_exit_code,
                final_state=telemetry_final_state,
            )
        trainer.close()
    # ^^^ THOG


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
# vvv THOG preserved superseded source lines for exact history audit
# f"topology={plan['topology']}  compressor={plan['compressor_family']}@{plan['compressor_version']}  coefficients={plan['coefficient_counts']['total']:,}  matrix_dense/coefficient={hyperblock['compression_ratio']:.2f}x",
# ^^^ THOG

# vvv THOG retired PLASTIC DEPTH hold-controller source preserved for history audit
# parser.add_argument("--plastic-layer-count-hold-updates", dest="plastic__layer_count_hold_updates", type=int, default=100)
# plastic__layer_count_hold_updates=arguments.plastic__layer_count_hold_updates,
# f"{config.plastic__layer_count_objective}  hold_updates={config.plastic__layer_count_hold_updates}",
# ^^^ THOG
