# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .basis import BASIS_VERSION
from .bases import basis_artifact_tag_for_family, basis_version_for_family
from .compact_identity import BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_CONVENTIONAL, GEOMETRY_PRESET_DEPTH, compact_identity_metadata
from .residual_init import (
    DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE,
    DEFAULT_RESIDUAL_INIT_DEPTH_VALUE,
    DEFAULT_RESIDUAL_INIT_POLICY,
    ResidualInitConfig,
)
from .run_naming import DEFAULT_COMPONENT_LIMIT, artifact_paths, dataset_label, normalize_component, truncate_component
from .training_config import ROW_ORDER_SCALING_RULE, TrainingConfig


PUBLIC_MODEL_TYPES = ("dense", "sheet")
INTERNAL_MODEL_TYPES = {"dense": "dense", "sheet": "thog2_sheet"}
# vvv THOG basis artifact tags now come from the basis-family registry
# ^^^ THOG
DEFAULT_O_ATTN_D_MODEL = 64
DEFAULT_O_ATTN_QKV_PER_CHANNEL = 6
DEFAULT_O_ATTN_OUT_PER_CHANNEL = 6
DEFAULT_O_MLP_D_MODEL = 64
DEFAULT_O_MLP_HIDDEN = 256
DEFAULT_MLP_CHANNEL_ORDER = DEFAULT_O_MLP_HIDDEN                                                                                                      # <<< THOG retained module constant name for callers while public configuration uses o_mlp_hidden
DEFAULT_EXPERIMENT_PREFIX = "NEL" + "SON"


@dataclass(frozen=True)
class OwtRunConfig:
    model_type: str
    run_mode: str = "fresh"
    host_label: str = "scruffy"
    run_name: str = "AKAROA"
    dataset: str = "openwebtext"
    data_dir: str = "data/openwebtext"
    checkpoint_root: str = "checkpoints"
    log_root: str = "logs"
    result_root: str = "results"
    wandb_root: str = "wandb"

    max_iters: int = 100
    # vvv THOG optional wall-clock stop for equal-time geometry grids
    max_wall_minutes: int = 0
    # ^^^ THOG
    eval_interval: int = 50
    eval_iters: int = 5
    log_interval: int = 10
    checkpoint_interval: int = 0

    batch_size: int = 12
    gradient_accumulation_steps: int = 160
    block_size: int = 256
    n_layer: int = 72
    n_head: int = 12
    n_embd: int = 768
    o_depth: int = 16
    o_attn_d_model: int = DEFAULT_O_ATTN_D_MODEL
    o_attn_qkv_per_channel: int = DEFAULT_O_ATTN_QKV_PER_CHANNEL
    o_attn_out_per_channel: int = DEFAULT_O_ATTN_OUT_PER_CHANNEL
    o_mlp_d_model: int = DEFAULT_O_MLP_D_MODEL
    o_mlp_hidden: int = DEFAULT_O_MLP_HIDDEN

    geometry_preset: Optional[str] = GEOMETRY_PRESET_DEPTH
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = BASIS_FAMILY_CHEBYSHEV
    basis_version: str = BASIS_VERSION
    attention_backend: str = "auto"
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX
    run_start_label: Optional[str] = None

    residual_init_policy: str = DEFAULT_RESIDUAL_INIT_POLICY
    residual_init_depth_source: str = DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE
    residual_init_depth_value: int = DEFAULT_RESIDUAL_INIT_DEPTH_VALUE

    activation_checkpointing: bool = True
    checkpoint_segment_size: int = 12

    learning_rate: float = 6.0e-4
    min_lr: float = 6.0e-5
    warmup_iters: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # vvv THOG public bounded non-finite recovery controls
    nonfinite_update_policy: str = "skip"
    max_nonfinite_update_skips: int = 10
    # ^^^ THOG
    dropout: float = 0.0
    bias: bool = True

    model_seed: int = 1337
    data_seed: int = 7331
    device: str = "cuda"
    dtype: str = "bfloat16"

    wandb_enabled: bool = True
    wandb_project: str = "thog"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"

    artifact_suffix: Optional[str] = None
    artifact_name_limit: int = DEFAULT_COMPONENT_LIMIT

    def __post_init__(self) -> None:
        if self.model_type not in PUBLIC_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {PUBLIC_MODEL_TYPES}")
        if self.run_mode not in ("fresh", "resume"):
            raise ValueError("run_mode must be fresh or resume")
        if self.attention_backend not in ("auto", "flash2", "sdpa", "math"):
            raise ValueError("attention_backend must be auto, flash2, sdpa, or math")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.wandb_mode == "disabled" and self.wandb_enabled:
            object.__setattr__(self, "wandb_enabled", False)
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 training is not supported on CPU")

        object.__setattr__(self, "experiment_prefix", normalize_component(self.experiment_prefix, uppercase=True))
        if self.run_start_label is not None:
            object.__setattr__(self, "run_start_label", normalize_component(self.run_start_label))
        if self.basis_version == "auto":
            requested_family = self.basis_family or BASIS_FAMILY_CHEBYSHEV
            resolved_version = BASIS_VERSION if requested_family == BASIS_FAMILY_CONVENTIONAL else basis_version_for_family(requested_family)
            object.__setattr__(self, "basis_version", resolved_version)

        positive = (
            "max_iters",
            "eval_interval",
            "eval_iters",
            "log_interval",
            "batch_size",
            "gradient_accumulation_steps",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "checkpoint_segment_size",
            "artifact_name_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_wall_minutes", "checkpoint_interval", "warmup_iters", "model_seed", "data_seed", "max_nonfinite_update_skips"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be less than max_iters")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.model_type == "sheet":
            order_limits = {
                "o_depth": self.n_layer,
                "o_attn_d_model": self.n_embd,
                "o_attn_qkv_per_channel": self.head_dim,
                "o_attn_out_per_channel": self.head_dim,
                "o_mlp_d_model": self.n_embd,
                "o_mlp_hidden": 4 * self.n_embd,
            }
            for name, limit in order_limits.items():
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
                if value > limit:
                    raise ValueError(f"{name} must not exceed {limit}")
            object.__setattr__(self, "basis_version", str(self.compact_identity()["basis_version"]))
        residual_init = self.residual_init_config()
        object.__setattr__(self, "residual_init_depth_source", residual_init.depth_source)
        if self.model_type == "dense" and residual_init.depth_source == "dof_implied_depth":
            raise ValueError("dof_implied_depth residual init is only defined for SHEET")
        if not self.activation_checkpointing and self.checkpoint_segment_size < 1:
            raise ValueError("checkpoint_segment_size must remain positive")
        if self.learning_rate <= 0.0 or self.min_lr < 0.0:
            raise ValueError("learning rates must be non-negative and maximum must be positive")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr must not exceed learning_rate")
        if self.nonfinite_update_policy not in ("raise", "skip"):
            raise ValueError("nonfinite_update_policy must be raise or skip")
        if self.weight_decay < 0.0 or self.grad_clip < 0.0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(
            policy=self.residual_init_policy,
            depth_source=self.residual_init_depth_source,
            depth_value=self.residual_init_depth_value,
        )

    def residual_init_artifact_fragment(self) -> str:
        residual_init = self.residual_init_config()
        parts = [f"r_{self.residual_init_policy}", f"z_{residual_init.depth_source}"]
        if residual_init.depth_source == "user_forced_depth":
            parts.append(f"Z_{self.residual_init_depth_value}")
        return "_".join(parts)

    @property
    def internal_model_type(self) -> str:
        return INTERNAL_MODEL_TYPES[self.model_type]

    @property
    def artifact_prefix(self) -> str:
        return "DENSE2" if self.model_type == "dense" else "SHEET"

    def compact_identity(self) -> Dict[str, Any]:
        if self.model_type != "sheet":
            raise ValueError("compact identity is only defined for SHEET runs")
        return compact_identity_metadata(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            o_depth=self.o_depth,
            o_attn_d_model=self.o_attn_d_model,
            o_attn_qkv_per_channel=self.o_attn_qkv_per_channel,
            o_attn_out_per_channel=self.o_attn_out_per_channel,
            o_mlp_d_model=self.o_mlp_d_model,
            o_mlp_hidden=self.o_mlp_hidden,
            basis_version=self.basis_version,
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )

    def compact_artifact_fragment(self) -> Optional[str]:
        if self.model_type != "sheet":
            return None
        identity = self.compact_identity()
        basis_label = basis_artifact_tag_for_family(str(identity["basis_family"]))
        preset_label = str(identity["geometry_preset"]).upper()
        return f"{basis_label}_{preset_label}"

    def run_descriptor(self) -> str:
        model_fragment = self.compact_artifact_fragment() or "DENSE"
        body = f"{self.experiment_prefix}_{model_fragment}_{normalize_component(self.host_label)}"
        return f"{self.run_start_label}_{body}" if self.run_start_label else body

    def parameter_artifact_fragment(self) -> str:
        fields = [
            # vvv THOG stable artifact identity must not depend on the mutable target update count
            # f"n_{self.max_iters}",
            # ^^^ THOG
            f"b_{self.batch_size}",
            f"LR_{int(round(self.learning_rate / 1.0e-5)):02d}",                                                                                           # <<< THOG compact learning-rate code with e-04 convention left to the user
            f"d_{dataset_label(self.dataset)}",
            f"w_{self.warmup_iters}",
            f"k_{self.checkpoint_interval}",
            f"A_{self.gradient_accumulation_steps}",
            f"L_{self.n_layer}",
            f"H_{self.n_head}",
            f"D_{self.n_embd}",
            f"C_{self.block_size}",
        ]
        # vvv THOG include wall-clock budget in fresh-run identity only when active
        if self.max_wall_minutes > 0:
            fields.append(f"M_{self.max_wall_minutes}")
        # ^^^ THOG
        if self.model_type == "sheet":
            fields.extend([
                f"P_{self.o_depth}",
                f"Q_{self.o_attn_d_model}",
                f"J_{self.o_attn_qkv_per_channel}",
                f"O_{self.o_attn_out_per_channel}",
                f"X_{self.o_mlp_d_model}",
                f"Y_{self.o_mlp_hidden}",
            ])
        fields.append(self.residual_init_artifact_fragment())
        fields.append(f"S_{self.checkpoint_segment_size}")
        return "_".join(fields)

    @property
    def artifact_name(self) -> str:
        artifact_name = f"{self.run_descriptor()}__{self.parameter_artifact_fragment()}"
        if self.artifact_suffix:
            artifact_name = f"{artifact_name}__{normalize_component(self.artifact_suffix, uppercase=True)}"
        return truncate_component(artifact_name, max_length=self.artifact_name_limit)

    def paths(self, *, log_timestamp: Optional[str] = None) -> Dict[str, Path]:
        timestamp = None if self.run_start_label else log_timestamp
        return artifact_paths(
            self.artifact_name,
            checkpoint_root=Path(self.checkpoint_root),
            log_root=Path(self.log_root),
            result_root=Path(self.result_root),
            log_timestamp=timestamp,
        )

    def local_gradient_accumulation_steps(self, world_size: int) -> int:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if self.gradient_accumulation_steps % world_size != 0:
            raise ValueError("global gradient_accumulation_steps must be divisible by world_size")
        return self.gradient_accumulation_steps // world_size

    def tokens_per_iter(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.block_size

    def to_training_config(self, *, vocab_size: int, world_size: int, out_dir: Path) -> TrainingConfig:
        sheet_kwargs: Dict[str, Any]
        if self.model_type == "sheet":
            sheet_kwargs = {
                "depth_order": self.o_depth,
                "base_row_order": self.o_attn_d_model,
                "mlp_channel_order": self.o_mlp_hidden,
                "o_attn_d_model": self.o_attn_d_model,
                "o_attn_qkv_per_channel": self.o_attn_qkv_per_channel,
                "o_attn_out_per_channel": self.o_attn_out_per_channel,
                "o_mlp_d_model": self.o_mlp_d_model,
                "o_mlp_hidden": self.o_mlp_hidden,
                "basis_version": self.basis_version,
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
            }
        else:
            sheet_kwargs = {
                "depth_order": 1,
                "base_row_order": 1,
                "mlp_channel_order": None,
                "o_attn_d_model": None,
                "o_attn_qkv_per_channel": None,
                "o_attn_out_per_channel": None,
                "o_mlp_d_model": None,
                "o_mlp_hidden": None,
                "basis_version": BASIS_VERSION,
                "geometry_preset": None,
                "attention_geometry": None,
                "mlp_geometry": None,
                "basis_family": None,
            }
        return TrainingConfig(
            model_type=self.internal_model_type,
            block_size=self.block_size,
            vocab_size=vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            **sheet_kwargs,
            residual_init_policy=self.residual_init_policy,
            residual_init_depth_source=self.residual_init_depth_source,
            residual_init_depth_value=self.residual_init_depth_value,
            checkpoint_segment_size=self.checkpoint_segment_size if self.activation_checkpointing else 0,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.local_gradient_accumulation_steps(world_size),
            max_updates=self.max_iters,
            max_wall_minutes=self.max_wall_minutes,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_lr,
            warmup_updates=self.warmup_iters,
            decay_updates=self.max_iters,
            decay_learning_rate=True,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
            nonfinite_update_policy=self.nonfinite_update_policy,
            max_nonfinite_update_skips=self.max_nonfinite_update_skips,
            eval_interval=self.eval_interval,
            eval_batches=self.eval_iters,
            checkpoint_interval=self.checkpoint_interval,
            log_interval=self.log_interval,
            model_seed=self.model_seed,
            data_seed=self.data_seed,
            device=self.device,
            dtype=self.dtype,
            out_dir=str(out_dir),
        )

    def canonical_dict(self, *, world_size: int) -> Dict[str, Any]:
        values = asdict(self)
        if self.model_type == "dense":
            for name in (
                "o_depth",
                "o_attn_d_model",
                "o_attn_qkv_per_channel",
                "o_attn_out_per_channel",
                "o_mlp_d_model",
                "o_mlp_hidden",
                "geometry_preset",
                "attention_geometry",
                "mlp_geometry",
                "basis_family",
                "basis_version",
            ):
                values.pop(name, None)
        else:
            values["compact_identity"] = self.compact_identity()
            values["compact_artifact_fragment"] = self.compact_artifact_fragment()
        values.update({
            "artifact_name": self.artifact_name,
            "artifact_prefix": self.artifact_prefix,
            "run_descriptor": self.run_descriptor(),
            "world_size": world_size,
            "local_gradient_accumulation_steps": self.local_gradient_accumulation_steps(world_size),
            "tokens_per_iter": self.tokens_per_iter(),
        })
        return values


__all__ = [
    "DEFAULT_EXPERIMENT_PREFIX",
    "DEFAULT_MLP_CHANNEL_ORDER",
    "DEFAULT_O_ATTN_D_MODEL",
    "DEFAULT_O_ATTN_QKV_PER_CHANNEL",
    "DEFAULT_O_ATTN_OUT_PER_CHANNEL",
    "DEFAULT_O_MLP_D_MODEL",
    "DEFAULT_O_MLP_HIDDEN",
    "INTERNAL_MODEL_TYPES",
    "OwtRunConfig",
    "PUBLIC_MODEL_TYPES",
]
# ^^^ THOG