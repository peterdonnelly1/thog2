# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .basis import BASIS_VERSION
from .checkpointing import validate_checkpoint_segment_size
from .compact_identity import compact_identity_metadata, conventional_identity_metadata, validate_dense_compact_fields
from .geometry import derive_row_order
from .residual_init import DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE, DEFAULT_RESIDUAL_INIT_DEPTH_VALUE, DEFAULT_RESIDUAL_INIT_POLICY, ResidualInitConfig


CHECKPOINT_SCHEMA_VERSION = 2
ROW_ORDER_SCALING_RULE = "proportional_ceil_v1"
MODEL_TYPES = ("dense", "thog2_sheet")
EXECUTION_OVERRIDE_FIELDS = {"device", "dtype", "max_updates", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval"}
MODEL_COMPATIBILITY_FIELDS = (
    "model_type",
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "depth_order",
    "base_row_order",
    "mlp_channel_order",
    "o_attn_d_model",
    "o_attn_qkv_per_channel",
    "o_attn_out_per_channel",
    "o_mlp_d_model",
    "o_mlp_hidden",
    "residual_init_policy",
    "residual_init_depth_source",
    "residual_init_depth_value",
    "basis_version",
    "row_order_scaling_rule",
    "geometry_preset",
    "attention_geometry",
    "mlp_geometry",
    "basis_family",
)


@dataclass
class TrainingConfig:
    model_type: str = "dense"
    block_size: int = 128
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    depth_order: int = 4
    base_row_order: int = 32
    mlp_channel_order: Optional[int] = None
    o_attn_d_model: Optional[int] = None                                                                                                               # <<< THOG final attention model-axis order
    o_attn_qkv_per_channel: Optional[int] = None                                                                                                       # <<< THOG final QKV per-head channel order
    o_attn_out_per_channel: Optional[int] = None                                                                                                       # <<< THOG final output per-head channel order
    o_mlp_d_model: Optional[int] = None                                                                                                                # <<< THOG final MLP model-axis order
    o_mlp_hidden: Optional[int] = None                                                                                                                 # <<< THOG final MLP hidden-axis order
    residual_init_policy: str = DEFAULT_RESIDUAL_INIT_POLICY
    residual_init_depth_source: str = DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE
    residual_init_depth_value: int = DEFAULT_RESIDUAL_INIT_DEPTH_VALUE
    basis_version: str = BASIS_VERSION
    row_order_scaling_rule: str = ROW_ORDER_SCALING_RULE
    geometry_preset: Optional[str] = None
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = None
    checkpoint_segment_size: int = 0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_updates: int = 10
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    warmup_updates: int = 0
    decay_updates: int = 10
    decay_learning_rate: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 0
    eval_batches: int = 1
    checkpoint_interval: int = 0
    log_interval: int = 1
    model_seed: int = 1337
    data_seed: int = 7331
    device: str = "cpu"
    dtype: str = "float32"
    out_dir: str = "out-thog2"

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}; got {self.model_type!r}")
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd", "depth_order", "base_row_order", "batch_size", "gradient_accumulation_steps", "max_updates", "decay_updates", "eval_batches", "log_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        optional_positive = (
            "mlp_channel_order",
            "o_attn_d_model",
            "o_attn_qkv_per_channel",
            "o_attn_out_per_channel",
            "o_mlp_d_model",
            "o_mlp_hidden",
        )
        for name in optional_positive:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None; got {value!r}")
        for name in ("warmup_updates", "eval_interval", "checkpoint_interval", "model_seed", "data_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
        validate_checkpoint_segment_size(self.checkpoint_segment_size)
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd must be divisible by n_head; got {self.n_embd} and {self.n_head}")
        if self.depth_order > self.n_layer:
            raise ValueError("depth_order must not exceed n_layer")
        if self.base_row_order > self.n_embd:
            raise ValueError("base_row_order must not exceed n_embd")
        if self.mlp_channel_order is not None and self.mlp_channel_order > 4 * self.n_embd:
            raise ValueError("mlp_channel_order must not exceed 4*n_embd")
        limits = {
            "o_attn_d_model": self.n_embd,
            "o_attn_qkv_per_channel": self.head_dim,
            "o_attn_out_per_channel": self.head_dim,
            "o_mlp_d_model": self.n_embd,
            "o_mlp_hidden": 4 * self.n_embd,
        }
        for name, limit in limits.items():
            value = getattr(self, name)
            if value is not None and value > limit:
                raise ValueError(f"{name} must not exceed {limit}")
        residual_init = self.residual_init_config()
        self.residual_init_depth_source = residual_init.depth_source
        if self.model_type == "dense" and residual_init.depth_source == "dof_implied_depth":
            raise ValueError("dof_implied_depth residual init is only defined for SHEET")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.min_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative and maximum must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.weight_decay < 0.0 or self.grad_clip < 0.0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 training is not supported on CPU")
        if self.row_order_scaling_rule != ROW_ORDER_SCALING_RULE:
            raise ValueError(f"unsupported row_order_scaling_rule: {self.row_order_scaling_rule}")
        if self.model_type == "dense":
            if self.basis_version != BASIS_VERSION:
                raise ValueError(f"unsupported dense basis_version: {self.basis_version}")
            validate_dense_compact_fields(
                geometry_preset=self.geometry_preset,
                attention_geometry=self.attention_geometry,
                mlp_geometry=self.mlp_geometry,
                basis_family=self.basis_family,
            )
        else:
            identity = self.compact_identity_metadata()
            self.basis_version = str(identity["basis_version"])
        if not isinstance(self.bias, bool) or not isinstance(self.decay_learning_rate, bool):
            raise ValueError("bias and decay_learning_rate must be bool")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def resolved_o_attn_d_model(self) -> int:
        return self.base_row_order if self.o_attn_d_model is None else self.o_attn_d_model

    @property
    def resolved_o_attn_qkv_per_channel(self) -> int:
        return derive_row_order(self.head_dim, self.n_embd, self.base_row_order) if self.o_attn_qkv_per_channel is None else self.o_attn_qkv_per_channel

    @property
    def resolved_o_attn_out_per_channel(self) -> int:
        return derive_row_order(self.head_dim, self.n_embd, self.base_row_order) if self.o_attn_out_per_channel is None else self.o_attn_out_per_channel

    @property
    def resolved_o_mlp_d_model(self) -> int:
        return self.base_row_order if self.o_mlp_d_model is None else self.o_mlp_d_model

    @property
    def resolved_o_mlp_hidden(self) -> int:
        if self.o_mlp_hidden is not None:
            return self.o_mlp_hidden
        if self.mlp_channel_order is not None:
            return self.mlp_channel_order
        return derive_row_order(4 * self.n_embd, self.n_embd, self.base_row_order)

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(policy=self.residual_init_policy, depth_source=self.residual_init_depth_source, depth_value=self.residual_init_depth_value)

    def model_arguments(self) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
        }
        if self.model_type == "thog2_sheet":
            arguments.update({
                "depth_order": self.depth_order,
                "base_row_order": self.base_row_order,
                "mlp_channel_order": self.mlp_channel_order,
                "o_attn_d_model": self.resolved_o_attn_d_model,
                "o_attn_qkv_per_channel": self.resolved_o_attn_qkv_per_channel,
                "o_attn_out_per_channel": self.resolved_o_attn_out_per_channel,
                "o_mlp_d_model": self.resolved_o_mlp_d_model,
                "o_mlp_hidden": self.resolved_o_mlp_hidden,
                "basis_version": self.basis_version,
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
            })
        return arguments

    def compact_identity_metadata(self) -> Dict[str, Any]:
        if self.model_type == "dense":
            return conventional_identity_metadata(n_layer=self.n_layer, n_embd=self.n_embd, n_head=self.n_head)
        return compact_identity_metadata(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            o_depth=self.depth_order,
            o_attn_d_model=self.resolved_o_attn_d_model,
            o_attn_qkv_per_channel=self.resolved_o_attn_qkv_per_channel,
            o_attn_out_per_channel=self.resolved_o_attn_out_per_channel,
            o_mlp_d_model=self.resolved_o_mlp_d_model,
            o_mlp_hidden=self.resolved_o_mlp_hidden,
            basis_version=self.basis_version,
            row_order_scaling_rule=self.row_order_scaling_rule,
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )

    def compatibility_signature(self) -> Dict[str, Any]:
        values = asdict(self)
        return {name: values[name] for name in MODEL_COMPATIBILITY_FIELDS}
# ^^^ THOG
