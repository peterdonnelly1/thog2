# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from .basis import BASIS_VERSION
from .checkpointing import validate_checkpoint_segment_size


CHECKPOINT_SCHEMA_VERSION = 1
ROW_ORDER_SCALING_RULE = "proportional_ceil_v1"
MODEL_TYPES = ("dense", "thog2_sheet")
EXECUTION_OVERRIDE_FIELDS = {
    "device",
    "dtype",
    "max_updates",
    "eval_interval",
    "eval_batches",
    "checkpoint_interval",
    "checkpoint_segment_size",
    "out_dir",
    "log_interval",
}
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
    "basis_version",
    "row_order_scaling_rule",
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
    basis_version: str = BASIS_VERSION
    row_order_scaling_rule: str = ROW_ORDER_SCALING_RULE
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
            raise ValueError(
                f"model_type must be one of {MODEL_TYPES}; got {self.model_type!r}"
            )
        for name in (
            "block_size",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_embd",
            "depth_order",
            "base_row_order",
            "batch_size",
            "gradient_accumulation_steps",
            "max_updates",
            "decay_updates",
            "eval_batches",
            "log_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        for name in (
            "warmup_updates",
            "eval_interval",
            "checkpoint_interval",
            "model_seed",
            "data_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
        validate_checkpoint_segment_size(self.checkpoint_segment_size)
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd must be divisible by n_head; got {self.n_embd} and {self.n_head}"
            )
        if self.depth_order > self.n_layer:
            raise ValueError("depth_order must not exceed n_layer")
        if self.base_row_order > self.n_embd:
            raise ValueError("base_row_order must not exceed n_embd")
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
            raise ValueError(
                f"unsupported row_order_scaling_rule: {self.row_order_scaling_rule}"
            )
        if self.basis_version != BASIS_VERSION:
            raise ValueError(f"unsupported basis_version: {self.basis_version}")
        if not isinstance(self.bias, bool) or not isinstance(self.decay_learning_rate, bool):
            raise ValueError("bias and decay_learning_rate must be bool")

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
            arguments.update(
                {
                    "depth_order": self.depth_order,
                    "base_row_order": self.base_row_order,
                    "basis_version": self.basis_version,
                }
            )
        return arguments

    def compatibility_signature(self) -> Dict[str, Any]:
        values = asdict(self)
        return {name: values[name] for name in MODEL_COMPATIBILITY_FIELDS}
# ^^^ THOG
