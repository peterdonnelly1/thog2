# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .instrumentation import INSTRUMENTATION_BACKENDS
from .residual_init import (
    DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE,
    DEFAULT_RESIDUAL_INIT_DEPTH_VALUE,
    DEFAULT_RESIDUAL_INIT_POLICY,
    ResidualInitConfig,
)
from .run_naming import DEFAULT_COMPONENT_LIMIT, artifact_paths, build_artifact_name
from .training_config import TrainingConfig


PUBLIC_MODEL_TYPES = ("dense", "sheet")
INTERNAL_MODEL_TYPES = {
    "dense": "dense",
    "sheet": "thog2_sheet",
}


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
    curve_root: str = "curves"
    wandb_root: str = "curves/wandb"

    max_iters: int = 100
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
    depth_order: int = 16
    base_row_order: int = 64

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
    dropout: float = 0.0
    bias: bool = True

    model_seed: int = 1337
    data_seed: int = 7331
    device: str = "cuda"
    dtype: str = "bfloat16"

    instrumentation: str = "tensorboard"
    system_log_interval: int = 10
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
        if self.instrumentation not in INSTRUMENTATION_BACKENDS:
            raise ValueError(f"instrumentation must be one of {INSTRUMENTATION_BACKENDS}")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.instrumentation == "wandb" and (
            self.wandb_mode == "disabled" or not self.wandb_enabled
        ):
            object.__setattr__(self, "instrumentation", "none")
        if self.instrumentation != "wandb":
            object.__setattr__(self, "wandb_enabled", False)
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 training is not supported on CPU")

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
        for name in (
            "checkpoint_interval",
            "warmup_iters",
            "model_seed",
            "data_seed",
            "system_log_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be less than max_iters")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.model_type == "sheet":
            for name in ("depth_order", "base_row_order"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if self.depth_order > self.n_layer:
                raise ValueError("depth_order must not exceed n_layer")
            if self.base_row_order > self.n_embd:
                raise ValueError("base_row_order must not exceed n_embd")
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
        if self.weight_decay < 0.0 or self.grad_clip < 0.0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(
            policy=self.residual_init_policy,
            depth_source=self.residual_init_depth_source,
            depth_value=self.residual_init_depth_value,
        )

    @property
    def internal_model_type(self) -> str:
        return INTERNAL_MODEL_TYPES[self.model_type]

    @property
    def artifact_prefix(self) -> str:
        return "DENSE2" if self.model_type == "dense" else "SHEET"

    @property
    def artifact_name(self) -> str:
        return build_artifact_name(
            model_type=self.internal_model_type,
            host_label=self.host_label,
            run_name=self.run_name,
            dataset_name=self.dataset,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_iters=self.max_iters,
            checkpoint_interval=self.checkpoint_interval,
            warmup_iters=self.warmup_iters,
            checkpoint_segment_size=self.checkpoint_segment_size,
            depth_order=self.depth_order if self.model_type == "sheet" else None,
            base_row_order=self.base_row_order if self.model_type == "sheet" else None,
            suffix=self.artifact_suffix,
            max_length=self.artifact_name_limit,
        )

    def paths(self, *, log_timestamp: Optional[str] = None) -> Dict[str, Path]:
        paths = artifact_paths(
            self.artifact_name,
            checkpoint_root=Path(self.checkpoint_root),
            log_root=Path(self.log_root),
            result_root=Path(self.result_root),
            log_timestamp=log_timestamp,
        )
        curve_backend = self.instrumentation
        paths["curve_dir"] = Path(self.curve_root) / curve_backend / self.artifact_name
        return paths

    def local_gradient_accumulation_steps(self, world_size: int) -> int:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if self.gradient_accumulation_steps % world_size != 0:
            raise ValueError(
                "global gradient_accumulation_steps must be divisible by world_size"
            )
        return self.gradient_accumulation_steps // world_size

    def tokens_per_iter(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.block_size

    def to_training_config(
        self,
        *,
        vocab_size: int,
        world_size: int,
        out_dir: Path,
    ) -> TrainingConfig:
        return TrainingConfig(
            model_type=self.internal_model_type,
            block_size=self.block_size,
            vocab_size=vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            depth_order=(self.depth_order if self.model_type == "sheet" else 1),
            base_row_order=(self.base_row_order if self.model_type == "sheet" else 1),
            residual_init_policy=self.residual_init_policy,
            residual_init_depth_source=self.residual_init_depth_source,
            residual_init_depth_value=self.residual_init_depth_value,
            checkpoint_segment_size=(
                self.checkpoint_segment_size if self.activation_checkpointing else 0
            ),
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.local_gradient_accumulation_steps(world_size),
            max_updates=self.max_iters,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_lr,
            warmup_updates=self.warmup_iters,
            decay_updates=self.max_iters,
            decay_learning_rate=True,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
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
            values.pop("depth_order", None)
            values.pop("base_row_order", None)
        values.update({
            "artifact_name": self.artifact_name,
            "artifact_prefix": self.artifact_prefix,
            "world_size": world_size,
            "local_gradient_accumulation_steps": self.local_gradient_accumulation_steps(
                world_size
            ),
            "tokens_per_iter": self.tokens_per_iter(),
        })
        return values


__all__ = ["INTERNAL_MODEL_TYPES", "OwtRunConfig", "PUBLIC_MODEL_TYPES"]
# ^^^ THOG
