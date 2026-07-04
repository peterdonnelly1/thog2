# vvv THOG
from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .basis import BASIS_VERSION
from .geometry import SheetGeometryConfig
from .residual_init import (
    DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE,
    DEFAULT_RESIDUAL_INIT_DEPTH_VALUE,
    DEFAULT_RESIDUAL_INIT_POLICY,
    ResidualInitConfig,
)
from .trajectory import SheetTrajectory


class ConventionalLayerNorm(nn.Module):
    def __init__(self, width: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        return F.layer_norm(
            inputs,
            self.weight.shape,
            self.weight,
            self.bias,
            1.0e-5,
        )


@dataclass
class SheetGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    depth_order: int = 12
    base_row_order: int = 128
    residual_init_policy: str = DEFAULT_RESIDUAL_INIT_POLICY
    residual_init_depth_source: str = DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE
    residual_init_depth_value: int = DEFAULT_RESIDUAL_INIT_DEPTH_VALUE
    basis_version: str = BASIS_VERSION

    def __post_init__(self) -> None:
        for name in ("block_size", "vocab_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if not isinstance(self.basis_version, str) or not self.basis_version.strip():
            raise ValueError("basis_version must be a non-empty string")
        self.residual_init_config()
        self.sheet_geometry()

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(
            policy=self.residual_init_policy,
            depth_source=self.residual_init_depth_source,
            depth_value=self.residual_init_depth_value,
        )

    def sheet_geometry(self) -> SheetGeometryConfig:
        return SheetGeometryConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            depth_order=self.depth_order,
            base_row_order=self.base_row_order,
            bias=self.bias,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SheetGPT(nn.Module):
    """Sequential correctness-first GPT using compact Chebyshev Sheet weights."""

    def __init__(self, config: SheetGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "ln_f": ConventionalLayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.trajectory = SheetTrajectory(
            config.sheet_geometry(),
            runtime_dtype=torch.float32,
            basis_version=config.basis_version,
            residual_init_config=config.residual_init_config(),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_conventional_weights)
        self.trajectory.reset_parameters()

    @staticmethod
    def _init_conventional_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _optional_bias(self, name: str, layer_index: int) -> Optional[Tensor]:
        if not self.config.bias:
            return None
        return self.trajectory.materialize_vector(name, layer_index)

    def _sheet_layer_norm(
        self,
        inputs: Tensor,
        weight_name: str,
        bias_name: str,
        layer_index: int,
    ) -> Tensor:
        weight = self.trajectory.materialize_vector(weight_name, layer_index)
        bias = self._optional_bias(bias_name, layer_index)
        return F.layer_norm(inputs, (self.config.n_embd,), weight, bias, 1.0e-5)

    def _attention(self, inputs: Tensor, layer_index: int) -> Tensor:
        batch_size, sequence_length, embedding_width = inputs.shape
        attention_weight = self.trajectory.materialize(
            "attention_input_weight", layer_index
        )
        attention_bias = self._optional_bias("attention_input_bias", layer_index)
        query, key, value = F.linear(
            inputs,
            attention_weight,
            attention_bias,
        ).split(self.config.n_embd, dim=2)

        head_width = embedding_width // self.config.n_head
        key = key.view(
            batch_size,
            sequence_length,
            self.config.n_head,
            head_width,
        ).transpose(1, 2)
        query = query.view(
            batch_size,
            sequence_length,
            self.config.n_head,
            head_width,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.config.n_head,
            head_width,
        ).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_width))
            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=inputs.device,
                )
            )
            scores = scores.masked_fill(
                ~causal_mask.view(1, 1, sequence_length, sequence_length),
                float("-inf"),
            )
            probabilities = F.softmax(scores, dim=-1)
            probabilities = F.dropout(
                probabilities,
                p=self.config.dropout,
                training=self.training,
            )
            attended = probabilities @ value

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_width,
        )
        output_weight = self.trajectory.materialize(
            "attention_output_weight", layer_index
        )
        output_bias = self._optional_bias("attention_output_bias", layer_index)
        projected = F.linear(attended, output_weight, output_bias)
        return F.dropout(
            projected,
            p=self.config.dropout,
            training=self.training,
        )

    def _mlp(self, inputs: Tensor, layer_index: int) -> Tensor:
        expansion_weight = self.trajectory.materialize(
            "mlp_expansion_weight", layer_index
        )
        expansion_bias = self._optional_bias("mlp_expansion_bias", layer_index)
        hidden = F.linear(inputs, expansion_weight, expansion_bias)
        hidden = F.gelu(hidden)
        contraction_weight = self.trajectory.materialize(
            "mlp_contraction_weight", layer_index
        )
        contraction_bias = self._optional_bias("mlp_contraction_bias", layer_index)
        output = F.linear(hidden, contraction_weight, contraction_bias)
        return F.dropout(output, p=self.config.dropout, training=self.training)

    def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:
        normalized_attention = self._sheet_layer_norm(
            inputs,
            "ln_1_weight",
            "ln_1_bias",
            layer_index,
        )
        inputs = inputs + self._attention(normalized_attention, layer_index)
        normalized_mlp = self._sheet_layer_norm(
            inputs,
            "ln_2_weight",
            "ln_2_bias",
            layer_index,
        )
        return inputs + self._mlp(normalized_mlp, layer_index)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.config.block_size}"
            )
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=idx.device,
        )
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        for layer_index in range(self.config.n_layer):
            hidden = self._logical_block(hidden, layer_index)
        hidden = self.transformer.ln_f(hidden)

        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("cannot increase block size when cropping")
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        parameter_dict = {name: param for name, param in self.named_parameters()}
        parameter_dict = {name: param for name, param in parameter_dict.items() if param.requires_grad}
        decay_params = [param for name, param in parameter_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in parameter_dict.items() if param.dim() < 2]
        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        name_by_id = {id(param): name for name, param in parameter_dict.items()}
        for group in optimizer_groups:
            group["parameter_names"] = tuple(name_by_id[id(param)] for param in group["params"])
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

    def parameter_report(self) -> Dict[str, object]:
        conventional = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("trajectory.")
        )
        return {
            "persistent_parameters": sum(p.numel() for p in self.parameters()),
            "sheet_coefficients": self.trajectory.sheet_parameter_count(),
            "conventional_non_sheet_parameters": conventional,
            "dense_equivalent_repeated_parameters": self.trajectory.dense_equivalent_count(),
            "dense_equivalent_total_parameters": (
                conventional + self.trajectory.dense_equivalent_count()
            ),
            "matrix_sheet_coefficients": self.trajectory.matrix_sheet_parameter_count(),
            "matrix_dense_equivalent_parameters": self.trajectory.matrix_dense_equivalent_count(),
            "families": self.trajectory.family_report(),
        }

    def compact_state_violations(self) -> List[str]:
        state = self.state_dict()
        return sorted(key for key in state if key.startswith("trajectory.bases."))


__all__ = ["SheetGPT", "SheetGPTConfig"]
# ^^^ THOG
