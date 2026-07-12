# vvv THOG
from __future__ import annotations

import inspect
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .basis import BASIS_VERSION
from .block_trajectory import BlockTrajectory
from .compact_identity import (
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_MLP_BLOCK,
    resolve_compact_selectors,
    validate_current_sheet_support,
)
from .depth_trajectory import DepthTrajectory
from .geometry import SheetGeometryConfig
from .mlp_block_trajectory import MlpBlockTrajectory
from .semantic_materializer import LegacySheetColMaterializer
from .trajectory import SheetTrajectory


# vvv THOG
_FAST_DISCARD_TRUE_VALUES = {"1", "true", "yes", "on"}
_FAST_DISCARD_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in _FAST_DISCARD_TRUE_VALUES:
        return True
    if normalized_value in _FAST_DISCARD_FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false; got {raw_value!r}")
# ^^^ THOG


class ConventionalLayerNorm(nn.Module):
    def __init__(self, width: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1.0e-5)


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
    mlp_channel_order: Optional[int] = None
    o_attn_d_model: Optional[int] = None                                                                                                               # <<< THOG final attention model-axis order
    o_attn_qkv_per_channel: Optional[int] = None                                                                                                       # <<< THOG final QKV per-head channel order
    o_attn_out_per_channel: Optional[int] = None                                                                                                       # <<< THOG final output per-head channel order
    o_mlp_d_model: Optional[int] = None                                                                                                                # <<< THOG final MLP model-axis order
    o_mlp_hidden: Optional[int] = None                                                                                                                 # <<< THOG final MLP hidden-axis order
    basis_version: str = BASIS_VERSION
    geometry_preset: Optional[str] = None
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = None
    fast_discard: bool = field(default_factory=lambda: _env_bool("THOG2_FAST_DISCARD", False))

    def __post_init__(self) -> None:
        for name in ("block_size", "vocab_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.mlp_channel_order is not None:
            if isinstance(self.mlp_channel_order, bool) or not isinstance(self.mlp_channel_order, int) or self.mlp_channel_order <= 0:
                raise ValueError(f"mlp_channel_order must be a positive integer or None; got {self.mlp_channel_order!r}")
            if self.mlp_channel_order > 4 * self.n_embd:
                raise ValueError("mlp_channel_order must not exceed 4*n_embd")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if not isinstance(self.basis_version, str) or not self.basis_version.strip():
            raise ValueError("basis_version must be a non-empty string")
        if not isinstance(self.fast_discard, bool):
            raise ValueError(f"fast_discard must be bool; got {self.fast_discard!r}")
        self.sheet_geometry()
        self.compact_selectors()

    def sheet_geometry(self) -> SheetGeometryConfig:
        return SheetGeometryConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            depth_order=self.depth_order,
            base_row_order=self.base_row_order,
            mlp_channel_order=self.mlp_channel_order,
            o_attn_d_model=self.o_attn_d_model,
            o_attn_qkv_per_channel=self.o_attn_qkv_per_channel,
            o_attn_out_per_channel=self.o_attn_out_per_channel,
            o_mlp_d_model=self.o_mlp_d_model,
            o_mlp_hidden=self.o_mlp_hidden,
            bias=self.bias,
        )

    def compact_selectors(self):
        selectors = resolve_compact_selectors(
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )
        validate_current_sheet_support(selectors)
        return selectors

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SheetGPT(nn.Module):
    """Sequential correctness-first GPT using compact basis-generated weights."""

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
        selectors = config.compact_selectors()
        if selectors.attention_geometry == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK:
            self.trajectory = BlockTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
                compact_attention=True,
                compact_mlp=selectors.mlp_geometry == MLP_GEOMETRY_MLP_BLOCK,
            )
        elif selectors.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK:
            self.trajectory = MlpBlockTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
            )
        elif selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
            self.trajectory = DepthTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
            )
        else:
            self.trajectory = SheetTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
            )
        self.semantic_materializer = LegacySheetColMaterializer(self.trajectory)
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

    def _sheet_layer_norm(self, inputs: Tensor, weight_name: str, bias_name: str, layer_index: int) -> Tensor:
        weight = self.trajectory.materialize_vector(weight_name, layer_index)
        bias = self._optional_bias(bias_name, layer_index)
        output = F.layer_norm(inputs, (self.config.n_embd,), weight, bias, 1.0e-5)
        if self.config.fast_discard:
            del weight, bias
        return output

    def _attention(self, inputs: Tensor, layer_index: int) -> Tensor:
        batch_size, sequence_length, embedding_width = inputs.shape
        attention_weight = self.semantic_materializer.reconstructed_attention_input_weight(layer_index)
        attention_bias = None
        if self.config.bias:
            attention_bias = self.semantic_materializer.reconstructed_attention_input_bias(layer_index)
        query, key, value = F.linear(inputs, attention_weight, attention_bias).split(self.config.n_embd, dim=2)
        if self.config.fast_discard:
            del attention_weight, attention_bias
        head_width = embedding_width // self.config.n_head
        key = key.view(batch_size, sequence_length, self.config.n_head, head_width).transpose(1, 2)
        query = query.view(batch_size, sequence_length, self.config.n_head, head_width).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.config.n_head, head_width).transpose(1, 2)
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
            causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device))
            scores = scores.masked_fill(~causal_mask.view(1, 1, sequence_length, sequence_length), float("-inf"))
            probabilities = F.softmax(scores, dim=-1)
            probabilities = F.dropout(probabilities, p=self.config.dropout, training=self.training)
            attended = probabilities @ value
            if self.config.fast_discard:
                del scores, causal_mask, probabilities
        if self.config.fast_discard:
            del query, key, value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_width)
        output_weight = self.trajectory.materialize("attention_output_weight", layer_index)
        output_bias = self._optional_bias("attention_output_bias", layer_index)
        projected = F.linear(attended, output_weight, output_bias)
        if self.config.fast_discard:
            del attended, output_weight, output_bias
        output = F.dropout(projected, p=self.config.dropout, training=self.training)
        if self.config.fast_discard:
            del projected
        return output

    def _mlp(self, inputs: Tensor, layer_index: int) -> Tensor:
        expansion_weight = self.trajectory.materialize("mlp_expansion_weight", layer_index)
        expansion_bias = self._optional_bias("mlp_expansion_bias", layer_index)
        hidden = F.linear(inputs, expansion_weight, expansion_bias)
        if self.config.fast_discard:
            del expansion_weight, expansion_bias
        hidden = F.gelu(hidden)
        contraction_weight = self.trajectory.materialize("mlp_contraction_weight", layer_index)
        contraction_bias = self._optional_bias("mlp_contraction_bias", layer_index)
        output = F.linear(hidden, contraction_weight, contraction_bias)
        if self.config.fast_discard:
            del hidden, contraction_weight, contraction_bias
        dropped = F.dropout(output, p=self.config.dropout, training=self.training)
        if self.config.fast_discard:
            del output
        return dropped

    def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:
        normalized_attention = self._sheet_layer_norm(inputs, "ln_1_weight", "ln_1_bias", layer_index)
        attention_output = self._attention(normalized_attention, layer_index)
        if self.config.fast_discard:
            del normalized_attention
        inputs = inputs + attention_output
        if self.config.fast_discard:
            del attention_output
        normalized_mlp = self._sheet_layer_norm(inputs, "ln_2_weight", "ln_2_bias", layer_index)
        mlp_output = self._mlp(normalized_mlp, layer_index)
        if self.config.fast_discard:
            del normalized_mlp
        output = inputs + mlp_output
        if self.config.fast_discard:
            del inputs, mlp_output
        return output

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {sequence_length}; block size is {self.config.block_size}")
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        for layer_index in range(self.config.n_layer):
            hidden = self._logical_block(hidden, layer_index)
        hidden = self.transformer.ln_f(hidden)
        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None
        return logits, loss

    def parameter_report(self) -> Dict[str, object]:
        total_persistent = sum(parameter.numel() for parameter in self.parameters())
        sheet_coefficients = self.trajectory.sheet_parameter_count()
        conventional = total_persistent - sheet_coefficients
        dense_equivalent_repeated = self.trajectory.dense_equivalent_count()
        dense_equivalent_total = conventional + dense_equivalent_repeated
        return {
            "persistent_parameters": total_persistent,
            "sheet_coefficients": sheet_coefficients,
            "conventional_non_sheet_parameters": conventional,
            "dense_equivalent_repeated_parameters": dense_equivalent_repeated,
            "dense_equivalent_total_parameters": dense_equivalent_total,
            "matrix_sheet_coefficients": self.trajectory.matrix_sheet_parameter_count(),
            "matrix_dense_equivalent_parameters": self.trajectory.matrix_dense_equivalent_count(),
            "families": self.trajectory.family_report(),
        }

    def get_num_params(self, non_embedding: bool = True) -> int:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            parameter_count -= self.transformer.wpe.weight.numel()
        return parameter_count

    def optimizer_parameter_groups(self, weight_decay: float) -> Tuple[Dict[str, object], Dict[str, object]]:
        decay: Dict[str, nn.Parameter] = {}
        no_decay: Dict[str, nn.Parameter] = {}
        for family_name, parameter, metadata in self.trajectory.named_semantic_parameters():
            target = decay if metadata.weight_decay else no_decay
            target[f"trajectory.coefficients.{family_name}"] = parameter
        sheet_parameter_ids = {id(parameter) for parameter in self.trajectory.coefficients.values()}
        for name, parameter in self.named_parameters():
            if id(parameter) in sheet_parameter_ids:
                continue
            if name in {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}:
                target = no_decay
            elif parameter.ndim >= 2:
                target = decay
            else:
                target = no_decay
            target[name] = parameter
        return (
            {"params": list(decay.values()), "parameter_names": tuple(decay.keys()), "weight_decay": weight_decay},
            {"params": list(no_decay.values()), "parameter_names": tuple(no_decay.keys()), "weight_decay": 0.0},
        )

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(self.optimizer_parameter_groups(weight_decay), lr=learning_rate, betas=betas, fused=use_fused)

    def compact_state_violations(self) -> Tuple[str, ...]:
        violations: List[str] = []
        compact_coefficient_prefixes = (
            "trajectory.coefficients.",
            "trajectory.depth.coefficients.",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(compact_coefficient_prefixes):
                continue
            if name.startswith("transformer.wte") or name.startswith("transformer.wpe") or name.startswith("transformer.ln_f") or name.startswith("lm_head"):
                continue
            violations.append(name)
        return tuple(violations)


__all__ = ["SheetGPT", "SheetGPTConfig", "ConventionalLayerNorm"]
# ^^^ THOG
