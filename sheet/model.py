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
from .semantic_materializer import LegacySheetColMaterializer
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
        self.sheet_geometry()

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
        attention_weight = self.semantic_materializer.reconstructed_attention_input_weight(layer_index)
        attention_bias = None
        if self.config.bias:
            attention_bias = self.semantic_materializer.reconstructed_attention_input_bias(layer_index)
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

    def optimizer_parameter_groups(
        self,
        weight_decay: float,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        decay: Dict[str, nn.Parameter] = {}
        no_decay: Dict[str, nn.Parameter] = {}

        for family_name, parameter, metadata in self.trajectory.named_semantic_parameters():
            target = decay if metadata.weight_decay else no_decay
            target[f"trajectory.coefficients.{family_name}"] = parameter

        sheet_parameter_ids = {
            id(parameter) for parameter in self.trajectory.coefficients.values()
        }
        for name, parameter in self.named_parameters():
            if id(parameter) in sheet_parameter_ids:
                continue
            if name in {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}:
                decay[name] = parameter
            else:
                no_decay[name] = parameter

        all_trainable = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        grouped_ids = [id(parameter) for parameter in decay.values()] + [
            id(parameter) for parameter in no_decay.values()
        ]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("a trainable parameter appears in more than one optimizer group")
        if set(grouped_ids) != all_trainable:
            missing = all_trainable - set(grouped_ids)
            extra = set(grouped_ids) - all_trainable
            raise RuntimeError(
                f"optimizer grouping does not cover trainable parameters exactly; "
                f"missing={len(missing)}, extra={len(extra)}"
            )

        decay_names = tuple(sorted(decay))
        no_decay_names = tuple(sorted(no_decay))
        return (
            {
                "params": [decay[name] for name in decay_names],
                "weight_decay": weight_decay,
                "group_name": "decay",
                "parameter_names": decay_names,
            },
            {
                "params": [no_decay[name] for name in no_decay_names],
                "weight_decay": 0.0,
                "group_name": "no_decay",
                "parameter_names": no_decay_names,
            },
        )

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        groups = list(self.optimizer_parameter_groups(weight_decay))
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

    def compact_state_violations(self) -> Tuple[str, ...]:
        violations: List[str] = []
        for key in self.state_dict():
            if key.startswith("trajectory.bases."):
                violations.append(f"persistent fixed basis: {key}")
        for name, parameter in self.named_parameters():
            if name.startswith("trajectory.coefficients."):
                continue
            if parameter.ndim == 3 and parameter.shape[0] == self.config.n_layer:
                violations.append(
                    f"possible persistent dense logical stack: {name} {tuple(parameter.shape)}"
                )
        return tuple(violations)

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
# ^^^ THOG
