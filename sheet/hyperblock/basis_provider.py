# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol

import torch
from torch import Tensor, nn

from sheet.bases import build_registered_basis

from .plan import ResolvedHyperblockPlan


class AxisBasisProvider(Protocol):
    family: str
    version: str

    def build(
        self,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> Tensor: ...

    def describe(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class RegisteredAxisBasisProvider:
    family: str
    version: str

    def build(
        self,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        return build_registered_basis(
            sample_count,
            order,
            runtime_dtype=runtime_dtype,
            device=device,
            version=self.version,
            basis_family=self.family,
        )

    def describe(self) -> Mapping[str, object]:
        return {
            "provider": "registered_axis_basis_provider_v1",
            "basis_family": self.family,
            "basis_version": self.version,
        }


class HyperblockBasisTables(nn.Module):
    def __init__(
        self,
        plan: ResolvedHyperblockPlan,
        *,
        runtime_dtype: torch.dtype,
        provider: Optional[AxisBasisProvider] = None,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.provider = provider or RegisteredAxisBasisProvider(
            plan.compressor_family,
            plan.compressor_version,
        )
        tables = self._build_tables(runtime_dtype)
        for name, table in tables.items():
            self.register_buffer(name, table, persistent=False)

    def _build_tables(self, runtime_dtype: torch.dtype) -> Dict[str, Tensor]:
        physical = self.plan.physical_axis_lengths
        retained = self.plan.retained_axis_orders
        keys = (
            "WEIGHT_FAMILY_COMMON",
            "WEIGHT_FAMILY_ATTENTION",
            "WEIGHT_FAMILY_MLP",
            "DEPTH",
            "D_MODEL",
            "MLP_HIDDEN",
            "ATTENTION_HEAD",
            "ATTENTION_HEAD_CHANNEL",
        )
        tables = {}
        for key in keys:
            tables[key.lower()] = self.provider.build(
                physical[key],
                retained[key],
                runtime_dtype=runtime_dtype,
                device=torch.device("cpu"),
            )
        return tables

    def as_mapping(self) -> Dict[str, Tensor]:
        return {
            "family_common": self.weight_family_common,
            "family_attention": self.weight_family_attention,
            "family_mlp": self.weight_family_mlp,
            "depth": self.depth,
            "d_model": self.d_model,
            "mlp_hidden": self.mlp_hidden,
            "attention_head": self.attention_head,
            "attention_head_channel": self.attention_head_channel,
        }

    def diagnostics(self) -> Dict[str, object]:
        rows: Dict[str, object] = {
            "provider": dict(self.provider.describe()),
        }
        for name, table in self.as_mapping().items():
            gram = table.float().transpose(0, 1) @ table.float()
            identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            rows[name] = {
                "shape": tuple(table.shape),
                "dtype": str(table.dtype),
                "max_orthogonality_error": float((gram - identity).abs().max().item()),
            }
        return rows
# ^^^ THOG
