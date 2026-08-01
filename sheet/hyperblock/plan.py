# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Dict, Mapping, Tuple

from sheet.bases import (
    CHEBYSHEV_BASIS_VERSION,
    normalize_basis_version,
    normalize_registered_basis_family,
)


HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE = "coupled_field_machine"
HYPERBLOCK_TOPOLOGY_VERSION = "coupled_field_machine_v1"
HYPERBLOCK_MATERIALIZATION_VERSION = "coupled_field_machine_materialization_v1"
HYPERBLOCK_INITIALIZATION_VERSION = "orthogonal_mode_variance_split_v1"
HYPERBLOCK_PLAN_SCHEMA_VERSION = 1

WEIGHT_FAMILIES: Tuple[str, ...] = (
    "Q",
    "K",
    "V",
    "ATTENTION_OUTPUT",
    "MLP_UP",
    "MLP_DOWN",
)
ATTENTION_FAMILIES: Tuple[str, ...] = WEIGHT_FAMILIES[:4]
MLP_FAMILIES: Tuple[str, ...] = WEIGHT_FAMILIES[4:]

COMMON_FAMILY_ORDER = len(WEIGHT_FAMILIES)
ATTENTION_FAMILY_ORDER = len(ATTENTION_FAMILIES)
MLP_FAMILY_ORDER = len(MLP_FAMILIES)


@dataclass(frozen=True)
class HyperblockOrders:
    depth: int
    d_model: int
    mlp_hidden: int
    attention_head: int
    attention_head_channel: int
    common_family: int = COMMON_FAMILY_ORDER
    attention_family: int = ATTENTION_FAMILY_ORDER
    mlp_family: int = MLP_FAMILY_ORDER

    def as_axis_mapping(self) -> Dict[str, int]:
        return {
            "WEIGHT_FAMILY_COMMON": self.common_family,
            "WEIGHT_FAMILY_ATTENTION": self.attention_family,
            "WEIGHT_FAMILY_MLP": self.mlp_family,
            "DEPTH": self.depth,
            "D_MODEL": self.d_model,
            "MLP_HIDDEN": self.mlp_hidden,
            "ATTENTION_HEAD": self.attention_head,
            "ATTENTION_HEAD_CHANNEL": self.attention_head_channel,
        }


@dataclass(frozen=True)
class ResolvedHyperblockPlan:
    n_layer: int
    n_embd: int
    n_head: int
    mlp_hidden_multiplier: int
    orders: HyperblockOrders
    compressor_family: str = "chebyshev"
    compressor_version: str = CHEBYSHEV_BASIS_VERSION
    topology: str = HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
    topology_version: str = HYPERBLOCK_TOPOLOGY_VERSION
    schema_version: int = HYPERBLOCK_PLAN_SCHEMA_VERSION
    materialization_version: str = HYPERBLOCK_MATERIALIZATION_VERSION
    initialization_version: str = HYPERBLOCK_INITIALIZATION_VERSION

    def __post_init__(self) -> None:
        for name in ("n_layer", "n_embd", "n_head", "mlp_hidden_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head; "
                f"got n_embd={self.n_embd}, n_head={self.n_head}"
            )
        if self.topology != HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE:
            raise ValueError(
                "unsupported HYPERBLOCK topology; "
                f"expected {HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE!r}, got {self.topology!r}"
            )
        if self.topology_version != HYPERBLOCK_TOPOLOGY_VERSION:
            raise ValueError(
                "unsupported HYPERBLOCK topology version; "
                f"expected {HYPERBLOCK_TOPOLOGY_VERSION!r}, got {self.topology_version!r}"
            )
        if self.schema_version != HYPERBLOCK_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported HYPERBLOCK plan schema version; "
                f"expected {HYPERBLOCK_PLAN_SCHEMA_VERSION}, got {self.schema_version}"
            )
        canonical_family = normalize_registered_basis_family(self.compressor_family)
        canonical_version = normalize_basis_version(
            canonical_family,
            self.compressor_version,
        )
        object.__setattr__(self, "compressor_family", canonical_family)
        object.__setattr__(self, "compressor_version", canonical_version)
        self._validate_orders()

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        return self.mlp_hidden_multiplier * self.n_embd

    @property
    def physical_axis_lengths(self) -> Dict[str, int]:
        return {
            "WEIGHT_FAMILY_COMMON": len(WEIGHT_FAMILIES),
            "WEIGHT_FAMILY_ATTENTION": len(ATTENTION_FAMILIES),
            "WEIGHT_FAMILY_MLP": len(MLP_FAMILIES),
            "DEPTH": self.n_layer,
            "D_MODEL": self.n_embd,
            "MLP_HIDDEN": self.mlp_hidden,
            "ATTENTION_HEAD": self.n_head,
            "ATTENTION_HEAD_CHANNEL": self.head_dim,
        }

    @property
    def retained_axis_orders(self) -> Dict[str, int]:
        return self.orders.as_axis_mapping()

    def _validate_orders(self) -> None:
        requested = self.orders.as_axis_mapping()
        physical = self.physical_axis_lengths
        for axis_name, order in requested.items():
            if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
                raise ValueError(
                    f"HYPERBLOCK order {axis_name} must be a positive integer; got {order!r}"
                )
            if order > physical[axis_name]:
                raise ValueError(
                    f"HYPERBLOCK order {axis_name}={order} exceeds physical length "
                    f"{physical[axis_name]}"
                )

    @property
    def attention_unique_mode_count(self) -> int:
        return self.orders.attention_head * self.orders.attention_head_channel - 1

    @property
    def mlp_unique_mode_count(self) -> int:
        return self.orders.mlp_hidden - 1

    @property
    def coefficient_shapes(self) -> Dict[str, Tuple[int, ...]]:
        common = (
            self.orders.common_family,
            self.orders.depth,
            self.orders.d_model,
        )
        attention = (
            self.orders.attention_family,
            self.orders.depth,
            self.orders.d_model,
            self.attention_unique_mode_count,
        )
        mlp = (
            self.orders.mlp_family,
            self.orders.depth,
            self.orders.d_model,
            self.mlp_unique_mode_count,
        )
        return {"common": common, "attention": attention, "mlp": mlp}

    @property
    def coefficient_counts(self) -> Dict[str, int]:
        counts = {
            name: prod(shape)
            for name, shape in self.coefficient_shapes.items()
        }
        counts["total"] = sum(counts.values())
        return counts

    @property
    def dense_equivalent_matrix_count(self) -> int:
        return 12 * self.n_layer * self.n_embd * self.n_embd

    @property
    def compression_ratio(self) -> float:
        coefficient_count = self.coefficient_counts["total"]
        return self.dense_equivalent_matrix_count / coefficient_count

    def identity(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology": self.topology,
            "topology_version": self.topology_version,
            "covered_families": WEIGHT_FAMILIES,
            "attention_families": ATTENTION_FAMILIES,
            "mlp_families": MLP_FAMILIES,
            "physical_axis_lengths": self.physical_axis_lengths,
            "retained_axis_orders": self.retained_axis_orders,
            "coefficient_shapes": self.coefficient_shapes,
            "coefficient_counts": self.coefficient_counts,
            "compressor_family": self.compressor_family,
            "compressor_version": self.compressor_version,
            "materialization_version": self.materialization_version,
            "initialization_version": self.initialization_version,
            "family_coordinate_policy": "branch_local_linear_minus_one_to_one_v1",
            "support_policy": "fixed_full_valid_regions_without_duplicate_constant_modes_v1",
        }

    def to_dict(self) -> Dict[str, object]:
        values = asdict(self)
        values["identity"] = self.identity()
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ResolvedHyperblockPlan":
        order_values = values.get("orders")
        if not isinstance(order_values, Mapping):
            raise ValueError("resolved HYPERBLOCK plan requires an orders mapping")
        orders = HyperblockOrders(
            depth=int(order_values["depth"]),
            d_model=int(order_values["d_model"]),
            mlp_hidden=int(order_values["mlp_hidden"]),
            attention_head=int(order_values["attention_head"]),
            attention_head_channel=int(order_values["attention_head_channel"]),
            common_family=int(order_values.get("common_family", COMMON_FAMILY_ORDER)),
            attention_family=int(order_values.get("attention_family", ATTENTION_FAMILY_ORDER)),
            mlp_family=int(order_values.get("mlp_family", MLP_FAMILY_ORDER)),
        )
        keyword_values = dict(values)
        keyword_values.pop("orders", None)
        keyword_values.pop("identity", None)
        return cls(orders=orders, **keyword_values)
# ^^^ THOG
