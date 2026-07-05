# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass


RESIDUAL_INIT_POLICIES = ("depth_scaled", "unscaled")
RESIDUAL_INIT_DEPTH_SOURCES = (
    "true_layer_depth",
    "dof_implied_depth",
    "user_forced_depth",
)
RESIDUAL_INIT_DEPTH_SOURCE_ALIASES = {
    "logical_depth": "true_layer_depth",
    "basis_depth": "dof_implied_depth",
    "constant": "user_forced_depth",
}
RESIDUAL_INIT_DEPTH_SOURCE_CHOICES = RESIDUAL_INIT_DEPTH_SOURCES + tuple(
    RESIDUAL_INIT_DEPTH_SOURCE_ALIASES
)
DEFAULT_RESIDUAL_INIT_POLICY = "depth_scaled"
DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE = "true_layer_depth"
DEFAULT_RESIDUAL_INIT_DEPTH_VALUE = 12
DEFAULT_RESIDUAL_INIT_BASE_STD = 0.02


def canonical_residual_init_depth_source(depth_source: str) -> str:
    if depth_source in RESIDUAL_INIT_DEPTH_SOURCE_ALIASES:
        return RESIDUAL_INIT_DEPTH_SOURCE_ALIASES[depth_source]
    if depth_source in RESIDUAL_INIT_DEPTH_SOURCES:
        return depth_source
    raise ValueError(
        "residual init depth source must be one of "
        f"{RESIDUAL_INIT_DEPTH_SOURCE_CHOICES}; got {depth_source!r}"
    )


@dataclass(frozen=True)
class ResidualInitConfig:
    policy: str = DEFAULT_RESIDUAL_INIT_POLICY
    depth_source: str = DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE
    depth_value: int = DEFAULT_RESIDUAL_INIT_DEPTH_VALUE

    def __post_init__(self) -> None:
        if self.policy not in RESIDUAL_INIT_POLICIES:
            raise ValueError(
                f"residual init policy must be one of {RESIDUAL_INIT_POLICIES}; "
                f"got {self.policy!r}"
            )
        object.__setattr__(
            self,
            "depth_source",
            canonical_residual_init_depth_source(self.depth_source),
        )
        if (
            isinstance(self.depth_value, bool)
            or not isinstance(self.depth_value, int)
            or self.depth_value < 1
        ):
            raise ValueError(
                "residual init depth value must be a positive integer; "
                f"got {self.depth_value!r}"
            )

    def effective_depth(
        self,
        *,
        model_type: str,
        n_layer: int,
        depth_order: int | None,
    ) -> int:
        if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer < 1:
            raise ValueError(f"n_layer must be a positive integer; got {n_layer!r}")
        if self.depth_source == "true_layer_depth":
            return n_layer
        if self.depth_source == "dof_implied_depth":
            if model_type != "thog2_sheet":
                raise ValueError("dof_implied_depth residual init is only defined for SHEET")
            if (
                isinstance(depth_order, bool)
                or not isinstance(depth_order, int)
                or depth_order < 1
            ):
                raise ValueError(
                    "dof_implied_depth residual init requires a positive depth_order; "
                    f"got {depth_order!r}"
                )
            return depth_order
        if self.depth_source == "user_forced_depth":
            return self.depth_value
        raise RuntimeError(f"unreachable residual init depth source: {self.depth_source}")

    def residual_std(
        self,
        *,
        model_type: str,
        n_layer: int,
        depth_order: int | None,
        base_std: float = DEFAULT_RESIDUAL_INIT_BASE_STD,
    ) -> float:
        if not isinstance(base_std, (int, float)) or base_std <= 0.0:
            raise ValueError(f"base_std must be positive; got {base_std!r}")
        if self.policy == "unscaled":
            return float(base_std)
        if self.policy == "depth_scaled":
            depth = self.effective_depth(
                model_type=model_type,
                n_layer=n_layer,
                depth_order=depth_order,
            )
            return float(base_std) / math.sqrt(2.0 * depth)
        raise RuntimeError(f"unreachable residual init policy: {self.policy}")


__all__ = [
    "DEFAULT_RESIDUAL_INIT_BASE_STD",
    "DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE",
    "DEFAULT_RESIDUAL_INIT_DEPTH_VALUE",
    "DEFAULT_RESIDUAL_INIT_POLICY",
    "RESIDUAL_INIT_DEPTH_SOURCE_ALIASES",
    "RESIDUAL_INIT_DEPTH_SOURCE_CHOICES",
    "RESIDUAL_INIT_DEPTH_SOURCES",
    "RESIDUAL_INIT_POLICIES",
    "ResidualInitConfig",
    "canonical_residual_init_depth_source",
]
# ^^^ THOG
