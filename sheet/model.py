# vvv THOG
from __future__ import annotations

import inspect
import math
import os
from dataclasses import asdict, dataclass, field
# vvv THOG HYPERBLOCK layer bundles are passed as a typed matrix mapping
# from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Mapping, Optional, Tuple
# ^^^ THOG

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .basis import BASIS_VERSION
from .block_trajectory import BlockTrajectory
from .compact_identity import (
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    DEFAULT_MLP_HIDDEN_COMPRESSOR,
    DEFAULT_MLP_HIDDEN_GROUP_SIZE,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_JPEG_LIKE_V1,
    MLP_GEOMETRY_MLP_BLOCK,
    resolve_compact_selectors,
    validate_current_sheet_support,
)
from .depth_trajectory import DepthTrajectory
from .geometry import SheetGeometryConfig
# vvv THOG coupled field machine HYPERBLOCK is an architecture-wide trajectory, separate from legacy BLOCK geometries
from .hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    CoupledFieldTrajectory,
    FactorisedHyperblockMlpLayer,                                                                                                                        # <<< THOG typed compact UP/DOWN factor bundle
    apply_factorised_hyperblock_mlp,                                                                                                                    # <<< THOG exact direct HYPERBLOCK MLP application
    HyperblockOrders,
    ResolvedHyperblockPlan,
)
# ^^^ THOG
from .jpeg_like_v1_trajectory import JpegLikeV1Trajectory
from .mlp_block_trajectory import MlpBlockTrajectory
# vvv THOG PLASTIC DEPTH owns learned real-valued DEPTH sampling geometry and discrete active-count state
from .plastic_depth import (
    resolve_plastic_depth_counts,
    validate_plastic_layer_count_objective,
    validate_plastic_sampling_initialisation,
)
# ^^^ THOG
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
    mlp_hidden_group_size: int = DEFAULT_MLP_HIDDEN_GROUP_SIZE
    mlp_hidden_compressor: str = DEFAULT_MLP_HIDDEN_COMPRESSOR
    basis_version: str = BASIS_VERSION
    geometry_preset: Optional[str] = None
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = None
    # vvv THOG fixed, non-breathing coupled-field HYPERBLOCK controls
    hyperblock_topology: Optional[str] = None
    hyperblock_compressor: str = "chebyshev"
    hyperblock_compressor_version: str = "auto"
    hyperblock_common_family_order: int = 6
    hyperblock_attention_family_order: int = 4
    hyperblock_mlp_family_order: int = 2
    hyperblock_depth_order: int = 16
    hyperblock_d_model_order: int = 16
    hyperblock_mlp_hidden_order: int = 16
    hyperblock_attention_head_order: int = 16
    hyperblock_attention_head_channel_order: int = 16
    hyperblock_mlp_hidden_multiplier: int = 4
    hyperblock_residual_weight_std: Optional[float] = None
    hyperblock_loop_count: int = 1
    hyperblock_loop_decay: float = 1.0
    # ^^^ THOG
    # vvv THOG PLASTIC DEPTH controls preserve the existing DEPTH path exactly while disabled
    plastic__enabled: bool = False
    plastic__layers_to_sample: Optional[int] = None
    plastic__do_learn_layer_count: bool = False
    plastic__initial_layer_count: Optional[int] = None
    plastic__max_permitted_layers: Optional[int] = None
    plastic__layer_sampling_initialisation: str = "equidistant"
    plastic__layer_count_objective: str = "lowest_loss"
    plastic__layer_count_update_brake: int = 5
    plastic__layer_count_probe__window_size_as_number_of_probes: int = 50
    plastic__layer_count_probe_noise_lambda: float = 3.0
    plastic__layer_count_cost_weight: float = 0.0
    plastic__layer_memory_budget_gib: Optional[float] = None
    plastic__geometry_learning_rate_multiplier: float = 0.1
    plastic__freeze_geometry_during_warmup: bool = True
    plastic__sampling_seed: int = 1337
    plastic__initial_active_layers: int = 0
    # ^^^ THOG
    depth_compress_layer_norm_and_bias: bool = False                                                                                                   # <<< THOG DEPTH-only LayerNorm/bias depth-compression switch
    fast_discard: bool = field(default_factory=lambda: _env_bool("THOG2_FAST_DISCARD", False))
    bypass_semantic_qkv_adapter: bool = field(default_factory=lambda: _env_bool("THOG2_BYPASS_SEMANTIC_QKV_ADAPTER", True))                                       # <<< THOG selectable semantic-QKV adapter bypass
    # direct_thog_mlp_application: bool = field(default_factory=lambda: _env_bool("THOG2_DIRECT_THOG_MLP_APPLICATION", False))                              # <<< THOG retired old option name; retained for source history
    direct_factorised_mlp: bool = field(default_factory=lambda: _env_bool("THOG2_DIRECT_FACTORISED_MLP", True))                                              # <<< THOG default-on exact direct application of existing THOG MLP factors
    # vvv THOG separate default-off experiment; legacy direct-factorised MLP behaviour remains untouched
    direct_factorised_hyperblock_mlp: bool = field(default_factory=lambda: _env_bool("THOG2_DIRECT_FACTORISED_HYPERBLOCK_MLP", False))
    # ^^^ THOG
    vectorise_per_head_materialisation: bool = field(default_factory=lambda: _env_bool("THOG2_VECTORISE_PER_HEAD_MATERIALISATION", True))                    # <<< THOG default-on selectable batched head-aware materialisation

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
        if isinstance(self.mlp_hidden_group_size, bool) or not isinstance(self.mlp_hidden_group_size, int) or self.mlp_hidden_group_size <= 0:
            raise ValueError(f"mlp_hidden_group_size must be a positive integer; got {self.mlp_hidden_group_size!r}")
        if not isinstance(self.mlp_hidden_compressor, str) or not self.mlp_hidden_compressor.strip():
            raise ValueError("mlp_hidden_compressor must be a non-empty string")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if not isinstance(self.basis_version, str) or not self.basis_version.strip():
            raise ValueError("basis_version must be a non-empty string")
        # vvv THOG resolve PLASTIC DEPTH persistent lattice size before constructing any trajectory geometry
        if not isinstance(self.plastic__enabled, bool):
            raise ValueError(f"plastic__enabled must be bool; got {self.plastic__enabled!r}")
        if not isinstance(self.plastic__do_learn_layer_count, bool):
            raise ValueError(
                "plastic__do_learn_layer_count must be bool; "
                f"got {self.plastic__do_learn_layer_count!r}"
            )
        if not isinstance(self.plastic__freeze_geometry_during_warmup, bool):
            raise ValueError(
                "plastic__freeze_geometry_during_warmup must be bool; "
                f"got {self.plastic__freeze_geometry_during_warmup!r}"
            )
        if isinstance(self.plastic__sampling_seed, bool) or not isinstance(self.plastic__sampling_seed, int) or self.plastic__sampling_seed < 0:
            raise ValueError(
                "plastic__sampling_seed must be a non-negative integer; "
                f"got {self.plastic__sampling_seed!r}"
            )
        resolved_plastic_counts = resolve_plastic_depth_counts(
            n_layer=self.n_layer,
            enabled=self.plastic__enabled,
            layers_to_sample=self.plastic__layers_to_sample,
            do_learn_layer_count=self.plastic__do_learn_layer_count,
            initial_layer_count=self.plastic__initial_layer_count,
            max_permitted_layers=self.plastic__max_permitted_layers,
        )
        self.plastic__initial_active_layers = resolved_plastic_counts.initial_active_layers
        if self.plastic__enabled:
            self.n_layer = resolved_plastic_counts.maximum_layers
        validate_plastic_sampling_initialisation(self.plastic__layer_sampling_initialisation)
        validate_plastic_layer_count_objective(self.plastic__layer_count_objective)
        if (
            isinstance(self.plastic__layer_count_update_brake, bool)
            or not isinstance(self.plastic__layer_count_update_brake, int)
            or self.plastic__layer_count_update_brake < 0
        ):
            raise ValueError(
                "plastic__layer_count_update_brake must be a non-negative integer; "
                f"got {self.plastic__layer_count_update_brake!r}"
            )
        # vvv THOG PLASTIC DEPTH robust paired-score gate controls
        if (
            isinstance(self.plastic__layer_count_probe__window_size_as_number_of_probes, bool)
            or not isinstance(self.plastic__layer_count_probe__window_size_as_number_of_probes, int)
            or self.plastic__layer_count_probe__window_size_as_number_of_probes < 1
        ):
            raise ValueError(
                "plastic__layer_count_probe__window_size_as_number_of_probes must be a positive integer; "
                f"got {self.plastic__layer_count_probe__window_size_as_number_of_probes!r}"
            )
        if (
            isinstance(self.plastic__layer_count_probe_noise_lambda, bool)
            or not isinstance(self.plastic__layer_count_probe_noise_lambda, (int, float))
            or not math.isfinite(float(self.plastic__layer_count_probe_noise_lambda))
            or float(self.plastic__layer_count_probe_noise_lambda) < 0.0
        ):
            raise ValueError(
                "plastic__layer_count_probe_noise_lambda must be finite and non-negative; "
                f"got {self.plastic__layer_count_probe_noise_lambda!r}"
            )
        # ^^^ THOG
        if (
            isinstance(self.plastic__layer_count_cost_weight, bool)
            or not isinstance(self.plastic__layer_count_cost_weight, (int, float))
            or not math.isfinite(float(self.plastic__layer_count_cost_weight))
            or float(self.plastic__layer_count_cost_weight) < 0.0
        ):
            raise ValueError(
                "plastic__layer_count_cost_weight must be finite and non-negative; "
                f"got {self.plastic__layer_count_cost_weight!r}"
            )
        if (
            self.plastic__layer_memory_budget_gib is not None
            and (
                isinstance(self.plastic__layer_memory_budget_gib, bool)
                or not isinstance(self.plastic__layer_memory_budget_gib, (int, float))
                or not math.isfinite(float(self.plastic__layer_memory_budget_gib))
                or float(self.plastic__layer_memory_budget_gib) <= 0.0
            )
        ):
            raise ValueError(
                "plastic__layer_memory_budget_gib must be finite and positive or None; "
                f"got {self.plastic__layer_memory_budget_gib!r}"
            )
        if self.plastic__layer_count_objective == "memory_budget" and self.plastic__layer_memory_budget_gib is None:
            raise ValueError(
                "plastic__layer_memory_budget_gib is required for memory_budget"
            )
        if (
            isinstance(self.plastic__geometry_learning_rate_multiplier, bool)
            or not isinstance(self.plastic__geometry_learning_rate_multiplier, (int, float))
            or not math.isfinite(float(self.plastic__geometry_learning_rate_multiplier))
            or float(self.plastic__geometry_learning_rate_multiplier) < 0.0
        ):
            raise ValueError(
                "plastic__geometry_learning_rate_multiplier must be finite and non-negative; "
                f"got {self.plastic__geometry_learning_rate_multiplier!r}"
            )
        if self.plastic__enabled and self.hyperblock_enabled:
            raise ValueError("PLASTIC DEPTH may not be combined with HYPERBLOCK")
        # ^^^ THOG
        if self.hyperblock_residual_weight_std is not None and (
            isinstance(self.hyperblock_residual_weight_std, bool)
            or not isinstance(self.hyperblock_residual_weight_std, (int, float))
            or self.hyperblock_residual_weight_std <= 0.0
        ):
            raise ValueError(
                "hyperblock_residual_weight_std must be positive or None; "
                f"got {self.hyperblock_residual_weight_std!r}"
            )
        # vvv THOG shared HYPERBLOCK recurrence has one integer visit count and one exponential update-decay scalar
        if isinstance(self.hyperblock_loop_count, bool) or not isinstance(self.hyperblock_loop_count, int) or self.hyperblock_loop_count <= 0:
            raise ValueError(
                "hyperblock_loop_count must be a positive integer; "
                f"got {self.hyperblock_loop_count!r}"
            )
        if (
            isinstance(self.hyperblock_loop_decay, bool)
            or not isinstance(self.hyperblock_loop_decay, (int, float))
            or not math.isfinite(float(self.hyperblock_loop_decay))
            or not 0.0 < float(self.hyperblock_loop_decay) <= 1.0
        ):
            raise ValueError(
                "hyperblock_loop_decay must be finite and in (0, 1]; "
                f"got {self.hyperblock_loop_decay!r}"
            )
        if not self.hyperblock_enabled and (
            self.hyperblock_loop_count != 1
            or float(self.hyperblock_loop_decay) != 1.0
        ):
            raise ValueError(
                "HYPERBLOCK loop controls require HYPERBLOCK"
            )
        # ^^^ THOG
        if not isinstance(self.depth_compress_layer_norm_and_bias, bool):
            raise ValueError(
                "depth_compress_layer_norm_and_bias must be bool; "
                f"got {self.depth_compress_layer_norm_and_bias!r}"
            )
        if not isinstance(self.fast_discard, bool):
            raise ValueError(f"fast_discard must be bool; got {self.fast_discard!r}")
        if not isinstance(self.bypass_semantic_qkv_adapter, bool):
            raise ValueError(f"bypass_semantic_qkv_adapter must be bool; got {self.bypass_semantic_qkv_adapter!r}")                                         # <<< THOG validate selectable hot path
        # if not isinstance(self.direct_thog_mlp_application, bool):                                                                                      # <<< THOG retired old option validation
        #     raise ValueError(f"direct_thog_mlp_application must be bool; got {self.direct_thog_mlp_application!r}")
        if not isinstance(self.direct_factorised_mlp, bool):
            raise ValueError(f"direct_factorised_mlp must be bool; got {self.direct_factorised_mlp!r}")                                                    # <<< THOG validate renamed exact MLP application path
        # vvv THOG validate and scope the independent HYPERBLOCK direct-application option
        if not isinstance(self.direct_factorised_hyperblock_mlp, bool):
            raise ValueError(
                "direct_factorised_hyperblock_mlp must be bool; "
                f"got {self.direct_factorised_hyperblock_mlp!r}"
            )
        if self.direct_factorised_hyperblock_mlp and not self.hyperblock_enabled:
            raise ValueError(
                "direct_factorised_hyperblock_mlp requires HYPERBLOCK"
            )
        # ^^^ THOG
        if not isinstance(self.vectorise_per_head_materialisation, bool):
            raise ValueError(f"vectorise_per_head_materialisation must be bool; got {self.vectorise_per_head_materialisation!r}")                          # <<< THOG validate selectable vectorised materialisation path

        # vvv THOG HYPERBLOCK owns all covered matrix axes and must not overlap legacy DEPTH/BLOCK selectors
        if self.hyperblock_enabled:
            conflicting_fields = {
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
            }
            active_conflicts = {
                name: value
                for name, value in conflicting_fields.items()
                if value is not None
            }
            if active_conflicts:
                raise ValueError(
                    "HYPERBLOCK may not be combined with legacy geometry selectors; "
                    f"got {active_conflicts}"
                )
            if self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "HYPERBLOCK keeps LayerNorm and bias vectors conventional in v0; "
                    "depth_compress_layer_norm_and_bias must be false"
                )
            self.hyperblock_plan()
        else:
            # vvv THOG DEPTH is controlled only by P and the LayerNorm/bias participation switch; row/axis orders are semantically irrelevant.
            selectors = self.compact_selectors()
            if selectors.geometry_preset != GEOMETRY_PRESET_DEPTH and self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "depth_compress_layer_norm_and_bias may be enabled only for geometry_preset='depth'"
                )
            if selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
                # vvv THOG PLASTIC DEPTH v0.1 is defined only for the continuous Chebyshev DEPTH field
                if self.plastic__enabled and selectors.basis_family != "chebyshev":
                    raise ValueError(
                        "PLASTIC DEPTH v0.1 requires the Chebyshev DEPTH compressor; "
                        f"got {selectors.basis_family!r}"
                    )
                # ^^^ THOG
                self.base_row_order = 1
                self.mlp_channel_order = 1
                self.o_attn_d_model = 1
                self.o_attn_qkv_per_channel = 1
                self.o_attn_out_per_channel = 1
                self.o_mlp_d_model = 1
                self.o_mlp_hidden = 1
            # ^^^ THOG

            if self.plastic__enabled and selectors.geometry_preset != GEOMETRY_PRESET_DEPTH:
                raise ValueError("PLASTIC DEPTH requires geometry_preset='depth'")
            geometry = self.sheet_geometry()
            if selectors.mlp_geometry == MLP_GEOMETRY_JPEG_LIKE_V1:
                mlp_hidden_length = 4 * self.n_embd
                if mlp_hidden_length % self.mlp_hidden_group_size != 0:
                    raise ValueError(
                        "4*d_model must be divisible by mlp_hidden_group_size; "
                        f"got 4*d_model={mlp_hidden_length}, group_size={self.mlp_hidden_group_size}"
                    )
                if geometry.resolved_o_mlp_hidden > self.mlp_hidden_group_size:
                    raise ValueError(
                        "o_mlp_hidden/Y must not exceed mlp_hidden_group_size for JPEG_LIKE_V1; "
                        f"got Y={geometry.resolved_o_mlp_hidden}, group_size={self.mlp_hidden_group_size}"
                    )
        # ^^^ THOG

    # vvv THOG HYPERBLOCK topology is resolved identity; the v0 CLI need only enable it
    @property
    def hyperblock_enabled(self) -> bool:
        return self.hyperblock_topology is not None

    def hyperblock_plan(self) -> ResolvedHyperblockPlan:
        if not self.hyperblock_enabled:
            raise ValueError("HYPERBLOCK is not enabled")
        return ResolvedHyperblockPlan(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            mlp_hidden_multiplier=self.hyperblock_mlp_hidden_multiplier,
            orders=HyperblockOrders(
                depth=self.hyperblock_depth_order,
                d_model=self.hyperblock_d_model_order,
                mlp_hidden=self.hyperblock_mlp_hidden_order,
                attention_head=self.hyperblock_attention_head_order,
                attention_head_channel=self.hyperblock_attention_head_channel_order,
                common_family=self.hyperblock_common_family_order,
                attention_family=self.hyperblock_attention_family_order,
                mlp_family=self.hyperblock_mlp_family_order,
            ),
            compressor_family=self.hyperblock_compressor,
            compressor_version=self.hyperblock_compressor_version,
            topology=self.hyperblock_topology or HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        )
    # ^^^ THOG

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
        # vvv THOG preserve the pre-HYPERBLOCK trajectory-selection branch headers for source history
        # elif selectors.attention_geometry == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK:
        # elif selectors.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK:
        # elif selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
        # ^^^ THOG
        # vvv THOG HYPERBLOCK is a separate trajectory and leaves every legacy selector path unchanged
        if config.hyperblock_enabled:
            self.trajectory = CoupledFieldTrajectory(
                config.hyperblock_plan(),
                bias=config.bias,
                runtime_dtype=torch.float32,
                residual_weight_std=config.hyperblock_residual_weight_std,
            )
            selectors = None
        else:
            selectors = config.compact_selectors()
        # ^^^ THOG
        if selectors is not None and selectors.mlp_geometry == MLP_GEOMETRY_JPEG_LIKE_V1:
            self.trajectory = JpegLikeV1Trajectory(
                config.sheet_geometry(),
                mlp_hidden_group_size=config.mlp_hidden_group_size,
                mlp_hidden_compressor=config.mlp_hidden_compressor,
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
            )
        elif selectors is not None and selectors.attention_geometry == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK:
            self.trajectory = BlockTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
                compact_attention=True,
                compact_mlp=selectors.mlp_geometry == MLP_GEOMETRY_MLP_BLOCK,
                vectorise_per_head_materialisation=config.vectorise_per_head_materialisation,                                                            # <<< THOG pass selectable head vectorisation into block trajectory
            )
        elif selectors is not None and selectors.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK:
            self.trajectory = MlpBlockTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
            )
        elif selectors is not None and selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
            self.trajectory = DepthTrajectory(
                config.sheet_geometry(),
                runtime_dtype=torch.float32,
                basis_version=config.basis_version,
                basis_family=selectors.basis_family,
                depth_compress_layer_norm_and_bias=config.depth_compress_layer_norm_and_bias,                                                           # <<< THOG select conventional or pure-depth block vectors
                # vvv THOG PLASTIC DEPTH adds a persistent trainable sampling lattice without changing DEPTH coefficients
                plastic_enabled=config.plastic__enabled,
                plastic_initial_active_layers=config.plastic__initial_active_layers,
                plastic_learn_layer_count=config.plastic__do_learn_layer_count,
                plastic_sampling_initialisation=config.plastic__layer_sampling_initialisation,
                plastic_seed=config.plastic__sampling_seed,
                # ^^^ THOG
            )
        elif selectors is not None:
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

    # vvv THOG preserve the pre-bundle attention signature for source history
    # def _attention(self, inputs: Tensor, layer_index: int) -> Tensor:
    # ^^^ THOG
    def _attention(
        self,
        inputs: Tensor,
        layer_index: int,
        layer_materializations: Optional[Mapping[str, Tensor]] = None,
    ) -> Tensor:
        batch_size, sequence_length, embedding_width = inputs.shape
        # vvv THOG consume the already-batched HYPERBLOCK layer matrices before legacy materialisation paths
        if layer_materializations is not None:
            attention_weight = layer_materializations["attention_input_weight"]
            attention_bias = self._optional_bias("attention_input_bias", layer_index)
        # ^^^ THOG
        # vvv THOG preserve the pre-bundle semantic-QKV branch header for source history
        # if self.config.bypass_semantic_qkv_adapter:
        # ^^^ THOG
        # vvv THOG selectable semantic-QKV adapter bypass for exact A/B timing comparisons
        elif self.config.bypass_semantic_qkv_adapter:
            attention_weight = self.trajectory.materialize("attention_input_weight", layer_index)
            attention_bias = None
            if self.config.bias:
                attention_bias = self.trajectory.materialize_vector("attention_input_bias", layer_index)
        else:
            attention_weight = self.semantic_materializer.reconstructed_attention_input_weight(layer_index)
            attention_bias = None
            if self.config.bias:
                attention_bias = self.semantic_materializer.reconstructed_attention_input_bias(layer_index)
        # ^^^ THOG
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
        # vvv THOG preserve the pre-bundle output-weight line and reuse the layer bundle when present
        # output_weight = self.trajectory.materialize("attention_output_weight", layer_index)
        output_weight = (
            self.trajectory.materialize("attention_output_weight", layer_index)
            if layer_materializations is None
            else layer_materializations["attention_output_weight"]
        )
        # ^^^ THOG
        output_bias = self._optional_bias("attention_output_bias", layer_index)
        projected = F.linear(attended, output_weight, output_bias)
        if self.config.fast_discard:
            del attended, output_weight, output_bias
        output = F.dropout(projected, p=self.config.dropout, training=self.training)
        if self.config.fast_discard:
            del projected
        return output

    # vvv THOG exact direct application of the existing THOG MLP factorisation, selectable for A/B timing
    def _supports_direct_factorised_mlp(self) -> bool:
        if isinstance(self.trajectory, MlpBlockTrajectory):
            return True
        return isinstance(self.trajectory, BlockTrajectory) and self.trajectory.compact_mlp

    def _direct_factorised_mlp_linear(
        self,
        inputs: Tensor,
        weight_name: str,
        layer_index: int,
        bias: Optional[Tensor],
    ) -> Tensor:
        coefficient = self.trajectory.coefficients[weight_name]
        depth_row = self.trajectory.depth_basis[layer_index].to(coefficient)
        mixed = torch.einsum("p,pab->ab", depth_row, coefficient)
        input_basis = self.trajectory.input_basis(weight_name).to(coefficient)
        output_basis = self.trajectory.output_basis(weight_name).to(coefficient)
        projected = torch.matmul(inputs, input_basis)
        projected = torch.matmul(projected, mixed.transpose(0, 1))
        output = torch.matmul(projected, output_basis.transpose(0, 1))
        if bias is not None:
            output = output + bias
        return output

    # vvv THOG preserve the pre-bundle MLP signature for source history
    # def _mlp(self, inputs: Tensor, layer_index: int) -> Tensor:
    # ^^^ THOG
    # def _mlp(                                                                                                                                     # <<< THOG preserved pre-direct-HYPERBLOCK signature
    #     self,
    #     inputs: Tensor,
    #     layer_index: int,
    #     layer_materializations: Optional[Mapping[str, Tensor]] = None,
    # ) -> Tensor:
    def _mlp(
        self,
        inputs: Tensor,
        layer_index: int,
        layer_materializations: Optional[Mapping[str, Tensor]] = None,
        hyperblock_mlp_factors: Optional[FactorisedHyperblockMlpLayer] = None,
    ) -> Tensor:
        direct_application = (
            self.config.direct_factorised_mlp
            and self._supports_direct_factorised_mlp()
        )
        expansion_bias = self._optional_bias("mlp_expansion_bias", layer_index)
        # vvv THOG direct HYPERBLOCK expansion bypasses the dense [MLP_HIDDEN,D_MODEL] matrix
        if hyperblock_mlp_factors is not None:
            hidden = apply_factorised_hyperblock_mlp(
                inputs,
                hyperblock_mlp_factors,
                family_index=0,
                expansion=True,
                bias=expansion_bias,
            )
        # if direct_application:
        elif direct_application:
        # ^^^ THOG
            hidden = self._direct_factorised_mlp_linear(
                inputs,
                "mlp_expansion_weight",
                layer_index,
                expansion_bias,
            )
        else:
            # vvv THOG preserve the pre-bundle expansion materialisation and use the shared layer result for HYPERBLOCK
            # expansion_weight = self.trajectory.materialize("mlp_expansion_weight", layer_index)
            expansion_weight = (
                self.trajectory.materialize("mlp_expansion_weight", layer_index)
                if layer_materializations is None
                else layer_materializations["mlp_expansion_weight"]
            )
            # ^^^ THOG
            hidden = F.linear(inputs, expansion_weight, expansion_bias)
            if self.config.fast_discard:
                del expansion_weight
        if self.config.fast_discard:
            del expansion_bias
        hidden = F.gelu(hidden)
        contraction_bias = self._optional_bias("mlp_contraction_bias", layer_index)
        # vvv THOG direct HYPERBLOCK contraction bypasses the dense [D_MODEL,MLP_HIDDEN] matrix
        if hyperblock_mlp_factors is not None:
            output = apply_factorised_hyperblock_mlp(
                hidden,
                hyperblock_mlp_factors,
                family_index=1,
                expansion=False,
                bias=contraction_bias,
            )
        # if direct_application:
        elif direct_application:
        # ^^^ THOG
            output = self._direct_factorised_mlp_linear(
                hidden,
                "mlp_contraction_weight",
                layer_index,
                contraction_bias,
            )
        else:
            # vvv THOG preserve the pre-bundle contraction materialisation and use the shared layer result for HYPERBLOCK
            # contraction_weight = self.trajectory.materialize("mlp_contraction_weight", layer_index)
            contraction_weight = (
                self.trajectory.materialize("mlp_contraction_weight", layer_index)
                if layer_materializations is None
                else layer_materializations["mlp_contraction_weight"]
            )
            # ^^^ THOG
            output = F.linear(hidden, contraction_weight, contraction_bias)
            if self.config.fast_discard:
                del contraction_weight
        if self.config.fast_discard:
            del hidden, contraction_bias
        dropped = F.dropout(output, p=self.config.dropout, training=self.training)
        if self.config.fast_discard:
            del output
        return dropped
    # ^^^ THOG

    # vvv THOG one physical HYPERBLOCK bundle can be revisited without rematerialising or changing its coordinate system
    def _logical_block_once(
        self,
        inputs: Tensor,
        layer_index: int,
        layer_materializations: Optional[Mapping[str, Tensor]] = None,
        hyperblock_mlp_factors: Optional[FactorisedHyperblockMlpLayer] = None,
    ) -> Tensor:
        normalized_attention = self._sheet_layer_norm(inputs, "ln_1_weight", "ln_1_bias", layer_index)
        attention_output = self._attention(
            normalized_attention,
            layer_index,
            layer_materializations,
        )
        if self.config.fast_discard:
            del normalized_attention
        attention_residual = inputs + attention_output
        if self.config.fast_discard:
            del attention_output
        normalized_mlp = self._sheet_layer_norm(attention_residual, "ln_2_weight", "ln_2_bias", layer_index)
        mlp_output = self._mlp(
            normalized_mlp,
            layer_index,
            layer_materializations,
            hyperblock_mlp_factors,
        )
        if self.config.fast_discard:
            del normalized_mlp
        output = attention_residual + mlp_output
        if self.config.fast_discard:
            del attention_residual, mlp_output
        return output

    # def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:                                                                              # <<< THOG preserved pre-loop block entry point
    def _logical_block(self, inputs: Tensor, layer_index: int) -> Tensor:
        layer_materializations = None
        hyperblock_mlp_factors = None
        is_hyperblock = isinstance(self.trajectory, CoupledFieldTrajectory)
        if is_hyperblock:
            direct_hyperblock_mlp = self.config.direct_factorised_hyperblock_mlp
            if direct_hyperblock_mlp:
                layer_materializations = self.trajectory.materialize_layer_matrices(
                    layer_index,
                    include_mlp=False,
                )
                hyperblock_mlp_factors = self.trajectory.factorised_mlp_layer(
                    layer_index
                )
            else:
                layer_materializations = self.trajectory.materialize_layer_matrices(
                    layer_index
                )

        loop_count = self.config.hyperblock_loop_count if is_hyperblock else 1
        loop_decay = float(self.config.hyperblock_loop_decay)
        for loop_index in range(loop_count):
            loop_input = inputs
            block_output = self._logical_block_once(
                loop_input,
                layer_index,
                layer_materializations,
                hyperblock_mlp_factors,
            )
            loop_gain = loop_decay ** loop_index
            inputs = (
                block_output
                if loop_gain == 1.0
                else loop_input + loop_gain * (block_output - loop_input)
            )
            if self.config.fast_discard:
                del block_output, loop_input

        if self.config.fast_discard:
            del layer_materializations, hyperblock_mlp_factors
        return inputs
    # ^^^ THOG

    # vvv THOG PLASTIC DEPTH exposes its current sampled canonical-layer ranks without affecting any disabled path
    @property
    def plastic_depth_enabled(self) -> bool:
        return bool(getattr(self.trajectory, "plastic_enabled", False))

    def plastic_depth_active_layer_indices(self) -> Tuple[int, ...]:
        active_layer_indices = getattr(self.trajectory, "active_layer_indices", None)
        if self.plastic_depth_enabled and callable(active_layer_indices):
            return tuple(int(value) for value in active_layer_indices())
        return tuple(range(self.config.n_layer))

    def set_plastic_depth_active_layer_count(self, active_layers: int) -> None:
        setter = getattr(self.trajectory, "set_active_layer_count", None)
        if not self.plastic_depth_enabled or not callable(setter):
            raise RuntimeError("PLASTIC DEPTH is not enabled")
        setter(active_layers)

    def plastic_depth_report(self) -> Optional[Dict[str, object]]:
        report_builder = getattr(self.trajectory, "plastic_depth_report", None)
        if not self.plastic_depth_enabled or not callable(report_builder):
            return None
        return report_builder()

    # vvv THOG model-level atomic transition API; trainer integration follows only after optimiser-state policy is tested
    def prepare_plastic_depth_count_transition(self, active_layers: int):
        preparer = getattr(
            self.trajectory,
            "prepare_plastic_depth_count_transition",
            None,
        )
        if not self.plastic_depth_enabled or not callable(preparer):
            raise RuntimeError("PLASTIC DEPTH is not enabled")
        return preparer(active_layers)

    def commit_plastic_depth_count_transition(self, transition) -> Dict[str, object]:
        committer = getattr(
            self.trajectory,
            "commit_plastic_depth_count_transition",
            None,
        )
        if not self.plastic_depth_enabled or not callable(committer):
            raise RuntimeError("PLASTIC DEPTH is not enabled")
        return committer(transition)
    # ^^^ THOG
    # ^^^ THOG

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
        # vvv THOG PLASTIC DEPTH executes only the active ranks from its persistent learned sampling lattice
        layer_indices = (
            self.plastic_depth_active_layer_indices()
            if self.plastic_depth_enabled
            else range(self.config.n_layer)
        )
        # vvv THOG preserve the original fixed-depth traversal while allowing PLASTIC DEPTH active ranks
        # for layer_index in range(self.config.n_layer):
        for layer_index in layer_indices:
        # ^^^ THOG
            hidden = self._logical_block(hidden, layer_index)
        # ^^^ THOG
        hidden = self.transformer.ln_f(hidden)
        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None
        return logits, loss

    # vvv THOG preserve the pre-HYPERBLOCK parameter-report return line for source history
    # return {
    # ^^^ THOG

    def parameter_report(self) -> Dict[str, object]:
        total_persistent = sum(parameter.numel() for parameter in self.parameters())
        sheet_coefficients = self.trajectory.sheet_parameter_count()
        # vvv THOG PLASTIC DEPTH geometry is compact PE state, neither a DEPTH coefficient nor a conventional GPT parameter
        plastic_depth_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("trajectory.plastic_sampling.")
        )
        # conventional = total_persistent - sheet_coefficients
        conventional = total_persistent - sheet_coefficients - plastic_depth_parameters
        # ^^^ THOG
        dense_equivalent_repeated = self.trajectory.dense_equivalent_count()
        dense_equivalent_total = conventional + dense_equivalent_repeated
        report = {
            "persistent_parameters": total_persistent,
            "sheet_coefficients": sheet_coefficients,
            "conventional_non_sheet_parameters": conventional,
            "dense_equivalent_repeated_parameters": dense_equivalent_repeated,
            "dense_equivalent_total_parameters": dense_equivalent_total,
            "matrix_sheet_coefficients": self.trajectory.matrix_sheet_parameter_count(),
            "matrix_dense_equivalent_parameters": self.trajectory.matrix_dense_equivalent_count(),
            "families": self.trajectory.family_report(),
        }
        # vvv THOG expose PLASTIC DEPTH state separately from the unchanged sheet-coefficient count
        plastic_depth_report = self.plastic_depth_report()
        if plastic_depth_report is not None:
            report["plastic_depth"] = plastic_depth_report
            report["plastic_depth_parameters"] = plastic_depth_parameters
        # ^^^ THOG
        # vvv THOG expose the resolved HYPERBLOCK field identity without changing legacy report fields
        hyperblock_report = getattr(self.trajectory, "hyperblock_report", None)
        if callable(hyperblock_report):
            # report["hyperblock"] = hyperblock_report()                                                                                               # <<< THOG preserved pre-execution-metadata report
            report["hyperblock"] = hyperblock_report()
            report["hyperblock"]["direct_factorised_mlp"] = self.config.direct_factorised_hyperblock_mlp                                           # <<< THOG record exact HYPERBLOCK MLP execution path
            report["hyperblock"]["loop_count"] = self.config.hyperblock_loop_count
            report["hyperblock"]["loop_decay"] = float(self.config.hyperblock_loop_decay)
        # ^^^ THOG
        return report

    def get_num_params(self, non_embedding: bool = True) -> int:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            parameter_count -= self.transformer.wpe.weight.numel()
        return parameter_count

    # vvv THOG preserve the pre-HYPERBLOCK optimizer-group lines for source history
    # target[f"trajectory.coefficients.{family_name}"] = parameter
    # sheet_parameter_ids = {id(parameter) for parameter in self.trajectory.coefficients.values()}
    # if id(parameter) in sheet_parameter_ids:
    # ^^^ THOG

    # vvv THOG preserve the original two-group signature while adding a dedicated PLASTIC DEPTH geometry group
    # def optimizer_parameter_groups(self, weight_decay: float) -> Tuple[Dict[str, object], Dict[str, object]]:
    def optimizer_parameter_groups(self, weight_decay: float) -> Tuple[Dict[str, object], ...]:
        decay: Dict[str, nn.Parameter] = {}
        no_decay: Dict[str, nn.Parameter] = {}
        geometry: Dict[str, nn.Parameter] = {}
        semantic_parameter_ids = set()
        coefficient_parameter_ids = {
            id(parameter) for parameter in self.trajectory.coefficients.values()
        }
        geometry_parameter_ids = {
            id(parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("trajectory.plastic_sampling.") and parameter.numel() > 0
        }
        for family_name, parameter, metadata in self.trajectory.named_semantic_parameters():
            semantic_parameter_ids.add(id(parameter))
            target = decay if metadata.weight_decay else no_decay
            container_name = (
                "coefficients"
                if id(parameter) in coefficient_parameter_ids
                else "vector_parameters"
            )
            target[f"trajectory.{container_name}.{family_name}"] = parameter
        for name, parameter in self.named_parameters():
            if id(parameter) in semantic_parameter_ids:
                continue
            if id(parameter) in geometry_parameter_ids:
                geometry[name] = parameter
                continue
            if name in {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}:
                target = no_decay
            elif parameter.ndim >= 2:
                target = decay
            else:
                target = no_decay
            target[name] = parameter
        # return (
        groups = [
            {"params": list(decay.values()), "parameter_names": tuple(decay.keys()), "weight_decay": weight_decay},
            {"params": list(no_decay.values()), "parameter_names": tuple(no_decay.keys()), "weight_decay": 0.0},
        ]
        # vvv THOG PLASTIC DEPTH geometry receives a separate slower no-decay optimiser group only when enabled
        if geometry:
            groups.append(
                {
                    "params": list(geometry.values()),
                    "parameter_names": tuple(geometry.keys()),
                    "weight_decay": 0.0,
                    "thog2_lr_multiplier": float(self.config.plastic__geometry_learning_rate_multiplier),
                    "thog2_freeze_during_warmup": bool(self.config.plastic__freeze_geometry_during_warmup),
                    "thog2_plastic_depth_geometry": True,
                }
            )
        # ^^^ THOG
        return tuple(groups)
    # ^^^ THOG

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(self.optimizer_parameter_groups(weight_decay), lr=learning_rate, betas=betas, fused=use_fused)

    def compact_state_violations(self) -> Tuple[str, ...]:
        violations: List[str] = []
        compact_coefficient_prefixes = (
            "trajectory.coefficients.",
            "trajectory.depth.coefficients.",
            "trajectory.vector_parameters.",                                                                                                           # <<< THOG HYPERBLOCK conventional per-layer vectors remain inside the compact model state
            "trajectory.plastic_sampling.",                                                                                                           # <<< THOG PLASTIC DEPTH lattice parameters are compact Plasticity Engine state
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
# vvv THOG preserved superseded source lines for exact history audit
# if isinstance(self.trajectory, CoupledFieldTrajectory):
# inputs = inputs + attention_output
# normalized_mlp = self._sheet_layer_norm(inputs, "ln_2_weight", "ln_2_bias", layer_index)
# output = inputs + mlp_output
# del inputs, mlp_output, layer_materializations, hyperblock_mlp_factors                                                                      # <<< THOG release optional compact UP/DOWN factors
# ^^^ THOG

# vvv THOG retired PLASTIC DEPTH hold-controller source preserved for history audit
# plastic__layer_count_hold_updates: int = 100
# isinstance(self.plastic__layer_count_hold_updates, bool)
# or not isinstance(self.plastic__layer_count_hold_updates, int)
# or self.plastic__layer_count_hold_updates <= 0
# "plastic__layer_count_hold_updates must be a positive integer; "
# f"got {self.plastic__layer_count_hold_updates!r}"
# ^^^ THOG
