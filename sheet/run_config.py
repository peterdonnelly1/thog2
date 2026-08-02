# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, Optional

from .basis import BASIS_VERSION
from .bases import basis_version_for_family, normalize_registered_basis_family
# vvv THOG lapped cosine run controls and version identity
from .bases.lapped_cosine import (
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    lapped_cosine_basis_version,
    normalize_lapped_cosine_basis_version,
)
# ^^^ THOG
from .geometry_registry import validate_resolved_geometry_plan
# vvv THOG coupled field machine HYPERBLOCK run identity is separate from selector-based BLOCK geometry
from .hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    HyperblockOrders,
    ResolvedHyperblockPlan,
)
# ^^^ THOG
from .compact_identity import (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_CONVENTIONAL,
    DEFAULT_MLP_HIDDEN_COMPRESSOR,
    DEFAULT_MLP_HIDDEN_GROUP_SIZE,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_JPEG_LIKE_V1,
    MLP_GEOMETRY_JPEG_LIKE_V1,
    compact_identity_metadata,
    resolve_compact_selectors,
)
from .residual_init import (
    DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE,
    DEFAULT_RESIDUAL_INIT_DEPTH_VALUE,
    DEFAULT_RESIDUAL_INIT_POLICY,
    ResidualInitConfig,
)
from .run_naming import DEFAULT_COMPONENT_LIMIT, artifact_paths, dataset_label, normalize_component, truncate_component
from .training_config import ROW_ORDER_SCALING_RULE, TrainingConfig


PUBLIC_MODEL_TYPES = ("dense", "sheet")
INTERNAL_MODEL_TYPES = {"dense": "dense", "sheet": "thog2_sheet"}
DEFAULT_O_ATTN_D_MODEL = 64
DEFAULT_O_ATTN_QKV_PER_CHANNEL = 6
DEFAULT_O_ATTN_OUT_PER_CHANNEL = 6
DEFAULT_O_MLP_D_MODEL = 64
DEFAULT_O_MLP_HIDDEN = 256
DEFAULT_MLP_CHANNEL_ORDER = DEFAULT_O_MLP_HIDDEN                                                                                                      # <<< THOG retained module constant name for callers while public configuration uses o_mlp_hidden
DEFAULT_EXPERIMENT_PREFIX = "NEL" + "SON"

_RESIDUAL_POLICY_LABELS = {"depth_scaled": "ds", "unscaled": "us"}                                                                                 # <<< THOG descriptor v2 abbreviates long residual-init values only
_RESIDUAL_DEPTH_SOURCE_LABELS = {"true_layer_depth": "tld", "dof_implied_depth": "did", "user_forced_depth": "ufd"}                              # <<< THOG descriptor v2 keeps getopts fields and shortens values


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
    wandb_root: str = "wandb"

    max_iters: int = 100
    # vvv THOG optional wall-clock stop for equal-time geometry grids
    max_wall_minutes: int = 0
    # ^^^ THOG
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
    # vvv THOG stratified layer-dropout run controls; omitted values preserve all-layer execution
    layer_dropout_stratum_size: Optional[int] = None
    layer_dropout_active_per_stratum: Optional[int] = None
    layer_dropout_resample_steps: int = 1
    # ^^^ THOG
    o_depth: int = 16
    o_attn_d_model: int = DEFAULT_O_ATTN_D_MODEL
    o_attn_qkv_per_channel: int = DEFAULT_O_ATTN_QKV_PER_CHANNEL
    o_attn_out_per_channel: int = DEFAULT_O_ATTN_OUT_PER_CHANNEL
    o_mlp_d_model: int = DEFAULT_O_MLP_D_MODEL
    o_mlp_hidden: int = DEFAULT_O_MLP_HIDDEN
    mlp_hidden_group_size: int = DEFAULT_MLP_HIDDEN_GROUP_SIZE
    mlp_hidden_compressor: str = DEFAULT_MLP_HIDDEN_COMPRESSOR
    depth_compress_layer_norm_and_bias: bool = False                                                                                                   # <<< THOG DEPTH-only LayerNorm/bias participation switch

    geometry_preset: Optional[str] = GEOMETRY_PRESET_DEPTH
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = BASIS_FAMILY_CHEBYSHEV
    basis_version: str = BASIS_VERSION
    resolved_geometry_plan: Optional[Dict[str, Any]] = None
    # vvv THOG fixed coupled-field HYPERBLOCK controls
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
    hyperblock_loop_count: int = 1
    hyperblock_loop_decay: float = 1.0
    # ^^^ THOG
    lapped_cosine_window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                                                              # <<< THOG locality control
    lapped_cosine_overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION                                                                      # <<< THOG overlap control
    attention_backend: str = "auto"
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX
    run_start_label: Optional[str] = None

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
    # vvv THOG public bounded non-finite recovery controls
    nonfinite_update_policy: str = "skip"
    max_nonfinite_update_skips: int = 10
    # ^^^ THOG
    dropout: float = 0.0
    bias: bool = True

    model_seed: int = 1337
    data_seed: int = 7331
    device: str = "cuda"
    dtype: str = "bfloat16"

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
        if self.attention_backend not in ("auto", "flash2", "sdpa", "math"):
            raise ValueError("attention_backend must be auto, flash2, sdpa, or math")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.wandb_mode == "disabled" and self.wandb_enabled:
            object.__setattr__(self, "wandb_enabled", False)
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 training is not supported on CPU")
        if not isinstance(self.depth_compress_layer_norm_and_bias, bool):
            raise ValueError(
                "depth_compress_layer_norm_and_bias must be bool; "
                f"got {self.depth_compress_layer_norm_and_bias!r}"
            )

        object.__setattr__(self, "experiment_prefix", normalize_component(self.experiment_prefix, uppercase=True))
        if self.resolved_geometry_plan is not None:
            if self.model_type != "sheet":
                raise ValueError("resolved_geometry_plan is defined only for sheet model_type")
            object.__setattr__(self, "resolved_geometry_plan", validate_resolved_geometry_plan(self.resolved_geometry_plan))
        object.__setattr__(self, "mlp_hidden_compressor", normalize_registered_basis_family(self.mlp_hidden_compressor))
        if self.run_start_label is not None:
            object.__setattr__(self, "run_start_label", normalize_component(self.run_start_label))

        # vvv THOG the v0 --hyperblock flag resolves to the sole implemented coupled-field topology
        if self.hyperblock_enabled:
            if self.model_type != "sheet":
                raise ValueError("HYPERBLOCK requires model_type='sheet'")
            conflicts = {
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
                "resolved_geometry_plan": self.resolved_geometry_plan,
            }
            active_conflicts = {
                name: value for name, value in conflicts.items() if value is not None
            }
            if active_conflicts:
                raise ValueError(
                    "HYPERBLOCK may not be combined with legacy geometry controls; "
                    f"got {active_conflicts}"
                )
            if self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "HYPERBLOCK keeps LayerNorm and bias vectors conventional in v0"
                )
            plan = self.hyperblock_plan()
            object.__setattr__(self, "hyperblock_topology", plan.topology)
            object.__setattr__(self, "hyperblock_compressor", plan.compressor_family)
            object.__setattr__(self, "hyperblock_compressor_version", plan.compressor_version)
        # ^^^ THOG

        # vvv THOG derive and validate the parameterised lapped basis version from explicit controls
        requested_family = self.basis_family or BASIS_FAMILY_CHEBYSHEV
        canonical_family = (
            requested_family
            if requested_family == BASIS_FAMILY_CONVENTIONAL
            else normalize_registered_basis_family(requested_family)
        )
        # vvv THOG preserve the pre-HYPERBLOCK lapped-cosine branch line for source history
        # if canonical_family == BASIS_FAMILY_LAPPED_COSINE:
        # ^^^ THOG
        if not self.hyperblock_enabled and canonical_family == BASIS_FAMILY_LAPPED_COSINE:
            control_version = lapped_cosine_basis_version(
                self.lapped_cosine_window_length,
                self.lapped_cosine_overlap_fraction,
            )
            if self.basis_version in ("auto", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
                object.__setattr__(self, "basis_version", control_version)
            else:
                explicit_version = normalize_lapped_cosine_basis_version(self.basis_version)
                if explicit_version != control_version:
                    raise ValueError(
                        "lapped cosine controls do not match explicit basis_version: "
                        f"controls imply {control_version!r}, got {explicit_version!r}"
                    )
                object.__setattr__(self, "basis_version", explicit_version)
        elif not self.hyperblock_enabled:
            if (
                self.lapped_cosine_window_length != DEFAULT_LAPPED_COSINE_WINDOW_LENGTH
                or abs(float(self.lapped_cosine_overlap_fraction) - DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION) > 1.0e-12
            ):
                raise ValueError(
                    "lapped cosine controls may be changed only when basis_family='lapped_cosine'"
                )
            if self.basis_version == "auto":
                resolved_version = BASIS_VERSION if canonical_family == BASIS_FAMILY_CONVENTIONAL else basis_version_for_family(canonical_family)
                object.__setattr__(self, "basis_version", resolved_version)
        # ^^^ THOG

        # vvv THOG DEPTH has no within-tensor order controls; canonicalize them before validation, naming, and checkpoint construction.
        if self.model_type == "sheet" and not self.hyperblock_enabled:
            selectors = resolve_compact_selectors(
                geometry_preset=self.geometry_preset,
                attention_geometry=self.attention_geometry,
                mlp_geometry=self.mlp_geometry,
                basis_family=self.basis_family,
            )
            if selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
                object.__setattr__(self, "o_attn_d_model", 1)
                object.__setattr__(self, "o_attn_qkv_per_channel", 1)
                object.__setattr__(self, "o_attn_out_per_channel", 1)
                object.__setattr__(self, "o_mlp_d_model", 1)
                object.__setattr__(self, "o_mlp_hidden", 1)
            elif self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "depth_compress_layer_norm_and_bias may be enabled only for geometry_preset='depth'"
                )
        elif self.depth_compress_layer_norm_and_bias:
            raise ValueError(
                "depth_compress_layer_norm_and_bias may be enabled only for geometry_preset='depth'"
            )
        # ^^^ THOG

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
            "layer_dropout_resample_steps",
            "mlp_hidden_group_size",
            "hyperblock_common_family_order",
            "hyperblock_attention_family_order",
            "hyperblock_mlp_family_order",
            "hyperblock_depth_order",
            "hyperblock_d_model_order",
            "hyperblock_mlp_hidden_order",
            "hyperblock_attention_head_order",
            "hyperblock_attention_head_channel_order",
            "hyperblock_mlp_hidden_multiplier",
            "hyperblock_loop_count",
            "checkpoint_segment_size",
            "artifact_name_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # vvv THOG shared-factory loop controls are orthogonal to HYPERBLOCK basis geometry
        if (
            isinstance(self.hyperblock_loop_decay, bool)
            or not isinstance(self.hyperblock_loop_decay, (int, float))
            or not math.isfinite(float(self.hyperblock_loop_decay))
            or not 0.0 < float(self.hyperblock_loop_decay) <= 1.0
        ):
            raise ValueError("hyperblock_loop_decay must be finite and in (0, 1]")
        if not self.hyperblock_enabled and (
            self.hyperblock_loop_count != 1
            or float(self.hyperblock_loop_decay) != 1.0
        ):
            raise ValueError("HYPERBLOCK loop controls require HYPERBLOCK")
        # ^^^ THOG

        for name in ("max_wall_minutes", "checkpoint_interval", "warmup_iters", "model_seed", "data_seed", "max_nonfinite_update_skips"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        # vvv THOG resolve layer-dropout defaults from the canonical depth and validate exact equal strata
        stratum_size = self.n_layer if self.layer_dropout_stratum_size is None else self.layer_dropout_stratum_size
        if isinstance(stratum_size, bool) or not isinstance(stratum_size, int) or stratum_size < 1:
            raise ValueError("layer_dropout_stratum_size must be a positive integer")
        active_per_stratum = stratum_size if self.layer_dropout_active_per_stratum is None else self.layer_dropout_active_per_stratum
        if isinstance(active_per_stratum, bool) or not isinstance(active_per_stratum, int) or active_per_stratum < 1:
            raise ValueError("layer_dropout_active_per_stratum must be a positive integer")
        if self.n_layer % stratum_size != 0:
            raise ValueError(
                "n_layer must be divisible by layer_dropout_stratum_size; "
                f"got n_layer={self.n_layer}, stratum_size={stratum_size}"
            )
        if active_per_stratum > stratum_size:
            raise ValueError(
                "layer_dropout_active_per_stratum must not exceed layer_dropout_stratum_size; "
                f"got active={active_per_stratum}, stratum_size={stratum_size}"
            )
        object.__setattr__(self, "layer_dropout_stratum_size", stratum_size)
        object.__setattr__(self, "layer_dropout_active_per_stratum", active_per_stratum)
        # ^^^ THOG
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be less than max_iters")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.model_type == "sheet" and not self.hyperblock_enabled:
            order_limits = {
                "o_depth": self.n_layer,
                "o_attn_d_model": self.n_embd,
                "o_attn_qkv_per_channel": self.head_dim,
                "o_attn_out_per_channel": self.head_dim,
                "o_mlp_d_model": self.n_embd,
                "o_mlp_hidden": 4 * self.n_embd,
            }
            for name, limit in order_limits.items():
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
                if value > limit:
                    raise ValueError(f"{name} must not exceed {limit}")
            object.__setattr__(self, "basis_version", str(self.compact_identity()["basis_version"]))
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
        if self.nonfinite_update_policy not in ("raise", "skip"):
            raise ValueError("nonfinite_update_policy must be raise or skip")
        if self.weight_decay < 0.0 or self.grad_clip < 0.0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    # vvv THOG HYPERBLOCK topology is explicit resolved identity even though v0 exposes a Boolean user switch
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

    # vvv THOG derived layer-dropout quantities; these are not independent user controls
    @property
    def layer_dropout_n_strata(self) -> int:
        return self.n_layer // int(self.layer_dropout_stratum_size)

    @property
    def n_active_layers(self) -> int:
        return self.layer_dropout_n_strata * int(self.layer_dropout_active_per_stratum)

    @property
    def layer_dropout_enabled(self) -> bool:
        return int(self.layer_dropout_active_per_stratum) < int(self.layer_dropout_stratum_size)
    # ^^^ THOG

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(
            policy=self.residual_init_policy,
            depth_source=self.residual_init_depth_source,
            depth_value=self.residual_init_depth_value,
        )

    def residual_init_artifact_fragment(self) -> str:
        residual_init = self.residual_init_config()
        policy_label = _RESIDUAL_POLICY_LABELS.get(self.residual_init_policy, normalize_component(self.residual_init_policy))
        depth_source_label = _RESIDUAL_DEPTH_SOURCE_LABELS.get(residual_init.depth_source, normalize_component(residual_init.depth_source))
        parts = [f"r_{policy_label}", f"z_{depth_source_label}"]                                                                                       # <<< THOG descriptor v2 keeps getopts letters and abbreviates long values
        if residual_init.depth_source == "user_forced_depth":
            parts.append(f"Z_{self.residual_init_depth_value}")
        return "_".join(parts)

    @property
    def internal_model_type(self) -> str:
        return INTERNAL_MODEL_TYPES[self.model_type]

    @property
    def artifact_prefix(self) -> str:
        return "DENSE2" if self.model_type == "dense" else "SHEET"

    def compact_identity(self) -> Dict[str, Any]:
        if self.model_type != "sheet":
            raise ValueError("compact identity is only defined for SHEET runs")
        if self.hyperblock_enabled:
            return {
                "model_type": self.model_type,
                "hyperblock": self.hyperblock_plan().identity(),
                "hyperblock_loop": {
                    "count": self.hyperblock_loop_count,
                    "decay": float(self.hyperblock_loop_decay),
                },
                "n_layer": self.n_layer,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "bias": self.bias,
            }
        identity = compact_identity_metadata(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            o_depth=self.o_depth,
            o_attn_d_model=self.o_attn_d_model,
            o_attn_qkv_per_channel=self.o_attn_qkv_per_channel,
            o_attn_out_per_channel=self.o_attn_out_per_channel,
            o_mlp_d_model=self.o_mlp_d_model,
            o_mlp_hidden=self.o_mlp_hidden,
            mlp_hidden_group_size=self.mlp_hidden_group_size,
            mlp_hidden_compressor=self.mlp_hidden_compressor,
            basis_version=self.basis_version,
            lapped_cosine_window_length=self.lapped_cosine_window_length,                                                                                # <<< THOG compact identity locality control
            lapped_cosine_overlap_fraction=self.lapped_cosine_overlap_fraction,                                                                          # <<< THOG compact identity overlap control
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )
        if self.resolved_geometry_plan is not None:
            identity["resolved_geometry_plan"] = self.resolved_geometry_plan
        return identity

    def _selector_fragment(self, selector: str) -> str:
        return normalize_component(selector.replace(".", "_"))

    def _geometry_slots_from_resolved_plan(self, plan: Dict[str, Any]) -> str:
        fields: list[str] = []
        if plan.get("depth_enabled"):
            fields.append(f"G0_{plan.get('depth_compressor')}")
        for index, selection in enumerate(plan.get("selections", []), start=1):
            fields.append(f"G{index}_{selection['compressor']}_{self._selector_fragment(selection['selector'])}")
        if not fields:
            raise ValueError("resolved geometry plan produced no descriptor slots")
        return "_".join(fields)

    def _legacy_geometry_slots(self, identity: Dict[str, Any]) -> str:
        basis_family = str(identity["basis_family"])
        preset = str(identity["geometry_preset"])
        if preset == GEOMETRY_PRESET_DEPTH:
            return f"G0_{basis_family}"
        if preset == GEOMETRY_PRESET_JPEG_LIKE_V1 or identity["mlp_geometry"] == MLP_GEOMETRY_JPEG_LIKE_V1:
            return f"G0_{basis_family}_G1_jpeg_like_MLP_UP_MLP_HIDDEN"
        return f"G0_{basis_family}_{normalize_component(preset)}"                                                                                       # <<< THOG legacy presets wither as resolved-ish compatibility labels

    def compact_artifact_fragment(self) -> Optional[str]:
        if self.model_type != "sheet":
            return None
        if self.hyperblock_enabled:
            return f"HB_{normalize_component(self.hyperblock_compressor)}"
        if self.resolved_geometry_plan is not None:
            return self._geometry_slots_from_resolved_plan(self.resolved_geometry_plan)                                                                  # <<< THOG descriptor v2 names systematic geometry by ordered G slots
        return self._legacy_geometry_slots(self.compact_identity())

    def run_descriptor(self) -> str:
        geometry_fragment = self.compact_artifact_fragment() or "DENSE"
        host = normalize_component(self.host_label)
        # body = f"{host}_{self.experiment_prefix}_{geometry_fragment}"                                                                                  # <<< THOG preserved one-space descriptor separator
        body = f"{host}_{self.experiment_prefix}___{geometry_fragment}"                                                                                 # <<< THOG three descriptor spaces after RUN_NAME
        return f"{self.run_start_label}_{body}" if self.run_start_label else body                                                                        # <<< THOG descriptor v2 places host immediately after timestamp when present

    def _learning_rate_code(self, value: float) -> int:
        return int(round(value / 1.0e-5))

    # vvv THOG filename-safe compact scalar label for non-default HYPERBLOCK loop decay
    @staticmethod
    def _artifact_float(value: float) -> str:
        return format(float(value), ".6g").replace("-", "m").replace(".", "p")
    # ^^^ THOG

    def _order_label_for_axis(self, *, element: str, axis: str) -> str:
        if axis == "MLP_HIDDEN":
            return "Y"
        if axis == "MLP_D_MODEL":
            return "X"
        if axis == "ATTENTION_D_MODEL":
            return "Q"
        if axis == "ATTENTION_HEAD_CHANNEL" and element == "ATTENTION_QKV":
            return "J"
        if axis == "ATTENTION_HEAD_CHANNEL" and element == "ATTENTION_OUTPUT":
            return "O"
        raise ValueError(f"no descriptor order label for {element}.{axis}")

    def _resolved_order_fields(self, plan: Dict[str, Any]) -> list[str]:
        fields: list[str] = []
        if plan.get("depth_enabled"):
            fields.append(f"P_{self.o_depth}")
        emitted: set[str] = set()
        for selection in plan.get("selections", []):
            element = str(selection["element"])
            for axis in selection.get("compressed_axes", []):
                label = self._order_label_for_axis(element=element, axis=str(axis))
                if label not in emitted:
                    fields.append(f"{label}_{selection['orders'][axis]}")
                    emitted.add(label)
                group_size = selection.get("axis_options", {}).get(axis, {}).get("group_size")
                if group_size is not None and "s" not in emitted:
                    fields.append(f"s_{int(group_size)}")
                    emitted.add("s")
        if plan.get("depth_enabled"):
            fields.append(f"DLB_{int(self.depth_compress_layer_norm_and_bias)}")
        if self._uses_lapped_cosine(plan=plan):
            overlap_percent = int(round(self.lapped_cosine_overlap_fraction * 100.0))
            fields.extend([f"W_{self.lapped_cosine_window_length}", f"i_{overlap_percent}"])
        return fields

    def _uses_lapped_cosine(self, *, plan: Optional[Dict[str, Any]] = None, identity: Optional[Dict[str, Any]] = None) -> bool:
        if plan is not None:
            if plan.get("depth_compressor") == BASIS_FAMILY_LAPPED_COSINE:
                return True
            return any(selection.get("compressor") == BASIS_FAMILY_LAPPED_COSINE for selection in plan.get("selections", []))
        return bool(identity is not None and identity.get("basis_family") == BASIS_FAMILY_LAPPED_COSINE)

    def _legacy_order_fields(self, identity: Dict[str, Any]) -> list[str]:
        fields = [f"P_{self.o_depth}"]
        if identity["geometry_preset"] == GEOMETRY_PRESET_DEPTH:
            fields.append(f"DLB_{int(self.depth_compress_layer_norm_and_bias)}")
        else:
            fields.extend([
                f"Q_{self.o_attn_d_model}",
                f"J_{self.o_attn_qkv_per_channel}",
                f"O_{self.o_attn_out_per_channel}",
                f"X_{self.o_mlp_d_model}",
                f"Y_{self.o_mlp_hidden}",
            ])
            if identity["mlp_geometry"] == MLP_GEOMETRY_JPEG_LIKE_V1:
                fields.append(f"s_{self.mlp_hidden_group_size}")
        if self._uses_lapped_cosine(identity=identity):
            overlap_percent = int(round(self.lapped_cosine_overlap_fraction * 100.0))
            fields.extend([f"W_{self.lapped_cosine_window_length}", f"i_{overlap_percent}"])
        return fields

    def parameter_artifact_fragment(self) -> str:
        fit_fields = [
            f"A_{self.gradient_accumulation_steps}",
            f"b_{self.batch_size}",
            f"c_{self._learning_rate_code(self.learning_rate)}",
            f"d_{dataset_label(self.dataset)}",
            f"f_{self._learning_rate_code(self.min_lr)}",
            f"w_{self.warmup_iters}",
        ]
        # vvv THOG only stochastic layer-dropout runs acquire descriptor fields; all-active names remain unchanged
        if self.layer_dropout_enabled:
            fit_fields.extend([
                f"LDs_{self.layer_dropout_stratum_size}",
                f"LDa_{self.layer_dropout_active_per_stratum}",
                f"LDr_{self.layer_dropout_resample_steps}",
            ])
        # ^^^ THOG
        shape_fields = [
            f"C_{self.block_size}",
            f"D_{self.n_embd}",
            f"H_{self.n_head}",
            f"L_{self.n_layer}",
        ]
        sections = ["_".join(fit_fields), "_".join(shape_fields)]
        if self.model_type == "sheet":
            if self.hyperblock_enabled:
                order_fields = [
                    f"HFC_{self.hyperblock_common_family_order}",
                    f"HFA_{self.hyperblock_attention_family_order}",
                    f"HFM_{self.hyperblock_mlp_family_order}",
                    f"HL_{self.hyperblock_depth_order}",
                    f"HD_{self.hyperblock_d_model_order}",
                    f"HM_{self.hyperblock_mlp_hidden_order}",
                    f"HH_{self.hyperblock_attention_head_order}",
                    f"HC_{self.hyperblock_attention_head_channel_order}",
                ]
                # vvv THOG preserve baseline artifact names while naming only active recurrence experiments
                if self.hyperblock_loop_count != 1 or float(self.hyperblock_loop_decay) != 1.0:
                    order_fields.extend([
                        f"HLC_{self.hyperblock_loop_count}",
                        f"HLD_{self._artifact_float(self.hyperblock_loop_decay)}",
                    ])
                # ^^^ THOG
            elif self.resolved_geometry_plan is not None:
                order_fields = self._resolved_order_fields(self.resolved_geometry_plan)
            else:
                order_fields = self._legacy_order_fields(self.compact_identity())
            if order_fields:
                sections.append("_".join(order_fields))                                                                                                 # <<< THOG descriptor v2 separates fit, shape, orders, and init with double underscores
        sections.append("_".join([self.residual_init_artifact_fragment(), f"S_{self.checkpoint_segment_size}"]))
        return "__".join(sections)

    @property
    def artifact_name(self) -> str:
        artifact_name = f"{self.run_descriptor()}__{self.parameter_artifact_fragment()}"
        if self.artifact_suffix:
            artifact_name = f"{artifact_name}__{normalize_component(self.artifact_suffix, uppercase=True)}"
        return truncate_component(artifact_name, max_length=self.artifact_name_limit)

    def paths(self, *, log_timestamp: Optional[str] = None) -> Dict[str, Path]:
        timestamp = None if self.run_start_label else log_timestamp
        return artifact_paths(
            self.artifact_name,
            checkpoint_root=Path(self.checkpoint_root),
            log_root=Path(self.log_root),
            result_root=Path(self.result_root),
            log_timestamp=timestamp,
        )

    def local_gradient_accumulation_steps(self, world_size: int) -> int:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if self.gradient_accumulation_steps % world_size != 0:
            raise ValueError("global gradient_accumulation_steps must be divisible by world_size")
        return self.gradient_accumulation_steps // world_size

    def tokens_per_iter(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.block_size

    def to_training_config(self, *, vocab_size: int, world_size: int, out_dir: Path) -> TrainingConfig:
        sheet_kwargs: Dict[str, Any]
        if self.model_type == "sheet":
            if self.hyperblock_enabled:
                sheet_kwargs = {
                    "depth_compress_layer_norm_and_bias": False,
                    "geometry_preset": None,
                    "attention_geometry": None,
                    "mlp_geometry": None,
                    "basis_family": None,
                    "resolved_geometry_plan": None,
                    "hyperblock_topology": self.hyperblock_topology,
                    "hyperblock_compressor": self.hyperblock_compressor,
                    "hyperblock_compressor_version": self.hyperblock_compressor_version,
                    "hyperblock_common_family_order": self.hyperblock_common_family_order,
                    "hyperblock_attention_family_order": self.hyperblock_attention_family_order,
                    "hyperblock_mlp_family_order": self.hyperblock_mlp_family_order,
                    "hyperblock_depth_order": self.hyperblock_depth_order,
                    "hyperblock_d_model_order": self.hyperblock_d_model_order,
                    "hyperblock_mlp_hidden_order": self.hyperblock_mlp_hidden_order,
                    "hyperblock_attention_head_order": self.hyperblock_attention_head_order,
                    "hyperblock_attention_head_channel_order": self.hyperblock_attention_head_channel_order,
                    "hyperblock_mlp_hidden_multiplier": self.hyperblock_mlp_hidden_multiplier,
                    "hyperblock_loop_count": self.hyperblock_loop_count,
                    "hyperblock_loop_decay": float(self.hyperblock_loop_decay),
                }
            else:
                sheet_kwargs = {
                "depth_order": self.o_depth,
                "base_row_order": self.o_attn_d_model,
                "mlp_channel_order": self.o_mlp_hidden,
                "o_attn_d_model": self.o_attn_d_model,
                "o_attn_qkv_per_channel": self.o_attn_qkv_per_channel,
                "o_attn_out_per_channel": self.o_attn_out_per_channel,
                "o_mlp_d_model": self.o_mlp_d_model,
                "o_mlp_hidden": self.o_mlp_hidden,
                "mlp_hidden_group_size": self.mlp_hidden_group_size,
                "mlp_hidden_compressor": self.mlp_hidden_compressor,
                "depth_compress_layer_norm_and_bias": self.depth_compress_layer_norm_and_bias,                                                         # <<< THOG checkpoint and model vector mode
                "basis_version": self.basis_version,
                "lapped_cosine_window_length": self.lapped_cosine_window_length,                                                                         # <<< THOG checkpoint locality control
                "lapped_cosine_overlap_fraction": self.lapped_cosine_overlap_fraction,                                                                    # <<< THOG checkpoint overlap control
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
                    "resolved_geometry_plan": self.resolved_geometry_plan,
                }
        else:
            sheet_kwargs = {
                "depth_order": 1,
                "base_row_order": 1,
                "mlp_channel_order": None,
                "o_attn_d_model": None,
                "o_attn_qkv_per_channel": None,
                "o_attn_out_per_channel": None,
                "o_mlp_d_model": None,
                "o_mlp_hidden": None,
                "mlp_hidden_group_size": DEFAULT_MLP_HIDDEN_GROUP_SIZE,
                "mlp_hidden_compressor": DEFAULT_MLP_HIDDEN_COMPRESSOR,
                "depth_compress_layer_norm_and_bias": False,
                "basis_version": BASIS_VERSION,
                "geometry_preset": None,
                "attention_geometry": None,
                "mlp_geometry": None,
                "basis_family": None,
                "resolved_geometry_plan": None,
            }
        return TrainingConfig(
            model_type=self.internal_model_type,
            block_size=self.block_size,
            vocab_size=vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            **sheet_kwargs,
            residual_init_policy=self.residual_init_policy,
            residual_init_depth_source=self.residual_init_depth_source,
            residual_init_depth_value=self.residual_init_depth_value,
            checkpoint_segment_size=self.checkpoint_segment_size if self.activation_checkpointing else 0,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.local_gradient_accumulation_steps(world_size),
            layer_dropout_stratum_size=self.layer_dropout_stratum_size,                                                                                   # <<< THOG pass stratified layer-dropout controls into trainer config
            layer_dropout_active_per_stratum=self.layer_dropout_active_per_stratum,                                                                         # <<< THOG pass active cardinality per stratum
            layer_dropout_resample_steps=self.layer_dropout_resample_steps,                                                                                 # <<< THOG pass selection lifetime in optimizer updates
            max_updates=self.max_iters,
            max_wall_minutes=self.max_wall_minutes,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_lr,
            warmup_updates=self.warmup_iters,
            decay_updates=self.max_iters,
            decay_learning_rate=True,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
            nonfinite_update_policy=self.nonfinite_update_policy,
            max_nonfinite_update_skips=self.max_nonfinite_update_skips,
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
            for name in (
                "o_depth",
                "o_attn_d_model",
                "o_attn_qkv_per_channel",
                "o_attn_out_per_channel",
                "o_mlp_d_model",
                "o_mlp_hidden",
                "mlp_hidden_group_size",
                "mlp_hidden_compressor",
                "depth_compress_layer_norm_and_bias",
                "geometry_preset",
                "attention_geometry",
                "mlp_geometry",
                "basis_family",
                "basis_version",
                "hyperblock_topology",
                "hyperblock_compressor",
                "hyperblock_compressor_version",
                "hyperblock_common_family_order",
                "hyperblock_attention_family_order",
                "hyperblock_mlp_family_order",
                "hyperblock_depth_order",
                "hyperblock_d_model_order",
                "hyperblock_mlp_hidden_order",
                "hyperblock_attention_head_order",
                "hyperblock_attention_head_channel_order",
                "hyperblock_mlp_hidden_multiplier",
                "hyperblock_loop_count",
                "hyperblock_loop_decay",
            ):
                values.pop(name, None)
        else:
            values["compact_identity"] = self.compact_identity()
            values["compact_artifact_fragment"] = self.compact_artifact_fragment()
        values.update({
            "artifact_name": self.artifact_name,
            "artifact_prefix": self.artifact_prefix,
            "run_descriptor": self.run_descriptor(),
            "world_size": world_size,
            "local_gradient_accumulation_steps": self.local_gradient_accumulation_steps(world_size),
            "tokens_per_iter": self.tokens_per_iter(),
            "layer_dropout_n_strata": self.layer_dropout_n_strata,                                                                                      # <<< THOG expose derived stratum count in run identity/telemetry
            "n_active_layers": self.n_active_layers,                                                                                                    # <<< THOG expose exact active depth per training update
            "layer_dropout_enabled": self.layer_dropout_enabled,                                                                                        # <<< THOG make degenerate path explicit in resolved config
        })
        return values


__all__ = [
    "DEFAULT_EXPERIMENT_PREFIX",
    "DEFAULT_MLP_CHANNEL_ORDER",
    "DEFAULT_O_ATTN_D_MODEL",
    "DEFAULT_O_ATTN_QKV_PER_CHANNEL",
    "DEFAULT_O_ATTN_OUT_PER_CHANNEL",
    "DEFAULT_O_MLP_D_MODEL",
    "DEFAULT_O_MLP_HIDDEN",
    "INTERNAL_MODEL_TYPES",
    "OwtRunConfig",
    "PUBLIC_MODEL_TYPES",
]
# ^^^ THOG
# vvv THOG preserved superseded source lines for exact history audit
# body = f"{host}_{self.experiment_prefix}_{geometry_fragment}"
# ^^^ THOG
