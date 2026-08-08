# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional

from .basis import BASIS_VERSION
# vvv THOG lapped cosine controls survive training config and checkpoints
from .bases import normalize_registered_basis_family
from .bases.lapped_cosine import (
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    lapped_cosine_basis_version,
)
# ^^^ THOG
from .checkpointing import validate_checkpoint_segment_size
from .compact_identity import (
    DEFAULT_MLP_HIDDEN_COMPRESSOR,
    DEFAULT_MLP_HIDDEN_GROUP_SIZE,
    GEOMETRY_PRESET_DEPTH,
    compact_identity_metadata,
    conventional_identity_metadata,
    resolve_compact_selectors,
    validate_dense_compact_fields,
)
from .geometry import derive_row_order
from .geometry_registry import validate_resolved_geometry_plan
# vvv THOG coupled field machine HYPERBLOCK has checkpoint identity independent of legacy geometry selectors
from .hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    HyperblockOrders,
    ResolvedHyperblockPlan,
)
# ^^^ THOG
# vvv THOG PLASTIC COARSE/FINE lifecycle configuration and candidate resolution
from .plastic_depth_coarse import (
    resolve_plastic_coarse_config,
    resolve_plastic_probe_interval,
    validate_plastic_fine_count_controls,
)
# ^^^ THOG
from .residual_init import DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE, DEFAULT_RESIDUAL_INIT_DEPTH_VALUE, DEFAULT_RESIDUAL_INIT_POLICY, ResidualInitConfig
# vvv THOG PLASTIC DEPTH configuration resolves a fixed persistent lattice and optional discrete active-count controller
from .plastic_depth import (
    PLASTIC_DEPTH_VERSION,
    plastic_depth_identity_metadata,
    resolve_plastic_depth_counts,
    validate_plastic_layer_count_objective,
    validate_plastic_sampling_initialisation,
)
# ^^^ THOG


CHECKPOINT_SCHEMA_VERSION = 2
ROW_ORDER_SCALING_RULE = "proportional_ceil_v1"
MODEL_TYPES = ("dense", "thog2_sheet")
EXECUTION_OVERRIDE_FIELDS = {"device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips"}
# vvv THOG PLASTIC DEPTH fields are omitted from persistent disabled-run metadata to preserve the exact pre-feature identity
PLASTIC_TRAINING_CONFIG_FIELDS = (
    "plastic__enabled",
    "plastic__runtime_phase",
    "plastic__coarse_phase",
    "plastic__coarse_phase_roll_through",
    "plastic__log_interval_coarse",
    "plastic__phase_1_n_steps",
    "plastic__phase_1_starting_layer_count",
    "plastic__phase_1__number_of_trials",
    "plastic__phase_1_evaluation_steps_count",
    "plastic__layers_to_sample",
    "plastic__do_learn_layer_count",
    "plastic__initial_layer_count",
    "plastic__max_permitted_layers",
    "plastic__layer_sampling_initialisation",
    "plastic__layer_count_objective",
    "plastic__layer_count_update_brake",
    "plastic__layer_count_probe__probe_every_n_steps",
    "plastic__layer_count_probe__number_of_sampled_valid_tokens",
    "plastic__layer_count_probe_radius",
    "plastic__layer_count__max_allowable_layer_change",
    "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence",
    "plastic__layer_count_probe__window_size_as_number_of_probes",
    "plastic__layer_count_probe_noise_lambda",
    "plastic__wall_time_equivalent_time_gain_discount",
    "plastic__wall_time_equivalent_time_gain_loss_rate_window",
    "plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",
    "plastic__layer_count_cost_weight",
    "plastic__layer_memory_budget_gib",
    "plastic__cuda_allocator_reserve_gib",
    "plastic__geometry_learning_rate_multiplier",
    "plastic__freeze_geometry_during_warmup",
    "plastic__initial_active_layers",
)
# ^^^ THOG

# vvv THOG v0.541 accept superseded PLASTIC config keys only when reconstructing existing checkpoints; new writes use canonical names
PLASTIC_V0541_RENAMED_CONFIG_FIELDS = {
    "plastic__layer_count_extrapolation_weight": "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence",
    "plastic__layer_count_max_step": "plastic__layer_count__max_allowable_layer_change",
}

def normalize_plastic_v0541_config_fields(values: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(values)
    for old_name, new_name in PLASTIC_V0541_RENAMED_CONFIG_FIELDS.items():
        if old_name not in normalized:
            continue
        old_value = normalized.pop(old_name)
        if new_name in normalized and normalized[new_name] != old_value:
            raise ValueError(
                f"conflicting PLASTIC checkpoint fields {old_name} and {new_name}"
            )
        normalized[new_name] = old_value
    return normalized
# ^^^ THOG

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
    "mlp_channel_order",
    "o_attn_d_model",
    "o_attn_qkv_per_channel",
    "o_attn_out_per_channel",
    "o_mlp_d_model",
    "o_mlp_hidden",
    "mlp_hidden_group_size",
    "mlp_hidden_compressor",
    "depth_compress_layer_norm_and_bias",                                                                                                                  # <<< THOG DEPTH vector representation is checkpoint identity
    "residual_init_policy",
    "residual_init_depth_source",
    "residual_init_depth_value",
    "basis_version",
    "lapped_cosine_window_length",                                                                                                                         # <<< THOG checkpoint compatibility locality control
    "lapped_cosine_overlap_fraction",                                                                                                                      # <<< THOG checkpoint compatibility overlap control
    "row_order_scaling_rule",
    "geometry_preset",
    "attention_geometry",
    "mlp_geometry",
    "basis_family",
    "resolved_geometry_plan",
    # vvv THOG HYPERBLOCK topology, basis and anisotropic retained orders are model compatibility identity
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
    # ^^^ THOG
    # vvv THOG PLASTIC DEPTH compatibility is carried by compact_identity so pre-feature schema-2 checkpoints remain resumable
    # ^^^ THOG
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
    mlp_channel_order: Optional[int] = None
    o_attn_d_model: Optional[int] = None                                                                                                                   # <<< THOG final attention model-axis order
    o_attn_qkv_per_channel: Optional[int] = None                                                                                                           # <<< THOG final QKV per-head channel order
    o_attn_out_per_channel: Optional[int] = None                                                                                                           # <<< THOG final output per-head channel order
    o_mlp_d_model: Optional[int] = None                                                                                                                    # <<< THOG final MLP model-axis order
    o_mlp_hidden: Optional[int] = None                                                                                                                     # <<< THOG final MLP hidden-axis order
    mlp_hidden_group_size: int = DEFAULT_MLP_HIDDEN_GROUP_SIZE
    mlp_hidden_compressor: str = DEFAULT_MLP_HIDDEN_COMPRESSOR
    depth_compress_layer_norm_and_bias: bool = False                                                                                                       # <<< THOG DEPTH-only LayerNorm/bias participation switch
    residual_init_policy: str = DEFAULT_RESIDUAL_INIT_POLICY
    residual_init_depth_source: str = DEFAULT_RESIDUAL_INIT_DEPTH_SOURCE
    residual_init_depth_value: int = DEFAULT_RESIDUAL_INIT_DEPTH_VALUE
    basis_version: str = BASIS_VERSION
    lapped_cosine_window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                                                                 # <<< THOG explicit locality control
    lapped_cosine_overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION                                                                         # <<< THOG explicit overlap control
    row_order_scaling_rule: str = ROW_ORDER_SCALING_RULE
    geometry_preset: Optional[str] = None
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = None
    resolved_geometry_plan: Optional[Dict[str, Any]] = None
    # vvv THOG fixed coupled-field HYPERBLOCK model controls
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
    # vvv THOG PLASTIC DEPTH controls; disabled is the exact established path
    plastic__enabled: bool = False
    plastic__runtime_phase: str = "fine"
    plastic__coarse_phase: str = "disabled"
    plastic__coarse_phase_roll_through: bool = False
    plastic__log_interval_coarse: int = 10
    plastic__phase_1_n_steps: Optional[int] = None
    plastic__phase_1_starting_layer_count: Optional[int] = None
    plastic__phase_1__number_of_trials: Optional[int] = None
    plastic__phase_1_evaluation_steps_count: Optional[int] = None
    plastic__layers_to_sample: Optional[int] = None
    plastic__do_learn_layer_count: bool = False
    plastic__initial_layer_count: Optional[int] = None
    plastic__max_permitted_layers: Optional[int] = None
    plastic__layer_sampling_initialisation: str = "equidistant"
    plastic__layer_count_objective: str = "lowest_loss"
    plastic__layer_count_update_brake: int = 5
    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None
    plastic__layer_count_probe__number_of_sampled_valid_tokens: int = 1024
    plastic__layer_count_probe_radius: int = 1
    plastic__layer_count__max_allowable_layer_change: int = 1
    plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence: float = 0.8
    plastic__layer_count_probe__window_size_as_number_of_probes: int = 50
    plastic__layer_count_probe_noise_lambda: float = 3.0
    plastic__wall_time_equivalent_time_gain_discount: float = 0.9
    plastic__wall_time_equivalent_time_gain_loss_rate_window: int = 64
    plastic__wall_time_equivalent_time_gain_loss_rate_min_observations: int = 16
    plastic__layer_count_cost_weight: float = 0.0
    plastic__layer_memory_budget_gib: Optional[float] = None
    plastic__cuda_allocator_reserve_gib: float = 0.5
    plastic__geometry_learning_rate_multiplier: float = 0.1
    plastic__freeze_geometry_during_warmup: bool = True
    plastic__initial_active_layers: int = 0
    # ^^^ THOG
    checkpoint_segment_size: int = 0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    # vvv THOG stratified layer-dropout controls; None resolves to the all-active current behaviour
    layer_dropout_stratum_size: Optional[int] = None
    layer_dropout_active_per_stratum: Optional[int] = None
    layer_dropout_resample_steps: int = 1
    # ^^^ THOG
    max_updates: int = 10
    # vvv THOG optional wall-clock stop for equal-time geometry comparisons
    max_wall_minutes: int = 0
    # ^^^ THOG
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    warmup_updates: int = 0
    decay_updates: int = 10
    decay_learning_rate: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # vvv THOG bounded non-finite update recovery controls
    nonfinite_update_policy: str = "skip"
    max_nonfinite_update_skips: int = 99999
    # ^^^ THOG
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
            raise ValueError(f"model_type must be one of {MODEL_TYPES}; got {self.model_type!r}")
        if self.resolved_geometry_plan is not None:
            if self.model_type != "thog2_sheet":
                raise ValueError("resolved_geometry_plan is defined only for thog2_sheet model_type")
            self.resolved_geometry_plan = validate_resolved_geometry_plan(self.resolved_geometry_plan)
        if not isinstance(self.depth_compress_layer_norm_and_bias, bool):
            raise ValueError(
                "depth_compress_layer_norm_and_bias must be bool; "
                f"got {self.depth_compress_layer_norm_and_bias!r}"
            )

        # vvv THOG resolve PLASTIC DEPTH before selector and n_layer-dependent validation
        if not isinstance(self.plastic__enabled, bool):
            raise ValueError(f"plastic__enabled must be bool; got {self.plastic__enabled!r}")
        if self.plastic__runtime_phase not in {"coarse", "fine"}:
            raise ValueError(
                "plastic__runtime_phase must be coarse or fine; "
                f"got {self.plastic__runtime_phase!r}"
            )
        if not isinstance(self.plastic__coarse_phase_roll_through, bool):
            raise ValueError("plastic__coarse_phase_roll_through must be bool")
        if (
            isinstance(self.plastic__log_interval_coarse, bool)
            or not isinstance(self.plastic__log_interval_coarse, int)
            or self.plastic__log_interval_coarse < 1
        ):
            raise ValueError("plastic__log_interval_coarse must be a positive integer")
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
        # vvv THOG resolve one-shot COARSE scheduling and canonical FINE lookahead controls before active-count construction
        resolved_coarse = resolve_plastic_coarse_config(
            coarse_phase=self.plastic__coarse_phase,
            plastic_enabled=self.plastic__enabled,
            do_learn_layer_count=self.plastic__do_learn_layer_count,
            n_steps=self.plastic__phase_1_n_steps,
            starting_layer_count=self.plastic__phase_1_starting_layer_count,
            number_of_trials=self.plastic__phase_1__number_of_trials,
            evaluation_steps_count=self.plastic__phase_1_evaluation_steps_count,
            max_permitted_layers=self.plastic__max_permitted_layers,
        )
        resolved_probe_interval = resolve_plastic_probe_interval(
            probe_interval=self.plastic__layer_count_probe__probe_every_n_steps,
            update_brake=self.plastic__layer_count_update_brake,
            enabled=self.plastic__enabled,
            do_learn_layer_count=self.plastic__do_learn_layer_count,
        )
        self.plastic__layer_count_probe__probe_every_n_steps = resolved_probe_interval
        # vvv THOG v0.521 fresh probe-token sampling uses 1024 by default and zero as the explicit all-valid sentinel
        if (
            isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, bool)
            or not isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, int)
            or self.plastic__layer_count_probe__number_of_sampled_valid_tokens < 0
        ):
            raise ValueError(
                "plastic__layer_count_probe__number_of_sampled_valid_tokens must be a non-negative integer; "
                f"got {self.plastic__layer_count_probe__number_of_sampled_valid_tokens!r}"
            )
        # ^^^ THOG
        # vvv THOG v0.521 direct TrainingConfig paths reject impossible positive probe samples before trainer construction
        probe_token_capacity = self.batch_size * self.block_size
        if (
            self.plastic__enabled
            and self.plastic__do_learn_layer_count
            and self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity
        ):
            raise ValueError(
                "plastic__layer_count_probe__number_of_sampled_valid_tokens must not exceed "
                f"batch_size * block_size ({probe_token_capacity})"
            )
        # ^^^ THOG
        validate_plastic_fine_count_controls(
            probe_radius=self.plastic__layer_count_probe_radius,
            max_step=self.plastic__layer_count__max_allowable_layer_change,
        )
        if (
            isinstance(self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence, bool)
            or not isinstance(self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence, (int, float))
            or not math.isfinite(float(self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence))
            or not (0.5 < float(self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence) <= 1.0)
        ):
            raise ValueError(
                "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence must lie in (0.5, 1.0]; "
                f"got {self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence!r}"
            )
        initial_layer_count_for_resolution = (
            resolved_coarse.candidate_layers[0]
            if resolved_coarse.enabled
            else self.plastic__initial_layer_count
        )
        # ^^^ THOG
        resolved_plastic_counts = resolve_plastic_depth_counts(
            n_layer=self.n_layer,
            enabled=self.plastic__enabled,
            layers_to_sample=self.plastic__layers_to_sample,
            do_learn_layer_count=self.plastic__do_learn_layer_count,
            initial_layer_count=initial_layer_count_for_resolution,
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
        # vvv THOG v0.541 public equivalent-time-gain controls are explicit, bounded and checkpoint-persistent
        if (
            isinstance(self.plastic__wall_time_equivalent_time_gain_discount, bool)
            or not isinstance(self.plastic__wall_time_equivalent_time_gain_discount, (int, float))
            or not math.isfinite(float(self.plastic__wall_time_equivalent_time_gain_discount))
            or not (0.0 <= float(self.plastic__wall_time_equivalent_time_gain_discount) <= 1.0)
        ):
            raise ValueError(
                "plastic__wall_time_equivalent_time_gain_discount must be finite and lie in [0, 1]; "
                f"got {self.plastic__wall_time_equivalent_time_gain_discount!r}"
            )
        if (
            isinstance(self.plastic__wall_time_equivalent_time_gain_loss_rate_window, bool)
            or not isinstance(self.plastic__wall_time_equivalent_time_gain_loss_rate_window, int)
            or self.plastic__wall_time_equivalent_time_gain_loss_rate_window < 2
        ):
            raise ValueError(
                "plastic__wall_time_equivalent_time_gain_loss_rate_window must be an integer >= 2; "
                f"got {self.plastic__wall_time_equivalent_time_gain_loss_rate_window!r}"
            )
        if (
            isinstance(self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations, bool)
            or not isinstance(self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations, int)
            or self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations < 2
            or self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations > self.plastic__wall_time_equivalent_time_gain_loss_rate_window
        ):
            raise ValueError(
                "plastic__wall_time_equivalent_time_gain_loss_rate_min_observations must be an integer in [2, plastic__wall_time_equivalent_time_gain_loss_rate_window]; "
                f"got {self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations!r}"
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
            raise ValueError("plastic__layer_memory_budget_gib is required for memory_budget")
        # vvv THOG universal CUDA safety reserve is execution state, configured in GiB and allowed to be explicitly disabled with zero
        if (
            isinstance(self.plastic__cuda_allocator_reserve_gib, bool)
            or not isinstance(self.plastic__cuda_allocator_reserve_gib, (int, float))
            or not math.isfinite(float(self.plastic__cuda_allocator_reserve_gib))
            or float(self.plastic__cuda_allocator_reserve_gib) < 0.0
        ):
            raise ValueError(
                "plastic__cuda_allocator_reserve_gib must be finite and non-negative; "
                f"got {self.plastic__cuda_allocator_reserve_gib!r}"
            )
        # ^^^ THOG
        if self.plastic__enabled and self.plastic__layer_count_objective == "memory_budget" and not self.device.startswith("cuda"):
            raise ValueError("PLASTIC DEPTH memory_budget requires a CUDA device")
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
        if self.plastic__enabled and self.model_type != "thog2_sheet":
            raise ValueError("PLASTIC DEPTH requires model_type='thog2_sheet'")
        if self.plastic__enabled and self.hyperblock_enabled:
            raise ValueError("PLASTIC DEPTH may not be combined with HYPERBLOCK")
        # ^^^ THOG

        # vvv THOG HYPERBLOCK and legacy selector geometries are mutually exclusive persistent parameterisations
        if self.hyperblock_enabled:
            if self.model_type != "thog2_sheet":
                raise ValueError("HYPERBLOCK requires model_type='thog2_sheet'")
            conflicting_fields = {
                "geometry_preset": self.geometry_preset,
                "attention_geometry": self.attention_geometry,
                "mlp_geometry": self.mlp_geometry,
                "basis_family": self.basis_family,
                "resolved_geometry_plan": self.resolved_geometry_plan,
            }
            active_conflicts = {
                name: value
                for name, value in conflicting_fields.items()
                if value is not None
            }
            if active_conflicts:
                raise ValueError(
                    "HYPERBLOCK may not be combined with legacy geometry controls; "
                    f"got {active_conflicts}"
                )
            if self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "HYPERBLOCK keeps LayerNorm and bias vectors conventional in v0; "
                    "depth_compress_layer_norm_and_bias must be false"
                )
            resolved_hyperblock_plan = self.hyperblock_plan()
            self.hyperblock_topology = resolved_hyperblock_plan.topology
            self.hyperblock_compressor = resolved_hyperblock_plan.compressor_family
            self.hyperblock_compressor_version = resolved_hyperblock_plan.compressor_version
        else:
            # vvv THOG DEPTH ignores every within-tensor order; normalize before validation and checkpoint identity are derived.
            if self.model_type == "thog2_sheet":
                selectors = resolve_compact_selectors(
                    geometry_preset=self.geometry_preset,
                    attention_geometry=self.attention_geometry,
                    mlp_geometry=self.mlp_geometry,
                    basis_family=self.basis_family,
                )
                if selectors.geometry_preset == GEOMETRY_PRESET_DEPTH:
                    if self.plastic__enabled and selectors.basis_family != "chebyshev":
                        raise ValueError(
                            "PLASTIC DEPTH v0.1 requires the Chebyshev DEPTH compressor; "
                            f"got {selectors.basis_family!r}"
                        )
                    self.base_row_order = 1
                    self.mlp_channel_order = 1
                    self.o_attn_d_model = 1
                    self.o_attn_qkv_per_channel = 1
                    self.o_attn_out_per_channel = 1
                    self.o_mlp_d_model = 1
                    self.o_mlp_hidden = 1
                elif self.depth_compress_layer_norm_and_bias:
                    raise ValueError(
                        "depth_compress_layer_norm_and_bias may be enabled only for geometry_preset='depth'"
                    )
                if self.plastic__enabled and selectors.geometry_preset != GEOMETRY_PRESET_DEPTH:
                    raise ValueError("PLASTIC DEPTH requires geometry_preset='depth'")
            elif self.depth_compress_layer_norm_and_bias:
                raise ValueError(
                    "depth_compress_layer_norm_and_bias may be enabled only for geometry_preset='depth'"
                )
            # ^^^ THOG
        # ^^^ THOG

        # vvv THOG preserve the exact pre-HYPERBLOCK positive-integer validation line for source history
        # for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd", "depth_order", "base_row_order", "mlp_hidden_group_size", "batch_size", "gradient_accumulation_steps", "layer_dropout_resample_steps", "max_updates", "decay_updates", "eval_batches", "log_interval"):
        # ^^^ THOG
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd", "depth_order", "base_row_order", "mlp_hidden_group_size", "hyperblock_common_family_order", "hyperblock_attention_family_order", "hyperblock_mlp_family_order", "hyperblock_depth_order", "hyperblock_d_model_order", "hyperblock_mlp_hidden_order", "hyperblock_attention_head_order", "hyperblock_attention_head_channel_order", "hyperblock_mlp_hidden_multiplier", "hyperblock_loop_count", "batch_size", "gradient_accumulation_steps", "layer_dropout_resample_steps", "max_updates", "decay_updates", "eval_batches", "log_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        # vvv THOG v0.521 reject impossible configured samples rather than silently clipping to microbatch capacity
        probe_token_capacity = self.batch_size * self.block_size
        if (
            self.plastic__enabled
            and self.plastic__do_learn_layer_count
            and self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity
        ):
            raise ValueError(
                "plastic__layer_count_probe__number_of_sampled_valid_tokens must not exceed "
                f"batch_size * block_size ({probe_token_capacity}); "
                f"got {self.plastic__layer_count_probe__number_of_sampled_valid_tokens}"
            )
        # ^^^ THOG
        # vvv THOG validate shared-factory recurrence independently of the HYPERBLOCK basis orders
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
        optional_positive = (
            "mlp_channel_order",
            "o_attn_d_model",
            "o_attn_qkv_per_channel",
            "o_attn_out_per_channel",
            "o_mlp_d_model",
            "o_mlp_hidden",
        )
        for name in optional_positive:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None; got {value!r}")
        for name in ("warmup_updates", "max_wall_minutes", "eval_interval", "checkpoint_interval", "model_seed", "data_seed", "max_nonfinite_update_skips"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
        # vvv THOG resolve layer-dropout defaults only after n_layer itself has been validated
        if self.layer_dropout_stratum_size is None:
            self.layer_dropout_stratum_size = self.n_layer
        if isinstance(self.layer_dropout_stratum_size, bool) or not isinstance(self.layer_dropout_stratum_size, int) or self.layer_dropout_stratum_size <= 0:
            raise ValueError(f"layer_dropout_stratum_size must be a positive integer; got {self.layer_dropout_stratum_size!r}")
        if self.layer_dropout_active_per_stratum is None:
            self.layer_dropout_active_per_stratum = self.layer_dropout_stratum_size
        if isinstance(self.layer_dropout_active_per_stratum, bool) or not isinstance(self.layer_dropout_active_per_stratum, int) or self.layer_dropout_active_per_stratum <= 0:
            raise ValueError(f"layer_dropout_active_per_stratum must be a positive integer; got {self.layer_dropout_active_per_stratum!r}")
        if self.n_layer % self.layer_dropout_stratum_size != 0:
            raise ValueError(
                "n_layer must be divisible by layer_dropout_stratum_size; "
                f"got n_layer={self.n_layer}, stratum_size={self.layer_dropout_stratum_size}"
            )
        if self.layer_dropout_active_per_stratum > self.layer_dropout_stratum_size:
            raise ValueError(
                "layer_dropout_active_per_stratum must not exceed layer_dropout_stratum_size; "
                f"got active={self.layer_dropout_active_per_stratum}, stratum_size={self.layer_dropout_stratum_size}"
            )
        if self.plastic__enabled and self.layer_dropout_active_per_stratum < self.layer_dropout_stratum_size:
            raise ValueError("PLASTIC DEPTH v0.1 may not be combined with layer dropout")
        # ^^^ THOG
        validate_checkpoint_segment_size(self.checkpoint_segment_size)
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd must be divisible by n_head; got {self.n_embd} and {self.n_head}")
        # vvv THOG legacy geometry orders are inactive when HYPERBLOCK owns every covered matrix axis
        if not self.hyperblock_enabled:
            if self.depth_order > self.n_layer:
                raise ValueError("depth_order must not exceed n_layer")
            if self.base_row_order > self.n_embd:
                raise ValueError("base_row_order must not exceed n_embd")
            if self.mlp_channel_order is not None and self.mlp_channel_order > 4 * self.n_embd:
                raise ValueError("mlp_channel_order must not exceed 4*n_embd")
            limits = {
                "o_attn_d_model": self.n_embd,
                "o_attn_qkv_per_channel": self.head_dim,
                "o_attn_out_per_channel": self.head_dim,
                "o_mlp_d_model": self.n_embd,
                "o_mlp_hidden": 4 * self.n_embd,
            }
            for name, limit in limits.items():
                value = getattr(self, name)
                if value is not None and value > limit:
                    raise ValueError(f"{name} must not exceed {limit}")
        # ^^^ THOG
        self.mlp_hidden_compressor = normalize_registered_basis_family(self.mlp_hidden_compressor)
        residual_init = self.residual_init_config()
        self.residual_init_depth_source = residual_init.depth_source
        if self.model_type == "dense" and residual_init.depth_source == "dof_implied_depth":
            raise ValueError("dof_implied_depth residual init is only defined for SHEET")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.min_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative and maximum must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.nonfinite_update_policy not in ("raise", "skip"):
            raise ValueError("nonfinite_update_policy must be raise or skip")
        if self.weight_decay < 0.0 or self.grad_clip < 0.0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 training is not supported on CPU")
        if self.row_order_scaling_rule != ROW_ORDER_SCALING_RULE:
            raise ValueError(f"unsupported row_order_scaling_rule: {self.row_order_scaling_rule}")
        if self.model_type == "dense":
            if self.basis_version != BASIS_VERSION:
                raise ValueError(f"unsupported dense basis_version: {self.basis_version}")
            validate_dense_compact_fields(
                geometry_preset=self.geometry_preset,
                attention_geometry=self.attention_geometry,
                mlp_geometry=self.mlp_geometry,
                basis_family=self.basis_family,
            )
        elif self.hyperblock_enabled:
            resolved_hyperblock_plan = self.hyperblock_plan()
            self.hyperblock_topology = resolved_hyperblock_plan.topology
            self.hyperblock_compressor = resolved_hyperblock_plan.compressor_family
            self.hyperblock_compressor_version = resolved_hyperblock_plan.compressor_version
        else:
            # vvv THOG derive the parameterised lapped version before compact identity validation
            # vvv THOG preserve the established compact-identity error contract for unknown basis names
            try:
                canonical_basis_family = normalize_registered_basis_family(self.basis_family or "chebyshev")
            except ValueError:
                canonical_basis_family = None
            # ^^^ THOG
            if canonical_basis_family == BASIS_FAMILY_LAPPED_COSINE and self.basis_version in ("auto", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
                self.basis_version = lapped_cosine_basis_version(
                    self.lapped_cosine_window_length,
                    self.lapped_cosine_overlap_fraction,
                )
            identity = self.compact_identity_metadata()
            self.basis_version = str(identity["basis_version"])
            # ^^^ THOG
        if not isinstance(self.bias, bool) or not isinstance(self.decay_learning_rate, bool):
            raise ValueError("bias and decay_learning_rate must be bool")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    # vvv THOG the v0 user flag resolves to the sole implemented topology while checkpoints retain the explicit subtype
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

    # vvv THOG derived layer-dropout quantities are configuration metadata, not independent knobs
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

    @property
    def resolved_o_attn_d_model(self) -> int:
        return self.base_row_order if self.o_attn_d_model is None else self.o_attn_d_model

    @property
    def resolved_o_attn_qkv_per_channel(self) -> int:
        return derive_row_order(self.head_dim, self.n_embd, self.base_row_order) if self.o_attn_qkv_per_channel is None else self.o_attn_qkv_per_channel

    @property
    def resolved_o_attn_out_per_channel(self) -> int:
        return derive_row_order(self.head_dim, self.n_embd, self.base_row_order) if self.o_attn_out_per_channel is None else self.o_attn_out_per_channel

    @property
    def resolved_o_mlp_d_model(self) -> int:
        return self.base_row_order if self.o_mlp_d_model is None else self.o_mlp_d_model

    @property
    def resolved_o_mlp_hidden(self) -> int:
        if self.o_mlp_hidden is not None:
            return self.o_mlp_hidden
        if self.mlp_channel_order is not None:
            return self.mlp_channel_order
        return derive_row_order(4 * self.n_embd, self.n_embd, self.base_row_order)

    def residual_init_config(self) -> ResidualInitConfig:
        return ResidualInitConfig(policy=self.residual_init_policy, depth_source=self.residual_init_depth_source, depth_value=self.residual_init_depth_value)

    # vvv THOG serialize no dormant PLASTIC DEPTH fields when disabled so checkpoint and report metadata remain exact regressions
    def persistent_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        if not self.plastic__enabled:
            for name in PLASTIC_TRAINING_CONFIG_FIELDS:
                values.pop(name, None)
        return values
    # ^^^ THOG

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
            # vvv THOG HYPERBLOCK does not resolve or pass inactive legacy geometry orders
            if self.hyperblock_enabled:
                arguments.update({
                    "depth_compress_layer_norm_and_bias": False,
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
                    "hyperblock_residual_weight_std": self.residual_init_config().residual_std(
                        model_type=self.model_type,
                        n_layer=self.n_layer,
                        depth_order=self.hyperblock_depth_order,
                    ),
                })
            else:
                arguments.update({
                    "depth_order": self.depth_order,
                    "base_row_order": self.base_row_order,
                    "mlp_channel_order": self.mlp_channel_order,
                    "o_attn_d_model": self.resolved_o_attn_d_model,
                    "o_attn_qkv_per_channel": self.resolved_o_attn_qkv_per_channel,
                    "o_attn_out_per_channel": self.resolved_o_attn_out_per_channel,
                    "o_mlp_d_model": self.resolved_o_mlp_d_model,
                    "o_mlp_hidden": self.resolved_o_mlp_hidden,
                    "mlp_hidden_group_size": self.mlp_hidden_group_size,
                    "mlp_hidden_compressor": self.mlp_hidden_compressor,
                    # vvv THOG preserve the exact pre-HYPERBLOCK model-argument line for source history
                    # "depth_compress_layer_norm_and_bias": self.depth_compress_layer_norm_and_bias,                                                       # <<< THOG pass DEPTH vector mode into SheetGPTConfig
                    # ^^^ THOG
                    "depth_compress_layer_norm_and_bias": self.depth_compress_layer_norm_and_bias,                                                         # <<< THOG pass DEPTH vector mode into SheetGPTConfig
                    "basis_version": self.basis_version,
                    "geometry_preset": self.geometry_preset,
                    "attention_geometry": self.attention_geometry,
                    "mlp_geometry": self.mlp_geometry,
                    "basis_family": self.basis_family,
                })
                # vvv THOG disabled PLASTIC DEPTH passes no new model arguments; enabled runs carry the complete Plasticity Engine identity
                if self.plastic__enabled:
                    arguments.update({
                        "plastic__enabled": True,
                        "plastic__layers_to_sample": self.plastic__layers_to_sample,
                        "plastic__do_learn_layer_count": self.plastic__do_learn_layer_count,
                        "plastic__initial_layer_count": self.plastic__initial_layer_count,
                        "plastic__max_permitted_layers": self.plastic__max_permitted_layers,
                        "plastic__layer_sampling_initialisation": self.plastic__layer_sampling_initialisation,
                        "plastic__layer_count_objective": self.plastic__layer_count_objective,
                        "plastic__layer_count_update_brake": self.plastic__layer_count_update_brake,
                        "plastic__layer_count_probe__window_size_as_number_of_probes": self.plastic__layer_count_probe__window_size_as_number_of_probes,
                        "plastic__layer_count_probe_noise_lambda": float(self.plastic__layer_count_probe_noise_lambda),
                        "plastic__layer_count_cost_weight": float(self.plastic__layer_count_cost_weight),
                        "plastic__layer_memory_budget_gib": self.plastic__layer_memory_budget_gib,
                        "plastic__geometry_learning_rate_multiplier": float(self.plastic__geometry_learning_rate_multiplier),
                        "plastic__freeze_geometry_during_warmup": self.plastic__freeze_geometry_during_warmup,
                        "plastic__sampling_seed": self.model_seed,
                    })
                # ^^^ THOG
            # ^^^ THOG
        return arguments

    def compact_identity_metadata(self) -> Dict[str, Any]:
        if self.model_type == "dense":
            return conventional_identity_metadata(n_layer=self.n_layer, n_embd=self.n_embd, n_head=self.n_head)
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
            o_depth=self.depth_order,
            o_attn_d_model=self.resolved_o_attn_d_model,
            o_attn_qkv_per_channel=self.resolved_o_attn_qkv_per_channel,
            o_attn_out_per_channel=self.resolved_o_attn_out_per_channel,
            o_mlp_d_model=self.resolved_o_mlp_d_model,
            o_mlp_hidden=self.resolved_o_mlp_hidden,
            mlp_hidden_group_size=self.mlp_hidden_group_size,
            mlp_hidden_compressor=self.mlp_hidden_compressor,
            basis_version=self.basis_version,
            lapped_cosine_window_length=self.lapped_cosine_window_length,                                                                                  # <<< THOG compact identity locality control
            lapped_cosine_overlap_fraction=self.lapped_cosine_overlap_fraction,                                                                            # <<< THOG compact identity overlap control
            row_order_scaling_rule=self.row_order_scaling_rule,
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )
        if self.resolved_geometry_plan is not None:
            identity["resolved_geometry_plan"] = self.resolved_geometry_plan
        # vvv THOG PLASTIC DEPTH identity is explicit while disabled identity remains byte-for-byte unchanged
        if self.plastic__enabled:
            # identity["plastic_depth"] = {
            #     "version": PLASTIC_DEPTH_VERSION,
            #     "maximum_layers": self.n_layer,
            #     "initial_active_layers": self.plastic__initial_active_layers,
            #     "learn_layer_count": self.plastic__do_learn_layer_count,
            #     "sampling_initialisation": self.plastic__layer_sampling_initialisation,
            #     "count_objective": self.plastic__layer_count_objective,
            #     "count_update_brake": self.plastic__layer_count_update_brake,
            #     "probe_noise_window": self.plastic__layer_count_probe__window_size_as_number_of_probes,
            #     "probe_noise_lambda": float(self.plastic__layer_count_probe_noise_lambda),
            #     "count_cost_weight": float(self.plastic__layer_count_cost_weight),
            #     "memory_budget_gib": self.plastic__layer_memory_budget_gib,
            #     "cuda_allocator_reserve_gib": float(self.plastic__cuda_allocator_reserve_gib),
            #     "geometry_lr_multiplier": float(self.plastic__geometry_learning_rate_multiplier),
            #     "freeze_geometry_during_warmup": self.plastic__freeze_geometry_during_warmup,
            # }
            # vvv THOG persist every exposed control under its exact canonical plastic__ name
            identity["plastic_depth"] = plastic_depth_identity_metadata(
                coarse_phase=self.plastic__coarse_phase,
                coarse_phase_roll_through=self.plastic__coarse_phase_roll_through,
                log_interval_coarse=self.plastic__log_interval_coarse,
                phase_1_n_steps=self.plastic__phase_1_n_steps,
                phase_1_starting_layer_count=self.plastic__phase_1_starting_layer_count,
                phase_1_number_of_trials=self.plastic__phase_1__number_of_trials,
                phase_1_evaluation_steps_count=self.plastic__phase_1_evaluation_steps_count,
                layer_count_probe__probe_every_n_steps=self.plastic__layer_count_probe__probe_every_n_steps,
                layer_count_probe_radius=self.plastic__layer_count_probe_radius,
                layer_count_max_step=self.plastic__layer_count__max_allowable_layer_change,
                layer_count_extrapolation_weight=float(self.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence),
                wall_time_equivalent_time_gain_discount=float(self.plastic__wall_time_equivalent_time_gain_discount),
                wall_time_equivalent_time_gain_loss_rate_window=self.plastic__wall_time_equivalent_time_gain_loss_rate_window,
                wall_time_equivalent_time_gain_loss_rate_min_observations=self.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations,
                layers_to_sample=self.plastic__layers_to_sample,
                do_learn_layer_count=self.plastic__do_learn_layer_count,
                initial_layer_count=self.plastic__initial_layer_count,
                max_permitted_layers=self.plastic__max_permitted_layers,
                layer_sampling_initialisation=self.plastic__layer_sampling_initialisation,
                layer_count_objective=self.plastic__layer_count_objective,
                layer_count_update_brake=self.plastic__layer_count_update_brake,
                layer_count_probe__window_size_as_number_of_probes=self.plastic__layer_count_probe__window_size_as_number_of_probes,
                layer_count_probe_noise_lambda=float(self.plastic__layer_count_probe_noise_lambda),
                layer_count_cost_weight=float(self.plastic__layer_count_cost_weight),
                layer_memory_budget_gib=self.plastic__layer_memory_budget_gib,
                cuda_allocator_reserve_gib=float(self.plastic__cuda_allocator_reserve_gib),
                geometry_learning_rate_multiplier=float(self.plastic__geometry_learning_rate_multiplier),
                freeze_geometry_during_warmup=self.plastic__freeze_geometry_during_warmup,
                initial_active_layers=self.plastic__initial_active_layers,
            )
            # ^^^ THOG
        # ^^^ THOG
        return identity

    # vvv THOG schema-2 checkpoint signatures must stay available after execution-only fields such as max_wall_minutes are added
    def compatibility_signature(self) -> Dict[str, Any]:
        values = asdict(self)
        if self.model_type == "thog2_sheet" and not self.hyperblock_enabled:
            values.update(
                {
                    "o_attn_d_model": self.resolved_o_attn_d_model,
                    "o_attn_qkv_per_channel": self.resolved_o_attn_qkv_per_channel,
                    "o_attn_out_per_channel": self.resolved_o_attn_out_per_channel,
                    "o_mlp_d_model": self.resolved_o_mlp_d_model,
                    "o_mlp_hidden": self.resolved_o_mlp_hidden,
                    "basis_version": self.basis_version,
                }
            )
        return {name: values[name] for name in MODEL_COMPATIBILITY_FIELDS}
    # ^^^ THOG
# ^^^ THOG
# vvv THOG preserved superseded source lines for exact history audit
# for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd", "depth_order", "base_row_order", "mlp_hidden_group_size", "hyperblock_common_family_order", "hyperblock_attention_family_order", "hyperblock_mlp_family_order", "hyperblock_depth_order", "hyperblock_d_model_order", "hyperblock_mlp_hidden_order", "hyperblock_attention_head_order", "hyperblock_attention_head_channel_order", "hyperblock_mlp_hidden_multiplier", "batch_size", "gradient_accumulation_steps", "layer_dropout_resample_steps", "max_updates", "decay_updates", "eval_batches", "log_interval"):
# ^^^ THOG

# vvv THOG retired PLASTIC DEPTH hold-controller source preserved for history audit
# "plastic__layer_count_hold_updates",
# plastic__layer_count_hold_updates: int = 100
# isinstance(self.plastic__layer_count_hold_updates, bool)
# or not isinstance(self.plastic__layer_count_hold_updates, int)
# or self.plastic__layer_count_hold_updates <= 0
# "plastic__layer_count_hold_updates must be a positive integer; "
# f"got {self.plastic__layer_count_hold_updates!r}"
# "plastic__layer_count_hold_updates": self.plastic__layer_count_hold_updates,
# "count_hold_updates": self.plastic__layer_count_hold_updates,
# ^^^ THOG

# vvv THOG retired short PLASTIC identity aliases preserved for source history
# "version": PLASTIC_DEPTH_VERSION,
# "maximum_layers": self.n_layer,
# "initial_active_layers": self.plastic__initial_active_layers,
# "learn_layer_count": self.plastic__do_learn_layer_count,
# "sampling_initialisation": self.plastic__layer_sampling_initialisation,
# "count_objective": self.plastic__layer_count_objective,
# "count_update_brake": self.plastic__layer_count_update_brake,
# "probe_noise_window": self.plastic__layer_count_probe__window_size_as_number_of_probes,
# "probe_noise_lambda": float(self.plastic__layer_count_probe_noise_lambda),
# "count_cost_weight": float(self.plastic__layer_count_cost_weight),
# "memory_budget_gib": self.plastic__layer_memory_budget_gib,
# "geometry_lr_multiplier": float(self.plastic__geometry_learning_rate_multiplier),
# "freeze_geometry_during_warmup": self.plastic__freeze_geometry_during_warmup,
# ^^^ THOG
