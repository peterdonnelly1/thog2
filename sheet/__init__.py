# vvv THOG
from .approximation import (
    ProjectionError, fit_sampled_sheets, is_within_epsilon,
    project_sampled_sheets, projection_error, reconstruct_sampled_sheets,
)
from .basis import (
    BASIS_VERSION, SINGLE_POINT_COORDINATE, BasisCache, BasisCacheKey,
    BasisOwner, basis_sha256, build_stabilized_basis,
    chebyshev_first_kind_basis, deterministic_reduced_qr,
    estimated_peak_tensor_bytes, normalized_coordinates,
    orthonormality_max_error,
)
from .batch_source import Batch, DeterministicBatchSource
from .checkpoints import load_payload, validate_compatibility
from .geometry import (
    MATRIX_FAMILY_NAMES, FamilyGeometry, SheetGeometryConfig,
    derive_row_order, family_geometry_map, parameter_count_rows,
    total_dense_equivalent_count, total_sheet_parameter_count,
    transformer_family_geometries,
)
from .model import ConventionalLayerNorm, SheetGPT, SheetGPTConfig
from .model_factory import build_model, parameter_report
from .trainer import SharedTrainer, TrainerEvent, TrainerState
from .training_config import (
    CHECKPOINT_SCHEMA_VERSION, MODEL_TYPES, ROW_ORDER_SCALING_RULE,
    TrainingConfig,
)
from .trajectory import FamilyMetadata, SheetTrajectory, build_family_metadata

__all__ = [
    "BASIS_VERSION", "Batch", "BasisCache", "BasisCacheKey", "BasisOwner",
    "CHECKPOINT_SCHEMA_VERSION", "ConventionalLayerNorm", "DeterministicBatchSource",
    "FamilyGeometry", "FamilyMetadata", "MATRIX_FAMILY_NAMES", "MODEL_TYPES",
    "ProjectionError", "ROW_ORDER_SCALING_RULE", "SINGLE_POINT_COORDINATE",
    "SharedTrainer", "SheetGPT", "SheetGPTConfig", "SheetGeometryConfig",
    "SheetTrajectory", "TrainerEvent", "TrainerState", "TrainingConfig",
    "basis_sha256", "build_family_metadata", "build_model",
    "build_stabilized_basis", "chebyshev_first_kind_basis", "derive_row_order",
    "deterministic_reduced_qr", "estimated_peak_tensor_bytes",
    "family_geometry_map", "fit_sampled_sheets", "is_within_epsilon",
    "load_payload", "normalized_coordinates", "orthonormality_max_error",
    "parameter_count_rows", "parameter_report", "project_sampled_sheets",
    "projection_error", "reconstruct_sampled_sheets",
    "total_dense_equivalent_count", "total_sheet_parameter_count",
    "transformer_family_geometries", "validate_compatibility",
]
# ^^^ THOG

# vvv THOG accept underscore long-option aliases wherever argparse owns the CLI, while preserving all existing hyphen spellings
from . import argparse_underscore_alias_patch as _argparse_underscore_alias_patch
# ^^^ THOG

# vvv THOG append actual getopt and artifact descriptor keys to the complete registered help surface
from . import help_registry_descriptor_patch as _help_registry_descriptor_patch
# ^^^ THOG

# vvv THOG make descriptor-registry help idempotent across layered argparse compatibility overlays
from . import help_registry_descriptor_dedupe_patch as _help_registry_descriptor_dedupe_patch
# ^^^ THOG

# vvv THOG centralise explicit RGB console colours and elapsed-field layout without changing training semantics
from . import stage6_trainer as _stage6_trainer


def _progress_elapsed_hh_mm_ss(value, completed_updates):
    elapsed_seconds = max(0, int(round(float(str(value).strip()))))
    if int(str(completed_updates).strip()) == 1:
        return f"{elapsed_seconds:7d}s"
    hours, remainder_seconds = divmod(elapsed_seconds, 60 * 60)
    minutes, seconds = divmod(remainder_seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


_stage6_trainer._progress_elapsed = _progress_elapsed_hh_mm_ss
# _stage6_trainer._PROGRESS_LOSS_DECREASE_STYLE_START = "\033[1;92m"                                                                                       # <<< THOG preserve palette-dependent bright-green attempt
_stage6_trainer._PROGRESS_LOSS_DECREASE_STYLE_START = "\033[1;38;2;0;255;0m"                                                                               # <<< THOG force explicit RGB bright green for falling loss
# _stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;93m"                                                                                    # <<< THOG preserve palette-dependent bright-yellow attempt
_stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[0;1;93m"                                                                                    # <<< THOG reset then force terminal bright-yellow plus bold across validation-loss label and value
# ^^^ THOG

# vvv THOG install active-prefix PLASTIC gauge verification after DEPTH trajectory classes are loaded
from . import plastic_depth_active_gauge_patch as _plastic_depth_active_gauge_patch
# ^^^ THOG

# vvv THOG install exact-radius PLASTIC count lookahead and dynamic probe console labels
from . import plastic_depth_lookahead_patch as _plastic_depth_lookahead_patch
# ^^^ THOG

# vvv THOG preserve existing test patchability while exact-radius lookahead remains a surgical overlay
from . import plastic_depth_lookahead_fix_patch as _plastic_depth_lookahead_fix_patch
# ^^^ THOG

# vvv THOG replace relative N+1 learned-count geometry with a fixed max-capacity absolute depth ruler
from . import plastic_depth_absolute_ruler_patch as _plastic_depth_absolute_ruler_patch
# ^^^ THOG

# vvv THOG stabilise absolute-ruler endpoint and sample-layer display rounding
from . import plastic_depth_absolute_numeric_patch as _plastic_depth_absolute_numeric_patch
# ^^^ THOG

# vvv THOG restore dynamic selector patchability after absolute-ruler default selector installation
from . import plastic_depth_absolute_patchability_patch as _plastic_depth_absolute_patchability_patch
# ^^^ THOG

# vvv THOG finalise PLASTIC console labels, suppress loss-gain clutter and colour candidate probe losses
from . import plastic_depth_console_cleanup_patch as _plastic_depth_console_cleanup_patch
# ^^^ THOG

# vvv THOG let learned-count runs probe periodically instead of paying the L-1/L/L+1 inline-probe tax every optimizer update
from . import plastic_depth_probe_interval_patch as _plastic_depth_probe_interval_patch
# ^^^ THOG

# vvv THOG require robust history agreement and learn wall-time only from uncontaminated non-probe updates
from . import plastic_depth_controller_stability_patch as _plastic_depth_controller_stability_patch
# ^^^ THOG

# vvv THOG align PLASTIC validation suffixes, suppress stale probe fields and expose active count brakes
from . import plastic_depth_console_minor_patch as _plastic_depth_console_minor_patch
# ^^^ THOG

# vvv THOG final COARSE/FINE overlay wins after all compatibility patches and probes every valid integer count in the configured radius
from . import plastic_depth_coarse_fine_patch as _plastic_depth_coarse_fine_patch
# ^^^ THOG

# vvv THOG recover candidate-local CUDA OOMs across the contiguous full-radius upward suffix
from . import plastic_depth_full_radius_oom_patch as _plastic_depth_full_radius_oom_patch
# ^^^ THOG

# vvv THOG install replayable FINE count-decision audit after final selector and commit semantics are fixed
from . import plastic_depth_audit_patch as _plastic_depth_audit_patch
# ^^^ THOG

# vvv THOG hard-release each COARSE trainer before the next candidate or FINE state and keep telemetry phase axes independent
from . import plastic_depth_coarse_runtime_recovery_patch as _plastic_depth_coarse_runtime_recovery_patch
# ^^^ THOG

# vvv THOG make the final console compaction and FINE warmup count guard win after every earlier compatibility overlay
from . import plastic_depth_warmup_guard_patch as _plastic_depth_warmup_guard_patch
# ^^^ THOG

# vvv THOG set the public cost-weight default and compact long probe-offset vector labels after final console formatting
from . import plastic_depth_cli_cost_and_label_patch as _plastic_depth_cli_cost_and_label_patch
# ^^^ THOG

# vvv THOG install v0.52 goal-agnostic directional coherence and final PLASTIC progress formatting after every earlier overlay
from . import plastic_depth_directional_coherence_patch as _plastic_depth_directional_coherence_patch
_plastic_depth_console_minor_patch._ALIGNMENT_LABELS = ("sampled =", "probe_losses", "score_z", "change_z")
# ^^^ THOG

# vvv THOG v0.521 makes probe-token sampling configurable and renders arbitrary-radius probe losses as signed deltas around bold-white L
from . import plastic_depth_probe_sampling_v0521_patch as _plastic_depth_probe_sampling_v0521_patch
# ^^^ THOG

# vvv THOG v0.521 report paired per-token delta standard errors as diagnostics only; robust MAD/z-score decisions are unchanged
from . import plastic_depth_probe_se_v0521_patch as _plastic_depth_probe_se_v0521_patch
# ^^^ THOG

# vvv THOG keep sampled immediately after layers with the single tab already inserted by the v0.52 formatter; do not reserve a fixed terminal column
# def _align_sampled_to_minimum_tab_column(line):                                                                                                           # <<< THOG preserve the superseded fixed-column helper entry point concept
#     ...
def _plastic_depth_leave_sampled_after_layers(line):
    return line


_plastic_depth_directional_coherence_patch._align_sampled_to_minimum_tab_column = _plastic_depth_leave_sampled_after_layers
# ^^^ THOG

# vvv THOG keep postfix brake annotations at the physical end of the row and render neutral L/R/A outcomes as stet
from . import plastic_depth_console_postfix_patch as _plastic_depth_console_postfix_patch
# ^^^ THOG

# vvv THOG v0.541 install equivalent-time wall-time economics after established FINE scoring and console overlays
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _plastic_depth_wall_time_equivalent_time_gain_patch
# ^^^ THOG

# vvv THOG v0.53 install fixed-batch non-overlapping probe windows after equivalent-time scoring so cached evidence reuses the complete selector stack
from . import plastic_depth_same_batch_all_probes_patch as _plastic_depth_same_batch_all_probes_patch
# ^^^ THOG

# vvv THOG v0.54 install selectable Theil-Sen/Kendall gradient classification last so legacy behaviour remains the untouched default path
from . import plastic_depth_theil_sen_kendall_patch as _plastic_depth_theil_sen_kendall_patch
# ^^^ THOG

# vvv THOG v0.54 restore synthetic gradient-control fields before the dataclass constructor during checkpoint resume
from . import plastic_depth_theil_sen_kendall_resume_config_patch as _plastic_depth_theil_sen_kendall_resume_config_patch
# ^^^ THOG

# vvv THOG v0.54 align gradient diagnostics with the final fat-arrow/provenance console layer without altering controller semantics
from . import plastic_depth_theil_sen_kendall_console_fix_patch as _plastic_depth_theil_sen_kendall_console_fix_patch
# ^^^ THOG

# vvv THOG v0.54 keep raw-loss bootstrap exploration active until wall-time timing/loss models can provide real TSK economic scores
from . import plastic_depth_theil_sen_kendall_bootstrap_fix_patch as _plastic_depth_theil_sen_kendall_bootstrap_fix_patch
# ^^^ THOG

# vvv THOG final compact PLASTIC operator layout: seconds plus decimal-hour elapsed time, one-decimal step seconds, shorter labels and tighter sampled placement
from . import plastic_depth_console_compact_layout_patch as _plastic_depth_console_compact_layout_patch
# ^^^ THOG

# vvv THOG v0.55 install renamed LRA and whole-window stratified Sen/Kendall control after every older selector and console overlay
from . import plastic_depth_sen_kendall_v055_patch as _plastic_depth_sen_kendall_v055_patch
# ^^^ THOG

# vvv THOG v0.55 ignore infeasible lower decision points without weakening the adjacent-action requirement or far-right informational probing
from . import plastic_depth_sen_kendall_v055_boundary_fix_patch as _plastic_depth_sen_kendall_v055_boundary_fix_patch
# ^^^ THOG

# vvv THOG enforce the CUDA allocator reserve as a real growth headroom barrier after same-batch and final Sen/Kendall selection are installed
from . import plastic_depth_cuda_headroom_guard_patch as _plastic_depth_cuda_headroom_guard_patch
# ^^^ THOG

# vvv THOG v1.3 install sampling-only chaos bumps last so count freeze, console and checkpoint semantics are authoritative
from . import chaos_bump_sampling_patch as _chaos_bump_sampling_patch
# ^^^ THOG

# vvv THOG add standalone Sen/Kendall thresholds and direct raw-loss jumping after every retained selector, audit and safety overlay
from . import plastic_depth_decision_algorithms_v057_patch as _plastic_depth_decision_algorithms_v057_patch
# ^^^ THOG

# vvv THOG reuse the six depth-weight chart families for conventional DENSE blocks as discrete cross-marker series
from . import dense_weight_curves_patch as _dense_weight_curves_patch
# ^^^ THOG

# vvv THOG register local runs before their first chart record and keep model liveness independent of finite instrumentation windows
from . import local_chart_lifecycle_patch as _local_chart_lifecycle_patch
# ^^^ THOG
