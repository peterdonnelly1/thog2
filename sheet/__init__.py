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
# _stage6_trainer._PROGRESS_LOSS_DECREASE_STYLE_START = "\033[1;92m"                                                                               # <<< THOG preserve palette-dependent bright-green attempt
_stage6_trainer._PROGRESS_LOSS_DECREASE_STYLE_START = "\033[1;38;2;0;255;0m"                                                                       # <<< THOG force explicit RGB bright green for falling loss
# _stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;93m"                                                                            # <<< THOG preserve palette-dependent bright-yellow attempt
_stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\033[1;38;2;255;255;0m"                                                                 # <<< THOG force bold explicit RGB yellow across validation-loss label and value
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

# vvv THOG finalise PLASTIC console labels, suppress loss-gain clutter and colour candidate probe losses
from . import plastic_depth_console_cleanup_patch as _plastic_depth_console_cleanup_patch
# ^^^ THOG
