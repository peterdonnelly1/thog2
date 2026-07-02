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
