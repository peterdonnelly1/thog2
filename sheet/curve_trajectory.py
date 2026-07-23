# vvv THOG
"""Import-only bridge for modules that still use the mathematical 'depth curve' terminology.

The canonical implementation and public preset are DepthTrajectory / DEPTH. This module
must not be used for preset selection or checkpoint identity.
"""

from .depth_trajectory import DEPTH_MATRIX_FAMILIES, DepthFamilyMetadata, DepthTrajectory

CURVE_MATRIX_FAMILIES = DEPTH_MATRIX_FAMILIES
CurveFamilyMetadata = DepthFamilyMetadata
CurveTrajectory = DepthTrajectory

__all__ = (
    "CURVE_MATRIX_FAMILIES",
    "CurveFamilyMetadata",
    "CurveTrajectory",
)
# ^^^ THOG
