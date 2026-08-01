# vvv THOG
from .basis_provider import AxisBasisProvider, HyperblockBasisTables, RegisteredAxisBasisProvider
from .direct_mlp import FactorisedHyperblockMlpLayer, apply_factorised_hyperblock_mlp, factorise_hyperblock_mlp_layer                         # <<< THOG optional exact HYPERBLOCK MLP application without dense UP/DOWN matrices
from .materializer import (
    MaterializedHyperblockAttentionLayer,                                                                                                           # <<< THOG attention-only layer bundle for direct HYPERBLOCK MLP
    MaterializedHyperblockLayer,
    MaterializedHyperblockRegions,
    materialize_attention_family_layer,
    materialize_attention_layer_staged,                                                                                                                  # <<< THOG skip dense HYPERBLOCK MLP materialisation while retaining batched attention
    materialize_layer_staged,
    materialize_mlp_family_layer,
    materialize_regions_reference,
    materialize_regions_staged,
    route_attention_input_matrices,
    route_attention_matrix,
    route_mlp_matrix,
)
from .trajectory import CoupledFieldTrajectory
from .plan import (
    ATTENTION_FAMILIES,
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    HYPERBLOCK_TOPOLOGY_VERSION,
    MLP_FAMILIES,
    WEIGHT_FAMILIES,
    HyperblockOrders,
    ResolvedHyperblockPlan,
)

__all__ = [
    "ATTENTION_FAMILIES",
    "CoupledFieldTrajectory",
    "FactorisedHyperblockMlpLayer",                                                                                                                     # <<< THOG public direct-application factor bundle
    "AxisBasisProvider",
    "HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE",
    "HYPERBLOCK_TOPOLOGY_VERSION",
    "HyperblockBasisTables",
    "HyperblockOrders",
    "MLP_FAMILIES",
    "MaterializedHyperblockAttentionLayer",                                                                                                       # <<< THOG public attention-only layer bundle
    "MaterializedHyperblockLayer",
    "MaterializedHyperblockRegions",
    "RegisteredAxisBasisProvider",
    "ResolvedHyperblockPlan",
    "WEIGHT_FAMILIES",
    "apply_factorised_hyperblock_mlp",                                                                                                            # <<< THOG exact factorised operational application
    "factorise_hyperblock_mlp_layer",                                                                                                                  # <<< THOG per-layer compact MLP factors
    "materialize_attention_family_layer",
    "materialize_attention_layer_staged",                                                                                                             # <<< THOG attention-only staged layer materialisation
    "materialize_layer_staged",
    "materialize_mlp_family_layer",
    "materialize_regions_reference",
    "materialize_regions_staged",
    "route_attention_input_matrices",
    "route_attention_matrix",
    "route_mlp_matrix",
]
# ^^^ THOG
