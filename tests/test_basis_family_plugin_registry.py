# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

import run_thog2_owt
from sheet.bases import (
    BASIS_ARTIFACT_TAG_CHEBYSHEV,
    BASIS_ARTIFACT_TAG_DCT,
    BASIS_FAMILIES,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    CHEBYSHEV_BASIS_VERSION,
    DCT_BASIS_VERSION,
    BasisDefinition,
    BasisKernel,
    BasisRegistry,
    DeviceLike,
    basis_artifact_tag_for_family,
    basis_version_for_family,
    get_basis_definition,
    normalize_registered_basis_family,
)
from sheet.bases.haar import BASIS_ARTIFACT_TAG_HAAR, BASIS_FAMILY_HAAR, HAAR_BASIS_VERSION
from sheet.bases.lapped_cosine import BASIS_ARTIFACT_TAG_LAPPED_COSINE, BASIS_FAMILY_LAPPED_COSINE, LAPPED_COSINE_BASIS_VERSION                                  # <<< THOG appended plugin contract
from sheet.compact_identity import GEOMETRY_PRESET_DEPTH, resolve_compact_selectors
from sheet.run_config import OwtRunConfig


class SyntheticIdentityKernel(BasisKernel):
    def __init__(self, family: str = "synthetic_identity", version: str = "synthetic_identity_v1") -> None:
        super().__init__(family, version, "integer_sample_index_v1", "identity_columns_v1")

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        return torch.arange(sample_count, dtype=dtype, device=torch.device("cpu" if device is None else device))

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return torch.eye(coordinates.numel(), order, dtype=coordinates.dtype, device=coordinates.device)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        return raw_basis


def synthetic_definition(*, family: str = "synthetic_identity", aliases: tuple[str, ...] = ("synth",), version: str = "synthetic_identity_v1", artifact_tag: str = "SYNTH") -> BasisDefinition:
    return BasisDefinition(
        family=family,
        aliases=aliases,
        version=version,
        artifact_tag=artifact_tag,
        supports_weight_basis=True,
        supports_native_products=False,
        kernel=SyntheticIdentityKernel(family, version),
    )


class BasisFamilyPluginRegistryTests(unittest.TestCase):
    def test_01_builtin_registry_preserves_family_version_alias_and_artifact_contracts(self) -> None:
        self.assertEqual(BASIS_FAMILIES, (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, BASIS_FAMILY_HAAR, BASIS_FAMILY_LAPPED_COSINE))
        self.assertEqual(normalize_registered_basis_family("cheby"), BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(normalize_registered_basis_family(CHEBYSHEV_BASIS_VERSION), BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(normalize_registered_basis_family("dct_ii"), BASIS_FAMILY_DCT)
        self.assertEqual(normalize_registered_basis_family(DCT_BASIS_VERSION), BASIS_FAMILY_DCT)
        self.assertEqual(normalize_registered_basis_family("balanced_haar"), BASIS_FAMILY_HAAR)
        self.assertEqual(normalize_registered_basis_family(HAAR_BASIS_VERSION), BASIS_FAMILY_HAAR)
        self.assertEqual(normalize_registered_basis_family("lapped"), BASIS_FAMILY_LAPPED_COSINE)
        self.assertEqual(normalize_registered_basis_family(LAPPED_COSINE_BASIS_VERSION), BASIS_FAMILY_LAPPED_COSINE)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_CHEBYSHEV), CHEBYSHEV_BASIS_VERSION)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_DCT), DCT_BASIS_VERSION)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_HAAR), HAAR_BASIS_VERSION)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_LAPPED_COSINE), LAPPED_COSINE_BASIS_VERSION)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_CHEBYSHEV), BASIS_ARTIFACT_TAG_CHEBYSHEV)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_DCT), BASIS_ARTIFACT_TAG_DCT)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_HAAR), BASIS_ARTIFACT_TAG_HAAR)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_LAPPED_COSINE), BASIS_ARTIFACT_TAG_LAPPED_COSINE)
        self.assertEqual(get_basis_definition("cheby").family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(get_basis_definition("haar_balanced").family, BASIS_FAMILY_HAAR)

    def test_02_local_registry_accepts_a_new_basis_and_builds_it_without_geometry_code(self) -> None:
        registry = BasisRegistry((synthetic_definition(),))
        self.assertEqual(registry.families(), ("synthetic_identity",))
        self.assertEqual(registry.normalize("synth"), "synthetic_identity")
        basis = registry["synth"].build(8, 3, runtime_dtype=torch.float32)
        torch.testing.assert_close(basis, torch.eye(8, 3, dtype=torch.float32), rtol=0.0, atol=0.0)

    def test_03_registry_rejects_duplicate_family_alias_version_and_artifact_tag(self) -> None:
        registry = BasisRegistry((synthetic_definition(),))
        with self.assertRaisesRegex(ValueError, "duplicate basis family"):
            registry.register(synthetic_definition())
        with self.assertRaisesRegex(ValueError, "alias collision"):
            registry.register(synthetic_definition(family="other_a", aliases=("synth",), version="other_a_v1", artifact_tag="OTHER_A"))
        with self.assertRaisesRegex(ValueError, "duplicate basis version"):
            registry.register(synthetic_definition(family="other_b", aliases=("other_b",), version="synthetic_identity_v1", artifact_tag="OTHER_B"))
        with self.assertRaisesRegex(ValueError, "duplicate basis artifact tag"):
            registry.register(synthetic_definition(family="other_c", aliases=("other_c",), version="other_c_v1", artifact_tag="SYNTH"))

    def test_04_selector_aliases_canonicalize_through_the_registry(self) -> None:
        cheby = resolve_compact_selectors(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family="cheby")
        haar = resolve_compact_selectors(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family="balanced_haar")
        self.assertEqual(cheby.basis_family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(cheby.requested_basis_family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(haar.basis_family, BASIS_FAMILY_HAAR)
        self.assertEqual(haar.requested_basis_family, BASIS_FAMILY_HAAR)

    def test_05_python_cli_choices_are_registry_derived(self) -> None:
        parser = run_thog2_owt.build_parser()
        action = next(action for action in parser._actions if action.dest == "basis_family")
        self.assertEqual(tuple(action.choices), BASIS_FAMILIES)

    def test_06_run_artifact_tags_are_registry_derived_and_existing_tags_remain_unchanged(self) -> None:
        cheby = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_CHEBYSHEV, basis_version="auto")
        dct = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_DCT, basis_version="auto")
        haar = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_HAAR, basis_version="auto")
        lapped = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_LAPPED_COSINE, basis_version="auto")
        self.assertEqual(cheby.compact_artifact_fragment(), "CHEBY_DEPTH")
        self.assertEqual(dct.compact_artifact_fragment(), "DCT_DEPTH")
        self.assertEqual(haar.compact_artifact_fragment(), "HAAR_DEPTH")
        self.assertEqual(lapped.compact_artifact_fragment(), "LAPPED_COSINE_DEPTH")
        self.assertEqual(cheby.basis_version, CHEBYSHEV_BASIS_VERSION)
        self.assertEqual(dct.basis_version, DCT_BASIS_VERSION)
        self.assertEqual(haar.basis_version, HAAR_BASIS_VERSION)
        self.assertEqual(lapped.basis_version, LAPPED_COSINE_BASIS_VERSION)

    def test_07_primary_wrappers_have_no_family_specific_allow_list_or_tag_branch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
            with self.subTest(name=name):
                text = (root / name).read_text(encoding="utf-8")
                self.assertNotIn("chebyshev|dct", text)
                self.assertNotIn('BASIS_TAG="CHEBY"', text)
                self.assertNotIn('BASIS_TAG="HAAR"', text)
                self.assertIn("basis_artifact_tag_for_family", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
