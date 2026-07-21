from __future__ import annotations

import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def write_file(relative_path: str, content: str) -> None:
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def replace_once(relative_path: str, old: str, new: str) -> None:
    target = ROOT / relative_path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {relative_path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_regex_once(relative_path: str, pattern: str, replacement: str) -> None:
    target = ROOT / relative_path
    text = target.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count == 0 and replacement in text:
        return
    if count != 1:
        raise RuntimeError(f"expected one regex match in {relative_path}, found {count}: {pattern!r}")
    target.write_text(updated, encoding="utf-8")


write_file(
    "sheet/bases/protocol.py",
    r'''
    # vvv THOG
    from __future__ import annotations

    import re
    from dataclasses import dataclass
    from typing import Dict, Optional, Tuple, Union

    import torch
    from torch import Tensor


    DeviceLike = Union[str, torch.device]
    _FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
    _ARTIFACT_TAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


    def validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


    def validate_floating_dtype(dtype: torch.dtype) -> None:
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"dtype must be floating point; got {dtype}")


    def deterministic_reduced_qr_positive_diagonal(raw_basis: Tensor) -> Tuple[Tensor, Tensor]:
        if raw_basis.ndim != 2:
            raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
        row_count, column_count = raw_basis.shape
        if row_count < column_count:
            raise ValueError(f"reduced QR requires rows >= columns; got rows={row_count}, columns={column_count}")
        if not raw_basis.is_floating_point():
            raise ValueError(f"raw_basis must use a floating dtype; got {raw_basis.dtype}")
        if not torch.isfinite(raw_basis).all():
            raise ValueError("raw_basis must be finite")
        q_matrix, r_matrix = torch.linalg.qr(raw_basis, mode="reduced")
        diagonal = torch.diagonal(r_matrix)
        signs = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        q_matrix = q_matrix * signs.unsqueeze(0)
        r_matrix = r_matrix * signs.unsqueeze(1)
        if not torch.isfinite(q_matrix).all() or not torch.isfinite(r_matrix).all():
            raise FloatingPointError("QR stabilization produced a non-finite value")
        return q_matrix, r_matrix


    @dataclass(frozen=True)
    class BasisKernel:
        basis_family: str
        basis_version: str
        coordinate_policy: str
        stabilization_policy: str

        def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
            raise NotImplementedError

        def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
            raise NotImplementedError

        def stabilize(self, raw_basis: Tensor) -> Tensor:
            raise NotImplementedError

        def build(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None) -> Tensor:
            validate_positive_integer("sample_count", sample_count)
            validate_positive_integer("order", order)
            if order > sample_count:
                raise ValueError(f"order must not exceed sample_count; got order={order}, sample_count={sample_count}")
            validate_floating_dtype(runtime_dtype)
            basis_version = self.basis_version if version is None else version
            if not isinstance(basis_version, str) or not basis_version.strip():
                raise ValueError("version must be a non-empty string")
            if basis_version != self.basis_version:
                raise ValueError(f"basis_version mismatch for {self.basis_family}: expected {self.basis_version!r}, got {basis_version!r}")
            coordinates = self.coordinates(sample_count, dtype=torch.float64, device="cpu")
            raw_basis = self.raw_basis(coordinates, order)
            stabilized_basis = self.stabilize(raw_basis)
            target_device = torch.device("cpu" if device is None else device)
            runtime_basis = stabilized_basis.to(device=target_device, dtype=runtime_dtype)
            runtime_basis.requires_grad_(False)
            if runtime_basis.shape != (sample_count, order):
                raise RuntimeError(f"unexpected basis shape {tuple(runtime_basis.shape)}; expected {(sample_count, order)}")
            if not torch.isfinite(runtime_basis).all():
                raise FloatingPointError("runtime basis contains a non-finite value")
            return runtime_basis

        def metadata(self) -> Dict[str, str]:
            return {
                "basis_family": self.basis_family,
                "basis_version": self.basis_version,
                "coordinate_policy": self.coordinate_policy,
                "stabilization_policy": self.stabilization_policy,
            }


    @dataclass(frozen=True)
    class BasisDefinition:
        family: str
        aliases: Tuple[str, ...]
        version: str
        artifact_tag: str
        supports_weight_basis: bool
        supports_native_products: bool
        kernel: BasisKernel

        def __post_init__(self) -> None:
            family = self.family.strip().lower()
            aliases = tuple(alias.strip().lower() for alias in self.aliases)
            version = self.version.strip().lower()
            artifact_tag = self.artifact_tag.strip().upper()
            if not _FAMILY_PATTERN.fullmatch(family):
                raise ValueError(f"invalid basis family: {self.family!r}")
            if not version:
                raise ValueError("basis version must be non-empty")
            if not _ARTIFACT_TAG_PATTERN.fullmatch(artifact_tag):
                raise ValueError(f"invalid basis artifact tag: {self.artifact_tag!r}")
            if any(not alias for alias in aliases):
                raise ValueError("basis aliases must be non-empty")
            if len(set(aliases)) != len(aliases):
                raise ValueError(f"duplicate aliases within basis definition {family!r}")
            if self.kernel.basis_family != family:
                raise ValueError(f"kernel family mismatch: definition={family!r}, kernel={self.kernel.basis_family!r}")
            if self.kernel.basis_version != version:
                raise ValueError(f"kernel version mismatch: definition={version!r}, kernel={self.kernel.basis_version!r}")
            object.__setattr__(self, "family", family)
            object.__setattr__(self, "aliases", aliases)
            object.__setattr__(self, "version", version)
            object.__setattr__(self, "artifact_tag", artifact_tag)

        def build(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None) -> Tensor:
            return self.kernel.build(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version or self.version)

        def metadata(self) -> Dict[str, object]:
            return {
                "basis_family": self.family,
                "basis_aliases": self.aliases,
                "basis_version": self.version,
                "artifact_tag": self.artifact_tag,
                "supports_weight_basis": self.supports_weight_basis,
                "supports_native_products": self.supports_native_products,
                **self.kernel.metadata(),
            }
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/bases/chebyshev.py",
    r'''
    # vvv THOG
    from __future__ import annotations

    from typing import Optional

    import torch
    from torch import Tensor

    from .protocol import BasisDefinition, BasisKernel, DeviceLike, deterministic_reduced_qr_positive_diagonal, validate_floating_dtype, validate_positive_integer


    BASIS_FAMILY_CHEBYSHEV = "chebyshev"
    CHEBYSHEV_BASIS_VERSION = "chebyshev_first_kind_qr_v1"
    BASIS_ARTIFACT_TAG_CHEBYSHEV = "CHEBY"
    SINGLE_POINT_COORDINATE = 0.0


    def chebyshev_coordinates(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        validate_positive_integer("sample_count", sample_count)
        validate_floating_dtype(dtype)
        target_device = torch.device("cpu" if device is None else device)
        if sample_count == 1:
            return torch.tensor([SINGLE_POINT_COORDINATE], dtype=dtype, device=target_device)
        return torch.linspace(-1.0, 1.0, sample_count, dtype=dtype, device=target_device)


    def chebyshev_raw_basis(coordinates: Tensor, order: int) -> Tensor:
        validate_positive_integer("order", order)
        if coordinates.ndim != 1:
            raise ValueError(f"coordinates must be one-dimensional; got shape {tuple(coordinates.shape)}")
        if coordinates.numel() == 0:
            raise ValueError("coordinates must contain at least one sample")
        if not coordinates.is_floating_point():
            raise ValueError(f"coordinates must use a floating dtype; got {coordinates.dtype}")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        sample_count = coordinates.numel()
        basis = torch.empty((sample_count, order), dtype=coordinates.dtype, device=coordinates.device)
        basis[:, 0] = 1.0
        if order == 1:
            return basis
        basis[:, 1] = coordinates
        for term_index in range(2, order):
            basis[:, term_index] = 2.0 * coordinates * basis[:, term_index - 1] - basis[:, term_index - 2]
        return basis


    class ChebyshevQrBasisKernel(BasisKernel):
        def __init__(self) -> None:
            super().__init__(
                basis_family=BASIS_FAMILY_CHEBYSHEV,
                basis_version=CHEBYSHEV_BASIS_VERSION,
                coordinate_policy="linear_minus_one_to_one_single_point_zero_v1",
                stabilization_policy="deterministic_reduced_qr_positive_diagonal_v1",
            )

        def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
            return chebyshev_coordinates(sample_count, dtype=dtype, device=device)

        def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
            return chebyshev_raw_basis(coordinates, order)

        def stabilize(self, raw_basis: Tensor) -> Tensor:
            stabilized_basis, _ = deterministic_reduced_qr_positive_diagonal(raw_basis)
            return stabilized_basis


    BASIS_DEFINITION = BasisDefinition(
        family=BASIS_FAMILY_CHEBYSHEV,
        aliases=("cheby", "chebyshev_first_kind_qr"),
        version=CHEBYSHEV_BASIS_VERSION,
        artifact_tag=BASIS_ARTIFACT_TAG_CHEBYSHEV,
        supports_weight_basis=True,
        supports_native_products=False,
        kernel=ChebyshevQrBasisKernel(),
    )
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/bases/dct.py",
    r'''
    # vvv THOG
    from __future__ import annotations

    import math
    from typing import Optional

    import torch
    from torch import Tensor

    from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


    BASIS_FAMILY_DCT = "dct"
    DCT_BASIS_VERSION = "dct_ii_orthonormal_v1"
    BASIS_ARTIFACT_TAG_DCT = "DCT"


    def dct_sample_indices(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        validate_positive_integer("sample_count", sample_count)
        validate_floating_dtype(dtype)
        target_device = torch.device("cpu" if device is None else device)
        return torch.arange(sample_count, dtype=dtype, device=target_device)


    def dct_ii_orthonormal_raw_basis(sample_indices: Tensor, order: int) -> Tensor:
        validate_positive_integer("order", order)
        if sample_indices.ndim != 1:
            raise ValueError(f"sample_indices must be one-dimensional; got shape {tuple(sample_indices.shape)}")
        if sample_indices.numel() == 0:
            raise ValueError("sample_indices must contain at least one sample")
        if not sample_indices.is_floating_point():
            raise ValueError(f"sample_indices must use a floating dtype; got {sample_indices.dtype}")
        if not torch.isfinite(sample_indices).all():
            raise ValueError("sample_indices must be finite")
        sample_count = sample_indices.numel()
        column_indices = torch.arange(order, dtype=sample_indices.dtype, device=sample_indices.device)
        angles = math.pi * (sample_indices.unsqueeze(1) + 0.5) * column_indices.unsqueeze(0) / float(sample_count)
        basis = torch.cos(angles)
        basis[:, 0] *= math.sqrt(1.0 / float(sample_count))
        if order > 1:
            basis[:, 1:] *= math.sqrt(2.0 / float(sample_count))
        return basis


    class DctIiOrthonormalBasisKernel(BasisKernel):
        def __init__(self) -> None:
            super().__init__(
                basis_family=BASIS_FAMILY_DCT,
                basis_version=DCT_BASIS_VERSION,
                coordinate_policy="integer_sample_index_half_shifted_dct_ii_v1",
                stabilization_policy="closed_form_dct_ii_orthonormal_no_qr_v1",
            )

        def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
            return dct_sample_indices(sample_count, dtype=dtype, device=device)

        def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
            return dct_ii_orthonormal_raw_basis(coordinates, order)

        def stabilize(self, raw_basis: Tensor) -> Tensor:
            if raw_basis.ndim != 2:
                raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
            if not raw_basis.is_floating_point():
                raise ValueError(f"raw_basis must use a floating dtype; got {raw_basis.dtype}")
            if not torch.isfinite(raw_basis).all():
                raise FloatingPointError("DCT basis contains a non-finite value")
            return raw_basis


    BASIS_DEFINITION = BasisDefinition(
        family=BASIS_FAMILY_DCT,
        aliases=("dct_ii", "dct_ii_orthonormal"),
        version=DCT_BASIS_VERSION,
        artifact_tag=BASIS_ARTIFACT_TAG_DCT,
        supports_weight_basis=True,
        supports_native_products=False,
        kernel=DctIiOrthonormalBasisKernel(),
    )
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/bases/registry.py",
    r'''
    # vvv THOG
    from __future__ import annotations

    from importlib import import_module
    from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

    import torch
    from torch import Tensor

    from .protocol import BasisDefinition, BasisKernel, DeviceLike


    BUILTIN_BASIS_MODULES = (
        "chebyshev",
        "dct",
    )


    def _normalize_token(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"basis family must be a non-empty string; got {value!r}")
        return value.strip().lower()


    class BasisRegistry(Mapping[str, BasisDefinition]):
        def __init__(self, definitions: Iterable[BasisDefinition] = ()) -> None:
            self._definitions: Dict[str, BasisDefinition] = {}
            self._lookup: Dict[str, str] = {}
            self._versions: Dict[str, str] = {}
            self._artifact_tags: Dict[str, str] = {}
            for definition in definitions:
                self.register(definition)

        def register(self, definition: BasisDefinition) -> None:
            if not isinstance(definition, BasisDefinition):
                raise TypeError(f"definition must be BasisDefinition; got {type(definition).__name__}")
            family = definition.family
            if family in self._definitions:
                raise ValueError(f"duplicate basis family: {family!r}")
            if definition.version in self._versions:
                raise ValueError(f"duplicate basis version: {definition.version!r}")
            if definition.artifact_tag in self._artifact_tags:
                raise ValueError(f"duplicate basis artifact tag: {definition.artifact_tag!r}")
            tokens = (family, *definition.aliases, definition.version)
            local_tokens = set()
            for token in tokens:
                normalized = _normalize_token(token)
                if normalized in local_tokens:
                    raise ValueError(f"duplicate basis alias within {family!r}: {normalized!r}")
                if normalized in self._lookup:
                    raise ValueError(f"basis alias collision: {normalized!r}")
                local_tokens.add(normalized)
            self._definitions[family] = definition
            self._versions[definition.version] = family
            self._artifact_tags[definition.artifact_tag] = family
            for token in local_tokens:
                self._lookup[token] = family

        def normalize(self, basis_family: str) -> str:
            token = _normalize_token(basis_family)
            try:
                return self._lookup[token]
            except KeyError as error:
                raise ValueError(f"unknown basis_family: {basis_family!r}") from error

        def get_definition(self, basis_family: str) -> BasisDefinition:
            return self._definitions[self.normalize(basis_family)]

        def families(self) -> Tuple[str, ...]:
            return tuple(self._definitions)

        def definitions(self) -> Tuple[BasisDefinition, ...]:
            return tuple(self._definitions.values())

        def __getitem__(self, key: str) -> BasisDefinition:
            return self.get_definition(key)

        def __iter__(self) -> Iterator[str]:
            return iter(self._definitions)

        def __len__(self) -> int:
            return len(self._definitions)


    def _load_builtin_definitions() -> Tuple[BasisDefinition, ...]:
        definitions = []
        for module_name in BUILTIN_BASIS_MODULES:
            module = import_module(f"{__package__}.{module_name}")
            definition = getattr(module, "BASIS_DEFINITION", None)
            if not isinstance(definition, BasisDefinition):
                raise TypeError(f"{module.__name__}.BASIS_DEFINITION must be BasisDefinition")
            definitions.append(definition)
        return tuple(definitions)


    BASIS_REGISTRY = BasisRegistry(_load_builtin_definitions())
    BASIS_FAMILIES = BASIS_REGISTRY.families()


    def registered_basis_families() -> Tuple[str, ...]:
        return BASIS_REGISTRY.families()


    def normalize_registered_basis_family(basis_family: str) -> str:
        return BASIS_REGISTRY.normalize(basis_family)


    def normalize_basis_family(basis_family: str) -> str:
        return normalize_registered_basis_family(basis_family)


    def get_basis_definition(basis_family: str) -> BasisDefinition:
        return BASIS_REGISTRY.get_definition(basis_family)


    def get_basis_spec(basis_family: str) -> BasisDefinition:
        return get_basis_definition(basis_family)


    def get_basis_kernel(basis_family: str) -> BasisKernel:
        return get_basis_definition(basis_family).kernel


    def basis_version_for_family(basis_family: str) -> str:
        return get_basis_definition(basis_family).version


    def basis_artifact_tag_for_family(basis_family: str) -> str:
        return get_basis_definition(basis_family).artifact_tag


    def basis_kernel_metadata(basis_family: str) -> Dict[str, str]:
        return get_basis_kernel(basis_family).metadata()


    def basis_registry_metadata(basis_family: str) -> Dict[str, object]:
        return get_basis_definition(basis_family).metadata()


    def normalize_basis_version(basis_family: str, basis_version: str, *, legacy_default_version: Optional[str] = None) -> str:
        expected_version = basis_version_for_family(basis_family)
        if basis_version == "auto":
            return expected_version
        if legacy_default_version is not None and basis_version == legacy_default_version and expected_version != legacy_default_version:
            return expected_version
        if basis_version != expected_version:
            raise ValueError(f"basis_version mismatch for basis_family={normalize_registered_basis_family(basis_family)!r}: expected {expected_version!r}, got {basis_version!r}")
        return expected_version


    def build_registered_basis(sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None, basis_family: str) -> Tensor:
        definition = get_basis_definition(basis_family)
        return definition.build(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version)
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/bases/__init__.py",
    r'''
    # vvv THOG
    from .protocol import BasisDefinition, BasisKernel, DeviceLike, deterministic_reduced_qr_positive_diagonal, validate_floating_dtype, validate_positive_integer
    from .chebyshev import BASIS_ARTIFACT_TAG_CHEBYSHEV, BASIS_FAMILY_CHEBYSHEV, CHEBYSHEV_BASIS_VERSION, SINGLE_POINT_COORDINATE, ChebyshevQrBasisKernel, chebyshev_coordinates, chebyshev_raw_basis
    from .dct import BASIS_ARTIFACT_TAG_DCT, BASIS_FAMILY_DCT, DCT_BASIS_VERSION, DctIiOrthonormalBasisKernel, dct_ii_orthonormal_raw_basis, dct_sample_indices
    from .registry import BASIS_FAMILIES, BASIS_REGISTRY, BUILTIN_BASIS_MODULES, BasisRegistry, basis_artifact_tag_for_family, basis_kernel_metadata, basis_registry_metadata, basis_version_for_family, build_registered_basis, get_basis_definition, get_basis_kernel, get_basis_spec, normalize_basis_family, normalize_basis_version, normalize_registered_basis_family, registered_basis_families


    BasisSpec = BasisDefinition


    __all__ = [
        "BASIS_ARTIFACT_TAG_CHEBYSHEV",
        "BASIS_ARTIFACT_TAG_DCT",
        "BASIS_FAMILIES",
        "BASIS_FAMILY_CHEBYSHEV",
        "BASIS_FAMILY_DCT",
        "BASIS_REGISTRY",
        "BUILTIN_BASIS_MODULES",
        "CHEBYSHEV_BASIS_VERSION",
        "DCT_BASIS_VERSION",
        "SINGLE_POINT_COORDINATE",
        "BasisDefinition",
        "BasisKernel",
        "BasisRegistry",
        "BasisSpec",
        "ChebyshevQrBasisKernel",
        "DctIiOrthonormalBasisKernel",
        "DeviceLike",
        "basis_artifact_tag_for_family",
        "basis_kernel_metadata",
        "basis_registry_metadata",
        "basis_version_for_family",
        "build_registered_basis",
        "chebyshev_coordinates",
        "chebyshev_raw_basis",
        "dct_ii_orthonormal_raw_basis",
        "dct_sample_indices",
        "deterministic_reduced_qr_positive_diagonal",
        "get_basis_definition",
        "get_basis_kernel",
        "get_basis_spec",
        "normalize_basis_family",
        "normalize_basis_version",
        "normalize_registered_basis_family",
        "registered_basis_families",
        "validate_floating_dtype",
        "validate_positive_integer",
    ]
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/basis_registry.py",
    r'''
    # vvv THOG
    from .bases import *  # noqa: F401,F403
    # ^^^ THOG
    ''',
)

write_file(
    "sheet/basis_kernel.py",
    r'''
    # vvv THOG
    from __future__ import annotations

    from types import MappingProxyType
    from typing import Mapping

    from .bases import (
        BASIS_FAMILIES,
        BASIS_FAMILY_CHEBYSHEV,
        BASIS_FAMILY_DCT,
        CHEBYSHEV_BASIS_VERSION,
        DCT_BASIS_VERSION,
        SINGLE_POINT_COORDINATE,
        BasisKernel,
        ChebyshevQrBasisKernel,
        DctIiOrthonormalBasisKernel,
        basis_artifact_tag_for_family,
        basis_kernel_metadata,
        basis_registry_metadata,
        basis_version_for_family,
        build_registered_basis,
        chebyshev_coordinates,
        chebyshev_raw_basis,
        dct_ii_orthonormal_raw_basis,
        dct_sample_indices,
        deterministic_reduced_qr_positive_diagonal,
        get_basis_definition,
        get_basis_kernel,
        get_basis_spec,
        normalize_basis_family,
        normalize_basis_version,
        normalize_registered_basis_family,
        registered_basis_families,
        validate_floating_dtype,
        validate_positive_integer,
    )


    _BASIS_KERNELS: Mapping[str, BasisKernel] = MappingProxyType({family: get_basis_kernel(family) for family in BASIS_FAMILIES})


    __all__ = [
        "BASIS_FAMILIES",
        "BASIS_FAMILY_CHEBYSHEV",
        "BASIS_FAMILY_DCT",
        "CHEBYSHEV_BASIS_VERSION",
        "DCT_BASIS_VERSION",
        "SINGLE_POINT_COORDINATE",
        "BasisKernel",
        "ChebyshevQrBasisKernel",
        "DctIiOrthonormalBasisKernel",
        "basis_artifact_tag_for_family",
        "basis_kernel_metadata",
        "basis_registry_metadata",
        "basis_version_for_family",
        "build_registered_basis",
        "chebyshev_coordinates",
        "chebyshev_raw_basis",
        "dct_ii_orthonormal_raw_basis",
        "dct_sample_indices",
        "deterministic_reduced_qr_positive_diagonal",
        "get_basis_definition",
        "get_basis_kernel",
        "get_basis_spec",
        "normalize_basis_family",
        "normalize_basis_version",
        "normalize_registered_basis_family",
        "registered_basis_families",
        "validate_floating_dtype",
        "validate_positive_integer",
    ]
    # ^^^ THOG
    ''',
)

replace_once(
    "sheet/compact_identity.py",
    "from .basis_kernel import BASIS_FAMILY_DCT as KERNEL_BASIS_FAMILY_DCT, DCT_BASIS_VERSION, basis_version_for_family\n",
    "from .bases import BASIS_FAMILIES as REGISTERED_BASIS_FAMILIES, BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, basis_version_for_family, normalize_basis_version, normalize_registered_basis_family\n",
)
replace_once(
    "sheet/compact_identity.py",
    'BASIS_FAMILY_CHEBYSHEV = "chebyshev"\nBASIS_FAMILY_DCT = KERNEL_BASIS_FAMILY_DCT\nBASIS_FAMILY_CONVENTIONAL = "conventional"\n',
    'BASIS_FAMILY_CONVENTIONAL = "conventional"\n',
)
replace_once(
    "sheet/compact_identity.py",
    "BASIS_FAMILIES = (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, BASIS_FAMILY_CONVENTIONAL)\n",
    "BASIS_FAMILIES = (*REGISTERED_BASIS_FAMILIES, BASIS_FAMILY_CONVENTIONAL)\n",
)
replace_once(
    "sheet/compact_identity.py",
    '''def _require_member(name: str, value: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    normalized = _canonical_optional_string(name, value)
    if normalized is not None and normalized not in allowed:
        raise ValueError(f"{name} must be one of {allowed} or None; got {value!r}")
    return normalized
''',
    '''def _require_member(name: str, value: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    normalized = _canonical_optional_string(name, value)
    if normalized is not None and normalized not in allowed:
        raise ValueError(f"{name} must be one of {allowed} or None; got {value!r}")
    return normalized


# vvv THOG basis-family canonicalisation is owned by the registry rather than duplicated selector allow-lists
def _require_basis_family(value: Optional[str]) -> Optional[str]:
    normalized = _canonical_optional_string("basis_family", value)
    if normalized is None or normalized == BASIS_FAMILY_CONVENTIONAL:
        return normalized
    try:
        return normalize_registered_basis_family(normalized)
    except ValueError as error:
        raise ValueError(f"basis_family must be one of {BASIS_FAMILIES} or None; got {value!r}") from error
# ^^^ THOG
''',
)
replace_once(
    "sheet/compact_identity.py",
    '    requested_basis = _require_member("basis_family", basis_family, BASIS_FAMILIES)\n',
    '    requested_basis = _require_basis_family(basis_family)\n',
)
replace_once(
    "sheet/compact_identity.py",
    '''def normalize_compact_basis_version(selectors: ResolvedCompactSelectors, basis_version: str) -> str:
    if selectors.basis_family == BASIS_FAMILY_CONVENTIONAL:
        return basis_version
    expected_version = basis_version_for_family(selectors.basis_family)
    if selectors.basis_family == BASIS_FAMILY_DCT and basis_version == BASIS_VERSION:
        return DCT_BASIS_VERSION
    if basis_version != expected_version:
        raise ValueError(
            f"basis_version mismatch for basis_family={selectors.basis_family!r}: "
            f"expected {expected_version!r}, got {basis_version!r}"
        )
    return basis_version
''',
    '''def normalize_compact_basis_version(selectors: ResolvedCompactSelectors, basis_version: str) -> str:
    if selectors.basis_family == BASIS_FAMILY_CONVENTIONAL:
        return basis_version
    return normalize_basis_version(selectors.basis_family, basis_version, legacy_default_version=BASIS_VERSION)
''',
)
replace_once(
    "sheet/compact_identity.py",
    "    supported_basis = selectors.basis_family in (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT)\n",
    "    supported_basis = selectors.basis_family in REGISTERED_BASIS_FAMILIES\n",
)
replace_once(
    "sheet/compact_identity.py",
    '        "with chebyshev or dct basis; "\n',
    '        f"with a registered basis family {REGISTERED_BASIS_FAMILIES}; "\n',
)

replace_once(
    "sheet/run_config.py",
    "from .basis import BASIS_VERSION\nfrom .compact_identity import BASIS_FAMILY_CHEBYSHEV, GEOMETRY_PRESET_DEPTH, compact_identity_metadata\n",
    "from .basis import BASIS_VERSION\nfrom .bases import basis_artifact_tag_for_family, basis_version_for_family\nfrom .compact_identity import BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_CONVENTIONAL, GEOMETRY_PRESET_DEPTH, compact_identity_metadata\n",
)
replace_once(
    "sheet/run_config.py",
    'BASIS_LABELS = {"chebyshev": "CHEBY", "dct": "DCT"}\n',
    '# vvv THOG basis artifact tags now come from the basis-family registry\n# ^^^ THOG\n',
)
replace_once(
    "sheet/run_config.py",
    '''        if self.basis_version == "auto":
            object.__setattr__(self, "basis_version", BASIS_VERSION)
''',
    '''        if self.basis_version == "auto":
            requested_family = self.basis_family or BASIS_FAMILY_CHEBYSHEV
            resolved_version = BASIS_VERSION if requested_family == BASIS_FAMILY_CONVENTIONAL else basis_version_for_family(requested_family)
            object.__setattr__(self, "basis_version", resolved_version)
''',
)
replace_once(
    "sheet/run_config.py",
    '        basis_label = BASIS_LABELS.get(str(identity["basis_family"]), str(identity["basis_family"]).upper())\n',
    '        basis_label = basis_artifact_tag_for_family(str(identity["basis_family"]))\n',
)

replace_once(
    "run_thog2_owt.py",
    "from sheet.basis import BASIS_VERSION\n",
    "from sheet.basis import BASIS_VERSION\nfrom sheet.bases import BASIS_FAMILIES\n",
)
replace_once(
    "run_thog2_owt.py",
    "from sheet.compact_identity import ATTENTION_GEOMETRIES, BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESETS, MLP_GEOMETRIES\n",
    "from sheet.compact_identity import ATTENTION_GEOMETRIES, BASIS_FAMILY_CHEBYSHEV, GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESETS, MLP_GEOMETRIES\n",
)
replace_once(
    "run_thog2_owt.py",
    '    parser.add_argument("--basis-family", choices=(BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT), default=BASIS_FAMILY_CHEBYSHEV)\n',
    '    parser.add_argument("--basis-family", choices=BASIS_FAMILIES, default=BASIS_FAMILY_CHEBYSHEV)\n',
)

for wrapper_path in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    replace_once(
        wrapper_path,
        '  -B BASIS_FAMILY=${BASIS_FAMILY}                   chebyshev | dct\n',
        '  -B BASIS_FAMILY=${BASIS_FAMILY}                   registered fixed basis family\n',
    )
    replace_once(
        wrapper_path,
        'case "$BASIS_FAMILY" in chebyshev|dct) ;; *) echo "BASIS_FAMILY must be chebyshev or dct." >&2; exit 2 ;; esac\n',
        '[[ "$BASIS_FAMILY" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "BASIS_FAMILY must be a lowercase registry name or alias." >&2; exit 2; }\n',
    )
    replace_once(
        wrapper_path,
        'BASIS_TAG="CHEBY"; [[ "$BASIS_FAMILY" == dct ]] && BASIS_TAG="DCT"\n',
        'BASIS_TAG="$("$PYTHON_BIN" -c \'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))\' "$BASIS_FAMILY")"                           # <<< THOG registry-derived basis validation and artifact tag\n',
    )

write_file(
    "tests/test_basis_family_plugin_registry.py",
    r'''
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
            self.assertEqual(BASIS_FAMILIES, (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT))
            self.assertEqual(normalize_registered_basis_family("cheby"), BASIS_FAMILY_CHEBYSHEV)
            self.assertEqual(normalize_registered_basis_family(CHEBYSHEV_BASIS_VERSION), BASIS_FAMILY_CHEBYSHEV)
            self.assertEqual(normalize_registered_basis_family("dct_ii"), BASIS_FAMILY_DCT)
            self.assertEqual(normalize_registered_basis_family(DCT_BASIS_VERSION), BASIS_FAMILY_DCT)
            self.assertEqual(basis_version_for_family(BASIS_FAMILY_CHEBYSHEV), CHEBYSHEV_BASIS_VERSION)
            self.assertEqual(basis_version_for_family(BASIS_FAMILY_DCT), DCT_BASIS_VERSION)
            self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_CHEBYSHEV), BASIS_ARTIFACT_TAG_CHEBYSHEV)
            self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_DCT), BASIS_ARTIFACT_TAG_DCT)
            self.assertEqual(get_basis_definition("cheby").family, BASIS_FAMILY_CHEBYSHEV)

        def test_02_local_registry_accepts_a_third_basis_and_builds_it_without_geometry_code(self) -> None:
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
            selectors = resolve_compact_selectors(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family="cheby")
            self.assertEqual(selectors.basis_family, BASIS_FAMILY_CHEBYSHEV)
            self.assertEqual(selectors.requested_basis_family, BASIS_FAMILY_CHEBYSHEV)

        def test_05_python_cli_choices_are_registry_derived(self) -> None:
            parser = run_thog2_owt.build_parser()
            action = next(action for action in parser._actions if action.dest == "basis_family")
            self.assertEqual(tuple(action.choices), BASIS_FAMILIES)

        def test_06_run_artifact_tags_remain_unchanged(self) -> None:
            cheby = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_CHEBYSHEV, basis_version="auto")
            dct = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_DCT, basis_version="auto")
            self.assertEqual(cheby.compact_artifact_fragment(), "CHEBY_DEPTH")
            self.assertEqual(dct.compact_artifact_fragment(), "DCT_DEPTH")
            self.assertEqual(cheby.basis_version, CHEBYSHEV_BASIS_VERSION)
            self.assertEqual(dct.basis_version, DCT_BASIS_VERSION)

        def test_07_primary_wrappers_have_no_chebyshev_dct_allow_list_or_tag_branch(self) -> None:
            root = Path(__file__).resolve().parents[1]
            for name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
                with self.subTest(name=name):
                    text = (root / name).read_text(encoding="utf-8")
                    self.assertNotIn("chebyshev|dct", text)
                    self.assertNotIn('BASIS_TAG="CHEBY"', text)
                    self.assertIn("basis_artifact_tag_for_family", text)


    if __name__ == "__main__":
        unittest.main(verbosity=2)
    # ^^^ THOG
    ''',
)

print("Applied THOG2 basis-family plug-in refactor")
