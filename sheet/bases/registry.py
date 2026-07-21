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
    "haar",
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
