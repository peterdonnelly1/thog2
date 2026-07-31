# vvv THOG
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, Tuple

from .bqrg import BQRG_DEFINITION
from .protocol import RecurrenceGeneratorDefinition


class RecurrenceGeneratorRegistry(Mapping[str, RecurrenceGeneratorDefinition]):
    def __init__(self, definitions: Iterable[RecurrenceGeneratorDefinition] = ()) -> None:
        self._definitions: Dict[str, RecurrenceGeneratorDefinition] = {}
        self._lookup: Dict[str, str] = {}
        self._versions: Dict[str, str] = {}
        self._artifact_tags: Dict[str, str] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: RecurrenceGeneratorDefinition) -> None:
        if not isinstance(definition, RecurrenceGeneratorDefinition):
            raise TypeError(f"definition must be RecurrenceGeneratorDefinition; got {type(definition).__name__}")
        if definition.family in self._definitions:
            raise ValueError(f"duplicate recurrence generator family: {definition.family!r}")
        if definition.version in self._versions:
            raise ValueError(f"duplicate recurrence generator version: {definition.version!r}")
        if definition.artifact_tag in self._artifact_tags:
            raise ValueError(f"duplicate recurrence generator artifact tag: {definition.artifact_tag!r}")
        tokens = (definition.family, *definition.aliases, definition.version)
        for token in tokens:
            normalized = token.strip().lower()
            if normalized in self._lookup:
                raise ValueError(f"recurrence generator alias collision: {normalized!r}")
            self._lookup[normalized] = definition.family
        self._definitions[definition.family] = definition
        self._versions[definition.version] = definition.family
        self._artifact_tags[definition.artifact_tag] = definition.family

    def normalize(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"recurrence generator family must be a non-empty string; got {value!r}")
        token = value.strip().lower()
        try:
            return self._lookup[token]
        except KeyError as error:
            raise ValueError(f"unknown recurrence generator family: {value!r}") from error

    def get_definition(self, value: str) -> RecurrenceGeneratorDefinition:
        return self._definitions[self.normalize(value)]

    def families(self) -> Tuple[str, ...]:
        return tuple(self._definitions)

    def definitions(self) -> Tuple[RecurrenceGeneratorDefinition, ...]:
        return tuple(self._definitions.values())

    def __getitem__(self, key: str) -> RecurrenceGeneratorDefinition:
        return self.get_definition(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    # vvv THOG keep Mapping membership total even though normalize() intentionally raises ValueError for public validation failures.
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or not key.strip():
            return False
        return key.strip().lower() in self._lookup
    # ^^^ THOG


RECURRENCE_GENERATOR_REGISTRY = RecurrenceGeneratorRegistry((BQRG_DEFINITION,))
RECURRENCE_GENERATOR_FAMILIES = RECURRENCE_GENERATOR_REGISTRY.families()


def normalize_recurrence_generator_family(value: str) -> str:
    return RECURRENCE_GENERATOR_REGISTRY.normalize(value)


def get_recurrence_generator_definition(value: str) -> RecurrenceGeneratorDefinition:
    return RECURRENCE_GENERATOR_REGISTRY.get_definition(value)


def recurrence_generator_version_for_family(value: str) -> str:
    return get_recurrence_generator_definition(value).version


def is_recurrence_generator_family(value: str) -> bool:
    try:
        normalize_recurrence_generator_family(value)
    except ValueError:
        return False
    return True


def validate_recurrence_generator_width(value: str, width: int) -> None:
    definition = get_recurrence_generator_definition(value)
    if not definition.accepts_persistent_width(width):
        allowed = ", ".join(str(item) for item in definition.persistent_widths)
        raise ValueError(
            f"recurrence generator {definition.family}@{definition.version} requires persistent width in ({allowed}); got {width}"
        )
# ^^^ THOG