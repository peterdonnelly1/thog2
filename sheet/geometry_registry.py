# vvv THOG
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .bases import BASIS_FAMILIES, basis_version_for_family, normalize_registered_basis_family
# vvv THOG recurrence generators are independent plugins consumed by the same geometry-facing compressor registry
from .recurrence_generators import RECURRENCE_GENERATOR_REGISTRY, get_recurrence_generator_definition, is_recurrence_generator_family, normalize_recurrence_generator_family, validate_recurrence_generator_width
# ^^^ THOG

GEOMETRY_REGISTRY_VERSION = "geometry_registry_v5"
GEOMETRY_PLAN_SCHEMA_VERSION = 1

ELEMENT_TYPE_CURVE = "CURVE"
ELEMENT_TYPE_SHEET = "SHEET"
ELEMENT_TYPE_SHEET_SET = "SHEET_SET"
ELEMENT_TYPES = (ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, ELEMENT_TYPE_SHEET_SET)

AXIS_DEPTH = "DEPTH"
AXIS_MLP_HIDDEN = "MLP_HIDDEN"
AXIS_MLP_D_MODEL = "MLP_D_MODEL"
AXIS_ATTENTION_D_MODEL = "ATTENTION_D_MODEL"
AXIS_ATTENTION_HEAD_CHANNEL = "ATTENTION_HEAD_CHANNEL"

ELEMENT_MLP_UP = "MLP_UP"
ELEMENT_MLP_DOWN = "MLP_DOWN"
ELEMENT_ATTENTION_QKV = "ATTENTION_QKV"
ELEMENT_ATTENTION_OUTPUT = "ATTENTION_OUTPUT"

COMPRESSOR_JPEG_LIKE = "jpeg_like"
JPEG_LIKE_VERSION = "jpeg_like_v1"
LEGACY_SEMANTICALLY_UNSOUND_PRESETS = frozenset({"legacy_sheet_col", "jpeg_like_v1", "mlp_block", "head_aware_block", "full_block"})

ORDER_OPTION_FIELDS: Mapping[Tuple[str, str], str] = {
    (ELEMENT_MLP_UP, AXIS_MLP_HIDDEN): "o_mlp_hidden",
    (ELEMENT_MLP_UP, AXIS_MLP_D_MODEL): "o_mlp_d_model",
    (ELEMENT_MLP_DOWN, AXIS_MLP_HIDDEN): "o_mlp_hidden",
    (ELEMENT_MLP_DOWN, AXIS_MLP_D_MODEL): "o_mlp_d_model",
    (ELEMENT_ATTENTION_QKV, AXIS_ATTENTION_D_MODEL): "o_attn_d_model",
    (ELEMENT_ATTENTION_QKV, AXIS_ATTENTION_HEAD_CHANNEL): "o_attn_qkv_per_channel",
    (ELEMENT_ATTENTION_OUTPUT, AXIS_ATTENTION_D_MODEL): "o_attn_d_model",
    (ELEMENT_ATTENTION_OUTPUT, AXIS_ATTENTION_HEAD_CHANNEL): "o_attn_out_per_channel",
}


@dataclass(frozen=True)
class GeometryEntry:
    selector: str
    element: str
    compressed_axes: Tuple[str, ...]
    implied_type: str
    permits_depth_companion: bool
    independent_indices: Tuple[str, ...] = ()
    description: str = ""
    implemented: bool = False
    legacy_only: bool = False
    implementation_note: str = ""

    @property
    def is_complete_element(self) -> bool:
        return self.selector == self.element

    def resolved_type(self, depth_enabled: bool) -> str:
        if depth_enabled and not self.permits_depth_companion:
            raise ValueError(
                f"{self.selector} is a permitted {self.implied_type} geometry, but combining it with DEPTH would imply "
                "a three-dimensional BLOCK/BLOCK_SET field, which is not part of the systematic geometry design"
            )
        return self.implied_type


@dataclass(frozen=True)
class CompressorCapability:
    family: str
    default_version: str
    element_types: Tuple[str, ...]
    implemented: bool
    legacy_only: bool = False
    supports_group_size: bool = False
    supported_selectors: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def semantic_dimensions(self) -> Tuple[int, ...]:
        return tuple(sorted({1 if element_type == ELEMENT_TYPE_CURVE else 2 for element_type in self.element_types}))

    @property
    def implemented_dimensions(self) -> Tuple[int, ...]:
        return self.semantic_dimensions if self.implemented else ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedOption:
    target: str
    property: str
    value: str
    source: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedGeometrySelection:
    selector: str
    element: str
    implied_type: str
    compressed_axes: Tuple[str, ...]
    independent_indices: Tuple[str, ...]
    compressor: str
    compressor_version: str
    orders: Dict[str, int]
    axis_options: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializerAdapter:
    implemented: bool
    legacy_geometry_preset: Optional[str]
    legacy_basis_family: Optional[str]
    legacy_basis_version: Optional[str]
    legacy_mlp_hidden_compressor: Optional[str]
    legacy_mlp_hidden_group_size: Optional[int]
    materialization_version: Optional[str]
    message: str
    legacy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedGeometryPlan:
    schema_version: int
    registry_version: str
    depth_enabled: bool
    depth_compressor: Optional[str]
    depth_compressor_version: Optional[str]
    depth_order: Optional[int]
    selections: Tuple[ResolvedGeometrySelection, ...]
    parsed_options: Tuple[ParsedOption, ...]
    shared_non_depth_compressor: Optional[str]
    shared_non_depth_compressor_version: Optional[str]
    materializer: MaterializerAdapter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "depth_enabled": self.depth_enabled,
            "depth_compressor": self.depth_compressor,
            "depth_compressor_version": self.depth_compressor_version,
            "depth_order": self.depth_order,
            "selections": [selection.to_dict() for selection in self.selections],
            "parsed_options": [option.to_dict() for option in self.parsed_options],
            "shared_non_depth_compressor": self.shared_non_depth_compressor,
            "shared_non_depth_compressor_version": self.shared_non_depth_compressor_version,
            "materializer": self.materializer.to_dict(),
        }


_GEOMETRY_ENTRIES = (
    GeometryEntry("MLP_UP.MLP_HIDDEN", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, True, (AXIS_MLP_D_MODEL,), "MLP expansion hidden-axis curve", False, False, "One legacy adapter covers DEPTH + jpeg_like; generic materialisation is Phase 2."),
    GeometryEntry("MLP_UP.MLP_D_MODEL", ELEMENT_MLP_UP, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, True, (AXIS_MLP_HIDDEN,), "MLP expansion model-axis curve"),
    GeometryEntry("MLP_UP", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, False, (), "Complete MLP expansion sheet"),
    GeometryEntry("MLP_DOWN.MLP_HIDDEN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, True, (AXIS_MLP_D_MODEL,), "MLP contraction hidden-axis curve"),
    GeometryEntry("MLP_DOWN.MLP_D_MODEL", ELEMENT_MLP_DOWN, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, True, (AXIS_MLP_HIDDEN,), "MLP contraction model-axis curve"),
    GeometryEntry("MLP_DOWN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, False, (), "Complete MLP contraction sheet"),
    GeometryEntry("ATTENTION_QKV.ATTENTION_D_MODEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_D_MODEL,), ELEMENT_TYPE_CURVE, True, ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL), "QKV input-model-axis curves"),
    GeometryEntry("ATTENTION_QKV.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_HEAD_CHANNEL,), ELEMENT_TYPE_CURVE, True, ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL), "QKV within-head channel curves"),
    GeometryEntry("ATTENTION_QKV", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, False, ("QKV_ROLE", "ATTENTION_HEAD"), "One QKV sheet per role and head"),
    GeometryEntry("ATTENTION_OUTPUT.ATTENTION_D_MODEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_D_MODEL,), ELEMENT_TYPE_CURVE, True, ("ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL), "Attention-output model-axis curves"),
    GeometryEntry("ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_HEAD_CHANNEL,), ELEMENT_TYPE_CURVE, True, ("ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL), "Attention-output within-head channel curves"),
    GeometryEntry("ATTENTION_OUTPUT", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, False, ("ATTENTION_HEAD",), "One attention-output sheet per head"),
)
GEOMETRY_REGISTRY: Mapping[str, GeometryEntry] = {entry.selector: entry for entry in _GEOMETRY_ENTRIES}


# vvv THOG first-class compressor capabilities are separated from current materializer executability
def _build_compressor_registry() -> Dict[str, CompressorCapability]:
    capabilities: Dict[str, CompressorCapability] = {}
    first_class_types = (ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, ELEMENT_TYPE_SHEET_SET)
    for family in BASIS_FAMILIES:
        canonical = normalize_registered_basis_family(family)
        capabilities[canonical] = CompressorCapability(
            canonical,
            basis_version_for_family(canonical),
            first_class_types,
            True,
            notes="First-class basis-backed compressor; current runtime materializers remain a separate capability boundary.",
        )
    # vvv THOG recurrence generators are independent plugins; BQRG v1 advertises only the DEPTH CURVE target
    for definition in RECURRENCE_GENERATOR_REGISTRY.definitions():
        if definition.family in capabilities:
            raise ValueError(f"compressor registry collision for recurrence generator {definition.family!r}")
        capabilities[definition.family] = CompressorCapability(
            definition.family,
            definition.version,
            (ELEMENT_TYPE_CURVE,),
            True,
            supported_selectors=definition.supported_targets,
            notes=f"Recurrence generator: {definition.description}",
        )
    # ^^^ THOG
    capabilities[COMPRESSOR_JPEG_LIKE] = CompressorCapability(
        COMPRESSOR_JPEG_LIKE,
        JPEG_LIKE_VERSION,
        first_class_types,
        True,
        legacy_only=False,
        supports_group_size=True,
        supported_selectors=(),
        notes="First-class grouped local compressor; only the DEPTH + MLP_UP.MLP_HIDDEN legacy adapter is executable today.",
    )
    return capabilities
# ^^^ THOG


COMPRESSOR_REGISTRY: Mapping[str, CompressorCapability] = _build_compressor_registry()


def permitted_geometry_selectors() -> Tuple[str, ...]:
    return tuple(GEOMETRY_REGISTRY)


def normalize_selector(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("geometry selector must be a non-empty string")
    normalized = ".".join(part.strip().upper() for part in value.strip().split("."))
    if normalized not in GEOMETRY_REGISTRY:
        raise ValueError(f"unregistered geometry selector {value!r}; permitted selectors are: {', '.join(GEOMETRY_REGISTRY)}")
    return normalized


def _normalize_option_target(value: str) -> str:
    normalized = ".".join(part.strip().upper() for part in value.strip().split("."))
    if not normalized:
        raise ValueError("geometry option target must not be empty")
    return normalized


def parse_option_assignment(assignment: str) -> ParsedOption:
    if not isinstance(assignment, str) or "=" not in assignment:
        raise ValueError(f"geometry option must use TARGET.PROPERTY=VALUE syntax; got {assignment!r}")
    left, value = assignment.split("=", 1)
    if "." not in left or not value.strip():
        raise ValueError(f"geometry option must use TARGET.PROPERTY=VALUE syntax; got {assignment!r}")
    target, property_name = left.rsplit(".", 1)
    property_normalized = property_name.strip().lower()
    if property_normalized not in ("compressor", "compressor_version", "version", "order", "group_size") and not property_normalized.startswith("generator_"):
        raise ValueError(f"unsupported geometry option property {property_name!r} in {assignment!r}")
    if property_normalized == "version":
        property_normalized = "compressor_version"
    return ParsedOption(_normalize_option_target(target), property_normalized, value.strip(), assignment)


def _positive_integer(label: str, value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer; got {value!r}") from error
    if parsed < 1:
        raise ValueError(f"{label} must be a positive integer; got {value!r}")
    return parsed


def normalize_compressor_family(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == COMPRESSOR_JPEG_LIKE:
        return normalized
    try:
        canonical = normalize_registered_basis_family(normalized)
    except ValueError:
        try:
            canonical = normalize_recurrence_generator_family(normalized)
        except ValueError as error:
            raise ValueError(f"unregistered compressor {value!r}; registered compressors are: {', '.join(COMPRESSOR_REGISTRY)}") from error
    if canonical not in COMPRESSOR_REGISTRY:
        raise ValueError(f"unregistered compressor {value!r}; registered compressors are: {', '.join(COMPRESSOR_REGISTRY)}")
    return canonical


def _split_compressor_value(value: str) -> Tuple[str, Optional[str]]:
    if "@" not in value:
        return normalize_compressor_family(value), None
    family, version = value.split("@", 1)
    if not version.strip():
        raise ValueError(f"compressor version is empty in {value!r}")
    return normalize_compressor_family(family), version.strip()


def _validate_selection_overlaps(entries: Sequence[GeometryEntry]) -> None:
    seen: Dict[str, GeometryEntry] = {}
    by_element: Dict[str, list[GeometryEntry]] = {}
    for entry in entries:
        if entry.selector in seen:
            raise ValueError(f"duplicate geometry selector: {entry.selector}")
        seen[entry.selector] = entry
        by_element.setdefault(entry.element, []).append(entry)
    for element, element_entries in by_element.items():
        complete = [entry for entry in element_entries if entry.is_complete_element]
        if complete and len(element_entries) > 1:
            others = ", ".join(entry.selector for entry in element_entries if not entry.is_complete_element)
            raise ValueError(f"overlapping geometry selectors for {element}: {complete[0].selector} already selects the complete registered geometry and cannot be combined with {others}")


def _validate_option_targets(parsed_options: Sequence[ParsedOption], entries: Sequence[GeometryEntry], *, depth_enabled: bool) -> None:
    selected_elements = {entry.element for entry in entries}
    selected_selectors = {entry.selector for entry in entries}
    selected_axis_targets = {f"{entry.element}.{axis}" for entry in entries for axis in entry.compressed_axes}
    for option in parsed_options:
        if option.property in ("compressor", "compressor_version") or option.property.startswith("generator_"):
            if option.target == AXIS_DEPTH:
                if not depth_enabled:
                    raise ValueError(f"{option.source!r} targets DEPTH, but --select-depth was not supplied")
            elif option.target not in selected_elements and option.target not in selected_selectors:
                raise ValueError(
                    f"{option.source!r} must target a selected element or selector; selected elements are "
                    f"{tuple(sorted(selected_elements))}; selected selectors are {tuple(sorted(selected_selectors))}"
                )
        elif option.property == "order":
            if option.target == AXIS_DEPTH:
                if not depth_enabled:
                    raise ValueError(f"{option.source!r} targets DEPTH, but --select-depth was not supplied")
            elif option.target not in selected_axis_targets:
                raise ValueError(f"{option.source!r} must target a compressed axis of a selected geometry; active axis targets are {tuple(sorted(selected_axis_targets))}")
        elif option.property == "group_size" and option.target not in selected_axis_targets:
            raise ValueError(f"{option.source!r} must target a compressed axis of a selected geometry; active axis targets are {tuple(sorted(selected_axis_targets))}")


def _option_map(parsed_options: Sequence[ParsedOption]) -> Dict[Tuple[str, str], str]:
    values: Dict[Tuple[str, str], str] = {}
    for option in parsed_options:
        key = (option.target, option.property)
        if key in values:
            raise ValueError(f"geometry option assigned more than once: {option.target}.{option.property}")
        values[key] = option.value
    return values


def _scoped_value(values: Mapping[Tuple[str, str], str], *, selector: str, element: str, property_name: str) -> Optional[str]:
    selector_value = values.get((selector, property_name))
    element_value = values.get((element, property_name))
    if selector_value is not None and element_value is not None and selector_value != element_value:
        raise ValueError(
            f"conflicting geometry options for {property_name}: {element}.{property_name}={element_value!r} "
            f"and {selector}.{property_name}={selector_value!r}"
        )
    return selector_value if selector_value is not None else element_value


def _compressor_for_target(target: str, values: Mapping[Tuple[str, str], str], *, default_family: str) -> Tuple[str, str]:
    family, inline_version = _split_compressor_value(values.get((target, "compressor"), default_family))
    capability = COMPRESSOR_REGISTRY[family]
    explicit_version = values.get((target, "compressor_version"))
    if inline_version is not None and explicit_version is not None and inline_version != explicit_version:
        raise ValueError(f"conflicting compressor versions for {target}: {inline_version!r} and {explicit_version!r}")
    return family, explicit_version or inline_version or capability.default_version


def _compressor_for_entry(entry: GeometryEntry, values: Mapping[Tuple[str, str], str], *, default_family: str) -> Tuple[str, str]:
    raw_family = _scoped_value(values, selector=entry.selector, element=entry.element, property_name="compressor") or default_family
    family, inline_version = _split_compressor_value(raw_family)
    capability = COMPRESSOR_REGISTRY[family]
    explicit_version = _scoped_value(values, selector=entry.selector, element=entry.element, property_name="compressor_version")
    if inline_version is not None and explicit_version is not None and inline_version != explicit_version:
        raise ValueError(f"conflicting compressor versions for {entry.selector}: {inline_version!r} and {explicit_version!r}")
    return family, explicit_version or inline_version or capability.default_version


def _generator_options_for_target(target: str, values: Mapping[Tuple[str, str], str]) -> Dict[str, str]:
    return {
        property_name: value
        for (option_target, property_name), value in values.items()
        if option_target == target and property_name.startswith("generator_")
    }


def _generator_options_for_entry(entry: GeometryEntry, values: Mapping[Tuple[str, str], str]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    properties = {property_name for (_, property_name) in values if property_name.startswith("generator_")}
    for property_name in properties:
        scoped = _scoped_value(values, selector=entry.selector, element=entry.element, property_name=property_name)
        if scoped is not None:
            options[property_name] = scoped
    return options


def _validate_generator_options(family: str, version: str, options: Dict[str, str]) -> None:
    if is_recurrence_generator_family(family):
        definition = get_recurrence_generator_definition(family)
        if version != definition.version:
            raise ValueError(f"recurrence generator version mismatch for {family}: expected {definition.version!r}, got {version!r}")
        definition.validate_options(options)
        return
    if options:
        raise ValueError(f"compressor {family}@{version} is not a recurrence generator and does not accept generator_* options")


def _order_for_axis(element: str, axis: str, values: Mapping[Tuple[str, str], str], legacy_orders: Mapping[str, int]) -> int:
    target = f"{element}.{axis}"
    if (target, "order") in values:
        return _positive_integer(f"{target}.order", values[(target, "order")])
    value = int(legacy_orders[ORDER_OPTION_FIELDS[(element, axis)]])
    if value < 1:
        raise ValueError(f"{ORDER_OPTION_FIELDS[(element, axis)]} must be positive; got {value!r}")
    return value


def _compatibility_message(selector: str, element_type: str, family: str, version: str) -> str:
    capability = COMPRESSOR_REGISTRY[family]
    if element_type not in capability.element_types:
        compatible = [name for name, item in COMPRESSOR_REGISTRY.items() if item.implemented and element_type in item.element_types]
        suffix = f" compatible {element_type} compressors: {', '.join(compatible)}" if compatible else f" no {element_type} compressor is currently registered"
        return f"compressor {family}@{version} is not registered for {element_type} geometry;{suffix}"
    if capability.supported_selectors and selector not in capability.supported_selectors:
        return f"compressor {family}@{version} is not valid for {selector}; supported selectors: {', '.join(capability.supported_selectors)}"
    return ""


def _legacy_semantic_warning(preset: str, *, stream=sys.stderr) -> str:
    text = f"WARNING: legacy geometry '{preset}' is not semantically sensible: it combines unrelated weight coordinates under one compact field. It remains runnable only for legacy comparison."
    if stream.isatty() and os.environ.get("NO_COLOR") is None:
        return f"\033[1;38;5;208m{text}\033[0m"
    return text


def _materializer_adapter(*, depth_enabled: bool, depth_compressor: Optional[str], depth_version: Optional[str], selections: Sequence[ResolvedGeometrySelection], incompatibilities: Sequence[str]) -> MaterializerAdapter:
    if incompatibilities:
        return MaterializerAdapter(False, None, None, None, None, None, None, "; ".join(incompatibilities))
    if depth_enabled and not selections:
        materialization_version = depth_version if depth_compressor is not None and is_recurrence_generator_family(depth_compressor) else "depth_v1"
        return MaterializerAdapter(True, "depth", depth_compressor, depth_version, None, None, materialization_version, "Implemented by the existing DEPTH trajectory.")
    if depth_enabled and len(selections) == 1 and selections[0].selector == "MLP_UP.MLP_HIDDEN" and selections[0].compressor == COMPRESSOR_JPEG_LIKE:
        group_size = int(selections[0].axis_options.get(AXIS_MLP_HIDDEN, {}).get("group_size", 256))
        return MaterializerAdapter(True, "jpeg_like_v1", depth_compressor, depth_version, "dct", group_size, "jpeg_like_v1", "Implemented through the legacy JPEG_LIKE_V1 adapter. The compressor is first-class; this materializer remains the old narrow runtime path.", True)
    if not depth_enabled and not selections:
        return MaterializerAdapter(False, None, None, None, None, None, None, "No geometry was selected; supply --select-depth and/or at least one --select-element.")
    unimplemented = ", ".join(selection.selector for selection in selections)
    return MaterializerAdapter(False, None, None, None, None, None, None, f"registered geometry is valid, but no current materializer exists for: {unimplemented}")


def _apply_jpeg_like_axis_options(*, entry: GeometryEntry, orders: Mapping[str, int], axis_options: Dict[str, Dict[str, Any]], default_mlp_hidden_group_size: int) -> None:
    for axis in entry.compressed_axes:
        if axis == AXIS_MLP_HIDDEN:
            axis_options.setdefault(axis, {}).setdefault("group_size", int(default_mlp_hidden_group_size))
        group_size = axis_options.get(axis, {}).get("group_size")
        if group_size is not None and orders[axis] > int(group_size):
            raise ValueError(f"{entry.element}.{axis}.order must not exceed group_size: order={orders[axis]}, group_size={group_size}")


def resolve_geometry_plan(*, select_depth: bool, selected_elements: Sequence[str], option_assignments: Sequence[str], legacy_orders: Mapping[str, int], default_depth_compressor: str = "chebyshev", default_non_depth_compressor: str = "dct", default_mlp_hidden_group_size: int = 256) -> ResolvedGeometryPlan:
    entries = tuple(GEOMETRY_REGISTRY[normalize_selector(value)] for value in selected_elements)
    _validate_selection_overlaps(entries)
    parsed_options = tuple(parse_option_assignment(value) for value in option_assignments)
    _validate_option_targets(parsed_options, entries, depth_enabled=bool(select_depth))
    values = _option_map(parsed_options)
    if not select_depth and not entries:
        raise ValueError("systematic geometry UI requires --select-depth and/or at least one --select-element")

    depth_compressor = depth_version = None
    depth_order: Optional[int] = None
    incompatibilities: list[str] = []
    if select_depth:
        depth_compressor, depth_version = _compressor_for_target(AXIS_DEPTH, values, default_family=default_depth_compressor)
        depth_order = _positive_integer("DEPTH.order", values[(AXIS_DEPTH, "order")]) if (AXIS_DEPTH, "order") in values else int(legacy_orders["o_depth"])
        if depth_order < 1:
            raise ValueError(f"o_depth must be positive; got {depth_order!r}")
        depth_capability = COMPRESSOR_REGISTRY[depth_compressor]
        if depth_capability.supported_selectors and AXIS_DEPTH not in depth_capability.supported_selectors:
            incompatibilities.append(
                f"DEPTH: compressor {depth_compressor}@{depth_version} is not valid for DEPTH; supported selectors: {', '.join(depth_capability.supported_selectors)}"
            )
        depth_generator_options = _generator_options_for_target(AXIS_DEPTH, values)
        _validate_generator_options(depth_compressor, str(depth_version), depth_generator_options)
        if is_recurrence_generator_family(depth_compressor):
            validate_recurrence_generator_width(depth_compressor, depth_order)

    resolved: list[ResolvedGeometrySelection] = []
    compressor_pairs: set[Tuple[str, str]] = set()
    for entry in entries:
        implied_type = entry.resolved_type(bool(select_depth))
        family, version = _compressor_for_entry(entry, values, default_family=default_non_depth_compressor)
        compressor_pairs.add((family, version))
        incompatibility = _compatibility_message(entry.selector, implied_type, family, version)
        if incompatibility:
            incompatibilities.append(f"{entry.selector}: {incompatibility}")
        _validate_generator_options(family, version, _generator_options_for_entry(entry, values))
        capability = COMPRESSOR_REGISTRY[family]
        orders = {axis: _order_for_axis(entry.element, axis, values, legacy_orders) for axis in entry.compressed_axes}
        axis_options: Dict[str, Dict[str, Any]] = {}
        for axis in entry.compressed_axes:
            target = f"{entry.element}.{axis}"
            if (target, "group_size") in values:
                if not capability.supports_group_size:
                    raise ValueError(f"compressor {family}@{version} does not accept group_size, but {target}.group_size was supplied")
                axis_options.setdefault(axis, {})["group_size"] = _positive_integer(f"{target}.group_size", values[(target, "group_size")])
        if family == COMPRESSOR_JPEG_LIKE:
            _apply_jpeg_like_axis_options(entry=entry, orders=orders, axis_options=axis_options, default_mlp_hidden_group_size=default_mlp_hidden_group_size)
        resolved.append(ResolvedGeometrySelection(entry.selector, entry.element, implied_type, entry.compressed_axes, entry.independent_indices, family, version, orders, axis_options))

    if len(compressor_pairs) > 1:
        assignments = ", ".join(f"{family}@{version}" for family, version in sorted(compressor_pairs))
        raise ValueError(f"Phase 1 requires every selected non-DEPTH element to use one shared compressor family/version; resolved assignments were {assignments}")
    shared_family, shared_version = (next(iter(compressor_pairs)) if compressor_pairs else (None, None))
    adapter = _materializer_adapter(depth_enabled=bool(select_depth), depth_compressor=depth_compressor, depth_version=depth_version, selections=resolved, incompatibilities=incompatibilities)
    return ResolvedGeometryPlan(GEOMETRY_PLAN_SCHEMA_VERSION, GEOMETRY_REGISTRY_VERSION, bool(select_depth), depth_compressor, depth_version, depth_order, tuple(resolved), parsed_options, shared_family, shared_version, adapter)


def validate_resolved_geometry_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("resolved_geometry_plan must be a mapping")
    values = dict(plan)
    if values.get("schema_version") != GEOMETRY_PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported resolved geometry plan schema: {values.get('schema_version')!r}")
    if values.get("registry_version") != GEOMETRY_REGISTRY_VERSION:
        raise ValueError(f"unsupported geometry registry version: {values.get('registry_version')!r}")
    selections = values.get("selections")
    if not isinstance(selections, list):
        raise ValueError("resolved geometry plan selections must be a list")
    for selection in selections:
        selector = selection.get("selector") if isinstance(selection, Mapping) else None
        if selector not in GEOMETRY_REGISTRY:
            raise ValueError(f"resolved geometry plan contains unregistered selector {selector!r}")
        if selection.get("implied_type") not in ELEMENT_TYPES:
            raise ValueError(f"resolved geometry plan contains invalid implied type {selection.get('implied_type')!r}")
    return values


def _format_field(label_width: int, label: str, value: Any) -> str:
    return f"  {label:<{label_width}}{value}"


def _geometry_plan_label_width(plan: ResolvedGeometryPlan) -> int:
    labels = ["implied element type:", "compressed axis:", "compressed axes:", "uncompressed axes:", "compressor:", "order:"]
    for selection in plan.selections:
        labels.extend(f"order {axis}:" for axis in selection.orders)
        labels.extend(f"{axis}.{name}:" for axis, options in selection.axis_options.items() for name in options)
    labels.extend(f"{option.property}:" for option in plan.parsed_options if option.property.startswith("generator_"))
    return max(len(label) for label in labels) + 4


def format_geometry_plan(plan: ResolvedGeometryPlan, *, detailed: bool = False) -> str:
    label_width = _geometry_plan_label_width(plan)
    lines = ["selected geometry", "-----------------"]
    if plan.depth_enabled:
        lines.extend([
            "DEPTH",
            _format_field(label_width, "implied element type:", ELEMENT_TYPE_CURVE),
            _format_field(label_width, "compressed axis:", AXIS_DEPTH),
            _format_field(label_width, "compressor:", f"{plan.depth_compressor}@{plan.depth_compressor_version}"),
            _format_field(label_width, "order:", plan.depth_order),
        ])
        for option in plan.parsed_options:
            if option.target == AXIS_DEPTH and option.property.startswith("generator_"):
                lines.append(_format_field(label_width, f"{option.property}:", option.value))
    for selection in plan.selections:
        axis_label = "compressed axis:" if len(selection.compressed_axes) == 1 else "compressed axes:"
        lines.extend([
            "",
            selection.selector,
            _format_field(label_width, "implied element type:", selection.implied_type),
            _format_field(label_width, axis_label, " × ".join(selection.compressed_axes)),
        ])
        if selection.independent_indices:
            lines.append(_format_field(label_width, "uncompressed axes:", " × ".join(selection.independent_indices)))
        lines.append(_format_field(label_width, "compressor:", f"{selection.compressor}@{selection.compressor_version}"))
        for axis, order in selection.orders.items():
            lines.append(_format_field(label_width, f"order {axis}:", order))
        for axis, options in selection.axis_options.items():
            for name, value in options.items():
                lines.append(_format_field(label_width, f"{axis}.{name}:", value))
        for option in plan.parsed_options:
            if option.target in (selection.selector, selection.element) and option.property.startswith("generator_"):
                lines.append(_format_field(label_width, f"{option.property}:", option.value))
    if plan.materializer.implemented and plan.materializer.legacy:
        status = "implemented through legacy adapter"
    else:
        status = "implemented" if plan.materializer.implemented else "not currently implemented"
    lines.extend(["", f"implementation status: {status}", f"  {plan.materializer.message}"])
    if plan.materializer.legacy:
        lines.append(f"  {_legacy_semantic_warning(str(plan.materializer.legacy_geometry_preset), stream=sys.stdout)}")
    if detailed:
        lines.extend([f"registry version:      {plan.registry_version}", f"plan schema:           {plan.schema_version}"])
        if plan.materializer.legacy_geometry_preset:
            lines.extend([f"legacy adapter preset: {plan.materializer.legacy_geometry_preset}", f"materialization:       {plan.materializer.materialization_version}"])
    return "\n".join(lines)


def format_geometry_registry() -> str:
    lines = [
        f"geometry registry ({GEOMETRY_REGISTRY_VERSION})",
        "=======================================",
        "",
        "selectors",
        "---------",
        "depth-compatible: selector may accompany DEPTH without implying a 3-D BLOCK/BLOCK_SET field",
        "materializer:     generic systematic materializer exists for this exact selector",
        "legacy-only:      compatibility-oriented entry rather than the systematic geometry path",
        "",
        f"{'selector':40} {'type':10} {'depth-compatible':16} {'materializer':12} {'legacy-only':11} {'compressed axes':36} uncompressed axes",
    ]
    # vvv THOG expose compressed and uncompressed axes independently and make status columns self-explanatory
    for entry in GEOMETRY_REGISTRY.values():
        depth_compatible = "yes" if entry.permits_depth_companion else "no"
        materializer = "yes" if entry.implemented else "no"
        legacy_only = "yes" if entry.legacy_only else "no"
        compressed_axes = " × ".join(entry.compressed_axes)
        uncompressed_axes = " × ".join(entry.independent_indices) if entry.independent_indices else "none"
        lines.append(f"{entry.selector:40} {entry.implied_type:10} {depth_compatible:16} {materializer:12} {legacy_only:11} {compressed_axes:36} {uncompressed_axes}")
    # ^^^ THOG
    lines.extend(["", "compressor registry", "-------------------", f"{'compressor':18} {'types':30} {'registered':12} {'legacy':7} {'group':7} notes"])
    for name, capability in COMPRESSOR_REGISTRY.items():
        lines.append(f"{name:18} {','.join(capability.element_types):30} {str(capability.implemented):12} {str(capability.legacy_only):7} {str(capability.supports_group_size):7} {capability.notes}")
    # vvv THOG dedicated RG subsection makes plugin identity, persistent widths, target guards, and option namespaces discoverable
    lines.extend(["", "recurrence generator registry", "-----------------------------", f"{'generator':18} {'version':16} {'persistent widths':18} {'targets':18} options"])
    for definition in RECURRENCE_GENERATOR_REGISTRY.definitions():
        widths = ",".join(str(width) for width in definition.persistent_widths)
        targets = ",".join(definition.supported_targets)
        options = ",".join(definition.option_names) if definition.option_names else "none"
        lines.append(f"{definition.family:18} {definition.version:16} {widths:18} {targets:18} {options}")
        lines.append(f"  {definition.description}")
    # ^^^ THOG
    lines.extend(["", "Compressor capability is first-class; generic SHEET/SHEET_SET materializers are still Phase 2."])
    return "\n".join(lines)


def _legacy_geometry_preset_from_argv(arguments: Sequence[str]) -> Optional[str]:
    for index, argument in enumerate(arguments):
        if argument in ("--geometry-preset", "-p") and index + 1 < len(arguments):
            return arguments[index + 1].strip().lower()
        if argument.startswith("--geometry-preset="):
            return argument.split("=", 1)[1].strip().lower()
    return None


def _warn_for_legacy_semantics() -> None:
    if any(argument in ("--select-depth", "--select-element", "--explain-geometry") or argument.startswith("--select-element=") for argument in sys.argv[1:]):
        return
    preset = _legacy_geometry_preset_from_argv(sys.argv[1:])
    if preset in LEGACY_SEMANTICALLY_UNSOUND_PRESETS:
        print(_legacy_semantic_warning(preset, stream=sys.stderr), file=sys.stderr, flush=True)


def _install_clean_cli_excepthook() -> None:
    if os.path.basename(sys.argv[0]) not in {"run_thog2_owt.py", "run_thog2_owt_residual.py"}:
        return
    previous = sys.excepthook

    def clean_excepthook(error_type, error, traceback):
        if issubclass(error_type, (ValueError, FileNotFoundError, FileExistsError)):
            print(f"error: {error}", file=sys.stderr, flush=True)
            return
        previous(error_type, error, traceback)

    sys.excepthook = clean_excepthook


def _print_geometry_registry_and_exit_if_requested() -> None:
    if "--print-geometry-registry" in sys.argv[1:]:
        print(format_geometry_registry())
        raise SystemExit(0)


def _consume_initial_eval_flags() -> None:
    retained = [sys.argv[0]]
    for argument in sys.argv[1:]:
        if argument == "--no-initial-eval":
            os.environ["THOG2_INITIAL_EVAL"] = "0"
        elif argument == "--initial-eval":
            os.environ["THOG2_INITIAL_EVAL"] = "1"
        else:
            retained.append(argument)
    sys.argv[:] = retained


_print_geometry_registry_and_exit_if_requested()
_consume_initial_eval_flags()
_warn_for_legacy_semantics()
_install_clean_cli_excepthook()

__all__ = [
    "AXIS_ATTENTION_D_MODEL", "AXIS_ATTENTION_HEAD_CHANNEL", "AXIS_DEPTH", "AXIS_MLP_D_MODEL", "AXIS_MLP_HIDDEN",
    "COMPRESSOR_JPEG_LIKE", "COMPRESSOR_REGISTRY", "ELEMENT_TYPE_CURVE", "ELEMENT_TYPE_SHEET", "ELEMENT_TYPE_SHEET_SET",
    "GEOMETRY_PLAN_SCHEMA_VERSION", "GEOMETRY_REGISTRY", "GEOMETRY_REGISTRY_VERSION", "GeometryEntry", "MaterializerAdapter",
    "ParsedOption", "ResolvedGeometryPlan", "ResolvedGeometrySelection", "format_geometry_plan", "format_geometry_registry", "normalize_compressor_family", "normalize_selector",
    "parse_option_assignment", "permitted_geometry_selectors", "resolve_geometry_plan", "validate_resolved_geometry_plan",
]
# ^^^ THOG