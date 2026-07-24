# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .bases import BASIS_FAMILIES, basis_version_for_family, normalize_registered_basis_family

GEOMETRY_REGISTRY_VERSION = "geometry_registry_v1"
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
    implied_type_with_depth: Optional[str]
    independent_indices: Tuple[str, ...] = ()
    description: str = ""

    @property
    def is_complete_element(self) -> bool:
        return self.selector == self.element

    def resolved_type(self, depth_enabled: bool) -> str:
        if not depth_enabled:
            return self.implied_type
        if self.implied_type_with_depth is None:
            raise ValueError(
                f"{self.selector} is a permitted {self.implied_type} geometry, but combining it "
                "with DEPTH would imply a three-dimensional BLOCK/BLOCK_SET field, which is not "
                "part of the systematic geometry design"
            )
        return self.implied_type_with_depth


@dataclass(frozen=True)
class CompressorCapability:
    family: str
    default_version: str
    semantic_dimensions: Tuple[int, ...]
    implemented_dimensions: Tuple[int, ...]
    supports_group_size: bool = False
    notes: str = ""

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
    GeometryEntry(
        "MLP_UP.MLP_HIDDEN", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET,
        (AXIS_MLP_D_MODEL,), "MLP expansion hidden-axis curve",
    ),
    GeometryEntry(
        "MLP_UP.MLP_D_MODEL", ELEMENT_MLP_UP, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET,
        (AXIS_MLP_HIDDEN,), "MLP expansion model-axis curve",
    ),
    GeometryEntry(
        "MLP_UP", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, None,
        (), "Complete MLP expansion sheet",
    ),
    GeometryEntry(
        "MLP_DOWN.MLP_HIDDEN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET,
        (AXIS_MLP_D_MODEL,), "MLP contraction hidden-axis curve",
    ),
    GeometryEntry(
        "MLP_DOWN.MLP_D_MODEL", ELEMENT_MLP_DOWN, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET,
        (AXIS_MLP_HIDDEN,), "MLP contraction model-axis curve",
    ),
    GeometryEntry(
        "MLP_DOWN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, None,
        (), "Complete MLP contraction sheet",
    ),
    GeometryEntry(
        "ATTENTION_QKV.ATTENTION_D_MODEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_D_MODEL,),
        ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET,
        ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL),
        "QKV input-model-axis curves",
    ),
    GeometryEntry(
        "ATTENTION_QKV.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_HEAD_CHANNEL,),
        ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET,
        ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL),
        "QKV within-head channel curves",
    ),
    GeometryEntry(
        "ATTENTION_QKV", ELEMENT_ATTENTION_QKV,
        (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, None,
        ("QKV_ROLE", "ATTENTION_HEAD"), "One QKV sheet per role and head",
    ),
    GeometryEntry(
        "ATTENTION_OUTPUT.ATTENTION_D_MODEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_D_MODEL,),
        ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET,
        ("ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL),
        "Attention-output model-axis curves",
    ),
    GeometryEntry(
        "ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_HEAD_CHANNEL,),
        ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET,
        ("ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL),
        "Attention-output within-head channel curves",
    ),
    GeometryEntry(
        "ATTENTION_OUTPUT", ELEMENT_ATTENTION_OUTPUT,
        (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, None,
        ("ATTENTION_HEAD",), "One attention-output sheet per head",
    ),
)

GEOMETRY_REGISTRY: Mapping[str, GeometryEntry] = {entry.selector: entry for entry in _GEOMETRY_ENTRIES}


def _build_compressor_registry() -> Dict[str, CompressorCapability]:
    capabilities: Dict[str, CompressorCapability] = {}
    for family in BASIS_FAMILIES:
        canonical = normalize_registered_basis_family(family)
        capabilities[canonical] = CompressorCapability(
            family=canonical,
            default_version=basis_version_for_family(canonical),
            semantic_dimensions=(1, 2),
            implemented_dimensions=(1,),
            supports_group_size=False,
            notes="Registered separable basis compressor; systematic non-DEPTH materialisers arrive in Phase 2.",
        )
    capabilities[COMPRESSOR_JPEG_LIKE] = CompressorCapability(
        family=COMPRESSOR_JPEG_LIKE,
        default_version=JPEG_LIKE_VERSION,
        semantic_dimensions=(1, 2),
        implemented_dimensions=(1,),
        supports_group_size=True,
        notes="Phase 1 adapter supports the existing DEPTH + MLP_UP.MLP_HIDDEN JPEG-like path; 2-D is future-valid.",
    )
    return capabilities


COMPRESSOR_REGISTRY: Mapping[str, CompressorCapability] = _build_compressor_registry()


def permitted_geometry_selectors() -> Tuple[str, ...]:
    return tuple(GEOMETRY_REGISTRY)


def normalize_selector(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("geometry selector must be a non-empty string")
    normalized = ".".join(part.strip().upper() for part in value.strip().split("."))
    if normalized not in GEOMETRY_REGISTRY:
        allowed = ", ".join(GEOMETRY_REGISTRY)
        raise ValueError(f"unregistered geometry selector {value!r}; permitted selectors are: {allowed}")
    return normalized


def _normalize_option_target(value: str) -> str:
    normalized = ".".join(part.strip().upper() for part in value.strip().split("."))
    if not normalized:
        raise ValueError("geometry option target must not be empty")
    return normalized


def parse_option_assignment(assignment: str) -> ParsedOption:
    if not isinstance(assignment, str) or "=" not in assignment:
        raise ValueError(
            f"geometry option must use TARGET.PROPERTY=VALUE syntax; got {assignment!r}"
        )
    left, value = assignment.split("=", 1)
    if "." not in left or not value.strip():
        raise ValueError(
            f"geometry option must use TARGET.PROPERTY=VALUE syntax; got {assignment!r}"
        )
    target, property_name = left.rsplit(".", 1)
    property_normalized = property_name.strip().lower()
    if property_normalized not in ("compressor", "compressor_version", "version", "order", "group_size"):
        raise ValueError(f"unsupported geometry option property {property_name!r} in {assignment!r}")
    if property_normalized == "version":
        property_normalized = "compressor_version"
    return ParsedOption(
        target=_normalize_option_target(target),
        property=property_normalized,
        value=value.strip(),
        source=assignment,
    )


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
        return normalize_registered_basis_family(normalized)
    except ValueError as error:
        allowed = ", ".join(COMPRESSOR_REGISTRY)
        raise ValueError(f"unregistered compressor {value!r}; registered compressors are: {allowed}") from error


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
            raise ValueError(
                f"overlapping geometry selectors for {element}: {complete[0].selector} already selects the complete "
                f"registered geometry and cannot be combined with {others}"
            )


def _validate_option_targets(
    parsed_options: Sequence[ParsedOption],
    entries: Sequence[GeometryEntry],
    *,
    depth_enabled: bool,
) -> None:
    selected_elements = {entry.element for entry in entries}
    selected_axis_targets = {
        f"{entry.element}.{axis}"
        for entry in entries
        for axis in entry.compressed_axes
    }
    for option in parsed_options:
        if option.property in ("compressor", "compressor_version"):
            if option.target == AXIS_DEPTH:
                if not depth_enabled:
                    raise ValueError(f"{option.source!r} targets DEPTH, but --select-depth was not supplied")
            elif option.target not in selected_elements:
                raise ValueError(
                    f"{option.source!r} must target a selected element; selected elements are "
                    f"{tuple(sorted(selected_elements))}"
                )
        elif option.property == "order":
            if option.target == AXIS_DEPTH:
                if not depth_enabled:
                    raise ValueError(f"{option.source!r} targets DEPTH, but --select-depth was not supplied")
            elif option.target not in selected_axis_targets:
                raise ValueError(
                    f"{option.source!r} must target a compressed axis of a selected geometry; active axis targets are "
                    f"{tuple(sorted(selected_axis_targets))}"
                )
        elif option.property == "group_size":
            if option.target not in selected_axis_targets:
                raise ValueError(
                    f"{option.source!r} must target a compressed axis of a selected geometry; active axis targets are "
                    f"{tuple(sorted(selected_axis_targets))}"
                )


def _option_map(parsed_options: Sequence[ParsedOption]) -> Dict[Tuple[str, str], str]:
    values: Dict[Tuple[str, str], str] = {}
    for option in parsed_options:
        key = (option.target, option.property)
        if key in values:
            raise ValueError(f"geometry option assigned more than once: {option.target}.{option.property}")
        values[key] = option.value
    return values


def _compressor_for_target(
    target: str,
    values: Mapping[Tuple[str, str], str],
    *,
    default_family: str,
) -> Tuple[str, str]:
    raw_family = values.get((target, "compressor"), default_family)
    family, inline_version = _split_compressor_value(raw_family)
    capability = COMPRESSOR_REGISTRY[family]
    explicit_version = values.get((target, "compressor_version"))
    if inline_version is not None and explicit_version is not None and inline_version != explicit_version:
        raise ValueError(
            f"conflicting compressor versions for {target}: {inline_version!r} and {explicit_version!r}"
        )
    version = explicit_version or inline_version or capability.default_version
    return family, version


def _order_for_axis(
    element: str,
    axis: str,
    values: Mapping[Tuple[str, str], str],
    legacy_orders: Mapping[str, int],
) -> int:
    target = f"{element}.{axis}"
    explicit = values.get((target, "order"))
    if explicit is not None:
        return _positive_integer(f"{target}.order", explicit)
    try:
        field_name = ORDER_OPTION_FIELDS[(element, axis)]
    except KeyError as error:
        raise ValueError(f"no order source is registered for {target}") from error
    try:
        value = int(legacy_orders[field_name])
    except KeyError as error:
        raise ValueError(f"legacy order mapping is missing {field_name!r}") from error
    if value < 1:
        raise ValueError(f"{field_name} must be positive; got {value!r}")
    return value


def _materializer_adapter(
    *,
    depth_enabled: bool,
    depth_compressor: Optional[str],
    depth_version: Optional[str],
    selections: Sequence[ResolvedGeometrySelection],
) -> MaterializerAdapter:
    if depth_enabled and not selections:
        return MaterializerAdapter(
            True,
            "depth",
            depth_compressor,
            depth_version,
            None,
            None,
            "depth_v1",
            "Implemented by the existing DEPTH trajectory.",
        )
    if (
        depth_enabled
        and len(selections) == 1
        and selections[0].selector == "MLP_UP.MLP_HIDDEN"
        and selections[0].compressor == COMPRESSOR_JPEG_LIKE
    ):
        group_size = selections[0].axis_options.get(AXIS_MLP_HIDDEN, {}).get("group_size", 256)
        return MaterializerAdapter(
            True,
            "jpeg_like_v1",
            depth_compressor,
            depth_version,
            "dct",
            int(group_size),
            "jpeg_like_v1",
            "Implemented by the existing JPEG_LIKE_V1 trajectory adapter.",
        )
    if not depth_enabled and not selections:
        return MaterializerAdapter(
            False, None, None, None, None, None, None,
            "No geometry was selected; supply --select-depth and/or at least one --select-element.",
        )
    return MaterializerAdapter(
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        "The selection is semantically valid and registered, but its systematic materialiser is scheduled for Phase 2.",
    )


def resolve_geometry_plan(
    *,
    select_depth: bool,
    selected_elements: Sequence[str],
    option_assignments: Sequence[str],
    legacy_orders: Mapping[str, int],
    default_depth_compressor: str = "chebyshev",
    default_non_depth_compressor: str = "dct",
    default_mlp_hidden_group_size: int = 256,
) -> ResolvedGeometryPlan:
    normalized_selectors = tuple(normalize_selector(value) for value in selected_elements)
    entries = tuple(GEOMETRY_REGISTRY[selector] for selector in normalized_selectors)
    _validate_selection_overlaps(entries)
    parsed_options = tuple(parse_option_assignment(value) for value in option_assignments)
    _validate_option_targets(parsed_options, entries, depth_enabled=bool(select_depth))
    values = _option_map(parsed_options)

    if not select_depth and not entries:
        raise ValueError("systematic geometry UI requires --select-depth and/or at least one --select-element")

    depth_compressor: Optional[str] = None
    depth_version: Optional[str] = None
    depth_order: Optional[int] = None
    if select_depth:
        depth_compressor, depth_version = _compressor_for_target(
            AXIS_DEPTH, values, default_family=default_depth_compressor
        )
        explicit_order = values.get((AXIS_DEPTH, "order"))
        depth_order = (
            _positive_integer("DEPTH.order", explicit_order)
            if explicit_order is not None
            else int(legacy_orders["o_depth"])
        )
        if depth_order < 1:
            raise ValueError(f"o_depth must be positive; got {depth_order!r}")

    resolved: list[ResolvedGeometrySelection] = []
    compressor_pairs: set[Tuple[str, str]] = set()
    for entry in entries:
        implied_type = entry.resolved_type(bool(select_depth))
        family, version = _compressor_for_target(
            entry.element, values, default_family=default_non_depth_compressor
        )
        compressor_pairs.add((family, version))
        dimensions = len(entry.compressed_axes)
        capability = COMPRESSOR_REGISTRY[family]
        if dimensions not in capability.semantic_dimensions:
            raise ValueError(
                f"compressor {family}@{version} does not support the {dimensions}-D geometry selected by {entry.selector}"
            )
        orders = {
            axis: _order_for_axis(entry.element, axis, values, legacy_orders)
            for axis in entry.compressed_axes
        }
        axis_options: Dict[str, Dict[str, Any]] = {}
        for axis in entry.compressed_axes:
            target = f"{entry.element}.{axis}"
            group_size_raw = values.get((target, "group_size"))
            if group_size_raw is not None:
                if not capability.supports_group_size:
                    raise ValueError(
                        f"compressor {family}@{version} does not accept group_size, but {target}.group_size was supplied"
                    )
                axis_options.setdefault(axis, {})["group_size"] = _positive_integer(
                    f"{target}.group_size", group_size_raw
                )
        if family == COMPRESSOR_JPEG_LIKE and AXIS_MLP_HIDDEN in entry.compressed_axes:
            axis_options.setdefault(AXIS_MLP_HIDDEN, {}).setdefault(
                "group_size", int(default_mlp_hidden_group_size)
            )
            if orders[AXIS_MLP_HIDDEN] > int(axis_options[AXIS_MLP_HIDDEN]["group_size"]):
                raise ValueError(
                    f"{entry.element}.{AXIS_MLP_HIDDEN}.order must not exceed group_size: "
                    f"order={orders[AXIS_MLP_HIDDEN]}, group_size={axis_options[AXIS_MLP_HIDDEN]['group_size']}"
                )
        resolved.append(
            ResolvedGeometrySelection(
                selector=entry.selector,
                element=entry.element,
                implied_type=implied_type,
                compressed_axes=((AXIS_DEPTH,) + entry.compressed_axes) if select_depth else entry.compressed_axes,
                independent_indices=entry.independent_indices,
                compressor=family,
                compressor_version=version,
                orders=orders,
                axis_options=axis_options,
            )
        )

    if len(compressor_pairs) > 1:
        assignments = ", ".join(f"{family}@{version}" for family, version in sorted(compressor_pairs))
        raise ValueError(
            "Phase 1 requires every selected non-DEPTH element to use one shared compressor family/version; "
            f"resolved assignments were {assignments}"
        )
    shared_family: Optional[str] = None
    shared_version: Optional[str] = None
    if compressor_pairs:
        shared_family, shared_version = next(iter(compressor_pairs))

    adapter = _materializer_adapter(
        depth_enabled=bool(select_depth),
        depth_compressor=depth_compressor,
        depth_version=depth_version,
        selections=resolved,
    )
    return ResolvedGeometryPlan(
        schema_version=GEOMETRY_PLAN_SCHEMA_VERSION,
        registry_version=GEOMETRY_REGISTRY_VERSION,
        depth_enabled=bool(select_depth),
        depth_compressor=depth_compressor,
        depth_compressor_version=depth_version,
        depth_order=depth_order,
        selections=tuple(resolved),
        parsed_options=parsed_options,
        shared_non_depth_compressor=shared_family,
        shared_non_depth_compressor_version=shared_version,
        materializer=adapter,
    )


def validate_resolved_geometry_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("resolved_geometry_plan must be a mapping")
    values = dict(plan)
    if values.get("schema_version") != GEOMETRY_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported resolved geometry plan schema: {values.get('schema_version')!r}"
        )
    if values.get("registry_version") != GEOMETRY_REGISTRY_VERSION:
        raise ValueError(
            f"unsupported geometry registry version: {values.get('registry_version')!r}"
        )
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


def format_geometry_plan(plan: ResolvedGeometryPlan, *, detailed: bool = False) -> str:
    lines = ["selected geometry", "-----------------"]
    if plan.depth_enabled:
        lines.extend(
            [
                "DEPTH",
                f"  implied element type:  {ELEMENT_TYPE_CURVE}",
                f"  compressed axis:       {AXIS_DEPTH}",
                f"  compressor:            {plan.depth_compressor}@{plan.depth_compressor_version}",
                f"  order:                 {plan.depth_order}",
            ]
        )
    for selection in plan.selections:
        lines.append("")
        lines.append(selection.selector)
        lines.append(f"  implied element type:  {selection.implied_type}")
        lines.append(f"  compressed axes:       {' × '.join(selection.compressed_axes)}")
        if selection.independent_indices:
            lines.append(f"  independent instances: {' × '.join(selection.independent_indices)}")
        lines.append(f"  compressor:            {selection.compressor}@{selection.compressor_version}")
        for axis, order in selection.orders.items():
            lines.append(f"  order {axis}:          {order}")
        for axis, options in selection.axis_options.items():
            for name, value in options.items():
                lines.append(f"  {axis}.{name}:      {value}")
    lines.append("")
    lines.append(f"implementation status: {'implemented' if plan.materializer.implemented else 'not implemented'}")
    lines.append(f"  {plan.materializer.message}")
    if detailed:
        lines.append(f"registry version:      {plan.registry_version}")
        lines.append(f"plan schema:           {plan.schema_version}")
        if plan.materializer.legacy_geometry_preset:
            lines.append(f"legacy adapter preset: {plan.materializer.legacy_geometry_preset}")
            lines.append(f"materialization:       {plan.materializer.materialization_version}")
    return "\n".join(lines)


__all__ = [
    "AXIS_ATTENTION_D_MODEL",
    "AXIS_ATTENTION_HEAD_CHANNEL",
    "AXIS_DEPTH",
    "AXIS_MLP_D_MODEL",
    "AXIS_MLP_HIDDEN",
    "COMPRESSOR_JPEG_LIKE",
    "COMPRESSOR_REGISTRY",
    "ELEMENT_TYPE_CURVE",
    "ELEMENT_TYPE_SHEET",
    "ELEMENT_TYPE_SHEET_SET",
    "GEOMETRY_PLAN_SCHEMA_VERSION",
    "GEOMETRY_REGISTRY",
    "GEOMETRY_REGISTRY_VERSION",
    "GeometryEntry",
    "MaterializerAdapter",
    "ParsedOption",
    "ResolvedGeometryPlan",
    "ResolvedGeometrySelection",
    "format_geometry_plan",
    "normalize_selector",
    "parse_option_assignment",
    "permitted_geometry_selectors",
    "resolve_geometry_plan",
    "validate_resolved_geometry_plan",
]
# ^^^ THOG
