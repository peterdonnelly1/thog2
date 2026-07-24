# vvv THOG
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .bases import BASIS_FAMILIES, basis_version_for_family, normalize_registered_basis_family

GEOMETRY_REGISTRY_VERSION = "geometry_registry_v2"
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

LEGACY_SEMANTICALLY_UNSOUND_PRESETS = frozenset(
    {"legacy_sheet_col", "jpeg_like_v1", "mlp_block", "head_aware_block", "full_block"}
)

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
    implemented: bool = False
    implemented_with_depth: bool = False

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

    def is_implemented(self, depth_enabled: bool) -> bool:
        return self.implemented_with_depth if depth_enabled else self.implemented


@dataclass(frozen=True)
class CompressorCapability:
    family: str
    default_version: str
    element_types: Tuple[str, ...]
    implemented: bool
    supports_group_size: bool = False
    notes: str = ""

    @property
    def semantic_dimensions(self) -> Tuple[int, ...]:
        return tuple(sorted({1 if kind == ELEMENT_TYPE_CURVE else 2 for kind in self.element_types}))

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
    GeometryEntry("MLP_UP.MLP_HIDDEN", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, (AXIS_MLP_D_MODEL,), "MLP expansion hidden-axis curve"),
    GeometryEntry("MLP_UP.MLP_D_MODEL", ELEMENT_MLP_UP, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, (AXIS_MLP_HIDDEN,), "MLP expansion model-axis curve"),
    GeometryEntry("MLP_UP", ELEMENT_MLP_UP, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, None, (), "Complete MLP expansion sheet"),
    GeometryEntry("MLP_DOWN.MLP_HIDDEN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, (AXIS_MLP_D_MODEL,), "MLP contraction hidden-axis curve"),
    GeometryEntry("MLP_DOWN.MLP_D_MODEL", ELEMENT_MLP_DOWN, (AXIS_MLP_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET, (AXIS_MLP_HIDDEN,), "MLP contraction model-axis curve"),
    GeometryEntry("MLP_DOWN", ELEMENT_MLP_DOWN, (AXIS_MLP_HIDDEN, AXIS_MLP_D_MODEL), ELEMENT_TYPE_SHEET, None, (), "Complete MLP contraction sheet"),
    GeometryEntry("ATTENTION_QKV.ATTENTION_D_MODEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET, ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL), "QKV input-model-axis curves"),
    GeometryEntry("ATTENTION_QKV.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_HEAD_CHANNEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET, ("QKV_ROLE", "ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL), "QKV within-head channel curves"),
    GeometryEntry("ATTENTION_QKV", ELEMENT_ATTENTION_QKV, (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, None, ("QKV_ROLE", "ATTENTION_HEAD"), "One QKV sheet per role and head"),
    GeometryEntry("ATTENTION_OUTPUT.ATTENTION_D_MODEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_D_MODEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET, ("ATTENTION_HEAD", AXIS_ATTENTION_HEAD_CHANNEL), "Attention-output model-axis curves"),
    GeometryEntry("ATTENTION_OUTPUT.ATTENTION_HEAD_CHANNEL", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_HEAD_CHANNEL,), ELEMENT_TYPE_CURVE, ELEMENT_TYPE_SHEET_SET, ("ATTENTION_HEAD", AXIS_ATTENTION_D_MODEL), "Attention-output within-head channel curves"),
    GeometryEntry("ATTENTION_OUTPUT", ELEMENT_ATTENTION_OUTPUT, (AXIS_ATTENTION_D_MODEL, AXIS_ATTENTION_HEAD_CHANNEL), ELEMENT_TYPE_SHEET_SET, None, ("ATTENTION_HEAD",), "One attention-output sheet per head"),
)

GEOMETRY_REGISTRY: Mapping[str, GeometryEntry] = {entry.selector: entry for entry in _GEOMETRY_ENTRIES}


def _build_compressor_registry() -> Dict[str, CompressorCapability]:
    capabilities: Dict[str, CompressorCapability] = {}
    for family in BASIS_FAMILIES:
        canonical = normalize_registered_basis_family(family)
        capabilities[canonical] = CompressorCapability(
            family=canonical,
            default_version=basis_version_for_family(canonical),
            element_types=(ELEMENT_TYPE_CURVE,),
            implemented=True,
            notes="Implemented one-dimensional CURVE compressor.",
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
    if property_normalized not in ("compressor", "compressor_version", "version", "order", "group_size"):
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
        raise ValueError(
            "compressor 'jpeg_like' is not currently implemented as a semantically valid SHEET compressor; "
            "JPEG_LIKE_V1 remains available only through the legacy geometry preset"
        )
    try:
        canonical = normalize_registered_basis_family(normalized)
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
    selected_axis_targets = {f"{entry.element}.{axis}" for entry in entries for axis in entry.compressed_axes}
    for option in parsed_options:
        if option.property in ("compressor", "compressor_version"):
            if option.target == AXIS_DEPTH:
                if not depth_enabled:
                    raise ValueError(f"{option.source!r} targets DEPTH, but --select-depth was not supplied")
            elif option.target not in selected_elements:
                raise ValueError(f"{option.source!r} must target a selected element; selected elements are {tuple(sorted(selected_elements))}")
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


def _compressor_for_target(target: str, values: Mapping[Tuple[str, str], str], *, default_family: str) -> Tuple[str, str]:
    family, inline_version = _split_compressor_value(values.get((target, "compressor"), default_family))
    capability = COMPRESSOR_REGISTRY[family]
    explicit_version = values.get((target, "compressor_version"))
    if inline_version is not None and explicit_version is not None and inline_version != explicit_version:
        raise ValueError(f"conflicting compressor versions for {target}: {inline_version!r} and {explicit_version!r}")
    return family, explicit_version or inline_version or capability.default_version


def _order_for_axis(element: str, axis: str, values: Mapping[Tuple[str, str], str], legacy_orders: Mapping[str, int]) -> int:
    target = f"{element}.{axis}"
    explicit = values.get((target, "order"))
    if explicit is not None:
        return _positive_integer(f"{target}.order", explicit)
    field_name = ORDER_OPTION_FIELDS[(element, axis)]
    value = int(legacy_orders[field_name])
    if value < 1:
        raise ValueError(f"{field_name} must be positive; got {value!r}")
    return value


def _compatibility_message(element_type: str, family: str, version: str) -> str:
    capability = COMPRESSOR_REGISTRY[family]
    if element_type in capability.element_types:
        return ""
    implemented_for_type = [name for name, item in COMPRESSOR_REGISTRY.items() if item.implemented and element_type in item.element_types]
    suffix = f" implemented {element_type} compressors: {', '.join(implemented_for_type)}" if implemented_for_type else f" no {element_type} compressor is currently implemented"
    return f"compressor {family}@{version} is a {ELEMENT_TYPE_CURVE} compressor and cannot be used with a {element_type} geometry;{suffix}"


def _materializer_adapter(*, depth_enabled: bool, depth_compressor: Optional[str], depth_version: Optional[str], selections: Sequence[ResolvedGeometrySelection], incompatibilities: Sequence[str]) -> MaterializerAdapter:
    if incompatibilities:
        return MaterializerAdapter(False, None, None, None, None, None, None, "; ".join(incompatibilities))
    if depth_enabled and not selections:
        return MaterializerAdapter(True, "depth", depth_compressor, depth_version, None, None, "depth_v1", "Implemented by the existing DEPTH trajectory.")
    if not depth_enabled and not selections:
        return MaterializerAdapter(False, None, None, None, None, None, None, "No geometry was selected; supply --select-depth and/or at least one --select-element.")
    unimplemented = [selection.selector for selection in selections]
    return MaterializerAdapter(False, None, None, None, None, None, None, f"registered geometry is not currently implemented: {', '.join(unimplemented)}")


def resolve_geometry_plan(*, select_depth: bool, selected_elements: Sequence[str], option_assignments: Sequence[str], legacy_orders: Mapping[str, int], default_depth_compressor: str = "chebyshev", default_non_depth_compressor: str = "dct", default_mlp_hidden_group_size: int = 256) -> ResolvedGeometryPlan:
    del default_mlp_hidden_group_size
    entries = tuple(GEOMETRY_REGISTRY[normalize_selector(value)] for value in selected_elements)
    _validate_selection_overlaps(entries)
    parsed_options = tuple(parse_option_assignment(value) for value in option_assignments)
    _validate_option_targets(parsed_options, entries, depth_enabled=bool(select_depth))
    values = _option_map(parsed_options)
    if not select_depth and not entries:
        raise ValueError("systematic geometry UI requires --select-depth and/or at least one --select-element")

    depth_compressor = depth_version = None
    depth_order: Optional[int] = None
    if select_depth:
        depth_compressor, depth_version = _compressor_for_target(AXIS_DEPTH, values, default_family=default_depth_compressor)
        depth_order = _positive_integer("DEPTH.order", values[(AXIS_DEPTH, "order")]) if (AXIS_DEPTH, "order") in values else int(legacy_orders["o_depth"])

    resolved: list[ResolvedGeometrySelection] = []
    compressor_pairs: set[Tuple[str, str]] = set()
    incompatibilities: list[str] = []
    for entry in entries:
        implied_type = entry.resolved_type(bool(select_depth))
        family, version = _compressor_for_target(entry.element, values, default_family=default_non_depth_compressor)
        compressor_pairs.add((family, version))
        incompatibility = _compatibility_message(implied_type, family, version)
        if incompatibility:
            incompatibilities.append(f"{entry.selector}: {incompatibility}")
        orders = {axis: _order_for_axis(entry.element, axis, values, legacy_orders) for axis in entry.compressed_axes}
        axis_options: Dict[str, Dict[str, Any]] = {}
        for axis in entry.compressed_axes:
            target = f"{entry.element}.{axis}"
            if (target, "group_size") in values:
                raise ValueError(f"compressor {family}@{version} does not accept group_size, but {target}.group_size was supplied")
        resolved.append(ResolvedGeometrySelection(entry.selector, entry.element, implied_type, ((AXIS_DEPTH,) + entry.compressed_axes) if select_depth else entry.compressed_axes, entry.independent_indices, family, version, orders, axis_options))

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


def format_geometry_plan(plan: ResolvedGeometryPlan, *, detailed: bool = False) -> str:
    lines = ["selected geometry", "-----------------"]
    if plan.depth_enabled:
        lines.extend(["DEPTH", f"  implied element type:  {ELEMENT_TYPE_CURVE}", f"  compressed axis:       {AXIS_DEPTH}", f"  compressor:            {plan.depth_compressor}@{plan.depth_compressor_version}", f"  order:                 {plan.depth_order}"])
    for selection in plan.selections:
        lines.extend(["", selection.selector, f"  implied element type:  {selection.implied_type}", f"  compressed axes:       {' × '.join(selection.compressed_axes)}"])
        if selection.independent_indices:
            lines.append(f"  independent instances: {' × '.join(selection.independent_indices)}")
        lines.append(f"  compressor:            {selection.compressor}@{selection.compressor_version}")
        for axis, order in selection.orders.items():
            lines.append(f"  order {axis}:          {order}")
        for axis, options in selection.axis_options.items():
            for name, value in options.items():
                lines.append(f"  {axis}.{name}:      {value}")
    lines.extend(["", f"implementation status: {'implemented' if plan.materializer.implemented else 'not currently implemented'}", f"  {plan.materializer.message}"])
    if detailed:
        lines.extend([f"registry version:      {plan.registry_version}", f"plan schema:           {plan.schema_version}"])
        if plan.materializer.legacy_geometry_preset:
            lines.extend([f"legacy adapter preset: {plan.materializer.legacy_geometry_preset}", f"materialization:       {plan.materializer.materialization_version}"])
    return "\n".join(lines)


# vvv THOG keep legacy geometries runnable while making their semantic defect impossible to miss.
def _legacy_geometry_preset_from_argv(arguments: Sequence[str]) -> Optional[str]:
    for index, argument in enumerate(arguments):
        if argument == "--geometry-preset" and index + 1 < len(arguments):
            return arguments[index + 1].strip().lower()
        if argument.startswith("--geometry-preset="):
            return argument.split("=", 1)[1].strip().lower()
    return None


def _warn_for_legacy_semantics() -> None:
    if any(argument in ("--select-depth", "--select-element", "--explain-geometry") or argument.startswith("--select-element=") for argument in sys.argv[1:]):
        return
    preset = _legacy_geometry_preset_from_argv(sys.argv[1:])
    if preset not in LEGACY_SEMANTICALLY_UNSOUND_PRESETS:
        return
    orange = "\033[1;38;5;208m" if sys.stderr.isatty() and os.environ.get("NO_COLOR") is None else ""
    reset = "\033[0m" if orange else ""
    print(f"{orange}WARNING: legacy geometry '{preset}' is not semantically sensible: it combines unrelated weight coordinates under one compact field. It remains runnable only for legacy comparison.{reset}", file=sys.stderr, flush=True)


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


_warn_for_legacy_semantics()
_install_clean_cli_excepthook()
# ^^^ THOG


__all__ = [
    "AXIS_ATTENTION_D_MODEL", "AXIS_ATTENTION_HEAD_CHANNEL", "AXIS_DEPTH", "AXIS_MLP_D_MODEL", "AXIS_MLP_HIDDEN",
    "COMPRESSOR_JPEG_LIKE", "COMPRESSOR_REGISTRY", "ELEMENT_TYPE_CURVE", "ELEMENT_TYPE_SHEET", "ELEMENT_TYPE_SHEET_SET",
    "GEOMETRY_PLAN_SCHEMA_VERSION", "GEOMETRY_REGISTRY", "GEOMETRY_REGISTRY_VERSION", "GeometryEntry", "MaterializerAdapter",
    "ParsedOption", "ResolvedGeometryPlan", "ResolvedGeometrySelection", "format_geometry_plan", "normalize_selector",
    "parse_option_assignment", "permitted_geometry_selectors", "resolve_geometry_plan", "validate_resolved_geometry_plan",
]
# ^^^ THOG
