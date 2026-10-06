# vvv THOG width selector consumes the existing scoped parser and central basis registry
from __future__ import annotations

import argparse

from .bases import basis_version_for_family, normalize_basis_version, normalize_registered_basis_family
from .geometry_registry import (GEOMETRY_PLAN_SCHEMA_VERSION, GEOMETRY_REGISTRY_VERSION,
    MaterializerAdapter, ResolvedGeometryPlan, _option_map, parse_option_assignment)
from .width import WIDTH_PRESET, WIDTH_CAPTURE_DEFAULTS, WIDTH_CAPTURE_PREFIX, build_width_basis


class ExplicitLegacyBasisAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.width_explicit_legacy_basis = True


def add_width_arguments(parser):
    parser.add_argument('--select-width', action='store_true', help='select direct spectral residual coefficients; requires width-type-I and WIDTH.order')
    for suffix, default in WIDTH_CAPTURE_DEFAULTS.items():
        parser.add_argument('--' + WIDTH_CAPTURE_PREFIX + suffix, default=default,
                            type=str if isinstance(default, str) else int,
                            choices=('off', 'basic', 'probes') if suffix == 'mode' else None)


def _family_version(values, target, default):
    family_spec = values.get((target, 'compressor'), default)
    parts = family_spec.split('@')
    if len(parts) > 2 or not parts[0]:
        raise ValueError(f'invalid {target}.compressor={family_spec!r}')
    family = normalize_registered_basis_family(parts[0])
    inline = parts[1] if len(parts) == 2 else None
    separate = values.get((target, 'compressor_version'))
    if inline is not None and separate is not None and normalize_basis_version(family, inline) != normalize_basis_version(family, separate):
        raise ValueError(f'conflicting compressor versions for {target}')
    version = normalize_basis_version(family, separate or inline or 'auto')
    return family, version


def resolve_width_geometry(arguments):
    if arguments.geometry_preset != WIDTH_PRESET or not arguments.select_width:
        raise ValueError('--select-width requires --geometry-preset width-type-I and conversely')
    if arguments.select_element or arguments.hyperblock or arguments.plastic__enabled:
        raise ValueError('width-type-I rejects additional element geometries, HYPERBLOCK, and PLASTIC')
    if vars(arguments).get('width_explicit_legacy_basis', False):
        raise ValueError('explicit --basis-family and --basis-version are incompatible with width mode; use scoped WIDTH/DEPTH options')
    if arguments.attention_geometry is not None or arguments.mlp_geometry is not None:
        raise ValueError('width-type-I rejects additional attention/MLP geometries')
    if arguments.model_type == 'dense' or arguments.initialise_from_dense_snapshot is not None or arguments.save_dense_initialisation_snapshot:
        raise ValueError('width-type-I trains compact parameters directly; legacy dense snapshot conversion is not supported')
    parsed = tuple(parse_option_assignment(value) for value in arguments.geometry_options)
    for option in parsed:
        if option.target not in ('WIDTH', 'DEPTH') or option.property not in ('compressor', 'compressor_version', 'order'):
            raise ValueError(f'unsupported width-path scoped option: {option.source}')
        if option.target == 'DEPTH' and not arguments.select_depth:
            raise ValueError('DEPTH options require --select-depth')
    values = _option_map(parsed)
    if ('WIDTH', 'order') not in values:
        raise ValueError('WIDTH.order is required; no automatic retained width is selected')
    try:
        retained_width = int(values[('WIDTH', 'order')])
    except ValueError as error:
        raise ValueError('WIDTH.order must be an integer coefficient count') from error
    family, version = _family_version(values, 'WIDTH', 'chebyshev')
    _, metadata = build_width_basis(arguments.n_embd, retained_width, family, version)
    if arguments.n_embd % arguments.n_head:
        raise ValueError('attention width A = D must be divisible by n_head')
    depth_family = depth_version = depth_order = None
    if arguments.select_depth:
        depth_family, depth_version = _family_version(values, 'DEPTH', 'chebyshev')
        try:
            depth_order = int(values.get(('DEPTH', 'order'), arguments.o_depth))
        except ValueError as error:
            raise ValueError('DEPTH.order must be an integer') from error
        if not 1 <= depth_order <= arguments.n_layer:
            raise ValueError('DEPTH.order must satisfy 1 <= P <= L')
    width = {'reference_width': arguments.n_embd, 'residual_width': retained_width,
             'attention_width': arguments.n_embd, 'mlp_hidden_width': 4 * arguments.n_embd,
             'executed_blocks': arguments.n_layer, 'depth_order': depth_order,
             'active_axes': ['WIDTH'] + (['DEPTH'] if arguments.select_depth else []),
             'basis': metadata, 'matrix_orientation': 'output_by_input',
             'operator_shapes': {'qkv': [3 * arguments.n_embd, retained_width],
                'attention_output': [retained_width, arguments.n_embd],
                'mlp_up': [4 * arguments.n_embd, retained_width],
                'mlp_down': [retained_width, 4 * arguments.n_embd]}}
    adapter = MaterializerAdapter(True, WIDTH_PRESET, depth_family or 'chebyshev',
                depth_version or basis_version_for_family('chebyshev'), None, None,
                'width_type_I_direct_v1', 'Direct coefficient residuals, projected reference-space norms, and compact operator contractions.')
    return ResolvedGeometryPlan(GEOMETRY_PLAN_SCHEMA_VERSION, GEOMETRY_REGISTRY_VERSION,
        bool(arguments.select_depth), depth_family, depth_version, depth_order,
        (), parsed, None, None, adapter, width)
# ^^^ THOG
