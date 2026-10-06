# vvv THOG residual-stream spectral reparameterisation; authoritative attached v0.2 specification
"""Fixed, shared width representation and exactly folded reference-space LayerNorm.

Analysis is C x, synthesis is C.T c, and C.T C is a lossy projection for r < D.
No normal forward constructs represented token features or a diagonal gain matrix.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .bases import build_registered_basis, get_basis_definition, normalize_basis_version


WIDTH_PRESET = "width-type-I"
WIDTH_CONFIG_FIELDS = (
    "width_enabled", "width_order", "width_compressor", "width_compressor_version",
    "width_depth_enabled",
)
WIDTH_CAPTURE_DEFAULTS = {
    "mode": "off", "log_every_n_steps": 100, "probe_every_n_steps": 1000,
    "history_length": 20, "sample_tokens_per_layer": 8,
    "feature_evaluation_points": 256, "probe_orders": "auto",
    "start_step": 0, "end_step": -1,
}
WIDTH_CAPTURE_PREFIX = "instrumentation__width_activation_curves__"


# vvv THOG rank failures name a numerically certified lower-order alternative
def _resolved_lower_width_order(reference_width, retained_width, kernel, version):
    order = 2 ** int(math.log2(min(128, retained_width - 1)))
    coordinates = kernel.coordinates(reference_width, dtype=torch.float64, device="cpu")
    while order >= 2:
        raw = kernel.raw_basis_for_version(coordinates, order, version)
        singular_values = torch.linalg.svdvals(raw)
        tolerance = torch.finfo(torch.float64).eps * max(reference_width, order) * singular_values[0]
        if int((singular_values > tolerance).sum()) == order:
            columns = build_registered_basis(reference_width, order, runtime_dtype=torch.float64,
                                             basis_family=kernel.basis_family, version=version)
            error = float((columns.T @ columns - torch.eye(order, dtype=torch.float64)).abs().max())
            span_error = float((raw - columns @ (columns.T @ raw)).norm() / raw.norm())
            if error <= 1.0e-10 and span_error <= 1.0e-10:
                return order
        order //= 2
    return None
# ^^^ THOG


def build_width_basis(reference_width: int, retained_width: int, family: str, version: str = "auto"):
    if isinstance(retained_width, bool) or not isinstance(retained_width, int) or not 2 <= retained_width <= reference_width:
        raise ValueError(f"WIDTH.order must satisfy 2 <= r <= D; got D={reference_width}, r={retained_width}")
    definition = get_basis_definition(family)
    if not definition.supports_width_basis:
        raise ValueError(f"{family} does not provide an orthonormal width-basis capability")
    resolved_version = normalize_basis_version(definition.family, version)
    kernel = definition.kernel
    # Construction checks deliberately use the shared family's raw coordinates and stabilization.
    coordinates = kernel.coordinates(reference_width, dtype=torch.float64, device="cpu")
    raw_basis = kernel.raw_basis_for_version(coordinates, retained_width, resolved_version)
    singular_values = torch.linalg.svdvals(raw_basis)
    tolerance = torch.finfo(torch.float64).eps * max(reference_width, retained_width) * singular_values[0]
    rank = int((singular_values > tolerance).sum())
    condition = float(singular_values[0] / singular_values[-1])
    if rank != retained_width:
        # vvv THOG explain the specified uniform-grid Chebyshev limit before allocation
        remedy = ""
        if definition.family == "chebyshev":
            lower_order = _resolved_lower_width_order(reference_width, retained_width, kernel, resolved_version)
            remedy = "; equally spaced Chebyshev coordinates cannot resolve this polynomial order in float64. "
            if lower_order is not None:
                remedy += f"Try WIDTH.order={lower_order} at D={reference_width}, or "
            remedy += f"keep WIDTH.order={retained_width} with WIDTH.compressor=dct. No basis is substituted automatically."
        # ^^^ THOG
        raise ValueError(
            f"unresolved width basis {definition.family}@{resolved_version}: D={reference_width}, "
            f"r={retained_width}, float64 rank={rank}, tolerance={float(tolerance):.6g}, condition={condition:.6g}{remedy}"
        )
    columns = build_registered_basis(reference_width, retained_width, runtime_dtype=torch.float64,
                                     basis_family=definition.family, version=resolved_version)
    error = float((columns.T @ columns - torch.eye(retained_width, dtype=torch.float64)).abs().max())
    # Certify the requested raw span, as well as orthogonality, for QR-derived coordinates.
    span_error = float((raw_basis - columns @ (columns.T @ raw_basis)).norm() / raw_basis.norm())
    if error > 1.0e-10 or span_error > 1.0e-10:
        raise ValueError(f"width stabilization failed for {definition.family}@{resolved_version}, D={reference_width}, r={retained_width}: orthogonality={error}, span={span_error}")
    analysis = columns.T.contiguous()
    constant_first = bool(torch.allclose(analysis[0], torch.full_like(analysis[0], 1.0 / math.sqrt(reference_width)), atol=1.0e-12, rtol=0.0))
    metadata = {
        **definition.metadata(), "basis_family": definition.family, "basis_version": resolved_version,
        "reference_width": reference_width, "retained_width": retained_width,
        "analysis_shape": [retained_width, reference_width], "basis_dtype": "float64_construction",
        "fingerprint": hashlib.sha256(analysis.numpy().tobytes()).hexdigest(),
        "constant_first_mode": constant_first, "mode_order": list(range(retained_width)),
        "raw_condition_number": condition, "raw_numerical_rank": rank,
        "rank_tolerance": float(tolerance), "rank_tolerance_policy": "eps_float64 * max(D,r) * sigma_max",
        "orthogonality_max_error": error, "raw_span_relative_error": span_error,
    }
    return analysis, metadata


class WidthBasis(nn.Module):
    """One registered basis copy, shared by all normalization sites in a model."""

    def __init__(self, analysis: Tensor, metadata: Optional[dict] = None):
        super().__init__()
        self.metadata = dict(metadata or {})
        self.reference_width = analysis.shape[1]
        self.retained_width = analysis.shape[0]
        self.constant_first_mode = bool(self.metadata.get("constant_first_mode", False))
        self.register_buffer("analysis", analysis.clone())
        self.register_buffer("constant_analysis", analysis.sum(dim=1), persistent=False)

    def _apply(self, function, recurse=True):
        # Half precision must never destroy the sensitive fixed normalization operators.
        original_analysis = self.analysis
        original_constant = self.constant_analysis
        super()._apply(function, recurse)
        if self.analysis.dtype in (torch.float16, torch.bfloat16):
            self.analysis = original_analysis.to(device=self.analysis.device, dtype=torch.float32)
            self.constant_analysis = original_constant.to(device=self.analysis.device, dtype=torch.float32)
        return self


def projected_layer_norm(inputs: Tensor, basis: WidthBasis, gain: Tensor, bias: Optional[Tensor], epsilon: float = 1.0e-5) -> Tensor:
    precision = torch.float64 if inputs.dtype == torch.float64 or gain.dtype == torch.float64 else torch.float32
    # Explicitly disable autocast: merely casting operands does not keep matmul accumulation in float32.
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        coefficients = inputs.to(precision)
        analysis = basis.analysis.to(precision)
        gamma = gain.to(precision)
        with torch.profiler.record_function("width_norm_construct"):
            gain_operator = (analysis * gamma.unsqueeze(0)) @ analysis.T
            projected_bias = None if bias is None else analysis @ bias.to(precision)
        with torch.profiler.record_function("width_norm_apply"):
            if basis.constant_first_mode:
                centered = torch.cat((torch.zeros_like(coefficients[..., :1]), coefficients[..., 1:]), dim=-1)
                variance = centered.square().sum(dim=-1, keepdim=True) / basis.reference_width
                numerator = functional.linear(centered, gain_operator)
            else:
                constant_analysis = basis.constant_analysis.to(precision)
                mean = (coefficients * constant_analysis).sum(dim=-1, keepdim=True) / basis.reference_width
                variance = (coefficients.square().sum(dim=-1, keepdim=True) / basis.reference_width - mean.square()).clamp_min(0)
                gain_mean = analysis @ gamma
                numerator = functional.linear(coefficients, gain_operator) - mean * gain_mean
            output = numerator * torch.rsqrt(variance + epsilon)
            if projected_bias is not None:
                output = output + projected_bias
    return output.to(inputs.dtype)


class ProjectedLayerNorm(nn.Module):
    def __init__(self, basis: WidthBasis, bias: bool, *, external_affine: bool = False):
        super().__init__()
        # An ordinary reference avoids registering the shared basis at each site.
        object.__setattr__(self, "width_basis", basis)
        self.weight = None if external_affine else nn.Parameter(torch.ones(basis.reference_width))
        self.bias = None if external_affine or not bias else nn.Parameter(torch.zeros(basis.reference_width))
        self.epsilon = 1.0e-5

    def forward(self, inputs, gain=None, bias=None):
        return projected_layer_norm(inputs, self.width_basis, self.weight if gain is None else gain,
                                    self.bias if gain is None else bias, self.epsilon)


def width_identity(config, *, analysis_metadata=None):
    metadata = analysis_metadata
    if metadata is None:
        _, metadata = build_width_basis(config.n_embd, config.width_order, config.width_compressor, config.width_compressor_version)
    depth_order = config.depth_order if hasattr(config, "depth_order") else config.o_depth
    depth = None
    if config.width_depth_enabled:
        definition = get_basis_definition(config.basis_family or "chebyshev")
        depth_version = normalize_basis_version(definition.family, config.basis_version)
        depth_columns = build_registered_basis(config.n_layer, depth_order, basis_family=definition.family,
                                                version=depth_version, runtime_dtype=torch.float64)
        depth = {**definition.metadata(), "basis_version": depth_version, "order": depth_order,
                 "fingerprint": hashlib.sha256(depth_columns.numpy().tobytes()).hexdigest(),
                 "compress_layer_norm_and_bias": config.depth_compress_layer_norm_and_bias}
    retained_width, reference_width = config.width_order, config.n_embd
    return {
        "representation": WIDTH_PRESET, "geometry_preset": WIDTH_PRESET,
        "reference_width": reference_width, "residual_width": retained_width,
        "attention_width": reference_width, "mlp_hidden_width": 4 * reference_width,
        "n_layer": config.n_layer, "n_head": config.n_head, "head_dim": reference_width // config.n_head,
        "context_capacity": config.block_size, "matrix_orientation": "output_by_input",
        "vocabulary_size": vars(config).get("vocab_size"),
        "operator_shapes": {"qkv": [3 * reference_width, retained_width],
                            "attention_output": [retained_width, reference_width],
                            "mlp_up": [4 * reference_width, retained_width],
                            "mlp_down": [retained_width, 4 * reference_width]},
        "width_basis": metadata, "depth": depth, "bias": config.bias,
        "basis_family": config.width_compressor, "basis_version": metadata["basis_version"],
        "attention_geometry": WIDTH_PRESET, "mlp_geometry": WIDTH_PRESET,
        "normalization": "C_LN_D_C_transpose", "normalization_epsilon": 1.0e-5,
    }


def validate_width_configuration(config):
    # vvv THOG script and programmatic aliases persist the same canonical representation
    if config.geometry_preset == "width":
        config.geometry_preset = WIDTH_PRESET
    # ^^^ THOG
    if not isinstance(config.width_enabled, bool) or not isinstance(config.width_depth_enabled, bool):
        raise ValueError("width selection fields must be bool")
    if not config.width_enabled:
        if config.geometry_preset == WIDTH_PRESET or config.width_order is not None or config.width_depth_enabled:
            raise ValueError("width-type-I and WIDTH options require --select-width")
        return
    if config.geometry_preset != WIDTH_PRESET:
        raise ValueError("--select-width requires --geometry-preset width-type-I")
    for name in ("n_layer", "n_embd", "n_head", "block_size"):
        value = vars(config)[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.n_embd % config.n_head:
        raise ValueError("attention internal width A = D must be divisible by n_head; r need not be")
    if config.plastic__enabled or config.hyperblock_topology is not None:
        raise ValueError("width-type-I does not support PLASTIC or HYPERBLOCK")
    if config.attention_geometry is not None or config.mlp_geometry is not None:
        raise ValueError("width-type-I does not support additional element geometries")
    if config.premat != "disabled" or config.save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step:
        raise ValueError("width uses direct depth contractions; PREMAT and materialized-weight replay are incompatible")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if not config.width_depth_enabled and config.depth_compress_layer_norm_and_bias:
        raise ValueError("depth-compress-layer-norm-and-bias requires --select-depth in width mode")
    depth_order = config.depth_order if hasattr(config, "depth_order") else config.o_depth
    if config.width_depth_enabled and not 1 <= depth_order <= config.n_layer:
        raise ValueError("joint DEPTH.order must satisfy 1 <= P <= L")
    analysis, metadata = build_width_basis(config.n_embd, config.width_order, config.width_compressor, config.width_compressor_version)
    object.__setattr__(config, "width_compressor", metadata["basis_family"])
    object.__setattr__(config, "width_compressor_version", metadata["basis_version"])
    if not config.width_depth_enabled:
        object.__setattr__(config, "depth_order" if hasattr(config, "depth_order") else "o_depth", config.n_layer)
    return analysis, metadata


@dataclass
class WidthFamilyMetadata:
    name: str
    target_weight_std: float
    weight_decay: bool = True
    semantic_type: str = "matrix"


class WidthTrajectory(nn.Module):
    """Compact coefficient banks; normal application never calls materialize."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.coefficients = nn.ParameterDict()
        self.vector_parameters = nn.ParameterDict()
        reference_width, retained_width = config.n_embd, config.width_order
        shapes = {"attention_input_weight": (3 * reference_width, retained_width),
                  "attention_output_weight": (retained_width, reference_width),
                  "mlp_expansion_weight": (4 * reference_width, retained_width),
                  "mlp_contraction_weight": (retained_width, 4 * reference_width)}
        count = config.depth_order if config.width_depth_enabled else config.n_layer
        self.metadata = tuple(WidthFamilyMetadata(name, 0.02 / math.sqrt(2 * config.n_layer) if name in ("attention_output_weight", "mlp_contraction_weight") else 0.02) for name in shapes)
        for name, shape in shapes.items():
            self.coefficients[name] = nn.Parameter(torch.empty(*shape, count))
        if config.width_depth_enabled:
            self.register_buffer("depth_basis", build_registered_basis(config.n_layer, count, basis_family=config.basis_family or "chebyshev", version=config.basis_version, runtime_dtype=torch.float32), persistent=False)
        else:
            self.depth_basis = None
        vector_shapes = {"attention_input_bias": 3 * reference_width,
                         "attention_output_bias": retained_width, "mlp_expansion_bias": 4 * reference_width,
                         "mlp_contraction_bias": retained_width}
        if config.width_depth_enabled and config.depth_compress_layer_norm_and_bias:
            vector_shapes.update({"ln_1_weight": reference_width, "ln_2_weight": reference_width,
                                  "ln_1_bias": reference_width, "ln_2_bias": reference_width})
        for name, size in vector_shapes.items():
            if not config.bias and name.endswith("bias"):
                continue
            compressed = config.width_depth_enabled and config.depth_compress_layer_norm_and_bias
            container = self.coefficients if compressed else self.vector_parameters
            container[name] = nn.Parameter(torch.zeros(size, count if compressed else config.n_layer))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for item in self.metadata:
                parameter = self.coefficients[item.name]
                if self.config.width_depth_enabled:
                    parameter.zero_()
                    nn.init.normal_(parameter[..., 0], mean=0.0, std=item.target_weight_std * math.sqrt(self.config.n_layer))
                else:
                    nn.init.normal_(parameter, mean=0.0, std=item.target_weight_std)
            for name, parameter in self.coefficients.items():
                if name.startswith("ln_") and name.endswith("weight"):
                    parameter.zero_()
                    parameter[..., 0].fill_(math.sqrt(self.config.n_layer))

    def linear(self, inputs, name, layer_index, bias_name):
        parameter = self.coefficients[name]
        bias = self.vector(bias_name, layer_index)
        if not self.config.width_depth_enabled:
            return functional.linear(inputs, parameter[..., layer_index], bias)
        row = self.depth_basis[layer_index]
        output = functional.linear(inputs, parameter[..., 0]) * row[0]
        for mode_index in range(1, parameter.shape[-1]):
            output = output + functional.linear(inputs, parameter[..., mode_index]) * row[mode_index]
        return output if bias is None else output + bias

    def vector(self, name, layer_index):
        if name in self.vector_parameters:
            return self.vector_parameters[name][:, layer_index]
        if name in self.coefficients:
            return self.coefficients[name] @ self.depth_basis[layer_index]
        return None

    def materialize(self, name, layer_index):
        raise RuntimeError("width normal execution must use direct contractions; materialization is diagnostic only")

    def named_semantic_parameters(self):
        by_name = {item.name: item for item in self.metadata}
        for name, parameter in self.coefficients.items():
            yield name, parameter, by_name.get(name, WidthFamilyMetadata(name, 0.0, False, "vector"))
        for name, parameter in self.vector_parameters.items():
            yield name, parameter, WidthFamilyMetadata(name, 0.0, False, "vector")


def initialize_width_model(model, config):
    analysis, metadata = validate_width_configuration(config)
    model.width_basis = WidthBasis(analysis, metadata)
    model._width_identity = width_identity(config, analysis_metadata=metadata)
    model._width_capture = None
    model._width_probe_order = None
    model._premat_runtime = None
    model.trajectory = WidthTrajectory(config)
    external_affine = config.width_depth_enabled and config.depth_compress_layer_norm_and_bias
    model.transformer = nn.ModuleDict({
        "wte": nn.Embedding(config.vocab_size, config.width_order),
        "wpe": nn.Embedding(config.block_size, config.width_order), "drop": nn.Dropout(config.dropout),
        "ln_f": ProjectedLayerNorm(model.width_basis, config.bias),
        "width_norms": nn.ModuleList([nn.ModuleDict({
            "ln_1": ProjectedLayerNorm(model.width_basis, config.bias, external_affine=external_affine),
            "ln_2": ProjectedLayerNorm(model.width_basis, config.bias, external_affine=external_affine),
        }) for _ in range(config.n_layer)]),
    })
    model.lm_head = nn.Linear(config.width_order, config.vocab_size, bias=False)
    model.transformer.wte.weight = model.lm_head.weight
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.transformer.wpe.weight, mean=0.0, std=0.02)


def width_mask(model, inputs):
    order = model._width_probe_order
    if order is None:
        return inputs
    return torch.cat((inputs[..., :order], torch.zeros_like(inputs[..., order:])), dim=-1)


def width_site(model, site, inputs):
    output = width_mask(model, inputs)
    if model._width_capture is not None:
        model._width_capture.residual(site, output)
    return output


def width_normalize(model, inputs, layer_index, name):
    inputs = width_mask(model, inputs)
    norm = model.transformer.ln_f if name == "ln_f" else model.transformer.width_norms[layer_index][name]
    gain = norm.weight
    bias = norm.bias
    if gain is None:
        gain = model.trajectory.vector(name + "_weight", layer_index)
        bias = model.trajectory.vector(name + "_bias", layer_index)
    output = projected_layer_norm(inputs, model.width_basis, gain, bias, norm.epsilon)
    if model._width_capture is not None:
        model._width_capture.normalization("ln_f" if name == "ln_f" else f"block_{layer_index}.{name}", inputs, gain, bias, output)
    return width_mask(model, output)


def width_logical_block(model, inputs, layer_index):
    config = model.config
    normalized = width_normalize(model, inputs, layer_index, "ln_1")
    query, key, value = model.trajectory.linear(normalized, "attention_input_weight", layer_index, "attention_input_bias").split(config.n_embd, dim=-1)
    batch_size, sequence_length = inputs.shape[:2]
    head_width = config.n_embd // config.n_head
    query, key, value = [item.view(batch_size, sequence_length, config.n_head, head_width).transpose(1, 2) for item in (query, key, value)]
    attended = functional.scaled_dot_product_attention(query, key, value, dropout_p=config.dropout if model.training else 0.0, is_causal=True)
    attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, config.n_embd)
    branch = model.trajectory.linear(attended, "attention_output_weight", layer_index, "attention_output_bias")
    branch = width_mask(model, functional.dropout(branch, p=config.dropout, training=model.training))
    hidden = width_mask(model, inputs + branch)
    normalized = width_normalize(model, hidden, layer_index, "ln_2")
    expanded = model.trajectory.linear(normalized, "mlp_expansion_weight", layer_index, "mlp_expansion_bias")
    branch = model.trajectory.linear(functional.gelu(expanded), "mlp_contraction_weight", layer_index, "mlp_contraction_bias")
    branch = width_mask(model, functional.dropout(branch, p=config.dropout, training=model.training))
    return width_site(model, f"block_{layer_index}.residual_output", hidden + branch)


def width_forward(model, idx, targets=None):
    if idx.ndim != 2 or idx.shape[1] > model.config.block_size:
        raise ValueError("tokens must have [batch,time] shape within context capacity")
    positions = torch.arange(idx.shape[1], device=idx.device)
    hidden = width_site(model, "embedding.residual", model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(positions)))
    if hasattr(model, "checkpoint_segment_size"):
        from .checkpointing import execute_logical_layers
        active_indices = model._active_layer_indices if model.training and torch.is_grad_enabled() else None
        hidden, model.last_execution_report = execute_logical_layers(
            hidden, n_layer=model.config.n_layer, segment_size=model.checkpoint_segment_size,
            logical_block=model._logical_block, training=model.training, layer_indices=active_indices,
            regional_segment_runner_factory=model._regional_segment_runner if model._torch_compile_mode == "regional" else None)
    else:
        for layer_index in range(model.config.n_layer):
            hidden = width_logical_block(model, hidden, layer_index)
    hidden = width_site(model, "head_input.normalized", width_normalize(model, hidden, None, "ln_f"))
    logits = model.lm_head(hidden if targets is not None else hidden[:, [-1]])
    loss = None if targets is None else functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)
    return logits, loss


def width_parameter_report(model):
    config = model.config
    reference_width, retained_width = config.n_embd, config.width_order
    persistent = sum(parameter.numel() for parameter in model.parameters())
    matrix_parameters = sum(model.trajectory.coefficients[item.name].numel() for item in model.trajectory.metadata)
    dense_core = config.n_layer * (4 * reference_width ** 2 + 2 * reference_width * (4 * reference_width))
    categories = {"embeddings_and_tied_head": 0, "core_operators": 0, "normalization_affine": 0, "linear_bias": 0}
    for name, parameter in model.named_parameters():
        category = "embeddings_and_tied_head" if name in ("transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight") else "core_operators" if name.endswith("weight") and name.startswith("trajectory.") and "ln_" not in name else "normalization_affine" if "ln_" in name or "width_norms" in name else "linear_bias"
        categories[category] += parameter.numel() * parameter.element_size()
    basis_bytes = model.width_basis.analysis.numel() * model.width_basis.analysis.element_size()
    buffer_bytes = sum(value.numel() * value.element_size() for value in model.buffers())
    return {
        "persistent_parameters": persistent, "sheet_coefficients": sum(p.numel() for p in model.trajectory.coefficients.values()),
        "conventional_non_sheet_parameters": persistent - sum(p.numel() for p in model.trajectory.coefficients.values()),
        "dense_equivalent_repeated_parameters": dense_core, "dense_equivalent_total_parameters": dense_core + (config.vocab_size + config.block_size) * reference_width + (2 * config.n_layer + 1) * reference_width * (2 if config.bias else 1) + (9 * config.n_layer * reference_width if config.bias else 0),
        "matrix_sheet_coefficients": matrix_parameters, "matrix_dense_equivalent_parameters": dense_core,
        "families": [{"name": item.name, "coefficient_shape": list(model.trajectory.coefficients[item.name].shape), "sheet_parameters": model.trajectory.coefficients[item.name].numel(), "semantic_type": "matrix"} for item in model.trajectory.metadata],
        "width": model._width_identity,
        "dimensions": {"D": reference_width, "r": retained_width, "L": config.n_layer,
                       "P": config.depth_order if config.width_depth_enabled else None,
                       "A": reference_width, "F": 4 * reference_width,
                       "V": config.vocab_size, "T": config.block_size,
                       "N": None, "N_status": "operation-dependent batch*sequence; recorded per captured residual site"},
        "storage": {"learned_parameter_bytes": sum(categories.values()), "learned_categories_bytes": categories,
                    "fixed_width_basis_bytes": basis_bytes, "fixed_buffer_bytes": buffer_bytes,
                    "shared_width_basis_copies": 1, "basis_buffer_dtype": str(model.width_basis.analysis.dtype),
                    "basis_conversion_temporary_shape": [retained_width, reference_width] if model.width_basis.analysis.dtype != next(model.parameters()).dtype else None,
                    "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                    "gradient_class": "same compact parameter shapes; actual gradients measured during training",
                    "optimizer_class": "optimizer-dependent; actual allocated state measured during capture",
                    "core_matrix_retained_fraction": matrix_parameters / dense_core,
                    "normalization_construction": "column_scaling_untiled", "scaled_basis_temporary_shape": [retained_width, reference_width],
                    "gain_operator_shape_per_site": [retained_width, retained_width],
                    "normalization_site_count": 2 * config.n_layer + 1,
                    "derived_gain_operator_bytes_per_site": retained_width ** 2 * max(4, next(model.parameters()).element_size()),
                    "projected_bias_shape_per_site": [retained_width] if config.bias else None,
                    "projected_bias_bytes_per_site": retained_width * max(4, next(model.parameters()).element_size()) if config.bias else 0,
                    "normal_residual_activation_shape": ["N", retained_width],
                    "normal_residual_activation_bytes": "N*r*activation_element_size; N varies by operation",
                    "live_derived_operator_accounting": "autograd may retain G at multiple sites and replayed graphs; included in captured saved-tensor logical and distinct-storage counters",
                    "gain_construction_work_order": "D*r^2", "gain_application_work_order": "N*r^2",
                    "derived_operators": "rebuilt differentiably each forward; no inference cache",
                    "outside_width_reduction": ["attention_scores", "A_wide_QKV", "KV_cache", "F_wide_MLP_hidden", "vocabulary_logits", "executed_depth"]},
    }


@torch.no_grad()
def width_parameter_diagnostics(model):
    rows = {}
    for item in model.trajectory.metadata:
        bank = model.trajectory.coefficients[item.name].detach().float()
        width_axis = 1 if item.name in ('attention_input_weight', 'mlp_expansion_weight') else 0
        mode_energy = bank.square().sum(dim=tuple(axis for axis in range(3) if axis != width_axis))
        mode_fractions = mode_energy / mode_energy.sum().clamp_min(1e-30)
        depth_energy = bank.square().sum(dim=(0, 1)) if model.config.width_depth_enabled else None
        rows[item.name] = {
            'semantic_type': 'direct_width_depth_coefficients' if model.config.width_depth_enabled else 'compact_per_layer_operator',
            'shape': list(bank.shape), 'coefficient_rms': float(bank.square().mean().sqrt()),
            'coefficient_l2_norm': float(bank.norm()), 'coefficient_max_abs': float(bank.abs().max()),
            'width_mode_axis': width_axis, 'width_mode_energy_fraction': mode_fractions.cpu().tolist(),
            'depth_mode_axis': 2 if depth_energy is not None else None,
            'depth_order_energy_fraction': None if depth_energy is None else (depth_energy / depth_energy.sum().clamp_min(1e-30)).cpu().tolist(),
            'interpretation': 'learned-bank energy diagnostics; not evidence of language-model quality',
        }
    return {'coefficient_utilization': rows, 'generated_weights': {},
            'generated_weights_status': 'unavailable: direct width execution; legacy layer-matrix synthesis intentionally omitted',
            'width': width_parameter_report(model), 'compact_state_violations': list(model.compact_state_violations())}
# ^^^ THOG
