# vvv THOG
"""Immutable DENSE step-zero snapshots and shared Chebyshev baselining.

The snapshot is deliberately smaller and stricter than a checkpoint.  It
contains the unique physical model parameters, their alias manifest, the
physical compatibility contract, integrity hashes, and the paired RNG
boundary.  It contains no training-progress or optimiser state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from model import GPT

from .basis import BASIS_VERSION, basis_sha256, build_stabilized_basis
from .checkpoints import capture_rng_state, restore_rng_state
from .compact_identity import BASIS_FAMILY_CHEBYSHEV
from .depth_trajectory import DepthTrajectory
from .model import SheetGPT
from .semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


DENSE_SNAPSHOT_SCHEMA_VERSION = 1
DENSE_SNAPSHOT_TENSOR_MANIFEST_VERSION = 1
DENSE_SNAPSHOT_MAPPING_ALGORITHM_VERSION = "dense_snapshot_chebyshev_float32_v1"
DENSE_SNAPSHOT_DIRECTORY = "dense_baseline_snapshots"
DENSE_SNAPSHOT_SUFFIX = ".dense_snapshot.pt"

DENSE_SNAPSHOT_ROLE_A = "A Normal DENSE"
DENSE_SNAPSHOT_ROLE_B = "B Compressor-baselined DENSE"
DENSE_SNAPSHOT_ROLE_C = "C Compact Run"
DENSE_SNAPSHOT_ROLES = (
    DENSE_SNAPSHOT_ROLE_A,
    DENSE_SNAPSHOT_ROLE_B,
    DENSE_SNAPSHOT_ROLE_C,
)

_SUPPORTED_NUMERIC_DTYPES = ("float32", "float16", "bfloat16")
_DTYPE_FILENAME_CODES = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
}

_MAPPED_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)

_DENSE_MATRIX_SPECS = {
    ATTENTION_QUERY_WEIGHT: ("attn.c_attn.weight", 0),
    ATTENTION_KEY_WEIGHT: ("attn.c_attn.weight", 1),
    ATTENTION_VALUE_WEIGHT: ("attn.c_attn.weight", 2),
    ATTENTION_OUTPUT_WEIGHT: ("attn.c_proj.weight", None),
    MLP_EXPANSION_WEIGHT: ("mlp.c_fc.weight", None),
    MLP_CONTRACTION_WEIGHT: ("mlp.c_proj.weight", None),
}

_DENSE_VECTOR_SUFFIXES = {
    "attention_input_bias": "attn.c_attn.bias",
    "attention_output_bias": "attn.c_proj.bias",
    "mlp_expansion_bias": "mlp.c_fc.bias",
    "mlp_contraction_bias": "mlp.c_proj.bias",
    "ln_1_weight": "ln_1.weight",
    "ln_1_bias": "ln_1.bias",
    "ln_2_weight": "ln_2.weight",
    "ln_2_bias": "ln_2.bias",
}

_DIRECT_MODEL_PARAMETER_NAMES = (
    "transformer.wte.weight",
    "transformer.wpe.weight",
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
    "lm_head.weight",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _tensor_bytes(tensor: Tensor) -> bytes:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _named_parameters_with_aliases(model: nn.Module) -> Tuple[Tuple[str, nn.Parameter], ...]:
    try:
        return tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:
        rows: List[Tuple[str, nn.Parameter]] = []

        def visit(module: nn.Module, prefix: str) -> None:
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    rows.append((prefix + name, parameter))
            for name, child in module._modules.items():
                if child is not None:
                    visit(child, prefix + name + ".")

        visit(model, "")
        return tuple(rows)


def _parameter_alias_groups(model: nn.Module) -> Tuple[Tuple[str, ...], ...]:
    names_by_id: Dict[int, List[str]] = {}
    for name, parameter in _named_parameters_with_aliases(model):
        names_by_id.setdefault(id(parameter), []).append(name)
    return tuple(
        tuple(sorted(names))
        for _, names in sorted(
            names_by_id.items(),
            key=lambda item: sorted(item[1])[0],
        )
    )


def _semantic_parameter_identity(name: str) -> Tuple[str, Optional[int]]:
    if name in ("transformer.wte.weight", "lm_head.weight"):
        return "token_embedding_and_output_head", None
    if name == "transformer.wpe.weight":
        return "positional_embedding", None
    if name == "transformer.ln_f.weight":
        return "final_layer_norm_weight", None
    if name == "transformer.ln_f.bias":
        return "final_layer_norm_bias", None
    matched = re.fullmatch(r"transformer\.h\.(\d+)\.(.+)", name)
    if matched is None:
        raise ValueError(f"unrecognised DENSE parameter name: {name!r}")
    layer_index = int(matched.group(1))
    suffix = matched.group(2)
    semantic_by_suffix = {
        "ln_1.weight": "ln_1_weight",
        "ln_1.bias": "ln_1_bias",
        "attn.c_attn.weight": "attention_input_weight",
        "attn.c_attn.bias": "attention_input_bias",
        "attn.c_proj.weight": ATTENTION_OUTPUT_WEIGHT,
        "attn.c_proj.bias": "attention_output_bias",
        "ln_2.weight": "ln_2_weight",
        "ln_2.bias": "ln_2_bias",
        "mlp.c_fc.weight": MLP_EXPANSION_WEIGHT,
        "mlp.c_fc.bias": "mlp_expansion_bias",
        "mlp.c_proj.weight": MLP_CONTRACTION_WEIGHT,
        "mlp.c_proj.bias": "mlp_contraction_bias",
    }
    if suffix not in semantic_by_suffix:
        raise ValueError(f"unrecognised DENSE block parameter name: {name!r}")
    return semantic_by_suffix[suffix], layer_index


def _capture_unique_parameters(
    model: nn.Module,
) -> Tuple[Dict[str, Tensor], Tuple[Dict[str, Any], ...]]:
    if not isinstance(model, GPT):
        raise TypeError("DENSE snapshot creation requires a DENSE GPT model")
    aliases_by_id: Dict[int, List[str]] = {}
    parameters_by_id: Dict[int, nn.Parameter] = {}
    ordered_ids: List[int] = []
    for name, parameter in _named_parameters_with_aliases(model):
        parameter_id = id(parameter)
        if parameter_id not in aliases_by_id:
            aliases_by_id[parameter_id] = []
            parameters_by_id[parameter_id] = parameter
            ordered_ids.append(parameter_id)
        aliases_by_id[parameter_id].append(name)

    tensors: Dict[str, Tensor] = {}
    manifest: List[Dict[str, Any]] = []
    for parameter_id in ordered_ids:
        aliases = tuple(sorted(aliases_by_id[parameter_id]))
        canonical_name = aliases[0]
        parameter = parameters_by_id[parameter_id]
        semantic_family, layer_index = _semantic_parameter_identity(canonical_name)
        tensor = parameter.detach().to(device="cpu").contiguous().clone()
        tensors[canonical_name] = tensor
        manifest.append(
            {
                "canonical_name": canonical_name,
                "aliases": list(aliases),
                "semantic_family": semantic_family,
                "layer_index": layer_index,
                "physical_shape": list(tensor.shape),
                "dtype": _tensor_dtype_name(tensor),
                "storage_order": "contiguous_c",
            }
        )
    return tensors, tuple(manifest)


def _tensor_payload_hash(
    tensors: Mapping[str, Tensor],
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    by_name = {str(item["canonical_name"]): item for item in manifest}
    if set(by_name) != set(tensors):
        raise ValueError(
            "tensor manifest and tensor payload names differ: "
            f"manifest={sorted(by_name)}, tensors={sorted(tensors)}"
        )
    for name in sorted(tensors):
        tensor = tensors[name]
        item = by_name[name]
        digest.update(_canonical_json(dict(item)))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def _rng_boundary_identifier(state: Mapping[str, Any]) -> str:
    numpy_state = state["numpy"]
    normalized = {
        "torch_cpu": state["torch_cpu"].tolist(),
        "torch_cuda": [value.tolist() for value in state.get("torch_cuda", ())],
        "numpy": {
            "algorithm": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "python": repr(state["python"]),
    }
    return _sha256_json(normalized)


def _physical_family_schema(n_embd: int, bias: bool) -> Tuple[Dict[str, Any], ...]:
    rows = [
        {"semantic_family": ATTENTION_QUERY_WEIGHT, "shape": [n_embd, n_embd]},
        {"semantic_family": ATTENTION_KEY_WEIGHT, "shape": [n_embd, n_embd]},
        {"semantic_family": ATTENTION_VALUE_WEIGHT, "shape": [n_embd, n_embd]},
        {"semantic_family": ATTENTION_OUTPUT_WEIGHT, "shape": [n_embd, n_embd]},
        {"semantic_family": MLP_EXPANSION_WEIGHT, "shape": [4 * n_embd, n_embd]},
        {"semantic_family": MLP_CONTRACTION_WEIGHT, "shape": [n_embd, 4 * n_embd]},
        {"semantic_family": "ln_1_weight", "shape": [n_embd]},
        {"semantic_family": "ln_2_weight", "shape": [n_embd]},
    ]
    if bias:
        rows.extend(
            (
                {"semantic_family": "attention_input_bias", "shape": [3 * n_embd]},
                {"semantic_family": "attention_output_bias", "shape": [n_embd]},
                {"semantic_family": "mlp_expansion_bias", "shape": [4 * n_embd]},
                {"semantic_family": "mlp_contraction_bias", "shape": [n_embd]},
                {"semantic_family": "ln_1_bias", "shape": [n_embd]},
                {"semantic_family": "ln_2_bias", "shape": [n_embd]},
            )
        )
    return tuple(rows)


def physical_compatibility_payload(config: Any) -> Dict[str, Any]:
    return {
        "physical_layer_count": int(config.n_layer),
        "attention_head_count": int(config.n_head),
        "model_width": int(config.n_embd),
        "context_capacity": int(config.block_size),
        "vocabulary_size": int(config.vocab_size),
        "embedding_schema": {
            "token_embedding": [int(config.vocab_size), int(config.n_embd)],
            "positional_embedding": [int(config.block_size), int(config.n_embd)],
        },
        "resolved_training_dtype": str(config.dtype),
        "bias_enabled": bool(config.bias),
        "mlp_hidden_multiplier": 4,
        "tying_aliases": [["lm_head.weight", "transformer.wte.weight"]],
        "materialised_family_schema": list(
            _physical_family_schema(int(config.n_embd), bool(config.bias))
        ),
    }


def _validate_tensor_manifest(payload: Mapping[str, Any]) -> None:
    tensors = payload.get("tensors")
    manifest = payload.get("tensor_manifest")
    if not isinstance(tensors, Mapping):
        raise ValueError("snapshot tensor payload is not a mapping")
    if not isinstance(manifest, (list, tuple)):
        raise ValueError("snapshot tensor manifest is not a sequence")
    aliases_seen: Dict[str, str] = {}
    for item in manifest:
        if not isinstance(item, Mapping):
            raise ValueError("snapshot tensor manifest entry is not a mapping")
        name = str(item.get("canonical_name", ""))
        aliases = item.get("aliases")
        if not name or not isinstance(aliases, (list, tuple)) or name not in aliases:
            raise ValueError(f"invalid tensor manifest identity for {name!r}")
        if item.get("storage_order") != "contiguous_c":
            raise ValueError(f"unsupported storage order for {name!r}: {item.get('storage_order')!r}")
        if name not in tensors or not isinstance(tensors[name], Tensor):
            raise ValueError(f"tensor manifest entry {name!r} has no tensor payload")
        tensor = tensors[name]
        expected_shape = tuple(int(value) for value in item.get("physical_shape", ()))
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"tensor {name!r} shape mismatch: manifest={expected_shape}, payload={tuple(tensor.shape)}"
            )
        expected_dtype = str(item.get("dtype"))
        actual_dtype = _tensor_dtype_name(tensor)
        if expected_dtype != actual_dtype:
            raise ValueError(
                f"tensor {name!r} dtype mismatch: manifest={expected_dtype!r}, payload={actual_dtype!r}"
            )
        for alias in aliases:
            alias_name = str(alias)
            if alias_name in aliases_seen:
                raise ValueError(
                    f"tensor alias {alias_name!r} occurs in both {aliases_seen[alias_name]!r} and {name!r}"
                )
            aliases_seen[alias_name] = name
    if set(tensors) != {str(item["canonical_name"]) for item in manifest}:
        raise ValueError("snapshot contains tensors not described exactly once by the manifest")


def validate_snapshot_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("snapshot_schema_version") != DENSE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "dense snapshot schema mismatch: "
            f"expected={DENSE_SNAPSHOT_SCHEMA_VERSION}, got={payload.get('snapshot_schema_version')!r}"
        )
    if payload.get("tensor_manifest_version") != DENSE_SNAPSHOT_TENSOR_MANIFEST_VERSION:
        raise ValueError(
            "dense snapshot tensor manifest version mismatch: "
            f"expected={DENSE_SNAPSHOT_TENSOR_MANIFEST_VERSION}, got={payload.get('tensor_manifest_version')!r}"
        )
    compatibility = payload.get("compatibility_payload")
    if not isinstance(compatibility, Mapping):
        raise ValueError("dense snapshot compatibility payload is missing")
    expected_compatibility_hash = _sha256_json(compatibility)
    if payload.get("compatibility_hash") != expected_compatibility_hash:
        raise ValueError(
            "dense snapshot compatibility hash mismatch: "
            f"stored={payload.get('compatibility_hash')!r}, calculated={expected_compatibility_hash!r}"
        )
    _validate_tensor_manifest(payload)
    expected_tensor_hash = _tensor_payload_hash(
        payload["tensors"],
        payload["tensor_manifest"],
    )
    if payload.get("tensor_payload_hash") != expected_tensor_hash:
        raise ValueError(
            "dense snapshot tensor payload hash mismatch: "
            f"stored={payload.get('tensor_payload_hash')!r}, calculated={expected_tensor_hash!r}"
        )
    boundary = payload.get("paired_rng_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("dense snapshot paired RNG boundary is missing")
    expected_boundary_identifier = _rng_boundary_identifier(boundary)
    if payload.get("paired_rng_boundary_identifier") != expected_boundary_identifier:
        raise ValueError(
            "dense snapshot paired RNG boundary identifier mismatch: "
            f"stored={payload.get('paired_rng_boundary_identifier')!r}, calculated={expected_boundary_identifier!r}"
        )


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("dense snapshot root payload is not a mapping")
    return value


def load_dense_initialisation_snapshot(path: Union[str, Path]) -> Mapping[str, Any]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"dense snapshot file is missing: {target}")
    payload = _safe_torch_load(target)
    validate_snapshot_payload(payload)
    return payload


def _sanitize_filename_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    normalized = normalized.strip("-_")
    return normalized or "host"


def dense_snapshot_filename(
    config: Any,
    compatibility_hash: str,
    *,
    created_at: Optional[datetime] = None,
    host_name: Optional[str] = None,
) -> str:
    timestamp = (created_at or datetime.now()).strftime("%y%m%d-%H%M")
    host = _sanitize_filename_component(
        host_name or getattr(config, "dense_snapshot_host_label", None) or socket.gethostname().split(".")[0]
    )
    dtype = str(config.dtype)
    if dtype not in _DTYPE_FILENAME_CODES:
        raise ValueError(f"unsupported snapshot dtype for filename: {dtype!r}")
    return (
        f"{timestamp}_{host}_L{int(config.n_layer)}_H{int(config.n_head)}_"
        f"D{int(config.n_embd)}_C{int(config.block_size)}_T{_DTYPE_FILENAME_CODES[dtype]}_"
        f"VS{int(config.vocab_size)}_BI{int(bool(config.bias))}_MM4_TE1_"
        f"IF{DENSE_SNAPSHOT_SCHEMA_VERSION}_CH{compatibility_hash[:8]}"
        f"{DENSE_SNAPSHOT_SUFFIX}"
    )


def _snapshot_payload(model: nn.Module, config: Any) -> Dict[str, Any]:
    tensors, manifest = _capture_unique_parameters(model)
    compatibility = physical_compatibility_payload(config)
    boundary = capture_rng_state()
    return {
        "snapshot_schema_version": DENSE_SNAPSHOT_SCHEMA_VERSION,
        "tensor_manifest_version": DENSE_SNAPSHOT_TENSOR_MANIFEST_VERSION,
        "compatibility_payload": compatibility,
        "compatibility_hash": _sha256_json(compatibility),
        "tensor_manifest": list(manifest),
        "tensors": tensors,
        "tensor_payload_hash": _tensor_payload_hash(tensors, manifest),
        "paired_rng_boundary": boundary,
        "paired_rng_boundary_identifier": _rng_boundary_identifier(boundary),
    }


def save_dense_initialisation_snapshot(
    model: nn.Module,
    config: Any,
    *,
    root: Optional[Path] = None,
    created_at: Optional[datetime] = None,
    host_name: Optional[str] = None,
) -> Tuple[Path, Mapping[str, Any]]:
    rng_before = capture_rng_state()
    rng_before_identifier = _rng_boundary_identifier(rng_before)
    payload = _snapshot_payload(model, config)
    validate_snapshot_payload(payload)
    target_root = repository_root() if root is None else Path(root)
    directory = target_root / DENSE_SNAPSHOT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    filename = dense_snapshot_filename(
        config,
        str(payload["compatibility_hash"]),
        created_at=created_at,
        host_name=host_name,
    )
    target = directory / filename
    if target.exists():
        raise FileExistsError(f"dense snapshot refuses to overwrite existing file: {target}")
    temporary = directory / ("." + filename + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"dense snapshot temporary file already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        reloaded = _safe_torch_load(temporary)
        validate_snapshot_payload(reloaded)
        rng_after_identifier = _rng_boundary_identifier(capture_rng_state())
        if rng_after_identifier != rng_before_identifier:
            restore_rng_state(rng_before)
            raise RuntimeError("dense snapshot creation altered the training RNG boundary")
        if target.exists():
            raise FileExistsError(f"dense snapshot refuses to overwrite existing file: {target}")
        temporary.replace(target)
        target.chmod(0o444)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target.resolve(), payload


def _manifest_alias_lookup(payload: Mapping[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for item in payload["tensor_manifest"]:
        canonical_name = str(item["canonical_name"])
        for alias in item["aliases"]:
            lookup[str(alias)] = canonical_name
    return lookup


def _snapshot_tensor(
    payload: Mapping[str, Any],
    alias_lookup: Mapping[str, str],
    name: str,
) -> Tensor:
    if name not in alias_lookup:
        raise ValueError(f"dense snapshot is missing required parameter {name!r}")
    return payload["tensors"][alias_lookup[name]]


def validate_physical_compatibility(payload: Mapping[str, Any], config: Any) -> None:
    saved = payload["compatibility_payload"]
    requested = physical_compatibility_payload(config)
    mismatches = []
    for name in (
        "physical_layer_count",
        "attention_head_count",
        "model_width",
        "context_capacity",
        "vocabulary_size",
        "embedding_schema",
        "bias_enabled",
        "mlp_hidden_multiplier",
        "tying_aliases",
        "materialised_family_schema",
    ):
        if saved.get(name) != requested.get(name):
            mismatches.append(
                f"{name}: saved={saved.get(name)!r}, requested={requested.get(name)!r}, class=physical_source_structure"
            )
    saved_dtype = saved.get("resolved_training_dtype")
    requested_dtype = requested.get("resolved_training_dtype")
    if saved_dtype not in _SUPPORTED_NUMERIC_DTYPES:
        mismatches.append(
            f"resolved_training_dtype: saved={saved_dtype!r}, requested={requested_dtype!r}, class=unsupported_saved_numeric_conversion"
        )
    if requested_dtype not in _SUPPORTED_NUMERIC_DTYPES:
        mismatches.append(
            f"resolved_training_dtype: saved={saved_dtype!r}, requested={requested_dtype!r}, class=unsupported_target_numeric_conversion"
        )
    if mismatches:
        raise ValueError("incompatible dense snapshot: " + "; ".join(mismatches))


def _dense_matrix_stack(
    payload: Mapping[str, Any],
    alias_lookup: Mapping[str, str],
    family: str,
    *,
    n_layer: int,
    n_embd: int,
) -> Tensor:
    suffix, qkv_role = _DENSE_MATRIX_SPECS[family]
    rows = []
    for layer_index in range(n_layer):
        name = f"transformer.h.{layer_index}.{suffix}"
        tensor = _snapshot_tensor(payload, alias_lookup, name).to(dtype=torch.float32)
        if qkv_role is not None:
            start = int(qkv_role) * n_embd
            tensor = tensor[start : start + n_embd]
        rows.append(tensor)
    return torch.stack(rows, dim=0)


def _is_mapped_dense_parameter(name: str) -> bool:
    matched = re.fullmatch(r"transformer\.h\.\d+\.(.+)", name)
    if matched is None:
        return False
    return matched.group(1) in {
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    }


def _fit_chebyshev_family(
    source: Tensor,
    basis: Tensor,
    family: str,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    if source.ndim != 3:
        raise ValueError(f"mapped family {family} must have shape L x rows x columns; got {tuple(source.shape)}")
    if basis.ndim != 2 or basis.shape[0] != source.shape[0]:
        raise ValueError(
            f"Chebyshev basis/source mismatch for {family}: basis={tuple(basis.shape)}, source={tuple(source.shape)}"
        )
    source_float = source.to(device="cpu", dtype=torch.float32).contiguous()
    basis_float = basis.to(device="cpu", dtype=torch.float32).contiguous()
    flat = source_float.reshape(source_float.shape[0], -1)
    inverse = torch.linalg.pinv(basis_float)
    coefficient_flat = inverse @ flat
    reconstruction_flat = basis_float @ coefficient_flat
    rows, columns = source_float.shape[1:]
    coefficients = coefficient_flat.reshape(basis_float.shape[1], rows, columns).permute(1, 2, 0).contiguous()
    reconstruction = reconstruction_flat.reshape_as(source_float)
    residual = source_float - reconstruction
    source_norm = torch.linalg.vector_norm(source_float)
    coefficient_norm = torch.linalg.vector_norm(coefficients)
    reconstruction_norm = torch.linalg.vector_norm(reconstruction)
    error = torch.linalg.vector_norm(residual)
    relative_error = error / torch.clamp(source_norm, min=torch.finfo(torch.float32).tiny)
    optimality = torch.linalg.vector_norm(basis_float.transpose(0, 1) @ residual.reshape(source.shape[0], -1))
    values = (
        source_norm,
        coefficient_norm,
        reconstruction_norm,
        error,
        relative_error,
        optimality,
    )
    if not all(bool(torch.isfinite(value).item()) for value in values):
        raise FloatingPointError(f"non-finite Chebyshev diagnostic for {family}")
    if not bool(torch.isfinite(coefficients).all().item()):
        raise FloatingPointError(f"non-finite Chebyshev coefficients for {family}")
    if not bool(torch.isfinite(reconstruction).all().item()):
        raise FloatingPointError(f"non-finite Chebyshev reconstruction for {family}")
    diagnostics = {
        "family": family,
        "source_norm": float(source_norm.item()),
        "coefficient_norm": float(coefficient_norm.item()),
        "reconstruction_norm": float(reconstruction_norm.item()),
        "absolute_reconstruction_error": float(error.item()),
        "relative_reconstruction_error": float(relative_error.item()),
        "residual_optimality_norm": float(optimality.item()),
    }
    return coefficients, reconstruction, diagnostics


def _physical_state_hash(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().to(device="cpu").contiguous()
        identity = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": _tensor_dtype_name(tensor),
        }
        digest.update(_canonical_json(identity))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def _target_parameter_lookup(model: nn.Module) -> Tuple[Dict[str, nn.Parameter], Dict[int, Tuple[str, ...]]]:
    by_name: Dict[str, nn.Parameter] = {}
    names_by_id: Dict[int, List[str]] = {}
    for name, parameter in _named_parameters_with_aliases(model):
        by_name[name] = parameter
        names_by_id.setdefault(id(parameter), []).append(name)
    aliases = {parameter_id: tuple(sorted(names)) for parameter_id, names in names_by_id.items()}
    return by_name, aliases


def _set_prepared_parameter(
    prepared: Dict[int, Tensor],
    parameters_by_name: Mapping[str, nn.Parameter],
    name: str,
    value: Tensor,
) -> None:
    if name not in parameters_by_name:
        raise ValueError(f"consuming model is missing required parameter {name!r}")
    parameter = parameters_by_name[name]
    converted = value.detach().to(device="cpu", dtype=parameter.dtype).contiguous()
    if tuple(converted.shape) != tuple(parameter.shape):
        raise ValueError(
            f"parameter {name!r} shape mismatch: prepared={tuple(converted.shape)}, target={tuple(parameter.shape)}"
        )
    existing = prepared.get(id(parameter))
    if existing is not None and not torch.equal(existing, converted):
        raise ValueError(f"conflicting prepared values for aliased parameter {name!r}")
    prepared[id(parameter)] = converted


def _commit_prepared_parameters(
    model: nn.Module,
    prepared: Mapping[int, Tensor],
    aliases_by_id: Mapping[int, Tuple[str, ...]],
) -> None:
    missing = [aliases for parameter_id, aliases in aliases_by_id.items() if parameter_id not in prepared]
    if missing:
        raise ValueError(f"dense snapshot did not prepare every consuming parameter: {missing}")
    parameters_by_id = {id(parameter): parameter for parameter in model.parameters()}
    with torch.no_grad():
        for parameter_id, value in prepared.items():
            parameters_by_id[parameter_id].copy_(
                value.to(
                    device=parameters_by_id[parameter_id].device,
                    dtype=parameters_by_id[parameter_id].dtype,
                )
            )


def _mapping_basis(model: nn.Module, config: Any, order: int) -> Tensor:
    if isinstance(model, SheetGPT):
        if not isinstance(model.trajectory, DepthTrajectory):
            raise ValueError("C Compact Run snapshot initialisation requires the pure DEPTH trajectory")
        return model.trajectory.depth_basis.detach().to(device="cpu", dtype=torch.float32)
    return build_stabilized_basis(
        int(config.n_layer),
        order,
        runtime_dtype=torch.float32,
        device="cpu",
        version=str(config.dense_snapshot_chebyshev_version or BASIS_VERSION),
        basis_family=BASIS_FAMILY_CHEBYSHEV,
    )


def _validate_mapping_target(model: nn.Module, config: Any) -> Tuple[str, int, str]:
    if isinstance(model, GPT) and not isinstance(model, SheetGPT):
        role = DENSE_SNAPSHOT_ROLE_B
        order = config.dense_snapshot_chebyshev_order
        version = config.dense_snapshot_chebyshev_version
        if order is None or version is None:
            raise ValueError("B Compressor-baselined DENSE requires a complete Chebyshev DEPTH mapping")
    elif isinstance(model, SheetGPT):
        role = DENSE_SNAPSHOT_ROLE_C
        if not isinstance(model.trajectory, DepthTrajectory):
            raise ValueError("C Compact Run v1 supports only pure DEPTH")
        if model.trajectory.basis_family != BASIS_FAMILY_CHEBYSHEV:
            raise ValueError(
                "DENSE snapshot v1 supports only the Chebyshev compressor; "
                f"got {model.trajectory.basis_family!r}"
            )
        if model.trajectory.depth_compress_layer_norm_and_bias:
            raise ValueError("DENSE snapshot v1 keeps LayerNorm and bias outside DEPTH compression")
        if model.trajectory.plastic_enabled:
            raise ValueError("DENSE snapshot v1 does not support PLASTIC DEPTH sampling geometry")
        order = int(model.trajectory.config.depth_order)
        version = str(model.trajectory.basis_version)
    else:
        raise TypeError("dense snapshot initialisation requires a DENSE or pure DEPTH model")
    order = int(order)
    version = str(version)
    if version != BASIS_VERSION:
        raise ValueError(
            "DENSE snapshot v1 requires the current QR-stabilised Chebyshev version; "
            f"expected={BASIS_VERSION!r}, got={version!r}"
        )
    if order < 1 or order > int(config.n_layer):
        raise ValueError(
            f"DENSE snapshot Chebyshev order P must satisfy 1 <= P <= L; got P={order}, L={config.n_layer}"
        )
    return role, order, version


def initialise_model_from_dense_snapshot(
    model: nn.Module,
    config: Any,
    snapshot_path: Union[str, Path],
) -> Dict[str, Any]:
    payload = load_dense_initialisation_snapshot(snapshot_path)
    validate_physical_compatibility(payload, config)
    role, order, version = _validate_mapping_target(model, config)
    alias_lookup = _manifest_alias_lookup(payload)
    parameters_by_name, aliases_by_id = _target_parameter_lookup(model)
    prepared: Dict[int, Tensor] = {}
    physical_state: Dict[str, Tensor] = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, tensor in payload["tensors"].items()
    }

    basis = _mapping_basis(model, config, order)
    rank = int(torch.linalg.matrix_rank(basis.to(dtype=torch.float64)).item())
    if rank < order:
        raise ValueError(f"rank-deficient Chebyshev basis: rank={rank}, configured_order={order}")
    if not bool(torch.isfinite(basis).all().item()):
        raise FloatingPointError("non-finite Chebyshev basis")

    coefficients_by_family: Dict[str, Tensor] = {}
    reconstruction_by_family: Dict[str, Tensor] = {}
    diagnostics = []
    for family in _MAPPED_MATRIX_FAMILIES:
        source = _dense_matrix_stack(
            payload,
            alias_lookup,
            family,
            n_layer=int(config.n_layer),
            n_embd=int(config.n_embd),
        )
        coefficients, reconstruction, family_diagnostics = _fit_chebyshev_family(
            source,
            basis,
            family,
        )
        coefficients_by_family[family] = coefficients
        reconstruction_by_family[family] = reconstruction
        diagnostics.append(family_diagnostics)

    if order == int(config.n_layer):
        maximum_identity_error = max(
            float(torch.max(torch.abs(
                _dense_matrix_stack(
                    payload,
                    alias_lookup,
                    family,
                    n_layer=int(config.n_layer),
                    n_embd=int(config.n_embd),
                ).to(dtype=torch.float32) - reconstruction_by_family[family]
            )).item())
            for family in _MAPPED_MATRIX_FAMILIES
        )
        if maximum_identity_error > 2.0e-5:
            raise ValueError(
                "P = L Chebyshev identity reconstruction exceeded tolerance: "
                f"maximum_absolute_error={maximum_identity_error:.6e}, tolerance=2.000000e-05"
            )

    if role == DENSE_SNAPSHOT_ROLE_B:
        for target_name in parameters_by_name:
            if target_name in alias_lookup and not _is_mapped_dense_parameter(target_name):
                _set_prepared_parameter(
                    prepared,
                    parameters_by_name,
                    target_name,
                    _snapshot_tensor(payload, alias_lookup, target_name),
                )
        for layer_index in range(int(config.n_layer)):
            qkv = torch.cat(
                (
                    reconstruction_by_family[ATTENTION_QUERY_WEIGHT][layer_index],
                    reconstruction_by_family[ATTENTION_KEY_WEIGHT][layer_index],
                    reconstruction_by_family[ATTENTION_VALUE_WEIGHT][layer_index],
                ),
                dim=0,
            )
            qkv_name = f"transformer.h.{layer_index}.attn.c_attn.weight"
            _set_prepared_parameter(prepared, parameters_by_name, qkv_name, qkv)
            qkv_canonical = alias_lookup[qkv_name]
            physical_state[qkv_canonical] = qkv
            for family in (
                ATTENTION_OUTPUT_WEIGHT,
                MLP_EXPANSION_WEIGHT,
                MLP_CONTRACTION_WEIGHT,
            ):
                suffix, _ = _DENSE_MATRIX_SPECS[family]
                name = f"transformer.h.{layer_index}.{suffix}"
                value = reconstruction_by_family[family][layer_index]
                _set_prepared_parameter(prepared, parameters_by_name, name, value)
                physical_state[alias_lookup[name]] = value
    else:
        for name in _DIRECT_MODEL_PARAMETER_NAMES:
            if name in parameters_by_name and name in alias_lookup:
                _set_prepared_parameter(
                    prepared,
                    parameters_by_name,
                    name,
                    _snapshot_tensor(payload, alias_lookup, name),
                )
        for family, coefficients in coefficients_by_family.items():
            _set_prepared_parameter(
                prepared,
                parameters_by_name,
                f"trajectory.coefficients.{family}",
                coefficients,
            )
        trajectory = model.trajectory
        for item in trajectory.metadata:
            if item.semantic_type == "matrix":
                continue
            if item.name not in _DENSE_VECTOR_SUFFIXES:
                raise ValueError(f"unsupported direct-copy DEPTH family {item.name!r}")
            suffix = _DENSE_VECTOR_SUFFIXES[item.name]
            stacked = torch.stack(
                [
                    _snapshot_tensor(
                        payload,
                        alias_lookup,
                        f"transformer.h.{layer_index}.{suffix}",
                    ).to(dtype=torch.float32)
                    for layer_index in range(int(config.n_layer))
                ],
                dim=0,
            ).unsqueeze(1)
            _set_prepared_parameter(
                prepared,
                parameters_by_name,
                f"trajectory.coefficients.{item.name}",
                stacked,
            )
        for layer_index in range(int(config.n_layer)):
            qkv = torch.cat(
                (
                    reconstruction_by_family[ATTENTION_QUERY_WEIGHT][layer_index],
                    reconstruction_by_family[ATTENTION_KEY_WEIGHT][layer_index],
                    reconstruction_by_family[ATTENTION_VALUE_WEIGHT][layer_index],
                ),
                dim=0,
            )
            qkv_name = f"transformer.h.{layer_index}.attn.c_attn.weight"
            physical_state[alias_lookup[qkv_name]] = qkv
            for family in (
                ATTENTION_OUTPUT_WEIGHT,
                MLP_EXPANSION_WEIGHT,
                MLP_CONTRACTION_WEIGHT,
            ):
                suffix, _ = _DENSE_MATRIX_SPECS[family]
                name = f"transformer.h.{layer_index}.{suffix}"
                physical_state[alias_lookup[name]] = reconstruction_by_family[family][layer_index]

    mapping_identity = {
        "mapping_algorithm_version": DENSE_SNAPSHOT_MAPPING_ALGORITHM_VERSION,
        "chebyshev_version": version,
        "order": order,
        "physical_layer_count": int(config.n_layer),
        "basis_sha256": basis_sha256(basis),
        "family_plan": list(_MAPPED_MATRIX_FAMILIES),
        "target_dtype": str(config.dtype),
    }
    mapping_fingerprint = _sha256_json(mapping_identity)
    step_zero_manifest_identifier = _physical_state_hash(physical_state)
    metadata = {
        "lifecycle_role": role,
        "snapshot_path": str(Path(snapshot_path).expanduser().resolve()),
        "snapshot_schema_version": DENSE_SNAPSHOT_SCHEMA_VERSION,
        "compatibility_hash": str(payload["compatibility_hash"]),
        "tensor_payload_hash": str(payload["tensor_payload_hash"]),
        "paired_rng_boundary_identifier": str(payload["paired_rng_boundary_identifier"]),
        "chebyshev_version": version,
        "chebyshev_order": order,
        "family_plan": list(_MAPPED_MATRIX_FAMILIES),
        "mapping_algorithm_version": DENSE_SNAPSHOT_MAPPING_ALGORITHM_VERSION,
        "mapping_fingerprint": mapping_fingerprint,
        "basis_rank": rank,
        "basis_sha256": mapping_identity["basis_sha256"],
        "numerical_diagnostics": diagnostics,
        "step_zero_manifest_identifier": step_zero_manifest_identifier,
        "effective_initialisation": "dense_snapshot",
        "retained_state": "independent_W_hat" if role == DENSE_SNAPSHOT_ROLE_B else "trainable_C_star",
        "direct_copy_families": [
            "token_embedding_and_output_head",
            "positional_embedding",
            "final_layer_norm",
            "layer_norms",
            "biases",
        ],
    }

    original_parameters = {
        id(parameter): parameter.detach().to(device="cpu").contiguous().clone()
        for parameter in model.parameters()
    }
    try:
        _commit_prepared_parameters(model, prepared, aliases_by_id)
        if role == DENSE_SNAPSHOT_ROLE_C:
            with torch.no_grad():
                for family in _MAPPED_MATRIX_FAMILIES:
                    materialized = torch.stack(
                        [
                            model.trajectory.materialize(family, layer_index).detach().to(device="cpu", dtype=torch.float32)
                            for layer_index in range(int(config.n_layer))
                        ],
                        dim=0,
                    )
                    expected = reconstruction_by_family[family]
                    maximum_error = float(torch.max(torch.abs(materialized - expected)).item())
                    if maximum_error > 2.0e-5:
                        raise RuntimeError(
                            f"production DEPTH materialisation disagrees for {family}: "
                            f"maximum_absolute_error={maximum_error:.6e}"
                        )
        restore_rng_state(payload["paired_rng_boundary"])
    except BaseException:
        _commit_prepared_parameters(model, original_parameters, aliases_by_id)
        raise
    return metadata


def _saved_snapshot_metadata(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "lifecycle_role": DENSE_SNAPSHOT_ROLE_A,
        "snapshot_path": str(path),
        "snapshot_schema_version": DENSE_SNAPSHOT_SCHEMA_VERSION,
        "compatibility_hash": str(payload["compatibility_hash"]),
        "tensor_payload_hash": str(payload["tensor_payload_hash"]),
        "paired_rng_boundary_identifier": str(payload["paired_rng_boundary_identifier"]),
        "effective_initialisation": "ordinary_dense_initialisation",
        "retained_state": "independent_W_0",
    }


def print_dense_snapshot_startup(metadata: Mapping[str, Any]) -> None:
    print("DENSE Snapshot Baselining", flush=True)
    print(f"  lifecycle role:              {metadata['lifecycle_role']}", flush=True)
    print(f"  snapshot path:               {metadata['snapshot_path']}", flush=True)
    print(f"  snapshot schema:             {metadata['snapshot_schema_version']}", flush=True)
    print(f"  compatibility hash:          {str(metadata['compatibility_hash'])[:12]}", flush=True)
    print(f"  tensor payload hash:         {str(metadata['tensor_payload_hash'])[:12]}", flush=True)
    print(f"  paired RNG boundary:         {str(metadata['paired_rng_boundary_identifier'])[:12]}", flush=True)
    print(f"  effective initialisation:    {metadata['effective_initialisation']}", flush=True)
    if "mapping_fingerprint" in metadata:
        print(f"  Chebyshev mapping:           {metadata['chebyshev_version']}  P={metadata['chebyshev_order']}", flush=True)
        print(f"  mapping fingerprint:         {str(metadata['mapping_fingerprint'])[:12]}", flush=True)
        print(f"  step-zero manifest:          {str(metadata['step_zero_manifest_identifier'])[:12]}", flush=True)
        for row in metadata["numerical_diagnostics"]:
            print(
                "  "
                f"{row['family']}: source={row['source_norm']:.6e}  "
                f"coeff={row['coefficient_norm']:.6e}  "
                f"error={row['absolute_reconstruction_error']:.6e}  "
                f"relative={row['relative_reconstruction_error']:.6e}  "
                f"optimality={row['residual_optimality_norm']:.6e}",
                flush=True,
            )
    print(flush=True)


def apply_dense_snapshot_startup(
    model: nn.Module,
    config: Any,
    distributed: Any,
) -> Optional[Dict[str, Any]]:
    save_requested = bool(config.save_dense_initialisation_snapshot)
    snapshot_path = config.initialise_from_dense_snapshot
    if not save_requested and snapshot_path is None:
        return None
    metadata: Optional[Dict[str, Any]] = None
    if save_requested:
        if distributed.is_primary:
            path, payload = save_dense_initialisation_snapshot(model, config)
            metadata = _saved_snapshot_metadata(path, payload)
        gathered = distributed.all_gather_object(metadata)
        metadata = gathered[0]
        if metadata is None:
            raise RuntimeError("primary rank did not publish DENSE snapshot metadata")
        distributed.barrier()
    else:
        metadata = initialise_model_from_dense_snapshot(model, config, str(snapshot_path))
        distributed.assert_identical_object(metadata, "DENSE snapshot startup metadata")
    if distributed.is_primary:
        print_dense_snapshot_startup(metadata)
    return dict(metadata)


__all__ = [
    "DENSE_SNAPSHOT_DIRECTORY",
    "DENSE_SNAPSHOT_MAPPING_ALGORITHM_VERSION",
    "DENSE_SNAPSHOT_ROLE_A",
    "DENSE_SNAPSHOT_ROLE_B",
    "DENSE_SNAPSHOT_ROLE_C",
    "DENSE_SNAPSHOT_ROLES",
    "DENSE_SNAPSHOT_SCHEMA_VERSION",
    "apply_dense_snapshot_startup",
    "dense_snapshot_filename",
    "initialise_model_from_dense_snapshot",
    "load_dense_initialisation_snapshot",
    "physical_compatibility_payload",
    "print_dense_snapshot_startup",
    "save_dense_initialisation_snapshot",
    "validate_physical_compatibility",
    "validate_snapshot_payload",
]
# ^^^ THOG
