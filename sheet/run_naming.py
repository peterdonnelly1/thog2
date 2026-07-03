# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ARCHITECTURE_PREFIXES = {
    "dense": "DENSE2",
    "thog2_sheet": "SHEET",
}
DEFAULT_COMPONENT_LIMIT = 240
FILESYSTEM_COMPONENT_LIMIT = 255
_TRUNCATION_MARKER = "__TRUNC_"
_RUN_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]*$")
_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_DATASET_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_component(value: str, *, uppercase: bool = False) -> str:
    normalized = _COMPONENT_PATTERN.sub("_", value.strip()).strip("_")
    normalized = normalized.upper() if uppercase else normalized
    if not normalized:
        raise ValueError("name component must contain an alphanumeric character")
    return normalized


def validate_run_name(run_name: str) -> str:
    normalized = run_name.strip().upper()
    if not _RUN_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "run name must begin with a letter and contain only A-Z, 0-9, '_' or '-'"
        )
    return normalized


def dataset_label(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "openwebtext":
        return "owt"
    label = _DATASET_PATTERN.sub("_", normalized).strip("_")
    if not label:
        raise ValueError("dataset name must contain an alphanumeric character")
    return label


def architecture_prefix(model_type: str) -> str:
    try:
        return ARCHITECTURE_PREFIXES[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported THOG2 model type: {model_type!r}") from error


def _stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def truncate_component(value: str, *, max_length: int = DEFAULT_COMPONENT_LIMIT) -> str:
    """Bound one filesystem component while preserving identity with a digest."""

    if max_length < 48:
        raise ValueError("max_length must be at least 48")
    if len(value) <= max_length:
        return value
    marker = f"{_TRUNCATION_MARKER}{_stable_digest(value)}__"
    remaining = max_length - len(marker)
    tail_length = min(56, max(16, remaining // 3))
    head_length = remaining - tail_length
    if head_length < 16:
        raise ValueError("max_length is too small for stable truncation")
    return f"{value[:head_length]}{marker}{value[-tail_length:]}"


def bounded_filename(
    stem: str,
    suffix: str,
    *,
    max_length: int = FILESYSTEM_COMPONENT_LIMIT,
) -> str:
    if not suffix:
        raise ValueError("suffix must be non-empty")
    if len(suffix) >= max_length:
        raise ValueError("suffix leaves no room for a filename stem")
    bounded_stem = truncate_component(
        stem,
        max_length=max_length - len(suffix),
    )
    return bounded_stem + suffix


def build_artifact_name(
    *,
    model_type: str,
    host_label: str,
    run_name: str,
    dataset_name: str,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_iters: int,
    depth_order: int | None = None,
    base_row_order: int | None = None,
    suffix: str | None = None,
    max_length: int = DEFAULT_COMPONENT_LIMIT,
) -> str:
    integer_values = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_iters": max_iters,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    prefix = architecture_prefix(model_type)
    architecture = f"l_{n_layer}_h_{n_head}_d_{n_embd}_ctx_{block_size}"
    if model_type == "thog2_sheet":
        if depth_order is None or base_row_order is None:
            raise ValueError("SHEET naming requires depth_order and base_row_order")
        if not 1 <= depth_order <= n_layer:
            raise ValueError("depth_order must be in [1, n_layer]")
        if not 1 <= base_row_order <= n_embd:
            raise ValueError("base_row_order must be in [1, n_embd]")
        architecture += f"_p_{depth_order}_q_{base_row_order}"
    elif depth_order is not None or base_row_order is not None:
        raise ValueError("DENSE2 naming must not include SHEET orders")

    name = (
        f"{prefix}_{normalize_component(host_label)}__"
        f"{validate_run_name(run_name)}__{dataset_label(dataset_name)}__"
        f"{architecture}__b_{batch_size}_ga_{gradient_accumulation_steps}__"
        f"steps_{max_iters}"
    )
    if suffix:
        name += f"__{normalize_component(suffix, uppercase=True)}"
    return truncate_component(name, max_length=max_length)


def artifact_paths(
    artifact_name: str,
    *,
    checkpoint_root: Path = Path("checkpoints"),
    log_root: Path = Path("logs"),
    result_root: Path = Path("results"),
    log_timestamp: str | None = None,
) -> dict[str, Path]:
    checkpoint_dir = checkpoint_root / artifact_name
    log_dir = log_root / artifact_name
    result_dir = result_root / artifact_name
    log_suffix = (
        f"_train_{normalize_component(log_timestamp)}.log"
        if log_timestamp
        else "_train.log"
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_dir / "ckpt.pt",
        "log_dir": log_dir,
        "log_path": log_dir / bounded_filename(artifact_name, log_suffix),
        "result_dir": result_dir,
        "result_path": result_dir / "result.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical THOG2 artifact name")
    parser.add_argument("--model-type", choices=("dense", "thog2_sheet"), required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--max-iters", type=int, required=True)
    parser.add_argument("--depth-order", type=int)
    parser.add_argument("--base-row-order", type=int)
    parser.add_argument("--suffix")
    parser.add_argument("--max-length", type=int, default=DEFAULT_COMPONENT_LIMIT)
    arguments = parser.parse_args()
    print(
        build_artifact_name(
            model_type=arguments.model_type,
            host_label=arguments.host_label,
            run_name=arguments.run_name,
            dataset_name=arguments.dataset,
            n_layer=arguments.n_layer,
            n_head=arguments.n_head,
            n_embd=arguments.n_embd,
            block_size=arguments.block_size,
            batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            max_iters=arguments.max_iters,
            depth_order=arguments.depth_order,
            base_row_order=arguments.base_row_order,
            suffix=arguments.suffix,
            max_length=arguments.max_length,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
