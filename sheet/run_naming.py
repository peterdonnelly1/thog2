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
_TIMESTAMP_PATTERN = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})$")
_COMPACT_TIMESTAMP_PATTERN = re.compile(r"^\d{6}_\d{4}$")


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


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def compact_log_timestamp(log_timestamp: str) -> str:
    """Return YYMMDD_HHMM for timestamped local run directories."""

    normalized = normalize_component(log_timestamp)
    if _COMPACT_TIMESTAMP_PATTERN.fullmatch(normalized):
        return normalized
    match = _TIMESTAMP_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized
    return (
        f"{match.group('year')[-2:]}{match.group('month')}{match.group('day')}_"
        f"{match.group('hour')}{match.group('minute')}"
    )


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
    checkpoint_interval: int = 0,
    warmup_iters: int = 0,
    checkpoint_segment_size: int = 12,
    depth_order: int | None = None,
    base_row_order: int | None = None,
    suffix: str | None = None,
    max_length: int = DEFAULT_COMPONENT_LIMIT,
) -> str:
    for name, value in {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_iters": max_iters,
        "checkpoint_segment_size": checkpoint_segment_size,
    }.items():
        _require_positive_integer(name, value)
    for name, value in {
        "checkpoint_interval": checkpoint_interval,
        "warmup_iters": warmup_iters,
    }.items():
        _require_nonnegative_integer(name, value)
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    prefix = architecture_prefix(model_type)
    fields = [
        f"n_{max_iters}",
        f"b_{batch_size}",
        f"d_{dataset_label(dataset_name)}",
        f"w_{warmup_iters}",
        f"k_{checkpoint_interval}",
        f"A_{gradient_accumulation_steps}",
        f"L_{n_layer}",
        f"H_{n_head}",
        f"D_{n_embd}",
        f"C_{block_size}",
    ]
    if model_type == "thog2_sheet":
        if depth_order is None or base_row_order is None:
            raise ValueError("SHEET naming requires depth_order and base_row_order")
        if not 1 <= depth_order <= n_layer:
            raise ValueError("depth_order must be in [1, n_layer]")
        if not 1 <= base_row_order <= n_embd:
            raise ValueError("base_row_order must be in [1, n_embd]")
        fields.extend([f"P_{depth_order}", f"Q_{base_row_order}"])
    elif depth_order is not None or base_row_order is not None:
        raise ValueError("DENSE2 naming must not include SHEET orders")
    fields.append(f"S_{checkpoint_segment_size}")

    name = (
        f"{prefix}_{normalize_component(host_label)}__"
        f"{validate_run_name(run_name)}__"
        f"{'_'.join(fields)}"
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
    del result_root  # new runs keep inspectable result artifacts under logs/<run>/
    checkpoint_dir = checkpoint_root / artifact_name
    run_dir_name = (
        f"{compact_log_timestamp(log_timestamp)}_{artifact_name}"
        if log_timestamp
        else artifact_name
    )
    log_dir = log_root / truncate_component(run_dir_name)
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_dir / "ckpt.pt",
        "log_dir": log_dir,
        "log_path": log_dir / "train.log",
        "result_dir": log_dir,
        "result_path": log_dir / "result.json",
    }


def architecture_run_name(config) -> str:
    """Return the frozen Stage 4-6 selector name for legacy evidence paths."""

    if config.model_type == "thog2_sheet":
        return (
            f"THOG2_SHEET_L{config.n_layer}_H{config.n_head}_D{config.n_embd}"
            f"_C{config.block_size}_P{config.depth_order}_Q{config.base_row_order}"
        )
    return (
        f"DENSE_L{config.n_layer}_H{config.n_head}_D{config.n_embd}"
        f"_C{config.block_size}"
    )


def architecture_output_directory(config, root: str | Path) -> Path:
    """Retain the frozen Stage 4-6 output-directory contract."""

    return Path(root) / architecture_run_name(config)


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
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--checkpoint-segment-size", type=int, default=12)
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
            checkpoint_interval=arguments.checkpoint_interval,
            warmup_iters=arguments.warmup_iters,
            checkpoint_segment_size=arguments.checkpoint_segment_size,
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
