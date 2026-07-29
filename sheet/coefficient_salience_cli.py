# vvv THOG
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .checkpoints import load_payload, strip_compiled_prefix
from .coefficient_salience import (
    CoefficientBank,
    coefficient_rms_by_order,
    concentration_statistics,
    discover_coefficient_banks,
    gradient_diagnostics_by_order,
    select_banks,
    spearman_rho,
    zero_order_temporarily,
)
from .training_config import TrainingConfig
from .training_model_factory import build_training_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure post-hoc THOG coefficient salience by exact order ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="checkpoint file, checkpoint directory, or artifact name under checkpoints/",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/openwebtext"),
        help="dataset directory containing val.bin",
    )
    parser.add_argument(
        "--scope",
        default="depth",
        help="depth aggregate, FAMILY:AXIS, or a comma-separated list of exact scopes",
    )
    parser.add_argument(
        "--list-scopes",
        action="store_true",
        help="load the checkpoint, list available coefficient-order scopes, and exit",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="evaluation autocast dtype",
    )
    parser.add_argument(
        "--eval-shards",
        type=int,
        default=4,
        help="fixed validation batches used as independent salience shards",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="sequences per validation shard")
    parser.add_argument(
        "--gradient-shards",
        type=int,
        default=1,
        help="leading validation shards used for gradient diagnostics; 0 disables them",
    )
    parser.add_argument(
        "--no-gradients",
        action="store_true",
        help="skip secondary gradient diagnostics; useful for very large checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="validation sampling seed; defaults deterministically from the checkpoint data seed",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="output path without extension; default is <checkpoint_dir>/salience/<scope>_u<updates>",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing CSV/JSON output files")
    return parser


def _resolve_checkpoint_path(requested: Path) -> Path:
    candidates = [requested]
    if requested.is_dir():
        candidates.append(requested / "ckpt.pt")
    candidates.append(Path("checkpoints") / requested / "ckpt.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"checkpoint is missing; tried: {attempted}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(requested: str, checkpoint_dtype: str, device: torch.device) -> str:
    if requested != "auto":
        resolved = requested
    elif device.type == "cuda":
        resolved = checkpoint_dtype
    else:
        resolved = "float32"
    if device.type == "cpu" and resolved == "float16":
        raise ValueError("float16 salience evaluation is not supported on CPU; use --dtype float32 or bfloat16")
    return resolved


def _autocast_context(device: torch.device, dtype: str):
    if dtype == "float32":
        return nullcontext()
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch_dtype)


def _load_model(checkpoint_path: Path, device: torch.device, requested_dtype: str):
    payload = load_payload(checkpoint_path, map_location="cpu")
    trainer_config_payload = payload.get("trainer_config")
    if not isinstance(trainer_config_payload, dict):
        raise ValueError("checkpoint has no THOG trainer_config; legacy nanoGPT dense checkpoints are not salience inputs")
    stored_config = TrainingConfig(**trainer_config_payload)
    if stored_config.model_type != "thog2_sheet":
        raise ValueError(f"coefficient salience requires model_type='thog2_sheet'; got {stored_config.model_type!r}")

    evaluation_dtype = _resolve_dtype(requested_dtype, stored_config.dtype, device)
    config_values = asdict(stored_config)
    config_values["device"] = str(device)
    config_values["dtype"] = evaluation_dtype
    config_values["checkpoint_segment_size"] = 0
    config = TrainingConfig(**config_values)

    model = build_training_model(config, device=device)
    model.load_state_dict(strip_compiled_prefix(payload["model"]), strict=True)
    model.eval()
    return payload, config, model, evaluation_dtype


def _load_validation_tokens(data_dir: Path) -> np.memmap:
    path = data_dir / "val.bin"
    if not path.is_file():
        raise FileNotFoundError(f"validation token file is missing: {path}")
    if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise ValueError(f"validation token file size is not uint16-aligned: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def _fixed_starts(
    token_count: int,
    block_size: int,
    batch_size: int,
    shard_count: int,
    seed: int,
) -> Tuple[Tuple[int, ...], ...]:
    if token_count <= block_size:
        raise ValueError("validation split is not longer than the checkpoint block_size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(
        token_count - block_size,
        (shard_count, batch_size),
        generator=generator,
    )
    return tuple(tuple(int(value) for value in row.tolist()) for row in starts)


def _batch_from_starts(
    tokens: np.memmap,
    starts: Sequence[int],
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack(
        [
            torch.from_numpy(np.asarray(tokens[start : start + block_size], dtype=np.int64).copy())
            for start in starts
        ]
    )
    targets = torch.stack(
        [
            torch.from_numpy(np.asarray(tokens[start + 1 : start + block_size + 1], dtype=np.int64).copy())
            for start in starts
        ]
    )
    if device.type == "cuda":
        return (
            inputs.pin_memory().to(device, non_blocking=True),
            targets.pin_memory().to(device, non_blocking=True),
        )
    return inputs.to(device), targets.to(device)


def _evaluate_shards(
    model: torch.nn.Module,
    tokens: np.memmap,
    starts_by_shard: Sequence[Sequence[int]],
    block_size: int,
    device: torch.device,
    dtype: str,
) -> Tuple[float, ...]:
    losses: List[float] = []
    with torch.inference_mode():
        for starts in starts_by_shard:
            inputs, targets = _batch_from_starts(tokens, starts, block_size, device)
            with _autocast_context(device, dtype):
                _, loss = model(inputs, targets)
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite validation loss during salience analysis")
            losses.append(float(loss.detach().to(dtype=torch.float64, device="cpu").item()))
    return tuple(losses)


def _collect_gradients(
    model: torch.nn.Module,
    tokens: np.memmap,
    starts_by_shard: Sequence[Sequence[int]],
    block_size: int,
    device: torch.device,
    dtype: str,
) -> None:
    model.zero_grad(set_to_none=True)
    shard_count = len(starts_by_shard)
    if shard_count == 0:
        return
    for starts in starts_by_shard:
        inputs, targets = _batch_from_starts(tokens, starts, block_size, device)
        with _autocast_context(device, dtype):
            _, loss = model(inputs, targets)
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite calibration loss during salience gradient collection")
            scaled_loss = loss / shard_count
        scaled_loss.backward()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _rank_stability(per_order_delta: Sequence[Sequence[float]]) -> Optional[float]:
    if not per_order_delta or len(per_order_delta[0]) < 2:
        return None
    shard_count = len(per_order_delta[0])
    split = shard_count // 2
    if split == 0 or split == shard_count:
        return None
    first = [_mean(values[:split]) for values in per_order_delta]
    second = [_mean(values[split:]) for values in per_order_delta]
    return spearman_rho(first, second)


def _git_commit() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def _safe_scope_label(scope: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope.strip())
    return label or "depth"


def _output_prefix(arguments: argparse.Namespace, completed_updates: int) -> Path:
    if arguments.output_prefix is not None:
        return arguments.output_prefix
    return arguments.checkpoint.parent / "salience" / f"{_safe_scope_label(arguments.scope)}_u{completed_updates}"


def _write_outputs(
    prefix: Path,
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    shard_count: int,
    overwrite: bool,
) -> Tuple[Path, Path]:
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    for path in (csv_path, json_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}; use --overwrite or --output-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "order",
        "salience_rank",
        "coefficient_count",
        "coefficient_rms",
        "gradient_rms",
        "first_order_delta_loss_proxy",
        "baseline_loss",
        "ablated_loss",
        "delta_loss_zero",
        "delta_loss_zero_shard_std",
    ]
    for shard in range(shard_count):
        fieldnames.extend(
            (
                f"baseline_loss_shard_{shard}",
                f"ablated_loss_shard_{shard}",
                f"delta_loss_zero_shard_{shard}",
            )
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _print_scope_listing(banks: Sequence[CoefficientBank]) -> None:
    depth_banks = [bank for bank in banks if bank.axis_label == "depth"]
    if depth_banks and len({bank.order_count for bank in depth_banks}) == 1:
        print(f"depth{'':<34} aggregate P={depth_banks[0].order_count}")
    for bank in banks:
        print(
            f"{bank.scope_name:<40} P={bank.order_count:<4d} axis={bank.order_axis}  "
            f"shape={tuple(bank.parameter.shape)}"
        )


def _print_summary(rows: Sequence[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
    print("\nNatural order")
    print(" order      delta_loss     coeff_rms      grad_rms")
    for row in rows:
        grad = row["gradient_rms"]
        grad_text = "-" if grad is None else f"{grad:.3e}"
        print(
            f" {row['order']:>5d}  {row['delta_loss_zero']:>+14.7e}  "
            f"{row['coefficient_rms']:>12.3e}  {grad_text:>12}"
        )

    print("\nSalience rank")
    print(" rank  order      delta_loss")
    for row in sorted(rows, key=lambda item: item["salience_rank"]):
        print(f" {row['salience_rank']:>4d}  {row['order']:>5d}  {row['delta_loss_zero']:>+14.7e}")

    effective = statistics["effective_salience_dimension"]
    ratio = statistics["effective_salience_dimension_ratio"]
    top_fraction = statistics["top_quartile_positive_salience_fraction"]
    stability = statistics["rank_stability_spearman_rho"]
    print("\nConcentration")
    print(f" effective dimension:       {'n/a' if effective is None else f'{effective:.3f}'}")
    print(f" effective dimension / P:   {'n/a' if ratio is None else f'{ratio:.3f}'}")
    print(f" top-quartile salience:      {'n/a' if top_fraction is None else f'{top_fraction:.3f}'}")
    print(f" half-split rank stability:  {'n/a' if stability is None else f'{stability:.3f}'}")


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.checkpoint = _resolve_checkpoint_path(arguments.checkpoint)
    if arguments.no_gradients:
        arguments.gradient_shards = 0
    if arguments.eval_shards < 1:
        raise ValueError("--eval-shards must be at least 1")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if arguments.gradient_shards < 0 or arguments.gradient_shards > arguments.eval_shards:
        raise ValueError("--gradient-shards must be between 0 and --eval-shards")

    device = _resolve_device(arguments.device)
    payload, config, model, evaluation_dtype = _load_model(arguments.checkpoint, device, arguments.dtype)
    all_banks = discover_coefficient_banks(model)
    if arguments.list_scopes:
        _print_scope_listing(all_banks)
        return 0
    banks = select_banks(all_banks, arguments.scope)
    order_count = banks[0].order_count

    validation_tokens = _load_validation_tokens(arguments.data_dir)
    seed = config.data_seed + 1 if arguments.seed is None else arguments.seed
    starts_by_shard = _fixed_starts(
        len(validation_tokens),
        config.block_size,
        arguments.batch_size,
        arguments.eval_shards,
        seed,
    )

    print(f"checkpoint: {arguments.checkpoint}")
    print(f"device:     {device}")
    print(f"dtype:      {evaluation_dtype}")
    print(f"scope:      {arguments.scope}")
    print(f"banks:      {', '.join(bank.scope_name for bank in banks)}")
    print(f"orders:     {order_count}")
    print(f"shards:     {arguments.eval_shards} x batch {arguments.batch_size}")

    baseline_by_shard = _evaluate_shards(
        model,
        validation_tokens,
        starts_by_shard,
        config.block_size,
        device,
        evaluation_dtype,
    )
    baseline_loss = _mean(baseline_by_shard)
    print(f"baseline:   {baseline_loss:.7f}\n")

    coefficient_rms = coefficient_rms_by_order(banks)
    if arguments.gradient_shards > 0:
        _collect_gradients(
            model,
            validation_tokens,
            starts_by_shard[: arguments.gradient_shards],
            config.block_size,
            device,
            evaluation_dtype,
        )
        gradient_rms, first_order_proxy = gradient_diagnostics_by_order(banks)
        model.zero_grad(set_to_none=True)
    else:
        gradient_rms = tuple(None for _ in range(order_count))
        first_order_proxy = tuple(None for _ in range(order_count))

    ablated_by_order: List[Tuple[float, ...]] = []
    delta_by_order: List[Tuple[float, ...]] = []
    for order in range(order_count):
        with zero_order_temporarily(banks, order):
            ablated = _evaluate_shards(
                model,
                validation_tokens,
                starts_by_shard,
                config.block_size,
                device,
                evaluation_dtype,
            )
        delta = tuple(
            ablated_loss - baseline
            for ablated_loss, baseline in zip(ablated, baseline_by_shard)
        )
        ablated_by_order.append(ablated)
        delta_by_order.append(delta)
        print(
            f"[{order + 1:>{len(str(order_count))}d}/{order_count}] "
            f"order {order:>3d}: delta_loss={_mean(delta):+.7e}",
            flush=True,
        )

    mean_delta = [_mean(values) for values in delta_by_order]
    ranked_orders = sorted(range(order_count), key=lambda order: (-mean_delta[order], order))
    rank_by_order = {order: rank + 1 for rank, order in enumerate(ranked_orders)}
    coefficients_per_order = sum(bank.coefficients_per_order for bank in banks)

    rows: List[Dict[str, Any]] = []
    for order in range(order_count):
        row: Dict[str, Any] = {
            "order": order,
            "salience_rank": rank_by_order[order],
            "coefficient_count": coefficients_per_order,
            "coefficient_rms": coefficient_rms[order],
            "gradient_rms": gradient_rms[order],
            "first_order_delta_loss_proxy": first_order_proxy[order],
            "baseline_loss": baseline_loss,
            "ablated_loss": _mean(ablated_by_order[order]),
            "delta_loss_zero": mean_delta[order],
            "delta_loss_zero_shard_std": _sample_standard_deviation(delta_by_order[order]),
        }
        for shard in range(arguments.eval_shards):
            row[f"baseline_loss_shard_{shard}"] = baseline_by_shard[shard]
            row[f"ablated_loss_shard_{shard}"] = ablated_by_order[order][shard]
            row[f"delta_loss_zero_shard_{shard}"] = delta_by_order[order][shard]
        rows.append(row)

    statistics: Dict[str, Any] = concentration_statistics(mean_delta)
    statistics["rank_stability_spearman_rho"] = _rank_stability(delta_by_order)
    ratio = statistics["effective_salience_dimension_ratio"]
    stability = statistics["rank_stability_spearman_rho"]
    statistics["provisional_strong_signal"] = bool(
        ratio is not None and stability is not None and ratio <= 0.5 and stability >= 0.8
    )

    completed_updates = int(
        payload.get(
            "completed_updates",
            payload.get("trainer_state", {}).get("completed_updates", 0),
        )
    )
    prefix = _output_prefix(arguments, completed_updates)
    metadata: Dict[str, Any] = {
        "analysis": "coefficient_salience_zero_ablation_v1",
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_schema_version": payload.get("schema_version"),
        "checkpoint_completed_updates": completed_updates,
        "artifact_name": arguments.checkpoint.parent.name,
        "analysis_git_commit": _git_commit(),
        "model_type": config.model_type,
        "device": str(device),
        "dtype": evaluation_dtype,
        "scope": arguments.scope,
        "banks": [
            {
                "scope_name": bank.scope_name,
                "family": bank.name,
                "axis_label": bank.axis_label,
                "shape": list(bank.parameter.shape),
                "order_axis": bank.order_axis,
                "coefficients_per_order": bank.coefficients_per_order,
            }
            for bank in banks
        ],
        "order_count": order_count,
        "validation": {
            "data_dir": str(arguments.data_dir.resolve()),
            "block_size": config.block_size,
            "batch_size": arguments.batch_size,
            "shard_count": arguments.eval_shards,
            "token_count": arguments.eval_shards * arguments.batch_size * config.block_size,
            "seed": seed,
            "starts_by_shard": [list(starts) for starts in starts_by_shard],
            "baseline_loss": baseline_loss,
            "baseline_loss_by_shard": list(baseline_by_shard),
        },
        "gradient_diagnostics": {
            "shard_count": arguments.gradient_shards,
            "enabled": arguments.gradient_shards > 0,
        },
        "statistics": statistics,
        "orders": rows,
        "compact_identity": payload.get("compact_identity"),
    }
    csv_path, json_path = _write_outputs(
        prefix,
        rows,
        metadata,
        arguments.eval_shards,
        arguments.overwrite,
    )
    _print_summary(rows, statistics)
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


__all__ = ["build_parser", "main"]
# ^^^ THOG
