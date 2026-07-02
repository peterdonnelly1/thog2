# vvv THOG
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


RUN_LABELS = {
    "dense": "Dense",
    "q64": "Sheet Q64",
    "q128": "Sheet Q128",
    "q256": "Sheet Q256",
}


def selector_for_run(run: Mapping[str, Any]) -> str:
    if run["model_type"] == "dense":
        return "dense"
    return f"q{int(run['base_row_order'])}"


def load_results(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for run in manifest["runs"]:
        selector = selector_for_run(run)
        result_path = Path(run["out_dir"]) / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing Stage 6 result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results[selector] = result
    return results


def validate_controls(
    manifest: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_protocol = manifest["protocol_sha256"]
    expected_updates = int(manifest["budget"]["max_updates"])
    expected_tokens = int(manifest["consumed_tokens_per_run"])
    expected_training_trace = None
    expected_validation_trace = None
    expected_evaluation_updates = None
    checks: Dict[str, Any] = {}
    for selector, result in results.items():
        if result["protocol_sha256"] != expected_protocol:
            raise ValueError(f"{selector} used a different protocol")
        if int(result["budget"]["completed_updates"]) != expected_updates:
            raise ValueError(f"{selector} completed-update count differs")
        if int(result["budget"]["consumed_tokens"]) != expected_tokens:
            raise ValueError(f"{selector} consumed-token count differs")
        training_trace = result["trace"]["training_sha256"]
        validation_trace = result["trace"]["validation_sha256"]
        evaluation_updates = tuple(
            int(row["completed_updates"])
            for row in result["evaluations"]
        )
        if expected_training_trace is None:
            expected_training_trace = training_trace
            expected_validation_trace = validation_trace
            expected_evaluation_updates = evaluation_updates
        if training_trace != expected_training_trace:
            raise ValueError(f"{selector} training batch trace differs")
        if validation_trace != expected_validation_trace:
            raise ValueError(f"{selector} validation sample differs")
        if evaluation_updates != expected_evaluation_updates:
            raise ValueError(f"{selector} evaluation updates differ")
        checks[selector] = {
            "protocol_match": True,
            "completed_updates": expected_updates,
            "consumed_tokens": expected_tokens,
            "training_trace_sha256": training_trace,
            "validation_trace_sha256": validation_trace,
            "evaluation_updates": list(evaluation_updates),
        }
    return checks


def peak_memory(result: Mapping[str, Any], key: str) -> int:
    samples = result["memory"]["samples"]
    return max((int(sample[key]) for sample in samples), default=0)


def resource_rows(
    manifest: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    dense_parameters = int(results["dense"]["parameter_report"]["persistent_parameters"])
    rows = []
    for run in manifest["runs"]:
        selector = selector_for_run(run)
        result = results[selector]
        persistent = int(result["parameter_report"]["persistent_parameters"])
        evaluations = result["evaluations"]
        rows.append(
            {
                "selector": selector,
                "label": RUN_LABELS[selector],
                "model_type": run["model_type"],
                "base_row_order": (
                    int(run["base_row_order"])
                    if run["model_type"] == "thog2_sheet"
                    else None
                ),
                "persistent_parameters": persistent,
                "dense_equivalent_parameters": int(
                    result["parameter_report"]["dense_equivalent_total_parameters"]
                ),
                "persistent_parameter_fraction_of_dense": persistent / dense_parameters,
                "persistent_parameter_reduction_fraction": 1.0 - persistent / dense_parameters,
                "peak_allocated_bytes": peak_memory(result, "peak_allocated_bytes"),
                "peak_reserved_bytes": peak_memory(result, "peak_reserved_bytes"),
                "checkpoint_bytes": int(result["checkpoint"]["bytes"]),
                "training_seconds": float(result["timing"]["training_seconds"]),
                "evaluation_seconds": float(result["timing"]["evaluation_seconds"]),
                "checkpoint_seconds": float(result["timing"]["checkpoint_seconds"]),
                "wall_seconds": float(result["timing"]["wall_seconds"]),
                "tokens_per_training_second": float(
                    result["timing"]["tokens_per_training_second"]
                ),
                "initial_validation_loss": float(evaluations[0]["val"]),
                "final_validation_loss": float(evaluations[-1]["val"]),
                "best_validation_loss": min(float(row["val"]) for row in evaluations),
            }
        )
    return rows


def equal_update_rows(
    results: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    selectors = tuple(results)
    evaluations = {
        selector: {
            int(row["completed_updates"]): row
            for row in result["evaluations"]
        }
        for selector, result in results.items()
    }
    updates = sorted(set.intersection(*(
        set(rows) for rows in evaluations.values()
    )))
    output = []
    for update in updates:
        row: Dict[str, Any] = {
            "completed_updates": update,
            "consumed_tokens": int(evaluations[selectors[0]][update]["consumed_tokens"]),
        }
        for selector in selectors:
            evaluation = evaluations[selector][update]
            row[f"{selector}_validation_loss"] = float(evaluation["val"])
            row[f"{selector}_training_seconds"] = float(evaluation["training_seconds"])
        output.append(row)
    return output


def equal_time_rows(
    results: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    dense_evaluations = results["dense"]["evaluations"]
    output = []
    for dense_row in dense_evaluations:
        target_seconds = float(dense_row["training_seconds"])
        row: Dict[str, Any] = {
            "dense_completed_updates": int(dense_row["completed_updates"]),
            "target_training_seconds": target_seconds,
            "dense_validation_loss": float(dense_row["val"]),
        }
        for selector, result in results.items():
            if selector == "dense":
                continue
            candidate = min(
                result["evaluations"],
                key=lambda item: abs(float(item["training_seconds"]) - target_seconds),
            )
            candidate_seconds = float(candidate["training_seconds"])
            row[f"{selector}_completed_updates"] = int(candidate["completed_updates"])
            row[f"{selector}_training_seconds"] = candidate_seconds
            row[f"{selector}_time_delta_seconds"] = candidate_seconds - target_seconds
            row[f"{selector}_validation_loss"] = float(candidate["val"])
        output.append(row)
    return output


def utilization_summary(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for selector, result in results.items():
        diagnostics = result.get("sheet_diagnostics")
        if diagnostics is None:
            continue
        family_rows = diagnostics["coefficient_utilization"]
        output[selector] = {
            family: {
                "coefficient_rms": row["coefficient_rms"],
                "high_depth_order_energy_fraction": row[
                    "high_depth_order_energy_fraction"
                ],
                "high_row_order_energy_fraction": row[
                    "high_row_order_energy_fraction"
                ],
                "nonzero_fraction": row["nonzero_fraction"],
            }
            for family, row in family_rows.items()
        }
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _polyline_points(
    points: Iterable[Tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margin: int,
) -> str:
    transformed = []
    x_span = max(x_max - x_min, 1.0e-12)
    y_span = max(y_max - y_min, 1.0e-12)
    for x_value, y_value in points:
        x_pixel = margin + (x_value - x_min) / x_span * (width - 2 * margin)
        y_pixel = height - margin - (y_value - y_min) / y_span * (height - 2 * margin)
        transformed.append(f"{x_pixel:.2f},{y_pixel:.2f}")
    return " ".join(transformed)


def write_svg_curves(
    path: Path,
    *,
    title: str,
    x_label: str,
    curves: Mapping[str, Sequence[Tuple[float, float]]],
) -> None:
    all_points = [point for points in curves.values() for point in points]
    if not all_points:
        raise ValueError("cannot write an empty Stage 6 curve")
    width, height, margin = 960, 540, 70
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="480" y="28" text-anchor="middle" font-size="20">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="480" y="525" text-anchor="middle" font-size="14">{x_label}</text>',
        '<text x="18" y="270" text-anchor="middle" font-size="14" transform="rotate(-90 18 270)">Validation loss</text>',
    ]
    for index, (selector, points) in enumerate(curves.items()):
        colour = palette[index % len(palette)]
        polyline = _polyline_points(
            points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
            margin=margin,
        )
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{colour}" stroke-width="2"/>'
        )
        legend_y = 55 + index * 22
        parts.append(
            f'<line x1="760" y1="{legend_y}" x2="790" y2="{legend_y}" stroke="{colour}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="800" y="{legend_y + 5}" font-size="13">{RUN_LABELS[selector]}</text>'
        )
    parts.append(
        f'<text x="{margin}" y="{height-margin+20}" font-size="11">{x_min:.3g}</text>'
    )
    parts.append(
        f'<text x="{width-margin}" y="{height-margin+20}" text-anchor="end" font-size="11">{x_max:.3g}</text>'
    )
    parts.append(
        f'<text x="{margin-8}" y="{height-margin}" text-anchor="end" font-size="11">{y_min:.4f}</text>'
    )
    parts.append(
        f'<text x="{margin-8}" y="{margin+4}" text-anchor="end" font-size="11">{y_max:.4f}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def markdown_report(
    manifest: Mapping[str, Any],
    resources: Sequence[Mapping[str, Any]],
    control_checks: Mapping[str, Any],
) -> str:
    lines = [
        "# THOG2 Stage 6 Controlled Pilot Analysis",
        "",
        f"Protocol: `{manifest['protocol_sha256']}`",
        "",
        "## Control validation",
        "",
        "All completed runs used the same protocol, completed updates, consumed tokens, training batch trace, validation sample, and evaluation updates.",
        "",
        "## Resource and final-loss summary",
        "",
        "| Run | Persistent parameters | Reduction vs dense | Peak allocated GiB | Tokens/s | Final validation loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in resources:
        lines.append(
            "| {label} | {persistent_parameters:,} | {reduction:.2%} | {memory:.3f} | {throughput:.1f} | {loss:.6f} |".format(
                label=row["label"],
                persistent_parameters=row["persistent_parameters"],
                reduction=row["persistent_parameter_reduction_fraction"],
                memory=row["peak_allocated_bytes"] / (1024 ** 3),
                throughput=row["tokens_per_training_second"],
                loss=row["final_validation_loss"],
            )
        )
    lines.extend([
        "",
        "## Scientific classification",
        "",
        "**Pending review.** This generated analysis validates controls and tabulates measurements; it does not choose the final Stage 6 classification automatically.",
        "",
        "Allowed final classifications:",
        "",
        "- viable for further study;",
        "- viable only at weak compression;",
        "- inconclusive;",
        "- not viable under the tested design.",
        "",
        "## Scope limitation",
        "",
        f"This pilot is matched at `{manifest['scientific_scope']['matched_geometry']}`. It does not establish a matched dense comparison at L144.",
        "",
        "## Trace digests",
        "",
        "```json",
        json.dumps(control_checks, indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def analyze_pilot(manifest_path: Path, output_dir: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = load_results(manifest)
    required = {"dense", "q64", "q128", "q256"}
    if set(results) != required:
        raise ValueError(f"Stage 6 analysis requires {sorted(required)}")
    checks = validate_controls(manifest, results)
    resources = resource_rows(manifest, results)
    equal_updates = equal_update_rows(results)
    equal_times = equal_time_rows(results)
    utilization = utilization_summary(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "resource_comparison.csv", resources)
    write_csv(output_dir / "equal_update_comparison.csv", equal_updates)
    write_csv(output_dir / "equal_time_comparison.csv", equal_times)
    update_curves = {
        selector: [
            (float(row["completed_updates"]), float(row["val"]))
            for row in result["evaluations"]
        ]
        for selector, result in results.items()
    }
    time_curves = {
        selector: [
            (float(row["training_seconds"]), float(row["val"]))
            for row in result["evaluations"]
        ]
        for selector, result in results.items()
    }
    write_svg_curves(
        output_dir / "validation_loss_by_update.svg",
        title="Validation loss by completed update",
        x_label="Completed optimizer updates",
        curves=update_curves,
    )
    write_svg_curves(
        output_dir / "validation_loss_by_training_time.svg",
        title="Validation loss by clean training time",
        x_label="Training seconds excluding evaluation and checkpointing",
        curves=time_curves,
    )
    analysis = {
        "stage": 6,
        "suite": "controlled_pilot_analysis",
        "status": "awaiting_scientific_classification",
        "protocol_sha256": manifest["protocol_sha256"],
        "control_validation": checks,
        "resource_comparison": resources,
        "equal_update_comparison": equal_updates,
        "equal_time_comparison": equal_times,
        "coefficient_utilization": utilization,
        "scientific_classification": None,
        "classification_options": manifest["scientific_scope"]["classification_options"],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis.md").write_text(
        markdown_report(manifest, resources, checks),
        encoding="utf-8",
    )
    return analysis


__all__ = [
    "analyze_pilot",
    "equal_time_rows",
    "equal_update_rows",
    "load_results",
    "resource_rows",
    "validate_controls",
]
# ^^^ THOG
