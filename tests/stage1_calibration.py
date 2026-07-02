# vvv THOG
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CALIBRATION_GEOMETRIES = (
    (768, 128),
    (1536, 256),
    (3072, 512),
    (3072, 1024),
)
MAX_SECONDS_PER_GEOMETRY = 300.0
MAX_ESTIMATED_PEAK_BYTES = 2 * 1024**3
FLOAT64_ORTHONORMALITY_TOLERANCE = 2.0e-11
FLOAT32_ORTHONORMALITY_TOLERANCE = 5.0e-5


def _worker(sample_count: int, order: int) -> Dict[str, object]:
    import torch

    from sheet.basis import (
        basis_sha256,
        chebyshev_first_kind_basis,
        deterministic_reduced_qr,
        estimated_peak_tensor_bytes,
        normalized_coordinates,
        orthonormality_max_error,
    )

    thread_count = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(thread_count)
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    coordinates = normalized_coordinates(sample_count, dtype=torch.float64)
    raw_basis = chebyshev_first_kind_basis(coordinates, order)
    stabilized_float64, r_matrix = deterministic_reduced_qr(raw_basis)
    stabilized_float32 = stabilized_float64.to(dtype=torch.float32)
    elapsed_seconds = time.perf_counter() - started
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    diagonal = torch.abs(torch.diagonal(r_matrix))
    maximum_diagonal = float(torch.max(diagonal).item())
    minimum_diagonal = float(torch.min(diagonal).item())
    relative_minimum_diagonal = minimum_diagonal / maximum_diagonal if maximum_diagonal else 0.0

    result: Dict[str, object] = {
        "sample_count": sample_count,
        "order": order,
        "threads": thread_count,
        "elapsed_seconds": elapsed_seconds,
        "rss_before_kib": int(rss_before_kib),
        "rss_after_kib": int(rss_after_kib),
        "rss_delta_kib": max(0, int(rss_after_kib - rss_before_kib)),
        "estimated_peak_tensor_bytes_float64": estimated_peak_tensor_bytes(
            sample_count, order, runtime_dtype=torch.float64
        ),
        "estimated_peak_tensor_bytes_float32": estimated_peak_tensor_bytes(
            sample_count, order, runtime_dtype=torch.float32
        ),
        "raw_finite": bool(torch.isfinite(raw_basis).all().item()),
        "basis_float64_finite": bool(torch.isfinite(stabilized_float64).all().item()),
        "basis_float32_finite": bool(torch.isfinite(stabilized_float32).all().item()),
        "orthonormality_max_error_float64": orthonormality_max_error(stabilized_float64),
        "orthonormality_max_error_float32": orthonormality_max_error(stabilized_float32),
        "minimum_abs_r_diagonal": minimum_diagonal,
        "maximum_abs_r_diagonal": maximum_diagonal,
        "relative_minimum_abs_r_diagonal": relative_minimum_diagonal,
        "full_rank_qr_proxy": bool(torch.all(diagonal > 0.0).item()),
        "basis_float64_sha256": basis_sha256(stabilized_float64),
        "basis_float32_sha256": basis_sha256(stabilized_float32),
    }
    result["accepted"] = bool(
        result["raw_finite"]
        and result["basis_float64_finite"]
        and result["basis_float32_finite"]
        and result["full_rank_qr_proxy"]
        and elapsed_seconds <= MAX_SECONDS_PER_GEOMETRY
        and result["estimated_peak_tensor_bytes_float64"] <= MAX_ESTIMATED_PEAK_BYTES
        and result["orthonormality_max_error_float64"] <= FLOAT64_ORTHONORMALITY_TOLERANCE
        and result["orthonormality_max_error_float32"] <= FLOAT32_ORTHONORMALITY_TOLERANCE
    )
    return result


def _run_worker(sample_count: int, order: int) -> Dict[str, object]:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--sample-count",
            str(sample_count),
            "--order",
            str(order),
        ],
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "calibration worker failed for "
            f"sample_count={sample_count}, order={order}, returncode={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _write_evidence(output_path: Path, evidence: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_calibration(output_path: Path) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    evidence: Dict[str, object] = {
        "basis_version": "chebyshev_first_kind_qr_v1",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "commit": os.environ.get("GITHUB_SHA"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME"),
        "thresholds": {
            "max_seconds_per_geometry": MAX_SECONDS_PER_GEOMETRY,
            "max_estimated_peak_bytes": MAX_ESTIMATED_PEAK_BYTES,
            "float64_orthonormality_tolerance": FLOAT64_ORTHONORMALITY_TOLERANCE,
            "float32_orthonormality_tolerance": FLOAT32_ORTHONORMALITY_TOLERANCE,
        },
        "planned_geometries": [
            {"sample_count": sample_count, "order": order}
            for sample_count, order in CALIBRATION_GEOMETRIES
        ],
        "results": results,
        "failures": [],
        "status": "running",
        "accepted": False,
    }
    _write_evidence(output_path, evidence)

    for sample_count, order in CALIBRATION_GEOMETRIES:
        try:
            result = _run_worker(sample_count, order)
        except Exception as error:
            evidence["failures"].append(
                {
                    "sample_count": sample_count,
                    "order": order,
                    "error": str(error),
                }
            )
            evidence["status"] = "failed"
            evidence["accepted"] = False
            _write_evidence(output_path, evidence)
            raise
        results.append(result)
        _write_evidence(output_path, evidence)

    evidence["status"] = "completed"
    evidence["accepted"] = all(bool(result["accepted"]) for result in results)
    _write_evidence(output_path, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--order", type=int)
    arguments = parser.parse_args()

    if arguments.worker:
        if arguments.sample_count is None or arguments.order is None:
            parser.error("--worker requires --sample-count and --order")
        print(json.dumps(_worker(arguments.sample_count, arguments.order), sort_keys=True))
        return

    if arguments.output is None:
        parser.error("--output is required")
    evidence = run_calibration(arguments.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["accepted"]:
        raise SystemExit("Stage 1 calibration failed an acceptance threshold")


if __name__ == "__main__":
    main()
# ^^^ THOG
