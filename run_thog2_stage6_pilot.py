# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sheet.stage6_analysis import analyze_pilot
from sheet.stage6_protocol import PilotBudget, protocol_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parent


def sampled_file_fingerprint(path: Path, *, chunk_bytes: int = 1024 * 1024) -> Dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes:
            handle.seek(max(0, size - chunk_bytes))
            digest.update(handle.read(chunk_bytes))
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sampled_sha256": digest.hexdigest(),
        "sample_bytes_per_end": chunk_bytes,
    }


def dataset_manifest(dataset_dir: Path) -> Dict[str, Any]:
    directory = dataset_dir.resolve()
    train_path = directory / "train.bin"
    validation_path = directory / "val.bin"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("dataset must contain train.bin and val.bin")
    meta_path = directory / "meta.pkl"
    vocab_size = 50304
    meta_fingerprint: Optional[Dict[str, Any]] = None
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            metadata = pickle.load(handle)
        vocab_size = int(metadata["vocab_size"])
        meta_fingerprint = sampled_file_fingerprint(meta_path)
    if train_path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise ValueError("train.bin byte size is not divisible by uint16 size")
    if validation_path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise ValueError("val.bin byte size is not divisible by uint16 size")
    return {
        "path": str(directory),
        "format": "uint16_token_ids",
        "vocab_size": vocab_size,
        "train_tokens": train_path.stat().st_size // np.dtype(np.uint16).itemsize,
        "validation_tokens": validation_path.stat().st_size // np.dtype(np.uint16).itemsize,
        "train_file": sampled_file_fingerprint(train_path),
        "validation_file": sampled_file_fingerprint(validation_path),
        "meta_file": meta_fingerprint,
    }


def current_device_total_bytes(device: str) -> Optional[int]:
    target = torch.device(device)
    if target.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pilot requested but CUDA is unavailable")
    index = target.index if target.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).total_memory)


def prepare_manifest(
    *,
    dataset_dir: Path,
    output_root: Path,
    device: str,
    dtype: str,
) -> Path:
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(
            f"Stage 6 output root already exists: {root}; use a new path to preserve artifact isolation"
        )
    root.mkdir(parents=True)
    dataset = dataset_manifest(dataset_dir)
    manifest = protocol_manifest(
        budget=PilotBudget(),
        vocab_size=int(dataset["vocab_size"]),
        device=device,
        dtype=dtype,
        output_root=root,
        dataset=dataset,
        device_total_bytes=current_device_total_bytes(device),
    )
    manifest_path = root / "protocol.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def execute_manifest(manifest_path: Path) -> Dict[str, Any]:
    path = manifest_path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    output_root = Path(manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    status: Dict[str, Any] = {
        "stage": 6,
        "protocol_sha256": manifest["protocol_sha256"],
        "status": "running",
        "runs": [],
    }
    status_path = output_root / "pilot_status.json"

    for run in manifest["runs"]:
        run_id = run["run_id"]
        command = [
            sys.executable,
            "-m",
            "run_thog2_stage6_one",
            "--manifest",
            str(path),
            "--run-id",
            run_id,
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=dict(os.environ),
            text=True,
            capture_output=True,
            check=False,
        )
        (logs_dir / f"{run_id}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (logs_dir / f"{run_id}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        run_status = {
            "run_id": run_id,
            "returncode": completed.returncode,
            "stdout_log": str(logs_dir / f"{run_id}.stdout.log"),
            "stderr_log": str(logs_dir / f"{run_id}.stderr.log"),
        }
        status["runs"].append(run_status)
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            status["status"] = "failed"
            status["failed_run_id"] = run_id
            status_path.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Stage 6 run {run_id} failed; inspect {run_status['stderr_log']}"
            )

    analysis = analyze_pilot(path, output_root / "analysis")
    status["status"] = "completed_awaiting_scientific_classification"
    status["analysis"] = str(output_root / "analysis" / "analysis.json")
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(path),
        "status": str(status_path),
        "analysis": status["analysis"],
        "scientific_classification": analysis["scientific_classification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the THOG2 Stage 6 controlled pilot"
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    arguments = parser.parse_args()

    if arguments.manifest is not None:
        if arguments.dataset_dir is not None or arguments.out_dir is not None:
            raise ValueError("--manifest cannot be combined with --dataset-dir or --out-dir")
        if arguments.prepare_only:
            raise ValueError("--prepare-only is invalid with an existing manifest")
        manifest_path = arguments.manifest
    else:
        if arguments.dataset_dir is None or arguments.out_dir is None:
            raise ValueError("--dataset-dir and --out-dir are required when creating a protocol")
        manifest_path = prepare_manifest(
            dataset_dir=arguments.dataset_dir,
            output_root=arguments.out_dir,
            device=arguments.device,
            dtype=arguments.dtype,
        )
        print(manifest_path.read_text(encoding="utf-8"))
        if arguments.prepare_only:
            return

    outcome = execute_manifest(manifest_path)
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
# ^^^ THOG
