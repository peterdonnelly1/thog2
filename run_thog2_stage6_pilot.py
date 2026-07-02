# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sheet.stage6_analysis import analyze_pilot
from sheet.stage6_protocol import PilotBudget
from sheet.stage6_protocol import protocol_digest
from sheet.stage6_protocol import protocol_manifest
from sheet.stage6_protocol import verify_protocol_manifest
from sheet.stage6_source import source_identity
from sheet.stage6_source import verify_manifest_source


REPOSITORY_ROOT = Path(__file__).resolve().parent
ARTIFACT_NAME_LIMIT = 240
ARCHITECTURE_PREFIXES = {
    "dense": "DENSE2",
    "thog2_sheet": "SHEET",
}
_NAME_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


def normalize_name_component(value: str, *, uppercase: bool = False) -> str:
    normalized = _NAME_COMPONENT.sub("_", value.strip()).strip("_")
    normalized = normalized.upper() if uppercase else normalized
    if not normalized:
        raise ValueError("artifact name component must contain an alphanumeric character")
    return normalized


def architecture_prefix(model_type: str) -> str:
    try:
        return ARCHITECTURE_PREFIXES[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported THOG2 model_type: {model_type!r}") from error


def default_artifact_suffix() -> str:
    return datetime.now().strftime("PILOT_%Y%m%d_%H%M%S")


def build_artifact_name(
    run: Dict[str, Any],
    budget: Dict[str, Any],
    *,
    dtype: str,
    host_label: str,
    run_name: str,
    suffix: str,
    source_commit: str,
    world_size: int = 1,
) -> str:
    model_type = str(run["model_type"])
    geometry = (
        f"L{budget['n_layer']}_H{budget['n_head']}_"
        f"D{budget['n_embd']}_C{budget['block_size']}"
    )
    if model_type == "thog2_sheet":
        geometry += (
            f"_P{budget['depth_order']}_Q{run['base_row_order']}_"
            f"Q4D{run['row_order_4d']}"
        )
    artifact_name = (
        f"{architecture_prefix(model_type)}_"
        f"{normalize_name_component(host_label)}_"
        f"{normalize_name_component(run_name, uppercase=True)}__OWT__{geometry}__"
        f"B{budget['batch_size']}_GA{budget['gradient_accumulation_steps']}_G{world_size}__"
        f"N{budget['max_updates']}_E{budget['eval_interval']}_U{budget['eval_batches']}_"
        f"LG{budget['log_interval']}_W{budget['warmup_updates']}__"
        f"LR{budget['learning_rate']}_MLR{budget['min_learning_rate']}_"
        f"WD{budget['weight_decay']}_B1{budget['beta1']}_"
        f"B2{budget['beta2']}_GC{budget['grad_clip']}__"
        f"MS{budget['model_seed']}_DS{budget['data_seed']}__"
        f"DT{dtype}_AC{run['checkpoint_segment_size']}__"
        f"SRC{normalize_name_component(source_commit[:12])}__"
        f"{normalize_name_component(suffix, uppercase=True)}"
    )
    if len(artifact_name) > ARTIFACT_NAME_LIMIT:
        raise ValueError(
            "THOG2 artifact name exceeds the safe 240-character limit; "
            f"length={len(artifact_name)}"
        )
    return artifact_name


def apply_artifact_naming(
    manifest: Dict[str, Any],
    *,
    host_label: str,
    run_name: str,
    suffix: str,
    source_commit: str,
    world_size: int = 1,
) -> None:
    root = Path(manifest["output_root"])
    normalized_run_name = normalize_name_component(run_name, uppercase=True)
    normalized_suffix = normalize_name_component(suffix, uppercase=True)
    comparison_group = f"{normalized_run_name}__{normalized_suffix}"
    for run in manifest["runs"]:
        artifact_name = build_artifact_name(
            run,
            manifest["budget"],
            dtype=str(manifest["dtype"]),
            host_label=host_label,
            run_name=run_name,
            suffix=suffix,
            source_commit=source_commit,
            world_size=world_size,
        )
        checkpoint_dir = root / "checkpoints" / artifact_name
        job_type = (
            "dense2"
            if run["model_type"] == "dense"
            else f"sheet_q{run['base_row_order']}"
        )
        run.update({
            "artifact_prefix": architecture_prefix(str(run["model_type"])),
            "artifact_name": artifact_name,
            "out_dir": str(checkpoint_dir),
            "checkpoint_path": str(checkpoint_dir / "ckpt.pt"),
            "result_path": str(root / "results" / f"{artifact_name}.json"),
            "log_path": str(root / "logs" / f"{artifact_name}.log"),
            "wandb": {
                "name": artifact_name,
                "group": comparison_group,
                "job_type": job_type,
            },
        })
        run["training_config"]["out_dir"] = str(checkpoint_dir)
    manifest["artifact_naming"] = {
        "schema_version": 1,
        "convention": "THOG architecture-first controlled artifact naming",
        "supported_prefixes": ["DENSE2", "SHEET"],
        "host_label": normalize_name_component(host_label),
        "run_name": normalized_run_name,
        "suffix": normalized_suffix,
        "comparison_group": comparison_group,
        "maximum_name_length": ARTIFACT_NAME_LIMIT,
    }


def sampled_file_fingerprint(
    path: Path,
    *,
    chunk_bytes: int = 1024 * 1024,
) -> Dict[str, Any]:
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
    token_bytes = np.dtype(np.uint16).itemsize
    if train_path.stat().st_size % token_bytes != 0:
        raise ValueError("train.bin byte size is not divisible by uint16 size")
    if validation_path.stat().st_size % token_bytes != 0:
        raise ValueError("val.bin byte size is not divisible by uint16 size")
    return {
        "path": str(directory),
        "format": "uint16_token_ids",
        "vocab_size": vocab_size,
        "train_tokens": train_path.stat().st_size // token_bytes,
        "validation_tokens": validation_path.stat().st_size // token_bytes,
        "train_file": sampled_file_fingerprint(train_path),
        "validation_file": sampled_file_fingerprint(validation_path),
        "meta_file": meta_fingerprint,
    }


def verify_dataset_manifest(manifest: Dict[str, Any]) -> None:
    expected = manifest.get("dataset")
    if not isinstance(expected, dict):
        raise ValueError("Stage 6 protocol dataset identity is missing")
    actual = dataset_manifest(Path(expected["path"]))
    if actual != expected:
        raise ValueError("Stage 6 dataset differs from the locked protocol")


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
    host_label: str,
    run_name: str,
    artifact_suffix: str,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_mode: str,
) -> Path:
    source = source_identity(REPOSITORY_ROOT)
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(
            f"Stage 6 output root already exists: {root}; "
            "use a new path to preserve artifact isolation"
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
    manifest["source"] = source
    apply_artifact_naming(
        manifest,
        host_label=host_label,
        run_name=run_name,
        suffix=artifact_suffix,
        source_commit=str(source["commit"]),
    )
    manifest["wandb"] = {
        "enabled": wandb_mode != "disabled",
        "project": normalize_name_component(wandb_project),
        "entity": wandb_entity,
        "mode": wandb_mode,
        "root": str(root / "wandb"),
        "group": manifest["artifact_naming"]["comparison_group"],
        "metric_schema_version": 1,
    }
    manifest["protocol_sha256"] = protocol_digest(manifest)
    verify_protocol_manifest(manifest)
    for directory in ("checkpoints", "results", "logs", "wandb"):
        (root / directory).mkdir(exist_ok=True)
    manifest_path = root / "protocol.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def result_path_for_run(run: Dict[str, Any]) -> Path:
    return Path(run.get("result_path", Path(run["out_dir"]) / "result.json"))


def completed_result(
    run: Dict[str, Any],
    protocol_sha256: str,
) -> Optional[Dict[str, Any]]:
    result_path = result_path_for_run(run)
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        return None
    if result.get("run_id") != run["run_id"]:
        return None
    if result.get("protocol_sha256") != protocol_sha256:
        return None
    return result


def archive_incomplete_run(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    for index in range(1, 1000):
        target = run_dir.with_name(f"{run_dir.name}.incomplete_{index:03d}")
        if not target.exists():
            run_dir.rename(target)
            return target
    raise RuntimeError(f"could not allocate incomplete-run archive name for {run_dir}")


def stream_run(command, log_path: Path, *, environment: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n=== Stage 6 process start ===\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Stage 6 subprocess stdout pipe is unavailable")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def write_status(path: Path, status: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_manifest(manifest_path: Path) -> Dict[str, Any]:
    path = manifest_path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_protocol_manifest(manifest)
    verify_manifest_source(manifest, REPOSITORY_ROOT)
    verify_dataset_manifest(manifest)
    if manifest.get("status") != "locked_before_training":
        raise ValueError("Stage 6 manifest is not locked_before_training")

    output_root = Path(manifest["output_root"])
    status: Dict[str, Any] = {
        "stage": 6,
        "protocol_sha256": manifest["protocol_sha256"],
        "source": manifest["source"],
        "artifact_naming": manifest["artifact_naming"],
        "wandb": manifest["wandb"],
        "status": "running",
        "runs": [],
    }
    status_path = output_root / "pilot_status.json"
    write_status(status_path, status)

    for run in manifest["runs"]:
        run_id = run["run_id"]
        existing = completed_result(run, manifest["protocol_sha256"])
        if existing is not None:
            run_status = {
                "run_id": run_id,
                "artifact_name": run["artifact_name"],
                "action": "reused_completed_result",
                "returncode": 0,
                "result": str(result_path_for_run(run)),
            }
            status["runs"].append(run_status)
            write_status(status_path, status)
            print(f"Stage 6 reusing completed run: {run['artifact_name']}", flush=True)
            continue

        run_dir = Path(run["out_dir"])
        archived = archive_incomplete_run(run_dir)
        log_path = Path(run["log_path"])
        command = [
            sys.executable,
            "-m",
            "run_thog2_stage6_one",
            "--manifest",
            str(path),
            "--run-id",
            run_id,
        ]
        environment = dict(os.environ)
        environment["WANDB_DIR"] = str(manifest["wandb"]["root"])
        environment["WANDB_MODE"] = str(manifest["wandb"]["mode"])
        print(f"Stage 6 starting run: {run['artifact_name']}", flush=True)
        returncode = stream_run(command, log_path, environment=environment)
        run_status = {
            "run_id": run_id,
            "artifact_name": run["artifact_name"],
            "action": "executed",
            "returncode": returncode,
            "log": str(log_path),
            "result": str(result_path_for_run(run)),
            "checkpoint": str(run["checkpoint_path"]),
            "archived_incomplete_run": str(archived) if archived is not None else None,
        }
        status["runs"].append(run_status)
        write_status(status_path, status)
        if returncode != 0:
            status["status"] = "failed"
            status["failed_run_id"] = run_id
            write_status(status_path, status)
            raise RuntimeError(
                f"Stage 6 run {run['artifact_name']} failed; inspect {log_path}"
            )
        if completed_result(run, manifest["protocol_sha256"]) is None:
            raise RuntimeError(
                f"Stage 6 run {run['artifact_name']} returned success without valid result evidence"
            )

    analysis = analyze_pilot(path, output_root / "analysis")
    status["status"] = "completed_awaiting_scientific_classification"
    status["analysis"] = str(output_root / "analysis" / "analysis.json")
    write_status(status_path, status)
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
    parser.add_argument("--host-label", default=socket.gethostname().split(".")[0])
    parser.add_argument("--run-name", default="STAGE6")
    parser.add_argument("--artifact-suffix")
    parser.add_argument("--wandb-project", default="thog2")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
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
            raise ValueError(
                "--dataset-dir and --out-dir are required when creating a protocol"
            )
        manifest_path = prepare_manifest(
            dataset_dir=arguments.dataset_dir,
            output_root=arguments.out_dir,
            device=arguments.device,
            dtype=arguments.dtype,
            host_label=arguments.host_label,
            run_name=arguments.run_name,
            artifact_suffix=arguments.artifact_suffix or default_artifact_suffix(),
            wandb_project=arguments.wandb_project,
            wandb_entity=arguments.wandb_entity,
            wandb_mode=arguments.wandb_mode,
        )
        print(manifest_path.read_text(encoding="utf-8"))
        if arguments.prepare_only:
            return

    outcome = execute_manifest(manifest_path)
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
# ^^^ THOG
