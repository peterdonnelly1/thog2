# vvv THOG
from __future__ import annotations

import argparse
import importlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sheet.stage6_protocol import verify_protocol_manifest
from sheet.stage6_source import evaluation_metric_payload
from sheet.stage6_source import init_resilient_telemetry
from sheet.stage6_source import telemetry_configuration
from sheet.stage6_source import training_metric_payload
from sheet.stage6_source import verify_manifest_source
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig


REPOSITORY_ROOT = Path(__file__).resolve().parent


def load_tokens(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_vocab_size(dataset_dir: Path) -> int:
    meta_path = dataset_dir / "meta.pkl"
    if not meta_path.exists():
        return 50304
    with meta_path.open("rb") as handle:
        return int(pickle.load(handle)["vocab_size"])


def find_run(manifest: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    matches = [row for row in manifest["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"manifest does not contain exactly one run {run_id!r}")
    return matches[0]


def start_telemetry(
    manifest: Dict[str, Any],
    run: Dict[str, Any],
    parameter_report: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any]]:
    if not manifest["wandb"]["enabled"]:
        return None, None
    module = importlib.import_module("wandb")
    control = manifest["wandb"]
    handle = init_resilient_telemetry(
        module,
        project=str(control["project"]),
        entity=control.get("entity"),
        name=str(run["wandb"]["name"]),
        group=str(run["wandb"]["group"]),
        job_type=str(run["wandb"]["job_type"]),
        config=telemetry_configuration(manifest, run, parameter_report),
    )
    define_metric = getattr(handle, "define_metric", module.define_metric)
    define_metric("optimizer_update")
    for metric in (
        "iter",
        "tokens_seen",
        "clean_training_seconds",
        "training_loss",
        "validation_loss",
        "training_evaluation_loss",
        "learning_rate",
        "gradient_norm",
    ):
        define_metric(metric, step_metric="optimizer_update")
    handle.summary.update({
        "artifact_name": run["artifact_name"],
        "artifact_prefix": run["artifact_prefix"],
        "model_type": run["model_type"],
        "protocol_sha256": manifest["protocol_sha256"],
        "source_commit": manifest["source"]["commit"],
        "comparison_group": run["wandb"]["group"],
        "persistent_parameters": parameter_report["persistent_parameters"],
        "dense_equivalent_parameters": parameter_report[
            "dense_equivalent_total_parameters"
        ],
    })
    return module, handle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute one isolated THOG2 Stage 6 controlled pilot run"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    verify_protocol_manifest(manifest)
    verify_manifest_source(manifest, REPOSITORY_ROOT)
    if manifest.get("status") != "locked_before_training":
        raise ValueError("Stage 6 manifest is not locked_before_training")
    run = find_run(manifest, arguments.run_id)
    dataset_dir = Path(manifest["dataset"]["path"])
    vocab_size = load_vocab_size(dataset_dir)
    if vocab_size != int(manifest["dataset"]["vocab_size"]):
        raise ValueError("dataset vocabulary changed after protocol lock")

    config_values = dict(run["training_config"])
    config_values["vocab_size"] = vocab_size
    config = TrainingConfig(**config_values)
    run_dir = Path(config.out_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"

    train_tokens = load_tokens(dataset_dir / "train.bin")
    validation_tokens = load_tokens(dataset_dir / "val.bin")
    if int(train_tokens.size) != int(manifest["dataset"]["train_tokens"]):
        raise ValueError("training-token count changed after protocol lock")
    if int(validation_tokens.size) != int(manifest["dataset"]["validation_tokens"]):
        raise ValueError("validation-token count changed after protocol lock")

    trainer = Stage6Trainer(config, train_tokens, validation_tokens)
    if trainer.distributed.is_primary:
        start_telemetry(manifest, run, trainer.parameter_report)
    try:
        result = trainer.run_pilot(
            run_id=arguments.run_id,
            protocol_sha256=manifest["protocol_sha256"],
            dataset=manifest["dataset"],
            result_path=result_path,
        )
        if trainer.distributed.is_primary:
            print(json.dumps({
                "run_id": arguments.run_id,
                "completed_updates": result["budget"]["completed_updates"],
                "consumed_tokens": result["budget"]["consumed_tokens"],
                "final_validation_loss": result["evaluations"][-1]["val"],
                "training_seconds": result["timing"]["training_seconds"],
                "tokens_per_training_second": result["timing"]["tokens_per_training_second"],
                "result": str(result_path),
            }, indent=2, sort_keys=True))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
# ^^^ THOG
