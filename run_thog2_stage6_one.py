# vvv THOG
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sheet.stage6_protocol import verify_protocol_manifest
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig


def load_tokens(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_vocab_size(dataset_dir: Path) -> int:
    meta_path = dataset_dir / "meta.pkl"
    if not meta_path.exists():
        return 50304
    with meta_path.open("rb") as handle:
        metadata = pickle.load(handle)
    return int(metadata["vocab_size"])


def find_run(manifest: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    matches = [row for row in manifest["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"manifest does not contain exactly one run {run_id!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute one isolated THOG2 Stage 6 controlled pilot run"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    verify_protocol_manifest(manifest)
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
    try:
        result = trainer.run_pilot(
            run_id=arguments.run_id,
            protocol_sha256=manifest["protocol_sha256"],
            dataset=manifest["dataset"],
            result_path=result_path,
        )
        if trainer.distributed.is_primary:
            print(json.dumps(
                {
                    "run_id": arguments.run_id,
                    "completed_updates": result["budget"]["completed_updates"],
                    "consumed_tokens": result["budget"]["consumed_tokens"],
                    "final_validation_loss": result["evaluations"][-1]["val"],
                    "training_seconds": result["timing"]["training_seconds"],
                    "tokens_per_training_second": result["timing"]["tokens_per_training_second"],
                    "result": str(result_path),
                },
                indent=2,
                sort_keys=True,
            ))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
# ^^^ THOG
