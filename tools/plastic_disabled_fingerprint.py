from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = f"{tuple(tensor.shape)}|{tensor.dtype}".encode("utf-8")
    return hashlib.sha256(metadata + tensor.numpy().tobytes(order="C")).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_hash": _tensor_hash(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return value


def _config() -> TrainingConfig:
    return TrainingConfig(
        model_type="dense",
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        depth_order=2,
        dropout=0.0,
        bias=True,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_updates=2,
        learning_rate=2.0e-3,
        min_learning_rate=2.0e-4,
        warmup_updates=1,
        decay_updates=2,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=101,
        data_seed=202,
        device="cpu",
        dtype="float32",
        plastic__enabled=False,
    )


def main() -> None:
    tokens = torch.arange(512, dtype=torch.long) % 32
    validation = torch.roll(tokens, shifts=7)
    trainer = SharedTrainer(_config(), tokens, validation)
    try:
        initial_model = _canonical(trainer.raw_model.state_dict())
        initial_optimizer = _canonical(trainer.optimizer.state_dict())
        initial_batch_source = _canonical(trainer.batch_source.state_dict())
        metrics = trainer.train_one_update()
        checkpoint = trainer.checkpoint_payload()
        result = {
            "persistent_config": trainer.config.persistent_dict(),
            "structure_signature": _canonical(trainer.distributed_structure_signature()),
            "initial_model": initial_model,
            "initial_optimizer": initial_optimizer,
            "initial_batch_source": initial_batch_source,
            "update_metrics": _canonical(metrics),
            "final_model": _canonical(trainer.raw_model.state_dict()),
            "final_optimizer": _canonical(trainer.optimizer.state_dict()),
            "final_batch_source": _canonical(trainer.batch_source.state_dict()),
            "trainer_state": _canonical(checkpoint["trainer_state"]),
            "checkpoint_keys": sorted(checkpoint),
            "checkpoint_trainer_config": checkpoint["trainer_config"],
            "event_names": [event.name for event in trainer.events],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
