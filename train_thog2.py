# vvv THOG
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig


def load_tokens(path: Path) -> torch.Tensor:
    array = np.memmap(path, dtype=np.uint16, mode="r")
    return torch.from_numpy(
        np.asarray(array, dtype=np.int64).copy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="THOG2 shared trainer with single-process and torchrun DDP execution"
    )
    parser.add_argument(
        "--model-type",
        choices=("dense", "thog2_sheet"),
        required=True,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--depth-order", type=int, default=3)
    parser.add_argument("--base-row-order", type=int, default=32)
    parser.add_argument("--checkpoint-segment-size", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--max-updates", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=6.0e-4)
    parser.add_argument("--data-seed", type=int, default=7331)
    parser.add_argument("--model-seed", type=int, default=1337)
    arguments = parser.parse_args()

    meta_path = arguments.dataset_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            vocab_size = int(pickle.load(handle)["vocab_size"])
    else:
        vocab_size = 50304
    train_tokens = load_tokens(arguments.dataset_dir / "train.bin")
    validation_tokens = load_tokens(arguments.dataset_dir / "val.bin")
    config = TrainingConfig(
        model_type=arguments.model_type,
        block_size=arguments.block_size,
        vocab_size=vocab_size,
        n_layer=arguments.n_layer,
        n_head=arguments.n_head,
        n_embd=arguments.n_embd,
        dropout=arguments.dropout,
        depth_order=arguments.depth_order,
        base_row_order=arguments.base_row_order,
        checkpoint_segment_size=arguments.checkpoint_segment_size,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        max_updates=arguments.max_updates,
        learning_rate=arguments.learning_rate,
        decay_updates=arguments.max_updates,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_batches,
        checkpoint_interval=arguments.checkpoint_interval,
        model_seed=arguments.model_seed,
        data_seed=arguments.data_seed,
        device=arguments.device,
        dtype=arguments.dtype,
        out_dir=str(arguments.out_dir),
    )
    if arguments.resume:
        trainer = SharedTrainer.from_checkpoint(
            arguments.resume,
            train_tokens,
            validation_tokens,
            overrides={
                "device": arguments.device,
                "dtype": arguments.dtype,
                "max_updates": arguments.max_updates,
                "checkpoint_segment_size": arguments.checkpoint_segment_size,
                "out_dir": str(arguments.out_dir),
            },
            expected_config=config,
        )
    else:
        trainer = SharedTrainer(
            config,
            train_tokens,
            validation_tokens,
        )
    try:
        if trainer.distributed.is_primary:
            print(trainer.startup_report_json())
        trainer.run()
        trainer.save_checkpoint(arguments.out_dir / "ckpt.pt")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
# ^^^ THOG
