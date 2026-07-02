# vvv THOG
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        arguments.evidence.write_text(
            json.dumps(
                {
                    "test": "S3-11",
                    "satisfied": False,
                    "reason": "CUDA is not available in this environment",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    from sheet.trainer import SharedTrainer
    from sheet.training_config import TrainingConfig

    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    tokens = torch.arange(1024, dtype=torch.long) % 64
    validation_tokens = torch.roll(tokens, 13)
    config = TrainingConfig(
        model_type="thog2_sheet",
        block_size=16,
        vocab_size=64,
        n_layer=3,
        n_head=4,
        n_embd=32,
        depth_order=3,
        base_row_order=16,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_updates=3,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=0,
        device="cuda",
        dtype=dtype,
    )
    trainer = SharedTrainer(config, tokens, validation_tokens)
    trainer.run(target_updates=2)
    checkpoint = arguments.evidence.parent / "stage3_gpu_ckpt.pt"
    trainer.save_checkpoint(checkpoint)
    resumed = SharedTrainer.from_checkpoint(
        checkpoint,
        tokens,
        validation_tokens,
        overrides={
            "max_updates": 3,
            "device": "cuda",
            "dtype": dtype,
        },
    )
    resumed.run(target_updates=3)
    evidence = {
        "test": "S3-11",
        "satisfied": True,
        "device": torch.cuda.get_device_name(),
        "dtype": dtype,
        "completed_updates": resumed.state.completed_updates,
        "latest_training_loss": resumed.state.latest_training_loss,
        "latest_validation_loss": resumed.state.latest_validation_loss,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
# ^^^ THOG
