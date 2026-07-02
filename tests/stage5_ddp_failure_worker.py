# vvv THOG
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from sheet.trainer import SharedTrainer
from tests.stage5_test_support import stage5_config, token_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("structure_mismatch", "nonfinite_rank"),
        required=True,
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    train_tokens, validation_tokens = token_splits()
    trainer = SharedTrainer(
        stage5_config(max_updates=1, decay_updates=1),
        train_tokens,
        validation_tokens,
    )
    try:
        if arguments.mode == "structure_mismatch":
            trainer.distributed.assert_identical_object(
                {"injected_rank_specific_value": trainer.distributed.rank},
                "injected structure",
            )
            raise RuntimeError("injected structure mismatch was not detected")
        if arguments.mode == "nonfinite_rank":
            if trainer.distributed.rank == 1:
                first_parameter = next(trainer.raw_model.parameters())
                with torch.no_grad():
                    first_parameter.fill_(float("nan"))
            trainer.train_one_update()
            raise RuntimeError("injected non-finite rank state was not detected")
        raise RuntimeError(f"unsupported failure mode: {arguments.mode}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
# ^^^ THOG
