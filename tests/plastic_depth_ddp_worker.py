# vvv THOG
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from sheet.trainer import SharedTrainer
from tests.stage5_test_support import stage5_config, token_splits


def maximum_model_state_delta(trainer: SharedTrainer) -> float:
    maximum = torch.zeros((), dtype=torch.float64, device=trainer.device)
    for value in trainer.raw_model.state_dict().values():
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        reference = value.detach().clone()
        dist.broadcast(reference, src=0)
        difference = value.detach() - reference
        if difference.is_floating_point():
            both_nan = torch.isnan(value.detach()) & torch.isnan(reference)
            difference = torch.where(both_nan, torch.zeros_like(difference), difference)
            if bool(torch.isnan(difference).any().item()):
                difference = torch.full_like(difference, float("inf"))
        delta = difference.abs().max().to(dtype=torch.float64)
        maximum = torch.maximum(maximum, delta)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def maximum_optimizer_state_delta(trainer: SharedTrainer) -> float:
    maximum = torch.zeros((), dtype=torch.float64, device=trainer.device)
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            for value in trainer.optimizer.state.get(parameter, {}).values():
                if not isinstance(value, torch.Tensor) or value.numel() == 0:
                    continue
                reference = value.detach().clone()
                dist.broadcast(reference, src=0)
                delta = (value.detach() - reference).abs().max().to(dtype=torch.float64)
                maximum = torch.maximum(maximum, delta)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    train_tokens, validation_tokens = token_splits()
    config = stage5_config(
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        checkpoint_segment_size=1,
        gradient_accumulation_steps=1,
        max_updates=2,
        decay_updates=2,
        eval_batches=1,
        plastic__enabled=True,
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=2,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe_noise_window=8,
        plastic__layer_count_min_probes=1,
        plastic__layer_count_probe_noise_lambda=0.0,
        plastic__freeze_geometry_during_warmup=False,
    )
    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        history = trainer.run()
        lattice = trainer.raw_model.trajectory.plastic_sampling
        if lattice is None:
            raise RuntimeError("PLASTIC DEPTH DDP worker has no lattice")
        active_layers = int(lattice.active_layer_count.item())
        coordinates = lattice.interval_report()["active_public_coordinates"]
        trainer.distributed.assert_identical_object(active_layers, "PLASTIC DEPTH DDP active count")
        trainer.distributed.assert_identical_object(coordinates, "PLASTIC DEPTH DDP coordinates")
        evidence = {
            "completed_updates": trainer.state.completed_updates,
            "active_layers": active_layers,
            "active_public_coordinates": coordinates,
            "count_decisions": int(lattice.count_decision_number.item()),
            "model_state_max_delta": maximum_model_state_delta(trainer),
            "optimizer_state_max_delta": maximum_optimizer_state_delta(trainer),
            "history": history,
        }
        trainer.distributed.barrier()
        if trainer.distributed.is_primary:
            arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
            arguments.evidence.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        trainer.distributed.barrier()
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
# ^^^ THOG
