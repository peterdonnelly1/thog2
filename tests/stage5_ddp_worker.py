# vvv THOG
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist

from sheet.trainer import SharedTrainer
from tests.stage5_test_support import stage5_config, token_splits


def maximum_model_state_delta(trainer: SharedTrainer) -> float:
    maximum = torch.zeros((), dtype=torch.float64, device=trainer.device)
    for parameter in trainer.raw_model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        delta = (parameter.detach() - reference).abs().max().to(dtype=torch.float64)
        maximum = torch.maximum(maximum, delta)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def maximum_optimizer_state_delta(trainer: SharedTrainer) -> float:
    maximum = torch.zeros((), dtype=torch.float64, device=trainer.device)
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            state = trainer.optimizer.state.get(parameter, {})
            for value in state.values():
                if not isinstance(value, torch.Tensor):
                    continue
                reference = value.detach().clone()
                dist.broadcast(reference, src=0)
                delta = (value.detach() - reference).abs().max().to(dtype=torch.float64)
                maximum = torch.maximum(maximum, delta)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def communication_profile(
    trainer: SharedTrainer,
    *,
    iterations: int = 8,
    element_count: int = 65536,
) -> Dict[str, object]:
    tensor = torch.ones(
        element_count,
        dtype=torch.float32,
        device=trainer.device,
    )
    trainer.distributed.barrier()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    for _ in range(iterations):
        tensor.fill_(1.0)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    elapsed_tensor = torch.tensor(
        elapsed,
        dtype=torch.float64,
        device=trainer.device,
    )
    maximum = elapsed_tensor.clone()
    minimum = elapsed_tensor.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    return {
        "backend": trainer.distributed.identity.backend,
        "world_size": trainer.distributed.world_size,
        "iterations": iterations,
        "element_count": element_count,
        "bytes_per_collective_per_rank": tensor.numel() * tensor.element_size(),
        "minimum_elapsed_seconds": float(minimum.item()),
        "maximum_elapsed_seconds": float(maximum.item()),
        "maximum_seconds_per_collective": float(maximum.item()) / iterations,
    }


def save_gradient_probe(trainer: SharedTrainer, path: Path) -> Dict[str, object]:
    batch_state = trainer.batch_source.state_dict()
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    batch = trainer.batch_source.get_batch("train", device=trainer.device)
    with trainer.autocast_context():
        _, loss = trainer.model(batch.inputs, batch.targets)
    local_finite = loss is not None and bool(torch.isfinite(loss).item())
    trainer.distributed.require_all_true(
        local_finite,
        "non-finite gradient-probe loss on at least one rank",
    )
    if loss is None:
        raise RuntimeError("gradient probe did not produce a loss")
    loss.backward()
    trainer.distributed.require_all_true(
        trainer._local_gradients_are_finite(),
        "non-finite gradient probe on at least one rank",
    )
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in trainer.raw_model.named_parameters()
        if parameter.grad is not None
    }
    trainer.distributed.assert_identical_object(
        tuple(gradients),
        "gradient registration",
    )
    global_loss = trainer.distributed.mean_float(loss.detach())
    if trainer.distributed.is_primary:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "loss": global_loss,
                "gradients": gradients,
                "global_starts": trainer.batch_source.training_trace()[-1],
            },
            path,
        )
    trainer.distributed.barrier()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.batch_source.load_state_dict(batch_state)
    return {
        "path": str(path),
        "loss": global_loss,
        "gradient_tensors": len(gradients),
    }


def write_evidence(trainer: SharedTrainer, path: Path, evidence: Dict[str, object]) -> None:
    gathered_reports = trainer.distributed.all_gather_object(trainer.distributed.report())
    evidence["rank_reports"] = gathered_reports
    evidence["world_size"] = trainer.distributed.world_size
    evidence["communication_profile"] = communication_profile(trainer)
    evidence["model_state_max_delta"] = maximum_model_state_delta(trainer)
    evidence["optimizer_state_max_delta"] = maximum_optimizer_state_delta(trainer)
    trainer.distributed.barrier()
    if trainer.distributed.is_primary:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    trainer.distributed.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("construction", "update", "resume", "boundary"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    train_tokens, validation_tokens = token_splits()

    if arguments.mode == "boundary":
        config = stage5_config(
            n_layer=5,
            depth_order=4,
            checkpoint_segment_size=2,
            gradient_accumulation_steps=1,
            max_updates=1,
            decay_updates=1,
        )
    else:
        config = stage5_config()

    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    evidence: Dict[str, object] = {
        "mode": arguments.mode,
        "structure_signature": trainer.distributed_structure_signature(),
        "global_batch_size": config.batch_size,
        "local_batch_size": trainer.batch_source.local_batch_size,
    }

    if arguments.mode == "construction":
        evidence["history"] = []
    elif arguments.mode == "update":
        gradient_path = arguments.output_dir / "ddp_gradient_probe.pt"
        gradient_probe = save_gradient_probe(trainer, gradient_path)
        history = trainer.run()
        checkpoint = arguments.output_dir / "ddp_update.pt"
        trainer.save_checkpoint(checkpoint)
        evidence.update(
            {
                "gradient_probe": gradient_probe,
                "history": history,
                "checkpoint": str(checkpoint),
                "completed_updates": trainer.state.completed_updates,
                "training_trace": trainer.batch_source.training_trace(),
            }
        )
    elif arguments.mode == "resume":
        first_history = trainer.run(target_updates=1)
        resume_checkpoint = arguments.output_dir / "ddp_resume_start.pt"
        trainer.save_checkpoint(resume_checkpoint)
        resumed = SharedTrainer.from_checkpoint(
            resume_checkpoint,
            train_tokens,
            validation_tokens,
        )
        second_history = resumed.run(target_updates=2)
        final_checkpoint = arguments.output_dir / "ddp_resume_final.pt"
        resumed.save_checkpoint(final_checkpoint)
        evidence.update(
            {
                "history": first_history + second_history,
                "checkpoint": str(final_checkpoint),
                "completed_updates": resumed.state.completed_updates,
                "training_trace": resumed.batch_source.training_trace(),
            }
        )
        trainer = resumed
    elif arguments.mode == "boundary":
        history = trainer.run()
        report = trainer.raw_model.last_execution_report
        checkpoint = arguments.output_dir / "ddp_boundary.pt"
        trainer.save_checkpoint(checkpoint)
        evidence.update(
            {
                "history": history,
                "checkpoint": str(checkpoint),
                "completed_updates": trainer.state.completed_updates,
                "checkpoint_segments": report.checkpoint_segments,
                "logical_layers": report.logical_layers,
                "segment_size": report.segment_size,
            }
        )
    else:
        raise RuntimeError(f"unsupported mode: {arguments.mode}")

    write_evidence(trainer, arguments.evidence, evidence)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
# ^^^ THOG
