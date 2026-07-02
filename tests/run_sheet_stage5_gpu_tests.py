# vvv THOG
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

from sheet.checkpoints import load_payload
from sheet.stage4_trainer import Stage4Trainer
from sheet.stage5_profile import (
    profile_basis_construction,
    profile_block_components,
    profile_checkpoint_recomputation,
    profile_materialisation,
    synchronize,
    timed_call,
)
from sheet.stage5_target import (
    principal_stage5_config,
    reduced_stage5_config,
    synthetic_token_splits,
)


class AcceptanceFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def cleanup_trainer(trainer) -> None:
    if trainer is not None:
        trainer.close()
        del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bounded-updates", type=int, default=4)
    parser.add_argument("--include-q1024", action="store_true")
    arguments = parser.parse_args()
    if arguments.bounded_updates < 3:
        raise ValueError("bounded-updates must be at least 3")

    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = arguments.evidence.parent / "stage5_gpu_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    shared: Dict[str, Any] = {}

    def run_case(identifier: str, name: str, function: Callable[[], Dict[str, Any]]) -> None:
        started = time.perf_counter()
        try:
            details = function()
            results.append(
                {
                    "id": identifier,
                    "name": name,
                    "status": "passed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "details": details,
                }
            )
        except Exception as error:
            results.append(
                {
                    "id": identifier,
                    "name": name,
                    "status": "error",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )

    if not torch.cuda.is_available():
        evidence = {
            "stage": 5,
            "suite": "gpu_acceptance",
            "satisfied": False,
            "reason": "CUDA is not available",
            "test_execution": {
                "tests_run": 0,
                "failure_count": 0,
                "error_count": 1,
                "skipped_count": 0,
                "successful": False,
            },
        }
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    torch.cuda.reset_peak_memory_stats(device)

    def construct_principal() -> Dict[str, Any]:
        config = principal_stage5_config(
            device=str(device),
            dtype=dtype,
            max_updates=arguments.bounded_updates,
            out_dir=str(runtime_dir),
        )
        train_tokens, validation_tokens = synthetic_token_splits(config.vocab_size)
        trainer, construction_seconds = timed_call(
            device,
            lambda: Stage4Trainer(config, train_tokens, validation_tokens),
        )
        shared["trainer"] = trainer
        shared["config"] = config
        shared["training_losses"] = []
        shared["construction_seconds"] = construction_seconds
        report = trainer.parameter_report
        family_rows = list(report["families"])
        row_orders = sorted({int(row["row_order"]) for row in family_rows})
        require(config.n_layer == 144, "principal layer count is not 144")
        require(config.n_embd == 768, "principal embedding width is not 768")
        require(config.depth_order == 16, "principal depth order is not 16")
        require(config.base_row_order == 128, "principal base row order is not 128")
        require(128 in row_orders, "principal Q_d=128 family is absent")
        require(512 in row_orders, "principal Q_4d=512 family is absent")
        require(not trainer.raw_model.compact_state_violations(), "compact-state violation")
        require(
            all(bool(torch.isfinite(parameter).all().item()) for parameter in trainer.raw_model.parameters()),
            "principal parameters contain non-finite values",
        )
        require(
            int(report["persistent_parameters"])
            == sum(parameter.numel() for parameter in trainer.raw_model.parameters()),
            "principal persistent parameter count does not reconcile",
        )
        return {
            "construction_seconds": construction_seconds,
            "model_args": config.model_arguments(),
            "row_orders": row_orders,
            "parameter_report": report,
            "memory": trainer.memory_telemetry.report(),
        }

    def first_update() -> Dict[str, Any]:
        trainer = shared["trainer"]
        initial_evaluation, evaluation_seconds = timed_call(device, trainer.evaluate)
        metrics, update_seconds = timed_call(device, trainer.train_one_update)
        shared["initial_evaluation"] = initial_evaluation
        shared["training_losses"].append(metrics["training_loss"])
        require(finite_number(metrics["training_loss"]), "first training loss is non-finite")
        require(finite_number(metrics["gradient_norm"]), "first gradient norm is non-finite")
        execution = trainer.raw_model.last_execution_report
        require(execution.checkpointing_used, "principal update did not use checkpointing")
        require(execution.checkpoint_segments == 36, "principal segment count is not 36")
        return {
            "initial_evaluation": initial_evaluation,
            "evaluation_seconds": evaluation_seconds,
            "update_seconds": update_seconds,
            "metrics": metrics,
            "execution": {
                "checkpointing_used": execution.checkpointing_used,
                "checkpoint_segments": execution.checkpoint_segments,
                "logical_layers": execution.logical_layers,
                "segment_size": execution.segment_size,
            },
            "memory": trainer.memory_telemetry.report(),
        }

    def bounded_smoke() -> Dict[str, Any]:
        trainer = shared["trainer"]
        update_seconds: List[float] = []
        while trainer.state.completed_updates < arguments.bounded_updates:
            metrics, elapsed = timed_call(device, trainer.train_one_update)
            shared["training_losses"].append(metrics["training_loss"])
            update_seconds.append(elapsed)
            require(finite_number(metrics["training_loss"]), "bounded loss is non-finite")
            require(finite_number(metrics["gradient_norm"]), "bounded gradient norm is non-finite")
        final_evaluation, evaluation_seconds = timed_call(device, trainer.evaluate)
        checkpoint_path = runtime_dir / "principal_ckpt.pt"
        _, checkpoint_seconds = timed_call(
            device,
            lambda: trainer.save_checkpoint(checkpoint_path),
        )
        payload = load_payload(checkpoint_path)
        require(payload["completed_updates"] == arguments.bounded_updates, "checkpoint update count mismatch")
        require(
            not any(key.startswith("trajectory.bases.") for key in payload["model"]),
            "fixed basis appeared in principal checkpoint",
        )
        losses = list(shared["training_losses"])
        learning_signal = min(losses[1:]) < losses[0] or (
            float(final_evaluation["val"])
            < float(shared["initial_evaluation"]["val"])
        )
        require(learning_signal, "principal bounded run showed no short-term learning signal")
        require(all(finite_number(value) for value in losses), "principal loss history is non-finite")
        shared["final_evaluation"] = final_evaluation
        return {
            "completed_updates": trainer.state.completed_updates,
            "training_losses": losses,
            "initial_evaluation": shared["initial_evaluation"],
            "final_evaluation": final_evaluation,
            "learning_signal": learning_signal,
            "update_seconds": update_seconds,
            "evaluation_seconds": evaluation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        }

    def memory_accounting() -> Dict[str, Any]:
        trainer = shared["trainer"]
        report = trainer.memory_telemetry.report()
        samples = report["samples"]
        phases = [sample["phase"] for sample in samples]
        required_phases = {
            "trainer_start",
            "model_construction",
            "optimizer_allocation",
            "first_optimizer_state",
            "steady_update",
            "evaluation",
            "checkpoint",
        }
        require(required_phases.issubset(phases), f"missing memory phases: {sorted(required_phases - set(phases))}")
        require(
            all(int(sample["peak_allocated_bytes"]) >= int(sample["allocated_bytes"]) for sample in samples),
            "memory phase has allocated bytes above peak allocated bytes",
        )
        require(
            max(int(sample["peak_allocated_bytes"]) for sample in samples) > 0,
            "principal memory telemetry reported no CUDA allocation",
        )
        return report

    def basis_profile() -> Dict[str, Any]:
        rows = profile_basis_construction(
            device,
            include_q1024=arguments.include_q1024,
        )
        require(all(bool(row["finite"]) for row in rows), "basis profile contains non-finite basis")
        names = {row["name"] for row in rows}
        require({"depth_p16", "row_q128", "row_4d_q512"}.issubset(names), "basis profile is incomplete")
        return {"profiles": rows}

    def materialisation_profile() -> Dict[str, Any]:
        trainer = shared["trainer"]
        model = trainer.raw_model
        rows = profile_materialisation(
            model,
            dtype_name=dtype,
            layer_indices=(0, 72, 143),
        )
        block = profile_block_components(
            model,
            dtype_name=dtype,
            batch_size=1,
            sequence_length=256,
            layer_index=72,
        )
        recomputation = profile_checkpoint_recomputation(
            device=str(device),
            dtype=dtype,
        )
        require(all(bool(row["finite"]) for row in rows), "materialisation profile contains non-finite values")
        require(bool(block["finite"]), "block profile contains non-finite values")
        require(float(recomputation["loss_delta"]) <= 1.0e-4, "checkpoint profile loss mismatch")
        return {
            "materialisation": rows,
            "block_components": block,
            "checkpoint_recomputation": recomputation,
        }

    def dtype_matrix() -> Dict[str, Any]:
        rows = []
        for dtype_name in ("float32", "bfloat16", "float16"):
            if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise AcceptanceFailure("bfloat16 is required but unsupported")
            config = reduced_stage5_config(device=str(device), dtype=dtype_name)
            train_tokens, validation_tokens = synthetic_token_splits(config.vocab_size, length=8192)
            trainer = Stage4Trainer(config, train_tokens, validation_tokens)
            try:
                metrics, elapsed = timed_call(device, trainer.train_one_update)
                require(finite_number(metrics["training_loss"]), f"{dtype_name} loss is non-finite")
                require(finite_number(metrics["gradient_norm"]), f"{dtype_name} gradient is non-finite")
                rows.append(
                    {
                        "dtype": dtype_name,
                        "elapsed_seconds": elapsed,
                        "metrics": metrics,
                        "memory": trainer.memory_telemetry.report(),
                    }
                )
            finally:
                cleanup_trainer(trainer)
        return {"dtypes": rows}

    def uneven_gpu_boundary() -> Dict[str, Any]:
        config = reduced_stage5_config(
            device=str(device),
            dtype=dtype,
            n_layer=17,
            depth_order=8,
            checkpoint_segment_size=4,
        )
        train_tokens, validation_tokens = synthetic_token_splits(config.vocab_size, length=8192)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        try:
            metrics, elapsed = timed_call(device, trainer.train_one_update)
            execution = trainer.raw_model.last_execution_report
            require(execution.checkpoint_segments == 5, "17-layer boundary did not create 5 segments")
            require(finite_number(metrics["training_loss"]), "boundary loss is non-finite")
            return {
                "elapsed_seconds": elapsed,
                "metrics": metrics,
                "checkpoint_segments": execution.checkpoint_segments,
                "logical_layers": execution.logical_layers,
                "segment_size": execution.segment_size,
            }
        finally:
            cleanup_trainer(trainer)

    def longer_stability() -> Dict[str, Any]:
        trainer = shared["trainer"]
        losses = list(shared["training_losses"])
        require(trainer.state.completed_updates >= 3, "stability run was too short")
        require(all(finite_number(value) for value in losses), "stability losses are non-finite")
        require(
            all(bool(torch.isfinite(parameter).all().item()) for parameter in trainer.raw_model.parameters()),
            "principal parameters became non-finite",
        )
        require(not trainer.raw_model.compact_state_violations(), "persistent dense-stack violation")
        return {
            "completed_updates": trainer.state.completed_updates,
            "training_losses": losses,
            "final_evaluation": shared["final_evaluation"],
            "compact_state_violations": list(trainer.raw_model.compact_state_violations()),
        }

    run_case("S5-01", "principal geometry construction", construct_principal)
    run_case("S5-02", "principal forward backward and first update", first_update)
    run_case("S5-03", "bounded principal smoke with validation and checkpoint", bounded_smoke)
    run_case("S5-04", "principal GPU memory accounting", memory_accounting)
    run_case("S5-05", "basis construction profiling", basis_profile)
    run_case("S5-06", "materialisation and checkpoint profiling", materialisation_profile)
    run_case("S5-07", "supported GPU dtype matrix", dtype_matrix)
    run_case("S5-12", "uneven GPU checkpoint boundary", uneven_gpu_boundary)
    run_case("S5-13", "longer principal stability and compact-state guard", longer_stability)

    cleanup_trainer(shared.get("trainer"))
    passed = [item for item in results if item["status"] == "passed"]
    errors = [item for item in results if item["status"] == "error"]
    evidence = {
        "stage": 5,
        "suite": "gpu_acceptance",
        "device": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "dtype": dtype,
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bounded_updates": arguments.bounded_updates,
        "include_q1024": arguments.include_q1024,
        "results": results,
        "test_execution": {
            "tests_run": len(results),
            "failure_count": 0,
            "error_count": len(errors),
            "skipped_count": 0,
            "successful": len(errors) == 0 and len(passed) == len(results),
        },
        "satisfied": len(errors) == 0 and len(passed) == len(results),
    }
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not evidence["satisfied"]:
        for error in errors:
            print(f"{error['id']} {error['name']}: {error['error']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
