# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from run_thog2_owt import OwtTrainer, load_tokens, source_identity, validate_dataset
from sheet.basis import BASIS_VERSION
from sheet.compact_identity import (
    BASIS_FAMILY_CHEBYSHEV,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
)
from sheet.training_config import TrainingConfig
from sheet.wandb_telemetry import WandbTelemetry, attach_telemetry


DEFAULT_CHECKPOINT = (
    "checkpoints/"
    "SHEET_dreedle__KARITANE_LONG_260706_145723__n_99999_b_12_d_owt_"
    "w_20_k_500_A_4_L_144_H_32_D_2048_C_256_P_80_Q_256_"
    "r_depth_scaled_z_dof_implied_depth_S_12/ckpt.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume the original schema-1 KARITANE_LONG SHEET checkpoint"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default="data/openwebtext")
    parser.add_argument("--max-iters", type=int, default=99999)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--wandb-project", default="thog")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser


def expected_training_config(
    *,
    checkpoint_path: Path,
    vocab_size: int,
    arguments: argparse.Namespace,
) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=256,
        vocab_size=vocab_size,
        n_layer=144,
        n_head=32,
        n_embd=2048,
        dropout=0.0,
        bias=True,
        depth_order=80,
        base_row_order=256,
        residual_init_policy="depth_scaled",
        residual_init_depth_source="dof_implied_depth",
        basis_version=BASIS_VERSION,
        geometry_preset=GEOMETRY_PRESET_LEGACY_SHEET_COL,
        basis_family=BASIS_FAMILY_CHEBYSHEV,
        checkpoint_segment_size=12,
        batch_size=12,
        gradient_accumulation_steps=4,
        max_updates=arguments.max_iters,
        learning_rate=6.0e-4,
        min_learning_rate=6.0e-5,
        warmup_updates=20,
        decay_updates=99999,
        decay_learning_rate=True,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        nonfinite_update_policy="skip",
        max_nonfinite_update_skips=10,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_iters,
        checkpoint_interval=arguments.checkpoint_interval,
        log_interval=arguments.log_interval,
        model_seed=1337,
        data_seed=7331,
        device="cuda",
        dtype="float16",
        out_dir=str(checkpoint_path.parent),
    )


def protocol_sha256(
    config: TrainingConfig,
    dataset: Dict[str, Any],
    checkpoint_path: Path,
) -> str:
    payload = {
        "config": asdict(config),
        "dataset": dataset,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    arguments = build_parser().parse_args()
    checkpoint_path = Path(arguments.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"KARITANE_LONG checkpoint is missing: {checkpoint_path}"
        )

    dataset_dir = Path(arguments.data_dir)
    dataset = validate_dataset(dataset_dir, 256)
    config = expected_training_config(
        checkpoint_path=checkpoint_path,
        vocab_size=int(dataset["vocab_size"]),
        arguments=arguments,
    )

    train_tokens = load_tokens(dataset_dir / "train.bin")
    validation_tokens = load_tokens(dataset_dir / "val.bin")
    trainer = OwtTrainer.from_checkpoint(
        checkpoint_path,
        train_tokens,
        validation_tokens,
        expected_config=config,
        overrides={
            "device": config.device,
            "dtype": config.dtype,
            "max_updates": config.max_updates,
            "eval_interval": config.eval_interval,
            "eval_batches": config.eval_batches,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_segment_size": config.checkpoint_segment_size,
            "out_dir": config.out_dir,
            "log_interval": config.log_interval,
            "nonfinite_update_policy": config.nonfinite_update_policy,
            "max_nonfinite_update_skips": config.max_nonfinite_update_skips,
        },
    )

    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    run_id = f"{timestamp}_KARITANE_LONG_RESUME"
    result_dir = Path("results") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "result.json"
    source = source_identity()

    telemetry = WandbTelemetry(
        enabled=arguments.wandb_mode != "disabled",
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        mode=arguments.wandb_mode,
        root=Path("wandb"),
        name=run_id,
        group="KARITANE_LONG",
        job_type="sheet",
        config={
            **asdict(config),
            "resume_checkpoint": str(checkpoint_path),
            "source_commit": source["commit"],
            "source_branch": source["branch"],
            "dataset_record": dataset,
            "parameter_report": trainer.parameter_report,
        },
    )

    try:
        if trainer.distributed.is_primary:
            telemetry.start()
            telemetry.add_initial_summary(trainer.parameter_report)
        attach_telemetry(trainer, telemetry)
        result = trainer.run_pilot(
            run_id=run_id,
            protocol_sha256=protocol_sha256(
                config,
                dataset,
                checkpoint_path,
            ),
            dataset=dataset,
            result_path=result_path,
        )
        result["resume_checkpoint"] = str(checkpoint_path)
        result["source"] = source
        if trainer.distributed.is_primary:
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            telemetry.add_final_result(result)
        return 0
    finally:
        if trainer.distributed.is_primary:
            telemetry.finish()
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
