# vvv THOG
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sheet.checkpoints import save_payload
from sheet.lr_schedule import COSINE_SCHEDULE
from sheet.run_config import OwtRunConfig
from sheet.run_lifecycle import fresh_lifecycle


def make_run_config(root: Path, *, start_label: str = "260715-1200", max_iters: int = 10, model_type: str = "dense") -> OwtRunConfig:
    return OwtRunConfig(
        model_type=model_type,
        run_mode="fresh",
        host_label="testhost",
        run_name="TEST",
        experiment_prefix="TEST",
        data_dir=str(root / "data"),
        checkpoint_root=str(root / "checkpoints"),
        log_root=str(root / "logs"),
        result_root=str(root / "results"),
        wandb_root=str(root / "wandb"),
        run_start_label=start_label,
        max_iters=max_iters,
        decay_iters=max_iters,
        eval_interval=1,
        eval_iters=1,
        log_interval=1,
        checkpoint_interval=0,
        batch_size=1,
        gradient_accumulation_steps=1,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        o_depth=1,
        o_attn_d_model=8,
        o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1,
        o_mlp_d_model=8,
        o_mlp_hidden=8,
        geometry_preset="depth",
        residual_init_depth_source="true_layer_depth" if model_type == "dense" else "dof_implied_depth",
        activation_checkpointing=False,
        checkpoint_segment_size=1,
        learning_rate=1.0e-3,
        min_lr=1.0e-4,
        warmup_iters=1,
        device="cpu",
        dtype="float32",
        instrumentation_backend="none",
        wandb_enabled=False,
    )


def make_lifecycle(config: OwtRunConfig, *, world_size: int = 1) -> Dict[str, Any]:
    paths = config.paths()
    paths["tensorboard_dir"] = Path(config.result_root) / "tensorboard" / config.artifact_name
    return fresh_lifecycle(
        config=config,
        paths=paths,
        world_size=world_size,
        instrumentation_backend=config.instrumentation_backend,
        execution_options={
            "fast_discard": True,
            "bypass_semantic_qkv_adapter": True,
            "direct_factorised_mlp": True,
            "vectorise_per_head_materialisation": True,
        },
        lr_phase={
            "phase_type": COSINE_SCHEDULE,
            "phase_start_update": 0,
            "phase_end_update": config.decay_iters,
            "phase_peak_lr": config.learning_rate,
            "phase_min_lr": config.min_lr,
            "phase_warmup_iters": config.warmup_iters,
        },
    )


def write_checkpoint_stub(root: Path, *, start_label: str = "260715-1200", completed_updates: int = 4, max_iters: int = 10, model_type: str = "dense") -> tuple[Path, OwtRunConfig, Dict[str, Any]]:
    config = make_run_config(root, start_label=start_label, max_iters=max_iters, model_type=model_type)
    lifecycle = make_lifecycle(config)
    training_config = config.to_training_config(vocab_size=32, world_size=1, out_dir=config.paths()["checkpoint_dir"])
    payload = {
        "schema_version": 3,
        "completed_updates": completed_updates,
        "trainer_config": training_config.__dict__.copy(),
        "lifecycle": lifecycle,
    }
    checkpoint_path = config.paths()["checkpoint_path"]
    save_payload(payload, checkpoint_path)
    return checkpoint_path, config, lifecycle


def write_tiny_dataset(path: Path, *, vocab_size: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    train = (np.arange(256, dtype=np.uint16) % vocab_size)
    val = (np.arange(128, dtype=np.uint16) % vocab_size)
    train.tofile(path / "train.bin")
    val.tofile(path / "val.bin")
    with (path / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": vocab_size}, handle)
# ^^^ THOG
