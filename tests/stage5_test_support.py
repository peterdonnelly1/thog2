# vvv THOG
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from sheet.training_config import TrainingConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def token_splits(vocab_size: int = 32, length: int = 512):
    base = torch.arange(length, dtype=torch.long) % vocab_size
    return base.clone(), torch.roll(base, shifts=7)


def stage5_config(**overrides: Any) -> TrainingConfig:
    values = dict(
        model_type="thog2_sheet",
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=8,
        checkpoint_segment_size=1,
        batch_size=4,
        gradient_accumulation_steps=2,
        max_updates=2,
        learning_rate=2.0e-3,
        min_learning_rate=2.0e-4,
        warmup_updates=0,
        decay_updates=2,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=101,
        data_seed=202,
        device="cpu",
        dtype="float32",
    )
    values.update(overrides)
    return TrainingConfig(**values)


def run_ddp_worker(mode: str, output_dir: Path, *, timeout: int = 180) -> Dict[str, Any]:
    evidence_path = output_dir / f"{mode}.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(REPOSITORY_ROOT / "tests" / "stage5_ddp_worker.py"),
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--evidence",
        str(evidence_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "DDP worker failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not evidence_path.exists():
        raise AssertionError(f"DDP worker did not create {evidence_path}")
    return json.loads(evidence_path.read_text(encoding="utf-8"))


def assert_nested_close(
    test,
    left: Any,
    right: Any,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertTrue(
            torch.allclose(left, right, atol=atol, rtol=rtol),
            msg=f"tensor mismatch; max_delta={float((left - right).abs().max())}",
        )
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_close(test, left[key], right[key], atol=atol, rtol=rtol)
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_nested_close(test, left_item, right_item, atol=atol, rtol=rtol)
    elif isinstance(left, float):
        test.assertAlmostEqual(left, right, delta=atol + rtol * abs(left))
    else:
        test.assertEqual(left, right)
# ^^^ THOG
