# vvv THOG
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from sheet.hyperblock import HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig


def _config(out_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
        hyperblock_compressor="chebyshev",
        hyperblock_compressor_version="auto",
        hyperblock_common_family_order=3,
        hyperblock_attention_family_order=2,
        hyperblock_mlp_family_order=1,
        hyperblock_depth_order=2,
        hyperblock_d_model_order=4,
        hyperblock_mlp_hidden_order=4,
        hyperblock_attention_head_order=2,
        hyperblock_attention_head_channel_order=4,
        checkpoint_segment_size=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        max_updates=1,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-3,
        decay_updates=1,
        decay_learning_rate=False,
        weight_decay=0.0,
        grad_clip=0.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        log_interval=1,
        model_seed=941,
        data_seed=149,
        device="cpu",
        dtype="float32",
        out_dir=str(out_dir),
    )


@pytest.mark.parametrize("fast_discard", (False, True))
def test_hyperblock_stage6_pilot_reports_persistent_regions(
    monkeypatch: pytest.MonkeyPatch,
    fast_discard: bool,
) -> None:
    monkeypatch.setenv("THOG2_FAST_DISCARD", str(fast_discard).lower())
    monkeypatch.setenv("THOG2_TORCH_COMPILE", "false")
    tokens = torch.arange(128, dtype=torch.long).remainder(32)
    with tempfile.TemporaryDirectory() as directory:
        out_dir = Path(directory)
        trainer = Stage6Trainer(_config(out_dir), tokens, tokens)
        expected_names = (
            set(trainer.raw_model.trajectory.coefficients)
            | set(trainer.raw_model.trajectory.vector_parameters)
        )
        try:
            result = trainer.run_pilot(
                run_id=f"hyperblock_stage6_{fast_discard}",
                protocol_sha256="protocol",
                dataset={"fixture": True},
                result_path=out_dir / "result.json",
            )
        finally:
            trainer.close()

    assert result["budget"]["completed_updates"] == 1
    assert result["checkpoint"]["bytes"] > 0
    assert len(result["gradient_diagnostics"]) == 1
    gradient_rows = result["gradient_diagnostics"][0]["families"]
    assert set(gradient_rows) == expected_names
    assert {"common", "attention", "mlp"} <= set(gradient_rows)

    sheet_diagnostics = result["sheet_diagnostics"]
    assert sheet_diagnostics is not None
    utilization_rows = sheet_diagnostics["coefficient_utilization"]
    assert set(utilization_rows) == expected_names
    for name in ("common", "attention", "mlp"):
        assert utilization_rows[name]["semantic_type"] == "coupled_coefficient_region"
    assert sheet_diagnostics["compact_state_violations"] == []
# ^^^ THOG
