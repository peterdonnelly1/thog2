# vvv THOG
from __future__ import annotations

import math

from run_thog2_owt import add_console_tokens_per_second
from sheet.run_config import OwtRunConfig
from sheet.run_naming import build_artifact_name
from sheet.stage6_trainer import _session_consumed_tokens


def test_owt_artifact_identity_does_not_change_when_only_target_update_count_changes() -> None:
    short = OwtRunConfig(
        model_type="sheet",
        run_start_label="260714-0800",
        max_iters=10_000,
        warmup_iters=100,
    )
    long = OwtRunConfig(
        model_type="sheet",
        run_start_label="260714-0800",
        max_iters=50_000,
        warmup_iters=100,
    )
    assert short.artifact_name == long.artifact_name
    assert "_n_10000_" not in f"_{short.artifact_name}_"
    assert "_n_50000_" not in f"_{long.artifact_name}_"


def test_legacy_artifact_builder_excludes_mutable_target_update_count() -> None:
    common = dict(
        model_type="thog2_sheet",
        host_label="dreedle",
        run_name="BALLAN",
        dataset_name="openwebtext",
        n_layer=64,
        n_head=16,
        n_embd=1024,
        block_size=256,
        batch_size=32,
        gradient_accumulation_steps=4,
        warmup_iters=100,
        checkpoint_interval=500,
        checkpoint_segment_size=12,
        depth_order=16,
        base_row_order=256,
    )
    assert build_artifact_name(max_iters=10_000, **common) == build_artifact_name(max_iters=50_000, **common)


def test_console_throughput_prefers_session_tokens_but_keeps_lifetime_total() -> None:
    payload = add_console_tokens_per_second(
        {
            "consumed_tokens": 330_000_000,
            "session_consumed_tokens": 327_680,
            "cumulative_training_seconds": 35.0,
        }
    )
    assert payload["consumed_tokens"] == 330_000_000
    assert "session_consumed_tokens" not in payload
    assert math.isclose(payload["tok/s"], 327_680 / 35.0)


def test_console_throughput_retains_fresh_run_fallback() -> None:
    payload = add_console_tokens_per_second(
        {
            "consumed_tokens": 655_360,
            "cumulative_training_seconds": 70.0,
        }
    )
    assert math.isclose(payload["tok/s"], 655_360 / 70.0)


def test_session_token_accounting_starts_at_resume_boundary() -> None:
    assert _session_consumed_tokens(10_000, 10_010, 32_768) == 327_680
    assert _session_consumed_tokens(0, 10, 32_768) == 327_680
# ^^^ THOG
