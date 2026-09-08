# vvv THOG dynamic pre-materialisation policy, configuration, and topology regression tests
from __future__ import annotations

import pytest
import torch

from run_thog2_owt_core import build_parser
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.premat import (
    CandidateEnvelope,
    PrematMemoryObservation,
    decide_candidate_admission,
    validate_premat_configuration,
)


def _observation(*, allocated: int = 100, reserved: int = 120, peak: int = 200) -> PrematMemoryObservation:
    return PrematMemoryObservation(
        process_allocated_bytes=allocated,
        process_reserved_bytes=reserved,
        process_ordinary_peak_bytes=peak,
        device_free_bytes=500,
        device_total_bytes=1000,
    )


def test_peak_free_admission_enforces_process_and_device_guards_separately() -> None:
    admitted = decide_candidate_admission(
        observation=_observation(),
        envelope=CandidateEnvelope(30, 50, 20),
        stay_below_current_peak=True,
        gpu_memory_buffer_bytes=400,
    )
    assert admitted.admitted
    assert admitted.process_guard_passed
    assert admitted.device_guard_passed

    process_blocked = decide_candidate_admission(
        observation=_observation(peak=130),
        envelope=CandidateEnvelope(30, 50, 20),
        stay_below_current_peak=True,
        gpu_memory_buffer_bytes=0,
    )
    assert not process_blocked.admitted
    assert process_blocked.reason == "ordinary_process_peak"
    assert process_blocked.device_guard_passed


def test_aggressive_admission_reuses_allocator_cache_without_double_counting() -> None:
    decision = decide_candidate_admission(
        observation=_observation(allocated=100, reserved=180),
        envelope=CandidateEnvelope(50, 75, 0),
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=480,
    )
    assert decision.predicted_device_used_bytes == 500
    assert decision.admitted


def test_premat_headroom_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_premat_configuration(
            premat="enabled",
            attention_mode="fused",
            stay_below_current_peak=True,
            stay_within_global_buffer=True,
            gpu_memory_buffer_gb=1.0,
            logging="disabled",
            instra="disabled",
        )


def test_retired_plastic_memory_budget_cli_names_replacement(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--plastic__layer_count__memory_budget_gib", "12"])
    assert "--premat_gpu_memory_buffer_gb" in capsys.readouterr().err


def test_unfused_attention_matches_fused_math_on_cpu() -> None:
    common = dict(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=1,
        geometry_preset="depth",
        basis_family="chebyshev",
        direct_factorised_mlp=False,
    )
    torch.manual_seed(7)
    fused = SheetGPT(SheetGPTConfig(**common, premat_attention_mode="fused"))
    unfused = SheetGPT(SheetGPTConfig(**common, premat_attention_mode="unfused"))
    unfused.load_state_dict(fused.state_dict())
    fused.eval()
    unfused.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    fused_logits, _ = fused(tokens)
    unfused_logits, _ = unfused(tokens)
    torch.testing.assert_close(unfused_logits, fused_logits, rtol=1.0e-5, atol=1.0e-6)
# ^^^ THOG
