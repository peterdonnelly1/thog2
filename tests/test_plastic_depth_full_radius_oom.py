from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sheet.training_model as training_model_module
from sheet.checkpointing import CheckpointExecutionReport
from sheet.plastic_depth_inline import PlasticDepthInlineProbeRequest
from sheet.training_model import TrainingSheetGPT


def _report(logical_layers: int) -> CheckpointExecutionReport:
    return CheckpointExecutionReport(
        checkpointing_used=False,
        checkpoint_segments=0,
        logical_layers=logical_layers,
        segment_size=0,
    )


def _dummy_model():
    return SimpleNamespace(
        config=SimpleNamespace(n_layer=5),
        checkpoint_segment_size=0,
        _logical_block=lambda hidden, layer_index: hidden + float(layer_index + 1),
        training=True,
        _plastic_depth_candidate_head_loss=(
            lambda hidden, targets, sampled_token_indices: hidden.float().mean()
        ),
    )


def _request(sync_calls, prepared):
    def prepare() -> None:
        prepared.append(True)

    def synchronize(count: int, local_feasible: bool) -> bool:
        sync_calls.append((count, local_feasible))
        return local_feasible

    return PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4, 5),
        sampled_token_indices=None,
        selector=lambda candidates: candidates[0][0],
        recoverable_upward_counts=(4, 5),
        prepare_recoverable_upward_counts=prepare,
        synchronize_recoverable_upward_candidate=synchronize,
    )


def test_request_requires_contiguous_final_upward_suffix() -> None:
    valid = _request([], [])
    valid.validate(maximum_count=5)

    invalid = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4, 5),
        sampled_token_indices=None,
        selector=lambda candidates: candidates[0][0],
        recoverable_upward_counts=(3, 5),
        prepare_recoverable_upward_counts=lambda: None,
        synchronize_recoverable_upward_candidate=lambda count, feasible: feasible,
    )
    with pytest.raises(ValueError, match="final candidate suffix"):
        invalid.validate(maximum_count=5)


def test_one_upward_candidate_oom_preserves_all_completed_lower_candidates(
    monkeypatch,
) -> None:
    hidden = torch.zeros(1, 1, 1)
    targets = torch.zeros(1, 1, dtype=torch.long)
    prepared = []
    sync_calls = []

    def execute_lower(
        hidden,
        *,
        checkpoint_counts,
        **kwargs,
    ):
        checkpoints = tuple(
            (count, torch.full_like(hidden, float(count)))
            for count in checkpoint_counts
        )
        return checkpoints, _report(checkpoint_counts[-1])

    def execute_upward(hidden, *, layer_indices, **kwargs):
        candidate_count = int(layer_indices[0]) + 1
        if candidate_count == 5:
            raise torch.OutOfMemoryError("synthetic candidate OOM")
        return torch.full_like(hidden, float(candidate_count)), _report(1)

    monkeypatch.setattr(
        training_model_module,
        "execute_logical_layer_checkpoints",
        execute_lower,
    )
    monkeypatch.setattr(
        training_model_module,
        "execute_logical_layers",
        execute_upward,
    )

    checkpoints, losses, report = (
        TrainingSheetGPT._plastic_depth_recoverable_probe_candidate_suffix(
            _dummy_model(),
            hidden,
            targets,
            _request(sync_calls, prepared),
        )
    )

    assert tuple(checkpoints) == (2, 3, 4)
    assert tuple(count for count, _ in losses) == (2, 3, 4)
    assert prepared == [True]
    assert sync_calls == [(4, True), (5, False)]
    assert report.logical_layers == 4


def test_all_feasible_upward_candidates_extend_the_shared_prefix(monkeypatch) -> None:
    hidden = torch.zeros(1, 1, 1)
    targets = torch.zeros(1, 1, dtype=torch.long)
    prepared = []
    sync_calls = []

    monkeypatch.setattr(
        training_model_module,
        "execute_logical_layer_checkpoints",
        lambda hidden, checkpoint_counts, **kwargs: (
            tuple(
                (count, torch.full_like(hidden, float(count)))
                for count in checkpoint_counts
            ),
            _report(checkpoint_counts[-1]),
        ),
    )
    monkeypatch.setattr(
        training_model_module,
        "execute_logical_layers",
        lambda hidden, layer_indices, **kwargs: (
            torch.full_like(hidden, float(int(layer_indices[0]) + 1)),
            _report(1),
        ),
    )

    checkpoints, losses, report = (
        TrainingSheetGPT._plastic_depth_recoverable_probe_candidate_suffix(
            _dummy_model(),
            hidden,
            targets,
            _request(sync_calls, prepared),
        )
    )

    assert tuple(checkpoints) == (2, 3, 4, 5)
    assert tuple(count for count, _ in losses) == (2, 3, 4, 5)
    assert prepared == [True]
    assert sync_calls == [(4, True), (5, True)]
    assert report.logical_layers == 5
