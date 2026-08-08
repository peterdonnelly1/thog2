from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sheet.batch_source import DeterministicBatchSource
from sheet.plastic_depth_coarse_runner import (
    coarse_trial_training_config,
    render_plastic_coarse_trial_header,
    run_fixed_plastic_coarse_trial,
)
from sheet.plastic_depth_fresh_state import PlasticFreshTrainingState


@dataclass(frozen=True)
class _Config:
    plastic__runtime_phase: str = "coarse"
    plastic__coarse_phase: str = "disabled"
    plastic__initial_layer_count: int = 4
    plastic__log_interval_coarse: int = 10
    max_updates: int = 3
    batch_size: int = 2
    block_size: int = 4
    gradient_accumulation_steps: int = 1


class _Distributed:
    def require_all_true(self, value: bool, message: str) -> None:
        if not value:
            raise RuntimeError(message)

    def mean_float(self, value: torch.Tensor) -> float:
        return float(value.item())

    def max_float(self, value: torch.Tensor) -> float:
        return float(value.item())


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = inputs.float().mean() * 0.0 + self.weight.square()
        return inputs, loss


class _Trainer:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.config = _Config()
        self.device = torch.device("cpu")
        self.model = _Model()
        self.raw_model = self.model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.distributed = _Distributed()
        self.state = SimpleNamespace(completed_updates=0)
        self.batch_source = DeterministicBatchSource(
            torch.arange(128),
            torch.arange(1000, 1128),
            block_size=4,
            batch_size=2,
            data_seed=17,
        )
        self.fail_at = fail_at
        self.train_calls = 0

    def autocast_context(self):
        return torch.autocast("cpu", enabled=False)

    def train_one_update(self):
        self.train_calls += 1
        if self.fail_at == self.train_calls:
            raise RuntimeError("synthetic failure")
        batch = self.batch_source.get_batch("train", device=self.device)
        _, loss = self.model(batch.inputs, batch.targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.state.completed_updates += 1
        return {
            "training_loss": float(loss.detach().item()),
            "skipped_update": 0.0,
        }


def _state(trainer: _Trainer, layers: int = 4) -> PlasticFreshTrainingState:
    return PlasticFreshTrainingState(
        trainer=trainer,
        phase="coarse",
        active_layer_count=layers,
        instrumentation_namespace="coarse/trial_1",
        fingerprint={},
    )


class _Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self) -> float:
        return float(next(self.values))


def test_header_is_compact_and_uses_required_wording() -> None:
    header = render_plastic_coarse_trial_header(
        trial_index=2,
        trial_count=4,
        layers=8,
        n_steps=500,
        evaluation_steps_count=10,
        objective="relative_training_wall_time",
        geometry_initialisation="equidistant",
    )

    assert header.startswith("TRIAL 2/4")
    assert "  layers:      8" in header
    assert "  steps:       500" in header
    assert "starting at step" not in header
    assert "validation mean over final 10 batches" in header
    assert "fixed equidistant" in header


def test_fixed_trial_runs_exact_local_steps_and_final_validation_batches() -> None:
    trainer = _Trainer()
    progress = []
    result = run_fixed_plastic_coarse_trial(
        _state(trainer),
        trial_index=1,
        n_steps=3,
        evaluation_steps_count=2,
        clock=_Clock((10.0, 16.0)),
        progress_clock=_Clock((20.0, 20.5, 21.5)),
        progress_sink=progress.append,
    )

    assert result.status == "success"
    assert result.training_steps == 3
    assert trainer.state.completed_updates == 3
    assert result.training_elapsed_seconds == 6.0
    assert len(result.validation_losses) == 2
    assert progress[0].startswith("C 01      0/3")
    assert progress[-1].startswith("C 01      3/3")
    assert "step" not in progress[-1]
    assert "layers=" not in progress[-1]
    assert progress[-1].endswith("     1.5s")
    assert trainer.batch_source.validation_trace()


def test_training_timing_excludes_final_validation() -> None:
    trainer = _Trainer()
    clock = _Clock((100.0, 104.5))

    result = run_fixed_plastic_coarse_trial(
        _state(trainer),
        trial_index=1,
        n_steps=3,
        evaluation_steps_count=2,
        clock=clock,
    )

    assert result.training_elapsed_seconds == 4.5


def test_recoverable_trial_failure_is_recorded() -> None:
    trainer = _Trainer(fail_at=2)
    result = run_fixed_plastic_coarse_trial(
        _state(trainer),
        trial_index=3,
        n_steps=3,
        evaluation_steps_count=2,
        clock=_Clock((1.0, 2.0)),
    )

    assert result.status == "failed"
    assert result.training_steps == 1
    assert result.error_class == "RuntimeError"
    assert result.error_message == "synthetic failure"


def test_coarse_config_is_fixed_to_candidate_and_local_budget() -> None:
    resolved = coarse_trial_training_config(
        _Config(plastic__runtime_phase="fine", plastic__coarse_phase="enabled"),
        active_layer_count=8,
        n_steps=12,
    )

    assert resolved.plastic__runtime_phase == "coarse"
    assert resolved.plastic__coarse_phase == "disabled"
    assert resolved.plastic__initial_layer_count == 8
    assert resolved.max_updates == 12
