from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sheet.batch_source import DeterministicBatchSource
from sheet.plastic_depth_fresh_state import (
    build_fresh_training_state,
    destroy_fresh_training_state,
)


@dataclass(frozen=True)
class _Config:
    model_seed: int = 123
    data_seed: int = 456
    device: str = "cpu"
    block_size: int = 4
    batch_size: int = 2
    plastic__initial_layer_count: int = 2
    plastic__runtime_phase: str = "fine"


class _Model(nn.Module):
    def __init__(self, active_count: int) -> None:
        super().__init__()
        self.capacity_geometry = nn.Parameter(torch.randn(8))
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.active_count = active_count


class _Trainer:
    def __init__(
        self,
        config: _Config,
        train_tokens: torch.Tensor,
        validation_tokens: torch.Tensor,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.state = SimpleNamespace(completed_updates=0)
        self.raw_model = _Model(config.plastic__initial_layer_count)
        self.model = self.raw_model
        self.optimizer = torch.optim.AdamW(self.raw_model.parameters(), lr=1.0e-3)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.batch_source = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            data_seed=config.data_seed,
        )
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _tokens(offset: int) -> torch.Tensor:
    return torch.arange(offset, offset + 128, dtype=torch.long)


def _build(config: _Config, *, phase: str, active_count: int, namespace: str):
    return build_fresh_training_state(
        trainer_factory=_Trainer,
        resolved_config=config,
        train_tokens=_tokens(0),
        validation_tokens=_tokens(1000),
        phase=phase,
        active_layer_count=active_count,
        instrumentation_namespace=namespace,
    )


def test_equal_inputs_reconstruct_identical_complete_state() -> None:
    first = _build(_Config(), phase="coarse", active_count=4, namespace="coarse/trial_1")
    second = _build(_Config(), phase="coarse", active_count=4, namespace="coarse/trial_2")

    assert first.fingerprint == second.fingerprint
    assert first.trainer.config.plastic__runtime_phase == "coarse"
    assert first.trainer.raw_model.active_count == 4

    destroy_fresh_training_state(first)
    destroy_fresh_training_state(second)


def test_different_active_counts_preserve_capacity_geometry_and_parameters() -> None:
    lower = _build(_Config(), phase="coarse", active_count=2, namespace="coarse/trial_1")
    upper = _build(_Config(), phase="coarse", active_count=6, namespace="coarse/trial_2")

    assert lower.fingerprint["model"] == upper.fingerprint["model"]
    assert torch.equal(
        lower.trainer.raw_model.capacity_geometry,
        upper.trainer.raw_model.capacity_geometry,
    )
    assert lower.trainer.raw_model.active_count == 2
    assert upper.trainer.raw_model.active_count == 6

    destroy_fresh_training_state(lower)
    destroy_fresh_training_state(upper)


def test_intervening_construction_cannot_perturb_later_state() -> None:
    first = _build(_Config(), phase="fine", active_count=4, namespace="fine/direct")
    disturbance = _build(
        _Config(model_seed=999, data_seed=888),
        phase="coarse",
        active_count=7,
        namespace="coarse/disturbance",
    )
    second = _build(_Config(), phase="fine", active_count=4, namespace="fine/after")

    assert first.fingerprint == second.fingerprint

    destroy_fresh_training_state(first)
    destroy_fresh_training_state(disturbance)
    destroy_fresh_training_state(second)


def test_post_coarse_fine_equals_direct_fine() -> None:
    direct = _build(_Config(), phase="fine", active_count=6, namespace="fine/direct")
    trial = _build(_Config(), phase="coarse", active_count=2, namespace="coarse/trial_1")
    with torch.no_grad():
        trial.trainer.raw_model.weight.add_(100.0)
    destroy_fresh_training_state(trial)
    after_coarse = _build(_Config(), phase="fine", active_count=6, namespace="fine/post_coarse")

    assert direct.fingerprint == after_coarse.fingerprint

    destroy_fresh_training_state(direct)
    destroy_fresh_training_state(after_coarse)


def test_destroy_closes_and_drops_trainer_reference() -> None:
    state = _build(_Config(), phase="coarse", active_count=2, namespace="coarse/trial_1")
    trainer = state.trainer

    destroy_fresh_training_state(state)

    assert trainer.closed
    assert state.trainer is None
