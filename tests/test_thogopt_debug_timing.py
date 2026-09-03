"""Forensic timing must not add clock work or console rows to normal runs."""
from types import SimpleNamespace

import constants
import pytest
import torch

from run_thog2_owt_core import OwtTrainer
from sheet.stage6_trainer import Stage6Trainer
from sheet import thogopt
from tests.test_thogopt import make_optimizer, supply


@pytest.mark.parametrize('debug_level', [99, 100])
def test_debug_threshold_gates_clocks_and_extra_row(monkeypatch, capsys, debug_level):
    monkeypatch.setattr(constants, 'DEBUG', debug_level)
    _, parameter, optimizer = make_optimizer(history=2)
    calls = []

    def clock():
        assert debug_level > 99, 'normal optimizer execution called a diagnostic clock'
        calls.append(True)
        return len(calls) * .001

    monkeypatch.setattr(thogopt, 'time', SimpleNamespace(perf_counter=clock))
    supply(optimizer, torch.ones(4, 4, dtype=torch.float64))
    optimizer.step()
    assert bool(calls) == (debug_level > 99)
    assert optimizer.last_step_metrics['timing_enabled'] == (debug_level > 99)
    assert optimizer.state[parameter]['step'] == 1

    trainer = object.__new__(OwtTrainer)
    trainer.optimizer = optimizer
    trainer.distributed = SimpleNamespace(is_primary=True, world_size=1)
    monkeypatch.setattr(Stage6Trainer, '_print_progress', lambda *args, **kwargs: print('ordinary progress'))
    trainer._print_progress('test', 'optimizer_progress')
    output = capsys.readouterr().out
    assert 'ordinary progress' in output
    assert ('THOGOPT last update:' in output) == (debug_level > 99)


def test_timing_setting_does_not_change_updates_or_checkpoint_compatibility(monkeypatch):
    monkeypatch.setattr(constants, 'DEBUG', 99)
    _, quiet_parameter, quiet = make_optimizer(history=2)
    monkeypatch.setattr(constants, 'DEBUG', 100)
    _, timed_parameter, timed = make_optimizer(history=2)
    with torch.no_grad():
        timed_parameter.copy_(quiet_parameter)
    for _ in range(3):
        gradient = torch.randn(4, 4, dtype=torch.float64)
        for optimizer in (quiet, timed):
            supply(optimizer, gradient)
            optimizer.step()
        torch.testing.assert_close(quiet_parameter, timed_parameter, rtol=0, atol=0)
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(quiet.state[quiet_parameter][key], timed.state[timed_parameter][key], rtol=0, atol=0)
    timed.load_state_dict(quiet.state_dict())
    assert timed.timing_enabled
    assert timed.state[timed_parameter]['step'] == 3
