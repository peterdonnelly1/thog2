# vvv THOG real width captures inside the existing isolated Instra fixture
from pathlib import Path
import copy
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sheet.local_chart_store import LocalChartStore
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig

config = TrainingConfig(model_type='thog2_sheet', geometry_preset='width-type-I',
    width_enabled=True, width_order=3, width_compressor='dct', n_embd=8, n_head=2,
    n_layer=2, block_size=5, vocab_size=17, batch_size=1, max_updates=3,
    instrumentation__width_activation_curves__mode='probes',
    instrumentation__width_activation_curves__log_every_n_steps=1,
    instrumentation__width_activation_curves__probe_every_n_steps=1)
trainer = Stage6Trainer(config, torch.arange(140) % 17, torch.arange(140) % 17)
for _ in range(3):
    trainer.train_one_update()
captures = list(trainer._width_instrumentation.history)
trainer.close()
original_close = LocalChartStore.close
def close_with_width(self, **kwargs):
    if self.path.parent.name in ('fixture_00', 'fixture_01'):
        for capture in captures:
            self.append_width_snapshot(copy.deepcopy(capture), history_length=3)
    return original_close(self, **kwargs)
LocalChartStore.close = close_with_width
runpy.run_path(str(Path(__file__).with_name('instra_sep30_fixture.py')), run_name='__main__')
# ^^^ THOG
