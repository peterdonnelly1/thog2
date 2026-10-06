# vvv THOG measure the cost of accidentally enabling DEPTH in a WIDTH-only Recipe
"""Matched CPU update timings for the erroneous joint path and intended WIDTH path.

Dimensions are deliberately smaller than the user's CUDA run. These measurements
diagnose the selection bug; they do not predict RTX 4090 Laptop update times.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(variant, updates, seed):
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
    import torch
    from sheet.stage6_trainer import Stage6Trainer
    from sheet.training_config import TrainingConfig

    os.environ['THOG2_OPTIMIZER']='adamw'
    os.environ['THOG2_FAST_DISCARD']='true'
    joint=variant=='accidental_width_depth'
    config=TrainingConfig(model_type='thog2_sheet', geometry_preset='width-type-I',
        width_enabled=True, width_order=64, width_compressor='dct', width_depth_enabled=joint,
        n_embd=128, n_head=4, n_layer=16, depth_order=12, block_size=32, vocab_size=64,
        batch_size=2, gradient_accumulation_steps=2, checkpoint_segment_size=4,
        max_updates=updates, dropout=0, bias=True, dtype='float32', device='cpu',
        model_seed=seed, data_seed=seed+100, warmup_updates=0,
        learning_rate=0.0009, min_learning_rate=0.00009, decay_updates=updates)
    tokens=torch.arange(8192)%64
    with contextlib.redirect_stdout(io.StringIO()):
        trainer=Stage6Trainer(config,tokens,tokens)
    elapsed=[]
    try:
        for _ in range(updates):
            started=time.perf_counter()
            trainer.train_one_update()
            elapsed.append(time.perf_counter()-started)
        steady=elapsed[2:]
        return {'variant':variant,'seed':seed,'D':128,'r':64,'L':16,'P':12 if joint else None,
            'sequence_length':32,'batch_size':2,'gradient_accumulation':2,'checkpoint_segment_size':4,
            'dtype':'float32','cpu_threads':2,'optimizer':'AdamW','instrumentation':'off',
            'timing_window':'complete train_one_update; forward, AC replay, backward, clipping, optimizer and schedule; excludes startup/evaluation',
            'completed_updates':trainer.state.completed_updates,'update_ms_excluding_two_warmups':statistics.mean(steady)*1000,
            'all_update_ms':[value*1000 for value in elapsed],
            'linear_contractions_per_executed_block_forward':48 if joint else 4,
            'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available()}
    finally:
        trainer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('accidental_width_depth','intended_width_only'))
    parser.add_argument('--updates',type=int,default=8)
    parser.add_argument('--seed',type=int,default=77)
    parser.add_argument('--output',type=Path,default=Path('evidence/runner_width_selection_cpu.json'))
    args=parser.parse_args()
    if args.updates<3:
        parser.error('at least three updates are required')
    if args.variant:
        print(json.dumps(worker(args.variant,args.updates,args.seed)))
        return
    rows=[]
    for seed in (77,1337):
        for variant in ('accidental_width_depth','intended_width_only'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--variant',variant,
                '--seed',str(seed),'--updates',str(args.updates)],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            row=json.loads(result.stdout)
            rows.append(row)
            print(f"seed={seed} {variant}: {row['update_ms_excluding_two_warmups']:.2f} ms/update",flush=True)
    accidental=statistics.mean(row['update_ms_excluding_two_warmups'] for row in rows if row['variant']=='accidental_width_depth')
    intended=statistics.mean(row['update_ms_excluding_two_warmups'] for row in rows if row['variant']=='intended_width_only')
    report={'scope':'matched small CPU fixtures; no CUDA timing or language-model-quality claim',
        'user_capture':'WIDTH dct r512 + unintended DEPTH chebyshev P12; D1024 L16 context1024 batch16 accumulation6',
        'root_cause':'Runner previously selected DEPTH merely because inherited DEPTH.order was nonblank',
        'mean_accidental_joint_update_ms':accidental,'mean_intended_width_update_ms':intended,
        'observed_cpu_update_ratio':accidental/intended,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
# ^^^ THOG
