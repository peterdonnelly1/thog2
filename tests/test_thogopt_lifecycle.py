# vvv THOG
"""Public runner integration with real checkpoints and local history capture."""
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from sheet.checkpoints import load_payload
from sheet.thogopt_telemetry import history_rows


def test_fresh_resume_reset_fork_and_capture(tmp_path):
    data=tmp_path/'data';data.mkdir()
    (np.arange(1024,dtype=np.uint16)%32).tofile(data/'train.bin')
    (np.arange(256,dtype=np.uint16)%32).tofile(data/'val.bin')
    with (data/'meta.pkl').open('wb') as target:pickle.dump({'vocab_size':32},target)
    checkpoints=tmp_path/'checkpoints'
    environment=dict(os.environ,OMP_NUM_THREADS='1',THOG2_FAST_DISCARD='true',THOG2_INSTRUMENTATION='none',THOG2_INSTRUMENTATION_LOCAL_ROOT=str(tmp_path/'local'))
    environment['THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS']='1'
    def run(arguments):
        result=subprocess.run([sys.executable,'-m','run_thog2_owt',*arguments],env=environment,cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=60)
        assert result.returncode==0,result.stdout[-6000:]+result.stderr[-6000:]
        assert 'THOGOPT last update: stage=' in result.stdout
        assert 'gpu_candidates=0MiB async_staging=no' in result.stdout
    run(['--model-type','sheet','--geometry-preset','depth','--run-mode','fresh','--run-start-label','260902-1000',
        '--optimizer','thogopt','--run-name','THOGOPT_TEST','--experiment-prefix','THOGOPT_TEST','--max-iters','1',
        '--data-dir',str(data),'--checkpoint-root',str(checkpoints),'--log-root',str(tmp_path/'logs'),'--result-root',str(tmp_path/'results'),
        '--wandb-root',str(tmp_path/'wandb'),'--instrumentation','none','--no-wandb','--device','cpu','--dtype','float32',
        '--batch-size','1','--gradient-accumulation-steps','2','--block-size','8','--n-layer','3','--n-head','1','--n-embd','4',
        '--o-depth','2','--warmup-iters','0','--learning-rate','.001','--min-lr','.0001','--eval-interval','1','--eval-iters','1',
        '--log-interval','1','--checkpoint-interval','1','--residual-init-depth-source','true_layer_depth','--no-activation-checkpointing',
        '--instrumentation__optimizer_histories__full_matrix_every_n_steps','1'])
    path=next(checkpoints.rglob('ckpt.pt'))
    assert load_payload(path)['completed_updates']==1
    run(['--resume',str(path),'--max-iters','2','--checkpoint-root',str(checkpoints),'--instrumentation','none'])
    payload=load_payload(path)
    assert payload['completed_updates']==2
    assert payload['optimizer']['thogopt']['optimizer']=='thogopt'
    assert len(payload['optimizer']['thogopt_reference'])==6
    captures=[row for database in tmp_path.rglob('charts.sqlite3') for row in history_rows(database)]
    assert {row['optimizer_update'] for row in captures}>={1,2}
    assert all(row['full_path'] for row in captures)
    run(['--fork',str(path),'--reset-optimizer','--optimizer','thogopt','--thogopt__momentum_history_coefficients','1',
        '--fork-lr-mode','restart_cosine','--fork-learning-rate','.001','--fork-min-lr','.0001','--fork-rewarm-iters','0',
        '--max-iters','3','--checkpoint-root',str(checkpoints),'--instrumentation','none'])
    fork_path=next(p for p in checkpoints.rglob('ckpt.pt') if p!=path)
    child=load_payload(fork_path)
    assert child['completed_updates']==3
    assert child['lifecycle']['optimizer_name']=='thogopt'
    assert child['lifecycle']['optimizer_reset']['history_origin_completed_update']==2
    assert child['optimizer']['thogopt']['momentum_history']['effective']==1
    assert {state['step'] for state in child['optimizer']['state'].values()}=={1}
# ^^^ THOG
