"""Actual THOG2 CPU checkpoint equivalence, schedule and measured-token contracts."""
import copy
import json
from pathlib import Path
from dataclasses import replace
import uuid
import torch
import pytest
import instra_recovery as recovery
from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig
from sheet.lr_schedule import restart_cosine_learning_rate


def config(tmp_path,model_type):
    return TrainingConfig(model_type=model_type,n_layer=2,n_head=2,n_embd=16,block_size=8,vocab_size=32,
        batch_size=2,gradient_accumulation_steps=2,max_updates=4,decay_updates=4,dropout=.1,
        depth_order=2,base_row_order=4,device='cpu',dtype='float32',eval_interval=1,eval_batches=1,out_dir=str(tmp_path))


def equal(one,two):
    if isinstance(one,torch.Tensor):assert torch.equal(one,two)
    elif isinstance(one,dict):
        assert one.keys()==two.keys()
        for key in one:equal(one[key],two[key])
    elif isinstance(one,(tuple,list)):
        for a,b in zip(one,two):equal(a,b)
    else:assert one==two


@pytest.mark.parametrize('model_type',['dense','thog2_sheet'])
def test_actual_interruption_matches_uninterrupted(tmp_path,monkeypatch,model_type):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    monkeypatch.setenv('THOG2_RUNNER_METADATA',json.dumps({'run_id':uuid.uuid4().hex,'attempt_id':uuid.uuid4().hex,
        'parameters':{'--max-iters':4},'dtype':'float32'}))
    tokens=torch.arange(512)%32;c=config(tmp_path,model_type)
    torch.manual_seed(c.model_seed)
    reference=SharedTrainer(c,tokens,tokens);reference.run(target_updates=4)
    reference_state=copy.deepcopy(reference.raw_model.state_dict());reference_optimizer=copy.deepcopy(reference.optimizer.state_dict())
    reference_batch=reference.batch_source.state_dict();reference.close()
    torch.manual_seed(c.model_seed)
    interrupted=SharedTrainer(c,tokens,tokens);interrupted.run(target_updates=2)
    checkpoint=interrupted.save_checkpoint(tmp_path/'source.pt');interrupted.close()
    restored=SharedTrainer.from_checkpoint(checkpoint,tokens,tokens);restored.run(target_updates=4)
    equal(reference_state,restored.raw_model.state_dict());equal(reference_optimizer,restored.optimizer.state_dict())
    equal(reference_batch,restored.batch_source.state_dict())
    assert restored.batch_source.training_tokens_seen==128
    assert restored.scaler.state_dict()=={}
    restored.close()


def test_continue_does_not_move_decay_endpoint(tmp_path):
    tokens=torch.arange(512)%32;c=config(tmp_path,'dense')
    trainer=SharedTrainer(c,tokens,tokens);trainer.run(target_updates=4);checkpoint=trainer.save_checkpoint(tmp_path/'source.pt');trainer.close()
    continued=SharedTrainer.from_checkpoint(checkpoint,tokens,tokens,overrides={'max_updates':6})
    assert continued.config.decay_updates==4
    history=continued.run(target_updates=6)
    assert all(point['learning_rate']==c.min_learning_rate for point in history if 'learning_rate' in point)
    assert continued.batch_source.training_tokens_seen==192
    continued.close()


def test_phase_boundaries():
    arguments=dict(phase_start_update=10,phase_start_lr=.0001,phase_peak_lr=.001,phase_rewarm_iters=2,phase_end_update=20,phase_min_lr=.00001,phase_endpoint_version=2)
    assert restart_cosine_learning_rate(completed_updates=10,**arguments)==.0001
    assert restart_cosine_learning_rate(completed_updates=12,**arguments)==.001
    assert restart_cosine_learning_rate(completed_updates=20,**arguments)==.00001
    assert restart_cosine_learning_rate(completed_updates=30,**arguments)==.00001


def test_managed_fresh_continue_fork_and_target_finalisation(tmp_path,monkeypatch):
    from tests.test_resume_and_fork_cpu_integration import _write_tiny_dataset
    from run_thog2_owt import main
    import run_thog2_lifecycle as lifecycle
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    monkeypatch.setattr(lifecycle,'__file__',str(tmp_path/'run_thog2_lifecycle.py'))
    monkeypatch.setenv('INSTRA_ENABLED','0');monkeypatch.setenv('INSTRA_RECOVERY_CONTEXT','')
    run_id=uuid.uuid4().hex;attempt_id=uuid.uuid4().hex
    metadata={'run_id':run_id,'attempt_id':attempt_id,'parameters':{'--max-iters':2},'dtype':'float32'}
    monkeypatch.setenv('THOG2_RUNNER_METADATA',json.dumps(metadata))
    monkeypatch.setenv('THOG2_CURVE_ROOT',str(tmp_path/'curves'))
    _write_tiny_dataset(tmp_path/'data')
    arguments=['--data-dir',str(tmp_path/'data'),'--checkpoint-root',str(tmp_path/'out'),'--log-root',str(tmp_path/'logs'),
        '--result-root',str(tmp_path/'results'),'--wandb-root',str(tmp_path/'wandb'),'--instrumentation','none','--no-wandb',
        '--device','cpu','--dtype','float32','--model-type','dense','--run-mode','fresh','--run-start-label','261101-0100',
        '--run-name','TEST','--experiment-prefix','TEST','--max-iters','2','--batch-size','1','--gradient-accumulation-steps','1',
        '--block-size','8','--n-layer','1','--n-head','1','--n-embd','8','--warmup-iters','0','--learning-rate','0.001',
        '--min-lr','0.0001','--residual-init-depth-source','true_layer_depth','--no-activation-checkpointing',
        '--eval-interval','1','--eval-iters','1','--checkpoint-interval','1']
    assert main(arguments)==0
    evidence=recovery.rpc({'protocol':1,'operation':'inspect','run_id':run_id,'parameters':metadata['parameters'],'dtype':'float32'})
    assert evidence['completed']
    descriptor=max(evidence['checkpoints'],key=lambda value:value['completed_steps']);before=Path(descriptor['path']).read_bytes()
    request={'protocol':1,'action':'resume','checkpoint':descriptor,'target':2,'run_id':run_id,'attempt_id':uuid.uuid4().hex,'phase':None,'parameters':metadata['parameters']}
    from unittest.mock import patch
    with patch.object(recovery,'data_signature',wraps=recovery.data_signature) as fingerprint:
        assert recovery.execute(request)==0
        assert fingerprint.call_count==1
    assert Path(descriptor['path']).read_bytes()==before
    request.update(action='continue',target=4,previous_target=2,parameters={'--max-iters':4},attempt_id=uuid.uuid4().hex)
    assert recovery.execute(request)==0
    request.update(action='resume',target=4,parameters={'--max-iters':4},attempt_id=uuid.uuid4().hex)
    assert recovery.execute(request)==0  # An older checkpoint may predate Continue's new target.
    import run_thog2_owt_core as core
    monkeypatch.delattr(core,'artifact_paths',raising=False)  # Direct adapter entry does not prime this alias.
    request.update(action='fork',target=6,run_id=uuid.uuid4().hex,attempt_id=uuid.uuid4().hex,
        phase={'kind':'restart_cosine','start':2,'start_lr':descriptor['next_lr'],'peak_lr':.001,'min_lr':.0001,'rewarm_steps':0,'end':6})
    assert recovery.execute(request)==0
    assert Path(descriptor['path']).read_bytes()==before


def test_nonfinite_skip_counts_actual_consumption(tmp_path,monkeypatch):
    tokens=torch.arange(512)%32;trainer=SharedTrainer(config(tmp_path,'dense'),tokens,tokens)
    original=trainer._local_gradients_are_finite;calls=[0]
    def finite():
        calls[0]+=1
        return False if calls[0]==1 else original()
    monkeypatch.setattr(trainer,'_local_gradients_are_finite',finite)
    trainer.run(target_updates=2)
    assert trainer.state.completed_updates==2 and trainer.state.skipped_nonfinite_updates==1
    assert trainer.batch_source.training_tokens_seen==96
    trainer.close()
