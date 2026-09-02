# vvv THOG
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from sheet.stage4_trainer import Stage4Trainer
from sheet.thogopt import Thogopt
from sheet.thogopt_telemetry import configure_reference, history_snapshot, full_history_snapshot, append_history, history_rows
from sheet.thogopt_dashboard import history_payload, matrix_window
from sheet.local_chart_store import LocalChartStore
from tests.stage4_test_support import stage4_training_config, stage4_tokens


torch.set_num_threads(1)


def trainer_and_telemetry(monkeypatch, optimizer="thogopt"):
    monkeypatch.setenv("THOG2_OPTIMIZER",optimizer)
    config = stage4_training_config(geometry_preset="depth" if optimizer=="thogopt" else None,
        model_type="thog2_sheet" if optimizer=="thogopt" else "dense",depth_order=3,max_updates=5)
    trainer = Stage4Trainer(config,*stage4_tokens())
    telemetry = SimpleNamespace(config={},name="thogopt_test",run_name="thogopt_test",run=None)
    if optimizer=="thogopt": configure_reference(trainer,telemetry)
    return trainer,telemetry


@pytest.mark.parametrize("optimizer",["thogopt","adamw"])
def test_captured_histories_and_full_matrix_windows(monkeypatch,tmp_path,optimizer):
    trainer,telemetry = trainer_and_telemetry(monkeypatch,optimizer)
    trainer.train_one_update()
    before = {name:value.detach().clone() for name,value in trainer.raw_model.named_parameters()}
    snapshot = history_snapshot(trainer,telemetry)
    full = full_history_snapshot(trainer)
    store = LocalChartStore(tmp_path/'charts.sqlite3',run_name='history',config={})
    append_history(store,snapshot,full=full)
    assert len(history_rows(store.path))==1
    payload = history_payload(store.path,quantity="scaling")
    assert len(payload["figures"])==6
    assert payload["full_steps"]==[1]
    chart = "attn_q_head_N"
    family = snapshot["families"][chart]
    row,column = family["coordinates"][0]
    actual = matrix_window(store.path,step=1,chart=chart,quantity="momentum",layer=2,row_start=row,column_start=column,row_count=1,column_count=1)
    assert actual["values"][0][0]==pytest.approx(family["values"]["momentum"][0][1],rel=1e-10,abs=1e-14)
    if optimizer=="thogopt":
        assert family["reference"]["origin"]==0
        assert "projected_adaptive_update" in family["errors"]
        coeff = matrix_window(store.path,step=1,chart=chart,quantity="momentum",coefficient=0,row_start=row,column_start=column,row_count=1,column_count=1)
        assert coeff["values"][0][0]==family["momentum_coefficients"][0][0]
    zero = matrix_window(store.path,step=1,chart=chart,reference_path=store.path,row_count=2,column_count=3)
    assert zero["values"]==[[0.,0.,0.],[0.,0.,0.]]
    with pytest.raises(FileNotFoundError): matrix_window(store.path,step=2,chart=chart)
    with pytest.raises(ValueError): matrix_window(store.path,step=1,chart=chart,row_count=1000)
    for name,value in trainer.raw_model.named_parameters(): torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    trainer.train_one_update()
    append_history(store,history_snapshot(trainer,telemetry),history_length=1)
    assert history_rows(store.path)[0]["optimizer_update"]==2
    assert not list((tmp_path/'optimizer_histories').glob('*.pt'))


def test_instrumentation_does_not_change_training(monkeypatch):
    first,telemetry = trainer_and_telemetry(monkeypatch)
    second,_ = trainer_and_telemetry(monkeypatch)
    second.optimizer.reference_histories.clear()
    for _ in range(3):
        first.train_one_update(); history_snapshot(first,telemetry)
        second.train_one_update()
    for a,b in zip(first.raw_model.parameters(),second.raw_model.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_history_comparison_rejects_mismatched_coordinates(monkeypatch,tmp_path):
    trainer,telemetry = trainer_and_telemetry(monkeypatch)
    trainer.train_one_update()
    snapshot = history_snapshot(trainer,telemetry)
    first = LocalChartStore(tmp_path/'one'/'charts.sqlite3',run_name='one',config={})
    second = LocalChartStore(tmp_path/'two'/'charts.sqlite3',run_name='two',config={})
    append_history(first,snapshot)
    snapshot = deepcopy(snapshot)
    snapshot['families']['mlp_up']['coordinates'][0][0] += 1
    append_history(second,snapshot)
    with pytest.raises(ValueError,match='coordinates'): history_payload(first.path,reference_path=second.path)


@pytest.mark.parametrize("optimizer_kind,dtype",[("thogopt",torch.float32),("thogopt",torch.float64),("sgd",torch.float32)])
def test_full_model_optimizer_equivalence_100_steps(monkeypatch,tmp_path,dtype,optimizer_kind):
    from tests.test_dense_snapshot_baselining import _dense_config,_compact_config
    from sheet.dense_snapshot import save_dense_initialisation_snapshot
    from sheet.dense_weight_curves_patch import _dense_family_weight
    from sheet.thogopt_telemetry import _families
    monkeypatch.setenv('THOG2_OPTIMIZER','adamw')
    tokens = tuple(value%16 for value in stage4_tokens())
    dense = Stage4Trainer(_dense_config(max_updates=100,decay_updates=100,weight_decay=.01,grad_clip=1),*tokens)
    path,_ = save_dense_initialisation_snapshot(dense.raw_model,dense.config,root=tmp_path)
    compact = Stage4Trainer(_compact_config(path,3,max_updates=100,decay_updates=100,weight_decay=.01,grad_clip=1),*tokens)
    if dtype==torch.float64:
        dense.raw_model.double(); compact.raw_model.double()
        basis=compact.raw_model.trajectory.depth_basis.double()
        analysis=torch.linalg.pinv(basis)
        with torch.no_grad():
            for chart,name in _families().items():
                matrices=[]
                for block in dense.raw_model.transformer.h:
                    parameter,offset=_dense_family_weight(block,chart,4)
                    rows=compact.raw_model.trajectory.coefficients[name].shape[0]
                    matrices.append(parameter[offset:offset+rows])
                coefficients=(analysis@torch.stack(matrices).flatten(1)).T.reshape(compact.raw_model.trajectory.coefficients[name].shape)
                compact.raw_model.trajectory.coefficients[name].copy_(coefficients)
    # Match THOG's existing embedding decay exclusion in the dense oracle.
    # Optimizer equivalence requires identical physical parameter-group policies.
    decay=[];no_decay=[]
    for name,parameter in dense.raw_model.named_parameters():
        target=no_decay if name in ('transformer.wte.weight','transformer.wpe.weight','lm_head.weight') or parameter.ndim<2 else decay
        target.append(parameter)
    dense.optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':dense.config.weight_decay},{'params':no_decay,'weight_decay':0.}],
        lr=dense.config.learning_rate,betas=(dense.config.beta1,dense.config.beta2))
    if optimizer_kind=='sgd':
        dense.optimizer=torch.optim.SGD(dense.optimizer.param_groups,lr=dense.config.learning_rate,momentum=.9)
    monkeypatch.setenv('THOG2_OPTIMIZER',optimizer_kind)
    monkeypatch.setenv('THOG2_OPTIMIZER_MOMENTUM','0.9')
    from sheet.optimizer_factory import build_optimizer
    compact.optimizer=build_optimizer(compact.raw_model,weight_decay=compact.config.weight_decay,learning_rate=compact.config.learning_rate,
        betas=(compact.config.beta1,compact.config.beta2),device_type='cpu',thogopt_config=compact.config)
    for step in range(100):
        a=dense.train_one_update();b=compact.train_one_update()
        for chart,name in _families().items():
            for layer,block in enumerate(dense.raw_model.transformer.h):
                dense_weight,offset = _dense_family_weight(block,chart,4)
                actual = compact.raw_model.trajectory.materialize(name,layer)
                expected = dense_weight[offset:offset+actual.shape[0]]
                torch.testing.assert_close(actual,expected,atol=1e-10 if dtype==torch.float64 else 5e-6,rtol=1e-9 if dtype==torch.float64 else 5e-4,msg=lambda message:f"step={step}, chart={chart}, layer={layer}: {message}")
    assert dense.state.completed_updates==compact.state.completed_updates==100
# ^^^ THOG


def test_explicit_reset_fork_keeps_weights_and_data_position(monkeypatch):
    from dataclasses import replace
    from sheet.thogopt_fork import reset_optimizer_fork
    trainer,_ = trainer_and_telemetry(monkeypatch)
    trainer.train_one_update()
    payload = deepcopy(trainer.checkpoint_payload())
    config = replace(trainer.config,thogopt__momentum_history_coefficients=1)
    child = reset_optimizer_fork(Stage4Trainer,payload,config,*stage4_tokens())
    assert child.state.completed_updates==1
    assert len(child.optimizer.state)==0
    assert child.optimizer.q_m.shape[1]==1
    assert child.optimizer_reset_origin==1
    for a,b in zip(trainer.raw_model.parameters(),child.raw_model.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    child.train_one_update()
    assert child.state.completed_updates==2
    assert {state['step'] for state in child.optimizer.state.values()}=={1}


def test_public_cli_history_arguments_and_reset_scope():
    import run_thog2_lifecycle as lifecycle
    import run_thog2_owt_core as core
    parser=lifecycle.build_parser()
    args=parser.parse_args(['--optimizer','thogopt','--thogopt__momentum_history_coefficients','2','--thogopt__scaling_history_coefficients','auto','--instrumentation__optimizer_histories__full_matrix_every_n_steps','50'])
    assert args.thogopt__momentum_history_coefficients==2
    assert args.instrumentation__optimizer_histories__full_matrix_every_n_steps==50
    args=parser.parse_args(['--reset-optimizer'])
    with pytest.raises(ValueError,match='explicit fork'):lifecycle.prepare_context(args,{'reset_optimizer'})


def test_dynamic_loss_scale_growth_tracker_survives_restart(monkeypatch,tmp_path):
    class CpuScaledTrainer(Stage4Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.scaler=torch.amp.GradScaler('cpu',init_scale=128.,growth_interval=2)
    monkeypatch.setenv('THOG2_OPTIMIZER','thogopt')
    trainer=CpuScaledTrainer(stage4_training_config(geometry_preset='depth',max_updates=5),*stage4_tokens())
    trainer.train_one_update()
    path=tmp_path/'scaled.pt';trainer.save_checkpoint(path)
    resumed=CpuScaledTrainer.from_checkpoint(path,*stage4_tokens())
    trainer.train_one_update();resumed.train_one_update()
    assert trainer.scaler.state_dict()==resumed.scaler.state_dict()
    assert resumed.scaler.get_scale()==256.
    for left,right in zip(trainer.raw_model.parameters(),resumed.raw_model.parameters()):torch.testing.assert_close(left,right,atol=0,rtol=0)
