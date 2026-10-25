from types import SimpleNamespace
import json
import sqlite3

import instra
from sheet.local_chart_store import ensure_local_chart_store, close_local_chart_store
from sheet.instra_integration import record_scalars


def test_opt_in_preserves_native_identity_and_records_local_evidence(tmp_path, monkeypatch):
    source = tmp_path/'source';source.mkdir()
    manifest = instra.initialize_source(source,'8f401816-678b-4fd1-9ff7-0b1387c6643e')
    monkeypatch.setenv('INSTRA_ENABLED','1')
    monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry = SimpleNamespace(name='test',config={},run=SimpleNamespace(id='cloud-id-must-not-be-used'))
    store = ensure_local_chart_store(telemetry)
    store.append_training_loss(1,0.0)
    record_scalars(telemetry,{'train/loss':0.0,'val/loss':2.0},1)
    producer = telemetry._instra_producer
    assert store.path.is_relative_to(source/'instra'/'runs')
    assert producer.run_id != 'cloud-id-must-not-be-used'
    close_local_chart_store(telemetry)
    with sqlite3.connect(producer.database_path) as connection:
        context = json.loads(connection.execute('SELECT payload FROM context').fetchone()[0])
        assert context['model_source_id'] == manifest['model_source_id']
        rows=connection.execute('SELECT kind FROM records ORDER BY sequence').fetchall()
        assert rows == [('start',),('measurements',),('finish',)]


def test_disabled_adapter_keeps_legacy_route(tmp_path,monkeypatch):
    monkeypatch.delenv('INSTRA_ENABLED',raising=False)
    monkeypatch.setenv('THOG2_INSTRUMENTATION_LOCAL_ROOT',str(tmp_path))
    telemetry=SimpleNamespace(name='legacy',config={},run=SimpleNamespace(id='legacy-id',url='',dir=''))
    store=ensure_local_chart_store(telemetry)
    assert store.path == tmp_path/'legacy'/'legacy-id'/'charts.sqlite3'
    assert '_instra_producer' not in vars(telemetry)
    close_local_chart_store(telemetry)


def test_scalar_only_run_finalizes_without_chart_store(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir()
    instra.initialize_source(source,'8f401816-678b-4fd1-9ff7-0b1387c6643e')
    monkeypatch.setenv('INSTRA_ENABLED','1')
    monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=SimpleNamespace(name='scalars',config={},run=None)
    record_scalars(telemetry, {'train/loss': 1.0}, 1)
    close_local_chart_store(telemetry)
    assert telemetry._instra_producer.health()['closed']
    with sqlite3.connect(telemetry._instra_producer.database_path) as connection:
        assert connection.execute('SELECT kind FROM records ORDER BY sequence DESC LIMIT 1').fetchone()[0]=='finish'


def test_native_scalars_record_with_legacy_backend_disabled(tmp_path, monkeypatch):
    from sheet.wandb_telemetry import WandbTelemetry
    source=tmp_path/'source';source.mkdir()
    instra.initialize_source(source,'8f401816-678b-4fd1-9ff7-0b1387c6643e')
    monkeypatch.setenv('INSTRA_ENABLED','1')
    monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=WandbTelemetry(enabled=False,project='test',entity=None,mode='offline',root=tmp_path,name='native',group='',job_type='train',config={'device':'cpu'})
    telemetry.backend='none'
    telemetry.start()
    telemetry.log_event('optimizer_progress',{'completed_updates':1,'training_loss':2.5,'consumed_tokens':128,'cumulative_training_seconds':1.0,'learning_rate':0.001,'gradient_norm':0.5})
    telemetry.finish()
    with sqlite3.connect(telemetry._instra_producer.database_path) as connection:
        rows=connection.execute("SELECT payload FROM records WHERE kind='measurements'").fetchall()
    assert rows, 'Native recording must not depend on legacy scalar backend'


def test_recording_does_not_change_dense_training_state_or_rng(tmp_path, monkeypatch):
    import torch
    from sheet.wandb_telemetry import WandbTelemetry
    from sheet.training_model_factory import build_training_model
    from tests.test_sheet_stage6_dense_checkpointing import DenseCheckpointingTests
    source=tmp_path/'source';source.mkdir();instra.initialize_source(source,'8f401816-678b-4fd1-9ff7-0b1387c6643e')
    monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    results=[]
    for active in (False,True):
        monkeypatch.setenv('INSTRA_ENABLED','1' if active else '0')
        torch.manual_seed(123)
        model=build_training_model(DenseCheckpointingTests().config(0))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        telemetry=WandbTelemetry(enabled=False,project='test',entity=None,mode='offline',root=tmp_path,name='comparison',group='',job_type='train',config={'device':'cpu'})
        telemetry.backend='none';telemetry.start()
        inputs=torch.arange(16,dtype=torch.long).view(2,8)%32
        targets=torch.roll(inputs,-1,1)
        losses=[]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            _,loss=model(inputs,targets);loss.backward();optimizer.step()
            value=float(loss.detach());losses.append(value)
            telemetry.log_event('optimizer_progress',{'completed_updates':step+1,'training_loss':value,'consumed_tokens':16*(step+1),'cumulative_training_seconds':float(step+1),'learning_rate':0.001,'gradient_norm':0.5})
        telemetry.finish()
        results.append((losses,{key:value.detach().clone() for key,value in model.state_dict().items()},torch.get_rng_state().clone()))
    assert results[0][0]==results[1][0]
    assert torch.equal(results[0][2],results[1][2])
    assert results[0][1].keys()==results[1][1].keys()
    for key in results[0][1]:
        assert torch.equal(results[0][1][key],results[1][1][key]),key
