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


def test_chart_storage_fault_warns_recovers_and_keeps_gap(tmp_path, monkeypatch):
    import pytest
    from sheet import local_chart_store
    source=tmp_path/'source';source.mkdir();instra.initialize_source(source, instra.THOG2_MODEL_TYPE_ID)
    monkeypatch.setenv('INSTRA_ENABLED','1');monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=SimpleNamespace(name='fault',config={},run=None)
    original=local_chart_store.LocalChartStore.append_training_loss
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('simulated chart outage')
    monkeypatch.setattr(local_chart_store.LocalChartStore,'append_training_loss',fail)
    store=ensure_local_chart_store(telemetry)
    with pytest.warns(RuntimeWarning,match='chart recording degraded'):
        store.append_training_loss(1, 4.)
    assert telemetry._instra_producer.health()['diagnostic_failures']==1
    # A separate healthy method remains usable and native measurements continue.
    store.append_processing_throughput(2, 123.)
    record_scalars(telemetry,{'train/loss':3.},2)
    close_local_chart_store(telemetry)
    health=json.loads((telemetry._instra_producer.root/'recording-health.json').read_text())
    assert health['diagnostic_failures']==1
    assert 'simulated chart outage' in health['last_diagnostic_error']
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM processing_throughput').fetchone()[0]==1


def test_chart_initialization_failure_does_not_abort_training_event(tmp_path, monkeypatch):
    import pytest
    from sheet.wandb_telemetry import WandbTelemetry
    from sheet import local_chart_store
    source=tmp_path/'source';source.mkdir();instra.initialize_source(source, instra.THOG2_MODEL_TYPE_ID)
    monkeypatch.setenv('INSTRA_ENABLED','1');monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=WandbTelemetry(enabled=False,project='test',entity=None,mode='offline',root=tmp_path,name='fault',group='',job_type='train',config={'device':'cpu'})
    telemetry.backend='none'
    original=local_chart_store._open_database
    with monkeypatch.context() as temporary:
        temporary.setattr(local_chart_store,'_open_database',lambda *args,**kwargs: (_ for _ in ()).throw(sqlite3.OperationalError('chart init unavailable')))
        with pytest.warns(RuntimeWarning):telemetry.start()
    telemetry.log_event('optimizer_progress',{'completed_updates':1,'training_loss':2.5,'consumed_tokens':128,'cumulative_training_seconds':1.,'learning_rate':.001,'gradient_norm':.5})
    telemetry.finish()
    assert telemetry._instra_producer.health()['diagnostic_failures']==1


def test_strict_chart_storage_fault_propagates(tmp_path, monkeypatch):
    import pytest
    from sheet import local_chart_store
    source=tmp_path/'source';source.mkdir();instra.initialize_source(source, instra.THOG2_MODEL_TYPE_ID)
    monkeypatch.setenv('INSTRA_ENABLED','1');monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    monkeypatch.setenv('INSTRA_RECORDING_FAILURE_POLICY','raise')
    telemetry=SimpleNamespace(name='strict',config={},run=None)
    monkeypatch.setattr(local_chart_store.LocalChartStore,'append_training_loss',lambda *a,**k: (_ for _ in ()).throw(sqlite3.OperationalError('strict failure')))
    store=ensure_local_chart_store(telemetry)
    with pytest.raises(sqlite3.OperationalError):store.append_training_loss(1, 4.)
    close_local_chart_store(telemetry)


def test_strict_finish_fault_is_not_swallowed(tmp_path, monkeypatch):
    import pytest
    from sheet import wandb_telemetry
    monkeypatch.setenv('INSTRA_ENABLED','1')
    monkeypatch.setenv('INSTRA_RECORDING_FAILURE_POLICY','raise')
    telemetry=SimpleNamespace(run=None,writer=None)
    monkeypatch.setattr(wandb_telemetry,'close_local_chart_store',lambda *a,**k: (_ for _ in ()).throw(OSError('finish disk failure')))
    with pytest.raises(OSError,match='finish disk failure'):
        wandb_telemetry.WandbTelemetry.finish(telemetry)


def test_invalid_identity_disables_recording_without_reidentifying_source(tmp_path, monkeypatch):
    import pytest
    from sheet.wandb_telemetry import WandbTelemetry
    from sheet.instra_integration import recording_health
    source=tmp_path/'source';source.mkdir()
    manifest=source/'instra_manifest.json';manifest.write_text('{invalid')
    monkeypatch.setenv('INSTRA_ENABLED','1');monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=WandbTelemetry(enabled=False,project='test',entity=None,mode='offline',root=tmp_path,name='identity',group='',job_type='train',config={'device':'cpu'})
    telemetry.backend='none'
    with pytest.warns(RuntimeWarning,match='identity requires explicit recovery'):
        telemetry.start()
    telemetry.finish()
    assert recording_health(telemetry)['error']
    assert not recording_health(telemetry)['recording_available']
    assert manifest.read_text()=='{invalid'
    assert not (source/'instra').exists()


def test_premat_writer_constructor_failure_is_observable(tmp_path, monkeypatch):
    import pytest
    from sheet import wandb_telemetry
    source=tmp_path/'source';source.mkdir();instra.initialize_source(source,instra.THOG2_MODEL_TYPE_ID)
    run=instra.init(source,model_type_id=instra.THOG2_MODEL_TYPE_ID)
    owner=SimpleNamespace(_instra_producer=run)
    monkeypatch.setattr(wandb_telemetry,'LocalPrematLiveWriter',lambda *a,**k: (_ for _ in ()).throw(sqlite3.OperationalError('premat init failed')))
    with pytest.warns(RuntimeWarning,match='PREMAT recording degraded'):
        sink=wandb_telemetry._PrematLiveSink(run.root/'charts.sqlite3',recording_owner=owner)
        sink._thread.join(timeout=3)
    assert sink._error is not None
    assert run.health()['diagnostic_failures']==1
    with pytest.raises(RuntimeError,match='writer failed'):sink.close()
    run.finish()


def test_attachment_storage_fault_is_contained(monkeypatch):
    import pytest
    from sheet import wandb_telemetry
    monkeypatch.setenv('INSTRA_ENABLED','1')
    trainer=SimpleNamespace(config=SimpleNamespace(premat_processing_logging='enabled'),
                            distributed=SimpleNamespace(is_primary=True),_print_progress=lambda *a,**k:None)
    owner=SimpleNamespace()
    monkeypatch.setattr(wandb_telemetry,'ensure_local_chart_store',lambda *a: (_ for _ in ()).throw(OSError('attachment disk failure')))
    with pytest.warns(RuntimeWarning,match='attachment disk failure'):
        wandb_telemetry._attach_recording(owner,trainer)
    assert owner._instra_dropped_events==1
    monkeypatch.setenv('INSTRA_RECORDING_FAILURE_POLICY','raise')
    with pytest.raises(OSError):wandb_telemetry._attach_recording(owner,trainer)


def test_premat_queue_overflow_records_gap_and_strict_raises(monkeypatch):
    import queue
    import pytest
    from sheet.wandb_telemetry import _PrematLiveSink
    errors=[]
    sink=object.__new__(_PrematLiveSink)
    sink._error=None;sink._closed=False;sink._retain_detailed_history=False
    sink._recording_owner=SimpleNamespace(_instra_producer=SimpleNamespace(report_diagnostic_failure=errors.append))
    sink._queue=queue.Queue(maxsize=1);sink._queue.put((0,{}))
    sink.publish(1,{'layers':[]})
    assert len(errors)==1 and sink._queue.get()[0]==1
    sink._queue.put((1,{}));monkeypatch.setenv('INSTRA_RECORDING_FAILURE_POLICY','raise')
    with pytest.raises(RuntimeError,match='queue full'):sink.publish(2,{'layers':[]})
    assert len(errors)==2
