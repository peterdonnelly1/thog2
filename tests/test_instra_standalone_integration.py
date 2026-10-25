from types import SimpleNamespace
import json
import sqlite3

import instra
from sheet.local_chart_store import ensure_local_chart_store, close_local_chart_store
from sheet.instra_integration import record_scalars


def test_opt_in_preserves_native_identity_and_records_local_evidence(tmp_path, monkeypatch):
    source = tmp_path/'source';source.mkdir()
    manifest = instra.initialize_source(source,'thog2')
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
    instra.initialize_source(source,'thog2')
    monkeypatch.setenv('INSTRA_ENABLED','1')
    monkeypatch.setenv('INSTRA_SOURCE_ROOT',str(source))
    telemetry=SimpleNamespace(name='scalars',config={},run=None)
    record_scalars(telemetry, {'train/loss': 1.0}, 1)
    close_local_chart_store(telemetry)
    assert telemetry._instra_producer.health()['closed']
    with sqlite3.connect(telemetry._instra_producer.database_path) as connection:
        assert connection.execute('SELECT kind FROM records ORDER BY sequence DESC LIMIT 1').fetchone()[0]=='finish'
