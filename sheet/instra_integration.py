"""Opt-in standalone Instra adapter; the legacy telemetry route remains the default."""
import os
from pathlib import Path


def enabled():
    return os.environ.get('INSTRA_ENABLED') == '1'


def producer_for(telemetry):
    existing = vars(telemetry).get('_instra_producer')
    if existing is not None:
        return existing
    import instra
    source_root = Path(os.environ.get('INSTRA_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
    context = telemetry.config.get('runner', {})
    producer = instra.init(source_root, model_type_id=instra.THOG2_MODEL_TYPE_ID,
                          run_id=context.get('run_id'), grid_id=context.get('grid_id'),
                          run_attempt_id=context.get('attempt_id'), config=telemetry.config)
    from instra.resources import ProcessResources
    telemetry._instra_resources = ProcessResources()
    telemetry._instra_resource_sample_at = 0.0
    telemetry._instra_producer = producer
    return producer


def chart_store_for(telemetry, store_class):
    producer = producer_for(telemetry)
    store = store_class(producer.artifact_path('charts.sqlite3'),
                        run_name=str(telemetry.name), run_id=producer.run_id,
                        config=telemetry.config)
    for key in ('host_id', 'model_source_id', 'model_type_id', 'run_id', 'grid_id', 'run_attempt_id'):
        store.connection.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)',
                                 (key, str(producer.context[key])))
    store.connection.commit()
    telemetry._thog_local_chart_store = store
    return store


def record_scalars(telemetry, scalars, step):
    import time
    producer = producer_for(telemetry)
    values = dict(scalars)
    now = time.monotonic()
    if now - telemetry._instra_resource_sample_at >= 2:
        values.update(telemetry._instra_resources.sample())
        telemetry._instra_resource_sample_at = now
    return producer.log(values, step=int(step))


def finish(telemetry, state):
    producer = vars(telemetry).get('_instra_producer')
    if producer is not None:
        health = producer.finish(state='finished' if state in {'finished', 'completed'} else 'cancelled' if state == 'cancelled' else 'failed')
        if health['error'] or health['dropped_records']:
            print(f'Instra recording health: {health}', flush=True)
