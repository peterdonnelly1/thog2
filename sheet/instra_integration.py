"""Opt-in standalone Instra adapter; the legacy telemetry route remains the default."""
import os
from pathlib import Path
from functools import wraps
import sqlite3
import warnings


def enabled():
    return os.environ.get('INSTRA_ENABLED') == '1'


def failure_policy():
    value = os.environ.get('INSTRA_RECORDING_FAILURE_POLICY', 'continue')
    if value not in {'continue', 'raise'}:
        raise ValueError('INSTRA_RECORDING_FAILURE_POLICY must be continue or raise')
    return value


def recording_guard(function):
    """Contain storage faults only; programmer/model errors still propagate."""
    @wraps(function)
    def guarded(telemetry, *args, **kwargs):
        if not enabled():
            return function(telemetry, *args, **kwargs)
        try:
            return function(telemetry, *args, **kwargs)
        except (OSError, sqlite3.Error) as error:
            if failure_policy() == 'raise':
                raise
            text = f'{type(error).__name__}: {error}'
            if vars(telemetry).get('_instra_last_storage_error') != text:
                warnings.warn('Instra chart recording degraded: ' + text, RuntimeWarning)
            telemetry._instra_last_storage_error = text
            telemetry._instra_dropped_events = vars(telemetry).get('_instra_dropped_events', 0) + 1
            producer = vars(telemetry).get('_instra_producer')
            if producer is not None:
                producer.report_diagnostic_failure(text)
            return None
    return guarded


def producer_for(telemetry):
    existing = vars(telemetry).get('_instra_producer')
    if existing is not None:
        return existing
    import instra
    source_root = Path(os.environ.get('INSTRA_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
    context = telemetry.config.get('runner', {})
    try:
        producer = instra.init(source_root, model_type_id=instra.THOG2_MODEL_TYPE_ID,
                              run_id=context.get('run_id'), grid_id=context.get('grid_id'),
                              run_attempt_id=context.get('attempt_id'), config=telemetry.config,
                              failure_policy=failure_policy())
    except instra.IdentityError as error:
        # A direct model invocation may continue without recording. Runner
        # preflight rejects this before starting the model. Never replace IDs.
        raise OSError('Recording identity requires explicit recovery: ' + str(error)) from error
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
    # Retain the established chart schema/implementation and contain only storage
    # failures at its write boundary. No model computations are wrapped.
    for name in ('configure_weight_capture', 'heartbeat', 'update_runtime_metrics',
                 'append_heatmap_records', 'append_width_snapshot',
                 'append_depth_weight_snapshot', 'append_premat_snapshot',
                 'append_processing_throughput', 'append_training_loss',
                 'update_premat_aggregate', 'close'):
        method = object.__getattribute__(store, name)
        def invoke(owner, *args, _method=method, **kwargs):
            try:
                return _method(*args, **kwargs)
            except (OSError, sqlite3.Error):
                try:
                    store.connection.rollback()
                except sqlite3.Error:
                    pass
                raise
        guarded = recording_guard(invoke)
        def call(*args, _guarded=guarded, **kwargs):
            return _guarded(telemetry, *args, **kwargs)
        setattr(store, name, call)
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


def recording_health(telemetry):
    producer = vars(telemetry).get('_instra_producer')
    if producer is not None:
        return producer.health()
    return {'error': vars(telemetry).get('_instra_last_storage_error'),
            'dropped_records': vars(telemetry).get('_instra_dropped_events', 0),
            'diagnostic_failures': 0, 'recording_available': False}


def finish(telemetry, state):
    producer = vars(telemetry).get('_instra_producer')
    if producer is not None:
        producer.finish(state='finished' if state in {'finished', 'completed'} else 'cancelled' if state == 'cancelled' else 'failed')
    health = recording_health(telemetry)
    if health['error'] or health['dropped_records'] or health['diagnostic_failures']:
        print(f'Instra recording health: {health}', flush=True)
