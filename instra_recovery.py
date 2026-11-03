"""Instra v2 checkpoint operations, executed solely in the THOG2 environment."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys
import uuid
from instra_checkpoint_store import CheckpointStore, management_queue, management_worker, management_sync_directory

ROOT = Path(__file__).resolve().parent
consumed_tokens = 0
FORK_PARAMETERS = ['--instrumentation__depth_weight_curves__destination', '--activation-checkpointing', '--checkpoint-interval', '--checkpoint-segment-size', '--eval-interval', '--eval-iters', '--instrumentation__delta_loss_v_layer_heatmap', '--instrumentation__delta_loss_v_layer_heatmap__destination', '--instrumentation__delta_loss_v_layer_heatmap_abs_limit', '--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes', '--instrumentation__optimizer_histories__full_matrix_every_n_steps', '--instrumentation__width_activation_curves__end_step', '--instrumentation__width_activation_curves__feature_evaluation_points', '--instrumentation__width_activation_curves__history_length', '--instrumentation__width_activation_curves__log_every_n_steps', '--instrumentation__width_activation_curves__mode', '--instrumentation__width_activation_curves__probe_every_n_steps', '--instrumentation__width_activation_curves__probe_orders', '--instrumentation__width_activation_curves__sample_tokens_per_layer', '--instrumentation__width_activation_curves__start_step', '--log-interval', '--max-wall-minutes', '--premat_instra_microsteps', '--premat_logging', '--premat_processing_logging', '--premat_processing_logging_capture_frequency_hz', '--premat_processing_logging_capture_update', '--save_and_reuse_final_activation_checkpoin_group_weights_on_next_forward_step', '--wandb-entity', '--wandb-mode', '--wandb-project']


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    management_sync_directory(path.parent)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity():
    metadata = json.loads(os.environ.get('THOG2_RUNNER_METADATA', '{}'))
    context = json.loads(os.environ.get('INSTRA_RECOVERY_CONTEXT') or '{}')
    run_id = context.get('run_id') or metadata.get('run_id')
    attempt_id = context.get('attempt_id') or metadata.get('attempt_id')
    if not run_id or not attempt_id:
        return None
    if context:
        metadata = {**metadata, 'parameters': {**context.get('parameters', metadata.get('parameters', {})), '--max-iters': context['target']}, 'dtype': context.get('dtype', metadata.get('dtype'))}
    return str(uuid.UUID(run_id)), str(uuid.UUID(attempt_id)), metadata


def data_signature(source):
    import numpy as np
    values = []
    for storage in (source.train_tokens, source.validation_tokens):
        digest = hashlib.sha256()
        for offset in range(0, len(storage), 1024*1024):
            block = storage[offset:offset+1024*1024]
            if hasattr(block, 'numpy'):
                block = block.detach().cpu().numpy()
            array = np.asarray(block)
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        values.append({'length': len(storage), 'sha256': digest.hexdigest()})
    return values


def _publish_unlocked(trainer, path):
    identity = _identity()
    if not identity:
        return
    from sheet.lr_schedule import learning_rate_for_lifecycle, active_lr_phase
    run_id, attempt_id, metadata = identity
    bundle_id = uuid.uuid4().hex
    output = ROOT/'instra'/'runs'/run_id/'attempts'/attempt_id/'checkpoints'
    output.mkdir(parents=True, exist_ok=True)
    target = output/(f'step_{trainer.state.completed_updates:012d}_{bundle_id}.pt')
    temporary = target.with_suffix('.tmp')
    # Published bundles retain an immutable inode. Atomic trainer saves replace
    # the source path, so a later checkpoint cannot overwrite an earlier bundle.
    import errno
    try:
        try:
            os.link(path, temporary)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            with Path(path).open('rb') as source, temporary.open('wb') as destination:
                shutil.copyfileobj(source, destination, 1024*1024)
                destination.flush()
                os.fsync(destination.fileno())
        os.replace(temporary, target)
        management_sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    lifecycle = vars(trainer).get('lifecycle_metadata')
    identity_hash = vars(trainer).get('_instra_data_identity')
    if identity_hash is None:
        identity_hash = data_signature(trainer.batch_source)
        trainer._instra_data_identity = identity_hash
    descriptor = {'bundle_id': bundle_id, 'run_id': run_id, 'attempt_id': attempt_id,
                  'completed_steps': trainer.state.completed_updates,
                  'tokens_seen': vars(trainer.batch_source).get('training_tokens_seen'),
                  'next_lr': learning_rate_for_lifecycle(trainer.config, lifecycle, trainer.state.completed_updates),
                  'schedule': dict(active_lr_phase(lifecycle) or {'kind':'cosine','end':trainer.config.decay_updates}),
                  'path': str(target), 'sha256': _sha(target), 'complete': True, 'compatible': True,
                  'schema': 1, 'world_size': trainer.distributed.world_size,
                  'submitted_parameters': metadata.get('parameters'), 'dtype': metadata.get('dtype'),
                  'data_identity': identity_hash,
                  'configuration_signature': trainer.config.compatibility_signature()}
    _atomic_json(ROOT/'instra'/'recovery'/run_id/(bundle_id+'.json'), descriptor)
    CheckpointStore(ROOT,run_id).register(descriptor,aliases=[path],context=json.loads(os.environ.get('INSTRA_RECOVERY_CONTEXT') or '{}'))
    return descriptor


def publish(trainer,path):
    identity=_identity()
    if not identity: return
    with CheckpointStore(ROOT,identity[0]).locked():
        descriptor=_publish_unlocked(trainer,path)
    management_queue(ROOT,'instra_recovery',{'run_id':identity[0],'parameters':identity[2].get('parameters',{}),'dtype':identity[2].get('dtype')})
    return descriptor


def _management_compatible(request,descriptor):
    operational={'--max-iters','--checkpoint-root','--log-root','--result-root','--wandb-root','--host-label'}
    submitted=descriptor.get('submitted_parameters')
    return (submitted is not None and not any(value!=submitted.get(key) for key,value in request['parameters'].items() if key not in operational)
            and descriptor.get('dtype')==request.get('dtype') and descriptor.get('tokens_seen') is not None
            and descriptor.get('world_size')==1 and descriptor.get('schema')==1)


def completed(target, completed_steps):
    identity = _identity()
    if identity and completed_steps >= target:
        run_id, attempt_id, _ = identity
        _atomic_json(ROOT/'instra'/'recovery'/run_id/'completion.json',
                     {'target': target, 'completed_steps': completed_steps, 'attempt_id': attempt_id, 'finalised': True})


def validate(request):
    descriptor = request['checkpoint']
    CheckpointStore(ROOT,descriptor['run_id']).check_readable(descriptor['bundle_id'])
    if not re.fullmatch(r'[a-f0-9]{32}', descriptor['bundle_id']):
        raise ValueError('Invalid checkpoint bundle identity')
    directory = ROOT/'instra'/'recovery'/str(uuid.UUID(descriptor['run_id']))
    old = json.loads((directory/(descriptor['bundle_id']+'.json')).read_text())
    material = lambda value: {key: item for key,item in value.items() if key not in {'compatible','incompatibility_reasons'}}
    if material(old) != material(descriptor):
        raise ValueError('Checkpoint descriptor changed')
    path = Path(descriptor['path'])
    owned = ROOT/'instra'/'runs'/str(uuid.UUID(descriptor['run_id']))/'attempts'/str(uuid.UUID(descriptor['attempt_id']))/'checkpoints'
    if path.is_symlink() or not owned.resolve().is_relative_to(ROOT.resolve()) or not path.resolve(strict=True).is_relative_to(owned.resolve()) or _sha(path) != descriptor['sha256']:
        raise ValueError('Checkpoint fingerprint or ownership changed')
    return descriptor


def evidence(request):
    import sqlite3
    from collections import deque
    run_id = str(uuid.UUID(request['run_id']))
    directory = ROOT/'instra'/'runs'/run_id/'attempts'
    series = {}
    count = 0
    for path in sorted(directory.glob('*/records.sqlite3'))[-128:]:
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            continue
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as connection:
            rows = connection.execute("SELECT wall_time,process_elapsed,step,payload FROM records WHERE kind='measurements' ORDER BY sequence DESC LIMIT 512").fetchall()
        remaining = 3000
        for wall, elapsed, step, encoded in reversed(rows):
            if step is None:
                continue
            values = json.loads(encoded)
            for name,value in values.items():
                if name == 'tokens_seen' or value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
                    continue
                if len(series)>=128 and name not in series:
                    continue
                if remaining <= 0:
                    continue
                series.setdefault(name,deque(maxlen=128)).append({'attempt_id':path.parent.name,'step':step,'value':value,
                    'tokens_seen':values.get('tokens_seen'),'timestamp':wall,'process_elapsed':elapsed})
                remaining-=1;count+=1
    return {'protocol':1,'record_count':count,'charts':[{'definition':{'id':name},'points':list(points)} for name,points in sorted(series.items())]}


def rpc(request):
    if request.get('protocol') != 1:
        raise ValueError('Unsupported checkpoint protocol')
    if request['operation'] == 'management':
        return {'protocol':1,**CheckpointStore(ROOT,request['run_id']).rpc(request,lambda d:_management_compatible(request,d))}
    if request['operation'] == 'evidence':
        return evidence(request)
    if request['operation'] == 'validate':
        validate(request)
        return {'protocol':1, 'valid':True}
    if request['operation'] == 'completion':
        directory = ROOT/'instra'/'recovery'/str(uuid.UUID(request['run_id']))
        receipt = json.loads((directory/'completion.json').read_text()) if (directory/'completion.json').is_file() else {}
        return {'protocol':1, 'capabilities':{'protocol':1}, 'completed':receipt.get('finalised') is True and receipt.get('target') == request['parameters'].get('--max-iters')}
    if request['operation'] != 'inspect':
        raise ValueError('Unknown checkpoint operation')
    directory = ROOT/'instra'/'recovery'/str(uuid.UUID(request['run_id']))
    store=CheckpointStore(ROOT,request['run_id'])
    with store.locked(),store.database() as db:
        rows=store.inventory(db,lambda d:_management_compatible(request,d),validate=True)
    offset=request.get('offset',0)
    if type(offset) is not int or offset<0: raise ValueError('Invalid recovery page')
    usable=[r['descriptor'] for r in rows if r['availability']=='available' and r['validated']]
    limit=request.get('limit',2000)
    if type(limit) is not int or not 1<=limit<=2000:raise ValueError('Invalid recovery page limit')
    if 'limit' not in request and len(usable)>2000:raise ValueError('Legacy checkpoint catalogue exceeds 2000; upgrade bridge for pagination')
    checkpoints=[{**d,'compatible':True,'incompatibility_reasons':[]} for d in usable[offset:offset+limit]]
    next_offset=offset+limit if offset+limit<len(usable) else None
    completion = json.loads((directory/'completion.json').read_text()) if (directory/'completion.json').is_file() else {}
    return {'protocol':1, 'capabilities':{'protocol':1,'restore':True,'finalisation_only':True,'fork_operations':['restart_cosine'], 'fork_parameters':FORK_PARAMETERS,'world_sizes':[1]},
            'management_version':1,'next_offset':next_offset,'checkpoints':checkpoints, 'completed':completion.get('finalised') is True and completion.get('target') == request['parameters'].get('--max-iters'),
            'completed_steps':completion.get('completed_steps', max((item['completed_steps'] for item in checkpoints),default=0))}


def _execute_unreserved(request):
    descriptor = validate(request)
    if request['action'] not in {'resume','continue','fork'} or type(request['target']) is not int:
        raise ValueError('Invalid recovery action or target')
    old_target = request.get('parameters', descriptor.get('submitted_parameters') or {}).get('--max-iters')
    if request['action'] == 'resume' and request['target'] != old_target:
        raise ValueError('Resume preserves the stopping target')
    previous_target = request.get('previous_target', (descriptor.get('submitted_parameters') or {}).get('--max-iters'))
    if request['action'] == 'continue' and (type(previous_target) is not int or request['target'] <= max(previous_target, descriptor['completed_steps'])):
        raise ValueError('Continue requires a strictly larger target')
    import run_thog2_lifecycle as lifecycle
    arguments = ['--resume-from', descriptor['path'], '-q', 'fork' if request['action']=='fork' else 'resume',
                 '-n', str(request['target']), '-I', 'none']
    if request['action']=='fork':
        phase = request['phase']
        arguments += ['--fork-lr-mode','restart_cosine','--fork-learning-rate',str(phase['peak_lr']),
                      '--fork-min-lr',str(phase['min_lr']),'--fork-rewarm-iters',str(phase['rewarm_steps'])]
    overrides = request.get('parameter_overrides', {})
    if set(overrides) - set(FORK_PARAMETERS):
        raise ValueError('Unsupported Fork parameters')
    if overrides and request['action'] != 'fork':
        raise ValueError('Only Fork may change parameters')
    os.environ['INSTRA_RECOVERY_CONTEXT'] = json.dumps(request)
    return lifecycle.main(arguments)


def execute(request):
    source=request['checkpoint']
    store=CheckpointStore(ROOT,source['run_id'])
    owner='reader:'+request.get('operation_id',uuid.uuid4().hex)
    store.reserve(owner,source['bundle_id'],'active_read','submitted')
    try: return _execute_unreserved(request)
    finally: store.release(owner)


if __name__=='__main__':
    try:
        if len(sys.argv)==3 and sys.argv[1]=='--instra-checkpoint-worker':
            management_worker(ROOT,sys.argv[2],_management_compatible)
        elif sys.argv[1:] == ['--instra-recovery']:
            print(json.dumps(rpc(json.load(sys.stdin)), allow_nan=False))
        elif len(sys.argv)==3 and sys.argv[1]=='--instra-recovery-execute':
            raise SystemExit(execute(json.loads(sys.argv[2])))
        else:
            raise ValueError('Unknown recovery invocation')
    except (ValueError, OSError, KeyError) as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(2)
