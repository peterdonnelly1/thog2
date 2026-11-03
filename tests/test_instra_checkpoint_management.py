"""Stage 3D pure service, ownership, reservations, failure and restart proofs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import pytest

MODEL_ROOT=Path(__file__).resolve().parents[1]
if not (MODEL_ROOT/'instra_checkpoint_store.py').is_file():pytest.skip('Requires the Stage 3D model-owned checkpoint service',allow_module_level=True)
spec=importlib.util.spec_from_file_location('stage3d_store',MODEL_ROOT/'instra_checkpoint_store.py')
store_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(store_module)
Store=store_module.CheckpointStore
policy=store_module.management_policy
evaluate=store_module.management_evaluate

@pytest.fixture
def environment(tmp_path):
    run_id=str(uuid.uuid4());attempt=str(uuid.uuid4());store=Store(tmp_path,run_id)
    def publish(step, *, attempt_id=attempt, registered=True, context=None, multi=False):
        bundle_id=uuid.uuid4().hex
        directory=tmp_path/'instra'/'runs'/run_id/'attempts'/attempt_id/'checkpoints';directory.mkdir(parents=True,exist_ok=True)
        payload=directory/(bundle_id+'.pt');payload.write_bytes(('state:'+str(step)).encode())
        descriptor={'bundle_id':bundle_id,'run_id':run_id,'attempt_id':attempt_id,'completed_steps':step,'tokens_seen':step*8,
                    'path':str(payload),'sha256':Store.hash_file(payload),'complete':True,'compatible':True,'schema':1,'world_size':1}
        store.directory.mkdir(parents=True,exist_ok=True);(store.directory/(bundle_id+'.json')).write_text(json.dumps(descriptor))
        if registered:
            members=[str(payload)]
            if multi:
                sidecar=directory/(bundle_id+'.sidecar');sidecar.write_bytes(b'optimizer');members.append(str(sidecar))
            with store.locked():store.register(descriptor,context=context,members=members)
        return descriptor
    def rpc(operation,**arguments):
        return store.rpc({'management_operation':operation,'active':False,**arguments},lambda d:d.get('schema')==1 and d.get('world_size')==1)
    return store,publish,rpc

def mutate(rpc,operation,ids,**arguments):
    return rpc(operation,bundle_ids=[d['bundle_id'] for d in ids],operation_id=uuid.uuid4().hex,**arguments)

def rows(rpc):return rpc('list',limit=200)['rows']

@pytest.mark.parametrize('value',[{'n':0},{'n':True},{'n':1.2},{'n':-1},{'mode':'guess'},{'automatic':'yes'},{'version':2},{'extra':1}])
def test_invalid_policy(value):
    with pytest.raises(ValueError):policy(value)

def test_default_keep_all_restart_and_manual_remove(environment):
    store,publish,rpc=environment;a=publish(1);b=publish(2)
    assert rpc('policy')['policy']==policy()
    assert all(d['decision']=='keep' and d['reason']=='keep_all' for d in rpc('preview')['decisions'])
    assert mutate(rpc,'remove',[a])['status']=='confirmed'
    assert Store(store.root,store.run_id).rpc({'management_operation':'list'},lambda d:True)['rows'][0]
    assert Path(a['path']).exists() and any(r['availability']=='removed' for r in rows(rpc))
    assert mutate(rpc,'restore',[a])['status']=='confirmed'
    assert all(r['availability']=='available' for r in rows(rpc))

@pytest.mark.parametrize('mode,n,count',[('all',1,0),('latest',1,3),('n',1,3),('n',2,2),('n',8,0)])
def test_latest_n_and_keep_all(environment,mode,n,count):
    store,publish,rpc=environment
    for step in range(4):publish(step)
    rpc('policy',policy=policy({'mode':mode,'n':n}))
    plan=rpc('preview');assert sum(d['decision']=='remove' for d in plan['decisions'])==count
    result=rpc('apply',token=plan['token'],operation_id=uuid.uuid4().hex)
    assert result['status']=='confirmed' and len(result['outcomes'])==count
    assert len([r for r in rows(rpc) if r['availability']=='removed'])==count

def test_multi_member_restore_conflict_and_no_file_effects(environment):
    store,publish,rpc=environment;a=publish(1,multi=True)
    assert len(rows(rpc)[0]['members'])==2
    mutate(rpc,'remove',[a]);Path(a['path']).write_bytes(b'changed')
    assert mutate(rpc,'restore',[a])['outcomes'][0]['status']=='blocked'
    assert Path(a['path']).read_bytes()==b'changed'

def test_malformed_missing_corrupt_unsupported_siblings(environment):
    store,publish,rpc=environment;good=publish(1);missing=publish(2);corrupt=publish(3);unsupported=publish(4)
    Path(missing['path']).unlink();Path(corrupt['path']).write_bytes(b'bad')
    unsupported['world_size']=2;(store.directory/(unsupported['bundle_id']+'.json')).write_text(json.dumps(unsupported))
    (store.directory/('a'*32+'.json')).write_text('{invalid')
    payload=Path(good['path']).parent/'partial.tmp';payload.write_bytes(b'partial')
    result=rpc('validate',limit=200);by_id={r['bundle_id']:r for r in result['rows']}
    assert by_id[good['bundle_id']]['availability']=='available'
    assert by_id[missing['bundle_id']]['availability']=='missing'
    assert by_id[corrupt['bundle_id']]['availability']=='corrupt'
    assert by_id[unsupported['bundle_id']]['availability']=='corrupt'
    assert any(r['availability']=='writing/incomplete' for r in result['rows'])

def test_unknown_ownership_retained_and_denied_manual_removal(environment):
    store,publish,rpc=environment;legacy=publish(1,registered=False);publish(2)
    rpc('validate');rpc('policy',policy=policy({'mode':'latest'}))
    assert next(d for d in rpc('preview')['decisions'] if d['bundle_id']==legacy['bundle_id'])['reason']=='external_or_unknown'
    assert mutate(rpc,'remove',[legacy])['status']=='partial' and Path(legacy['path']).exists()

def test_symlink_and_run_identity_rejected(environment,tmp_path):
    store,publish,rpc=environment;d=publish(1);path=Path(d['path']);target=tmp_path/'outside';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
    assert rpc('validate')['rows'][0]['availability']=='corrupt'
    d['run_id']=str(uuid.uuid4());(store.directory/(d['bundle_id']+'.json')).write_text(json.dumps(d))
    with pytest.raises(ValueError):store.descriptor(d['bundle_id'])


def test_pins_overlap_restart_and_no_expiry(environment):
    store,publish,rpc=environment;a=publish(1);publish(2)
    mutate(rpc,'protect',[a]);store.reserve('fork:'+uuid.uuid4().hex,a['bundle_id'],'fork_source','persistent')
    restarted=Store(store.root,store.run_id)
    with restarted.database() as db:assert len(restarted.records(db,'pin'))==2
    mutate(rpc,'unprotect',[a],reason='manual')
    assert mutate(rpc,'remove',[a])['status']=='partial'
    assert mutate(rpc,'unprotect',[a],reason='fork_source')['status']=='partial'
    mutate(rpc,'unprotect',[a],reason='fork_source',acknowledge_reproducibility=True)
    assert mutate(rpc,'remove',[a])['status']=='confirmed'


def test_draft_adoption_no_gap_and_discard_refuses_submitted(environment):
    store,publish,rpc=environment;a=publish(1);publish(2);owner=uuid.uuid4().hex
    store.reserve(owner,a['bundle_id'])
    store.reserve(owner,a['bundle_id'],status='submitted',require_existing=True)
    with pytest.raises(ValueError,match='submitted'):store.release(owner,draft_only=True)
    assert mutate(rpc,'remove',[a])['status']=='partial'
    store.release(owner)
    with pytest.raises(ValueError,match='lost'):store.reserve(owner,a['bundle_id'],status='submitted',require_existing=True)


def test_cleanup_reader_races_deterministic(environment):
    store,publish,rpc=environment;a=publish(1);publish(2)
    ready=threading.Event();proceed=threading.Event();result=[]
    def reader():
        store.reserve('reader:'+uuid.uuid4().hex,a['bundle_id'],'active_read','submitted');ready.set();proceed.wait(3)
    thread=threading.Thread(target=reader);thread.start();assert ready.wait(3)
    result=mutate(rpc,'remove',[a]);proceed.set();thread.join()
    assert result['status']=='partial'
    b=publish(3);assert mutate(rpc,'remove',[b])['status']=='confirmed'
    with pytest.raises(ValueError,match='removed'):store.reserve('reader:'+uuid.uuid4().hex,b['bundle_id'])


def test_stale_preview_pin_policy_and_changed_key(environment):
    store,publish,rpc=environment;a=publish(1);publish(2);rpc('policy',policy=policy({'mode':'latest'}));plan=rpc('preview')
    mutate(rpc,'protect',[a])
    with pytest.raises(ValueError,match='changed'):rpc('apply',token=plan['token'],operation_id=uuid.uuid4().hex)
    key=uuid.uuid4().hex;request={'bundle_ids':[a['bundle_id']],'operation_id':key}
    first=rpc('protect',**request);assert rpc('protect',**request)==first
    with pytest.raises(ValueError,match='another request'):rpc('unprotect',reason='manual',**request)


def test_lineage_rewind_superseded_higher_step(environment):
    store,publish,rpc=environment;source=publish(5);old=publish(20);new=publish(6,attempt_id=str(uuid.uuid4()),context={'checkpoint':source})
    rpc('policy',policy=policy({'mode':'latest'}));listing=rows(rpc)
    assert next(r for r in listing if r['bundle_id']==old['bundle_id'])['lineage']=='superseded'
    plan=rpc('preview');assert next(d for d in plan['decisions'] if d['bundle_id']==old['bundle_id'])['decision']=='remove'
    assert 'latest_recovery' in next(r for r in listing if r['bundle_id']==new['bundle_id'])['roles']


def test_hashing_not_repeated_on_inventory_poll(environment,monkeypatch):
    store,publish,rpc=environment;publish(1)
    monkeypatch.setattr(Store,'hash_file',staticmethod(lambda p:(_ for _ in ()).throw(AssertionError('poll must not hash'))))
    assert rows(rpc)[0]['validated']
    assert rpc('validate')['rows'][0]['validated']


def test_multimember_alias_storage_accounting(environment):
    store,publish,rpc=environment;a=publish(1,multi=True);payload=Path(a['path']);alias=payload.parent/'ckpt.pt';os.link(payload,alias)
    result=rpc('list');assert result['normal_bytes']==len(payload.read_bytes())+len(b'optimizer')
    mutate(rpc,'remove',[a]);assert alias.exists() and payload.exists()
    result=rpc('list');assert result['normal_bytes']==0 and result['removed_bytes']>0 and result['reclaimed_bytes']==0


def test_policy_persists_without_hidden_cleanup(environment):
    store,publish,rpc=environment;publish(1);publish(2)
    rpc('policy',policy=policy({'mode':'latest','automatic':True}))
    assert all(r['availability']=='available' for r in rows(rpc))
    with Store(store.root,store.run_id).database() as db:assert store.get(db,'policy','effective')['automatic']


def test_more_than_2000_pagination(environment):
    store,publish,rpc=environment
    start=time.monotonic()
    # Small real immutable members exercise the actual catalogue, not a mocked page.
    for step in range(2005):publish(step)
    ids=[];offset=0;snapshot=None
    while offset is not None:
        result=rpc('list',offset=offset,limit=200,**({'snapshot':snapshot} if snapshot else {}));ids.extend(r['bundle_id'] for r in result['rows']);offset=result['next_offset'];snapshot=result['snapshot']
    assert len(ids)==len(set(ids))==2005
    rpc('policy',policy=policy({'mode':'n','n':2}));assert sum(d['decision']=='remove' for d in rpc('preview')['decisions'])==2003
    import resource
    print(f'\n2005 bundles publish/page/preview: {time.monotonic()-start:.3f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KiB')


def test_explicit_external_registration_only_deregisters_reference(environment,tmp_path):
    store,publish,rpc=environment;descriptor=publish(1,registered=False)
    external=tmp_path.parent/('external-'+uuid.uuid4().hex+'.pt');external.write_bytes(b'external original')
    descriptor.update(path=str(external),sha256=Store.hash_file(external))
    (store.directory/(descriptor['bundle_id']+'.json')).write_text(json.dumps(descriptor))
    try:
        store.register_external(descriptor)
        assert rpc('validate')['rows'][0]['ownership']=='external'
        assert mutate(rpc,'remove',[descriptor])['status']=='confirmed'
        assert external.read_bytes()==b'external original'
        assert mutate(rpc,'restore',[descriptor])['status']=='confirmed'
        assert external.read_bytes()==b'external original'
    finally:external.unlink()


def test_policy_revision_rejects_changed_then_restored_policy(environment):
    store,publish,rpc=environment;publish(1);publish(2)
    rpc('policy',policy={'mode':'latest'});plan=rpc('preview')
    rpc('policy',policy={'mode':'all'});rpc('policy',policy={'mode':'latest'})
    with pytest.raises(ValueError,match='changed'):rpc('apply',token=plan['token'],operation_id=uuid.uuid4().hex)


def test_atomic_rollback_before_receipt_leaves_bundle_available(environment):
    store,publish,rpc=environment;descriptor=publish(1)
    with pytest.raises(RuntimeError):
        with store.locked(),store.database() as db:
            store.put(db,'removed',descriptor['bundle_id'],{'state':'removed'})
            raise RuntimeError('injected crash before transaction commit')
    assert rows(rpc)[0]['availability']=='available' and Path(descriptor['path']).exists()


def test_fork_pin_persists_child_provenance(environment):
    store,publish,rpc=environment;descriptor=publish(1);child=uuid.uuid4().hex;owner='fork:'+uuid.uuid4().hex
    rpc('fork_pin',owner=owner,bundle_id=descriptor['bundle_id'],child_run_id=child)
    with store.database() as db:assert store.get(db,'pin',owner)['child_run_ids']==[child]


@pytest.mark.parametrize('adapter',['thog2','nanogpt'])
def test_repeated_remove_restore_has_monotonic_receipts_and_stales_preview(environment,adapter):
    original,publish,_=environment
    if adapter=='nanogpt':
        helper=MODEL_ROOT.parent/'nanogpt'/'instra_producer.py'
        if not helper.is_file():pytest.skip('Requires the nanoGPT helper for cross-adapter verification')
        spec=importlib.util.spec_from_file_location('stage3d_nano_revision',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        candidate=module.CheckpointStore
    else:candidate=Store
    store=candidate(original.root,original.run_id)
    def rpc(operation,**arguments):return store.rpc({'management_operation':operation,'active':False,**arguments},lambda d:True)
    first=publish(1);publish(2);rpc('policy',policy={'mode':'latest'})
    preview=rpc('preview');key=uuid.uuid4().hex
    request={'bundle_ids':[first['bundle_id']],'operation_id':key}
    removed=rpc('remove',**request);assert removed['outcomes'][0]['revision']==1
    restored=mutate(rpc,'restore',[first]);assert restored['outcomes'][0]['revision']==2
    with pytest.raises(ValueError,match='changed'):rpc('apply',token=preview['token'],operation_id=uuid.uuid4().hex)
    assert mutate(rpc,'remove',[first])['outcomes'][0]['revision']==3
    assert mutate(rpc,'restore',[first])['outcomes'][0]['revision']==4
    assert rpc('remove',**request)==removed  # Lost-ack replay cannot re-remove a restored bundle.
    assert next(r for r in rpc('list')['rows'] if r['bundle_id']==first['bundle_id'])['management_revision']==4
