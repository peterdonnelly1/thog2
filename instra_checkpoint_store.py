"""Model-owned checkpoint management v1. No training framework dependency.

Logical removal preserves every byte. Per-Run flock precedes SQLite writes.
This module is embedded verbatim in nanoGPT's single integration helper.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid


def management_sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def management_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def management_now():
    return datetime.now(timezone.utc).isoformat()


def management_policy(value=None):
    policy = {'version': 1, 'mode': 'all', 'n': 1, 'keep_best': False, 'automatic': False}
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(policy):
            raise ValueError('Unknown checkpoint policy fields')
        policy.update(value)
    if (policy['version'] != 1 or policy['mode'] not in {'all', 'latest', 'n'} or
            type(policy['n']) is not int or policy['n'] < 1 or policy['n'] > 1000000 or
            type(policy['keep_best']) is not bool or type(policy['automatic']) is not bool):
        raise ValueError('Invalid retention policy; N must be a positive integer')
    return policy


def management_evaluate(rows, policy, *, complete=True, active=False):
    """Pure, stable retention evaluator. Unknown lineage/ownership is never guessed."""
    policy = management_policy(policy)
    valid = [r for r in rows if r.get('availability') == 'available' and r.get('validated') and
             r.get('compatible') and r.get('lineage') == 'active' and r.get('ownership') == 'managed']
    ranked = sorted(valid, key=lambda r: (r['completed_steps'], r['bundle_id']), reverse=True)
    keep = {r['bundle_id'] for r in (ranked if policy['mode'] == 'all' else ranked[:1 if policy['mode'] == 'latest' else policy['n']])}
    decisions = []
    for row in rows:
        code = 'candidate'
        if not complete: code = 'incomplete_snapshot'
        elif policy['mode'] == 'all': code = 'keep_all'
        elif row.get('availability') != 'available' or not row.get('validated') or not row.get('compatible'): code = 'unvalidated_or_unavailable'
        elif row.get('ownership') != 'managed': code = 'external_or_unknown'
        elif row.get('lineage') not in {'active', 'superseded'}: code = 'ambiguous_lineage'
        elif row.get('protections'): code = 'protected'
        elif active and ranked and row['bundle_id'] == ranked[0]['bundle_id']: code = 'active_recovery'
        elif row['bundle_id'] in keep: code = 'policy_count'
        elif policy['keep_best'] and row.get('best_evaluation'): code = 'declared_best'
        elif not ranked: code = 'no_valid_replacement'
        decisions.append({'bundle_id': row['bundle_id'], 'decision': 'remove' if code == 'candidate' else 'keep',
                          'reason': code, 'size_bytes': row.get('size_bytes')})
    return decisions


class CheckpointStore:
    version = 1

    def __init__(self, root, run_id):
        self.root = Path(root).resolve()
        self.run_id = str(uuid.UUID(run_id))
        self.directory = self.root/'instra'/'recovery'/self.run_id
        self.home = self.root/'instra'/'checkpoint_management'

    def safe(self, path):
        path = Path(path)
        if not path.is_absolute(): path = self.root/path
        if not path.is_relative_to(self.root): raise ValueError('Checkpoint path escapes Model Source')
        for parent in (path, *path.parents):
            if parent == self.root: break
            if parent.is_symlink(): raise ValueError('Checkpoint path contains a symlink')
        if not path.resolve().is_relative_to(self.root): raise ValueError('Checkpoint containment changed')
        return path

    @contextmanager
    def locked(self):
        self.safe(self.home).mkdir(parents=True, exist_ok=True)
        lock = self.safe(self.home/(self.run_id+'.lock'))
        with lock.open('a') as stream:
            end = time.monotonic()+5
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= end: raise ValueError('Checkpoint Run is busy; retry after current operation')
                    time.sleep(.02)
            try: yield
            finally: fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def database(self):
        self.safe(self.home).mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self.safe(self.home/'management.sqlite3'), timeout=5)
        database.execute('PRAGMA synchronous=FULL')
        database.execute('CREATE TABLE IF NOT EXISTS records (run TEXT, kind TEXT, key TEXT, value TEXT NOT NULL, PRIMARY KEY(run,kind,key))')
        try:
            with database: yield database
        finally: database.close()

    def get(self, db, kind, key, default=None):
        row = db.execute('SELECT value FROM records WHERE run=? AND kind=? AND key=?', (self.run_id, kind, key)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, db, kind, key, value):
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)', (self.run_id, kind, key, json.dumps(value, allow_nan=False)))

    def records(self, db, kind):
        return {key: json.loads(value) for key,value in db.execute('SELECT key,value FROM records WHERE run=? AND kind=?', (self.run_id, kind))}

    def descriptor(self, bundle_id, *, external=False):
        if not isinstance(bundle_id, str) or not re.fullmatch('[a-f0-9]{32}', bundle_id): raise ValueError('Invalid checkpoint bundle identity')
        file = self.safe(self.directory/(bundle_id+'.json'))
        if file.stat().st_size > 262144: raise ValueError('Checkpoint manifest exceeds limit')
        value = json.loads(file.read_text())
        if value.get('bundle_id') != bundle_id or str(uuid.UUID(value['run_id'])) != self.run_id:
            raise ValueError('Checkpoint Run or bundle binding changed')
        attempt = str(uuid.UUID(value['attempt_id']))
        parent = self.safe(self.root/'instra'/'runs'/self.run_id/'attempts'/attempt/'checkpoints')
        path = self.external_path(value['path']) if external else self.safe(value['path'])
        if not external and not path.is_relative_to(parent): raise ValueError('Payload not owned by this Attempt')
        return value

    @staticmethod
    def external_path(name):
        path=Path(name)
        if not path.is_absolute():raise ValueError('External declaration requires a canonical absolute path')
        for parent in (path,*path.parents):
            if parent.is_symlink():raise ValueError('External checkpoint contains a symlink')
        if path.resolve()!=path:raise ValueError('External binding is not canonical')
        return path

    def register_external(self, descriptor):
        """Explicit model registration only; not exposed as a browser operation.

        Deregistration never renames, rewrites or deletes external originals.
        These adapters do not claim external restoration support.
        """
        with self.locked(),self.database() as db:
            checked=self.descriptor(descriptor['bundle_id'],external=True)
            if checked!=descriptor:raise ValueError('External descriptor changed')
            path=self.external_path(descriptor['path'])
            if self.hash_file(path)!=descriptor['sha256']:raise ValueError('External fingerprint changed')
            if self.get(db,'bundle',descriptor['bundle_id']):raise ValueError('Bundle already registered')
            self.put(db,'bundle',descriptor['bundle_id'],{'version':1,'fingerprint':management_digest(descriptor),'descriptor':descriptor,
                'ownership':'external','declared_by':'explicit_model_registration','saved_at':None,
                'members':[{'path':str(path),'size_bytes':path.stat().st_size,'sha256':descriptor['sha256']}],
                'aliases':[],'context':{},'best_evaluation':None})

    def register(self, descriptor, *, aliases=(), context=None, members=None, best_evaluation=None):
        """Only the model publisher calls this, while holding its Run lock."""
        bundle_id = descriptor['bundle_id']
        self.descriptor(bundle_id)
        paths = members or [descriptor['path']]
        owned = self.root/'instra'/'runs'/self.run_id/'attempts'/str(uuid.UUID(descriptor['attempt_id']))/'checkpoints'
        manifest_members = []
        if descriptor['path'] not in [str(p) for p in paths]: raise ValueError('Payload must be a declared member')
        for name in paths:
            path = self.safe(name)
            if not path.is_relative_to(owned): raise ValueError('Member escapes owned checkpoint directory')
            manifest_members.append({'path':str(path), 'size_bytes':path.stat().st_size,
                                     'sha256':descriptor['sha256'] if str(path)==descriptor['path'] else self.hash_file(path)})
        context = context or {}
        record = {'version':1, 'fingerprint':management_digest(descriptor), 'descriptor':descriptor,
                  'ownership':'managed', 'declared_by':'model_publisher', 'saved_at':management_now(),
                  'members':manifest_members, 'aliases':[str(p) for p in aliases], 'context':context,
                  'best_evaluation':best_evaluation}
        with self.database() as db:
            old = self.get(db, 'bundle', bundle_id)
            if old and old['fingerprint'] != record['fingerprint']: raise ValueError('Immutable bundle registration changed')
            if not old: self.put(db, 'bundle', bundle_id, record)
            signatures=[]
            for member in record['members']:
                stat=self.safe(member['path']).stat()
                signatures.append([member['path'],stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
            self.put(db,'validation',bundle_id,{'fingerprint':record['fingerprint'],'at':management_now(),'signatures':signatures})
            self.put(db, 'lineage', 'active', {'attempt_id':descriptor['attempt_id'], 'context':context})

    @staticmethod
    def hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        return digest.hexdigest()

    def fresh(self, db, descriptor, record):
        fingerprint = management_digest(descriptor)
        if record and record['fingerprint'] != fingerprint: raise ValueError('Manifest fingerprint changed')
        members = record['members'] if record else [{'path':descriptor['path'], 'sha256':descriptor['sha256']}]
        signatures = []
        for member in members:
            external=record and record['ownership']=='external'
            path = self.external_path(member['path']) if external else self.safe(member['path'])
            owned=self.root/'instra'/'runs'/self.run_id/'attempts'/str(uuid.UUID(descriptor['attempt_id']))/'checkpoints'
            if not external and not path.is_relative_to(owned): raise ValueError('Member binding changed')
            stat = path.stat()
            if self.hash_file(path) != member['sha256']: raise ValueError('Checkpoint member hash changed')
            signatures.append([str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
        payload_member=next((m for m in members if m['path']==descriptor['path']),None)
        if not payload_member or payload_member['sha256']!=descriptor['sha256']: raise ValueError('Payload fingerprint differs from registration')
        validation = {'fingerprint':fingerprint,'at':management_now(),'signatures':signatures}
        self.put(db,'validation',descriptor['bundle_id'],validation)
        return validation

    def current_validation(self, descriptor, validation):
        if not validation or validation['fingerprint'] != management_digest(descriptor): return False
        try:
            for name,device,inode,size,mtime,ctime in validation['signatures']:
                stat = (self.safe(name) if Path(name).is_relative_to(self.root) else self.external_path(name)).stat()
                if [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns] != [device,inode,size,mtime,ctime]: return False
        except OSError: return False
        return True

    def inventory(self, db, compatible, *, validate=False):
        registered = self.records(db,'bundle')
        removed = self.records(db,'removed')
        pins = self.records(db,'pin')
        validations = self.records(db,'validation')
        active = self.get(db,'lineage','active',{})
        lineage_limits={str(active.get('attempt_id')):None} if active else {}
        context=active.get('context',{})
        visited=set()
        while context.get('checkpoint') and context['checkpoint'].get('run_id')==self.run_id:
            source=context['checkpoint'];source_id=source.get('bundle_id')
            if source_id in visited: break
            visited.add(source_id)
            lineage_limits[str(source['attempt_id'])]=source['completed_steps']
            context=registered.get(source_id,{}).get('context',{})
        rows = []
        paths = sorted(self.safe(self.directory).glob('*.json'))
        if len(paths) > 100000: raise ValueError('Catalogue exceeds management limit; cleanup refused')
        for file in paths:
            if file.name == 'completion.json': continue
            bundle_id = file.stem
            row = {'bundle_id':bundle_id,'run_id':self.run_id,'availability':'incomplete','compatible':False,
                   'validated':False,'size_bytes':None,'saved_at':None,'lineage':'unknown','ownership':'unknown','protections':[],'roles':[]}
            row['management_revision']=removed.get(bundle_id,{}).get('revision',0)
            try:
                record = registered.get(bundle_id)
                external=bool(record and record['ownership']=='external')
                descriptor = self.descriptor(bundle_id,external=external)
                row.update(descriptor=descriptor,attempt_id=descriptor['attempt_id'],completed_steps=descriptor['completed_steps'],
                           tokens_seen=descriptor.get('tokens_seen'),fingerprint=management_digest(descriptor),compatible=bool(compatible(descriptor)))
                if external:row['compatible']=False;row['diagnostic']='External registration: removal affects reference only; this adapter does not support external restoration'
                if record:
                    row.update(ownership=record['ownership'],saved_at=record['saved_at'],members=record['members'],
                               aliases=record['aliases'],ownership_evidence=record['declared_by'],best_evaluation=record.get('best_evaluation'))
                    # A new Attempt is a new active branch. Older evidence is retained as
                    # ambiguous unless its ancestry is explicitly supplied by the controller.
                    attempt=str(record['descriptor']['attempt_id'])
                    if attempt in lineage_limits:
                        limit=lineage_limits[attempt]
                        row['lineage']='active' if limit is None or descriptor['completed_steps']<=limit else 'superseded'
                row['protections'] = [v for v in pins.values() if v['bundle_id']==bundle_id]
                row['roles'] = sorted({p['reason'] for p in row['protections']})
                if row.get('best_evaluation'): row['roles'].append('best_evaluation')
                if bundle_id in removed and removed[bundle_id].get('state')=='removed':
                    row.update(availability='removed',removal=removed[bundle_id])
                else:
                    path = self.external_path(descriptor['path']) if external else self.safe(descriptor['path']); row['size_bytes'] = path.stat().st_size
                    member_paths=[m['path'] for m in record['members']] if record else [descriptor['path']]
                    storage={}
                    for member_path in member_paths:
                        stat=(self.external_path(member_path) if external else self.safe(member_path)).stat();storage[f'{stat.st_dev}:{stat.st_ino}']=stat.st_size
                    row['storage_members']=storage
                    row['size_bytes']=sum(storage.values())
                    if validate and not self.current_validation(descriptor,validations.get(bundle_id)): validations[bundle_id] = self.fresh(db,descriptor,record)
                    row['validated'] = self.current_validation(descriptor,validations.get(bundle_id))
                    row['validation'] = validations.get(bundle_id)
                    row['availability'] = 'incomplete' if descriptor.get('complete') is not True else 'incompatible' if not row['compatible'] else 'available' if row['validated'] else 'unvalidated'
                if row['availability']=='removed' and record:
                    storage={}
                    for member in record['members']:
                        stat=(self.external_path(member['path']) if external else self.safe(member['path'])).stat();storage[f'{stat.st_dev}:{stat.st_ino}']=stat.st_size
                    row['storage_members']=storage;row['size_bytes']=sum(storage.values())
            except FileNotFoundError as error: row.update(availability='missing',diagnostic=str(error))
            except (ValueError,TypeError,KeyError,OSError) as error: row.update(availability='corrupt',diagnostic=str(error))
            rows.append(row)
        for file in sorted((self.root/'instra'/'runs'/self.run_id/'attempts').glob('*/checkpoints/*.tmp')):
            rows.append({'bundle_id':'temporary:'+file.name,'run_id':self.run_id,'availability':'writing/incomplete','ownership':'unknown',
                         'size_bytes':None,'validated':False,'compatible':False,'lineage':'unknown','protections':[],'roles':[]})
        latest = sorted([r for r in rows if r['availability']=='available' and r['lineage']=='active'],key=lambda r:(r['completed_steps'],r['bundle_id']))
        if latest: latest[-1]['roles'].append('latest_recovery')
        return rows

    def reserve(self, owner, bundle_id, reason='recovery', status='draft', require_existing=False):
        if not isinstance(owner,str) or not re.fullmatch('[a-zA-Z0-9:_-]{16,128}',owner): raise ValueError('Invalid reservation owner')
        with self.locked(), self.database() as db:
            descriptor = self.descriptor(bundle_id)
            if self.get(db,'removed',bundle_id,{}).get('state')=='removed': raise ValueError('Checkpoint has been removed')
            self.fresh(db,descriptor,self.get(db,'bundle',bundle_id))
            old = self.get(db,'pin',owner)
            if require_existing and not old: raise ValueError('Draft protection was lost; select and reserve again')
            if old and old['fingerprint']!=management_digest(descriptor):raise ValueError('Reserved checkpoint fingerprint changed')
            if old and old['bundle_id']!=bundle_id: raise ValueError('Reservation owner changed bundle')
            if old and old['status']=='submitted' and status=='draft': return old
            pin={'owner':owner,'bundle_id':bundle_id,'fingerprint':management_digest(descriptor),'reason':reason,'status':status,'acquired_at':old['acquired_at'] if old else management_now()}
            self.put(db,'pin',owner,pin)
            return pin

    def release(self, owner, *, draft_only=False):
        with self.locked(), self.database() as db:
            pin=self.get(db,'pin',owner)
            if pin and draft_only and pin['status']!='draft': raise ValueError('Draft is attached to submitted recovery; reconcile before discarding')
            db.execute('DELETE FROM records WHERE run=? AND kind=? AND key=?',(self.run_id,'pin',owner))

    def check_readable(self, bundle_id):
        with self.locked(),self.database() as db:
            if self.get(db,'removed',bundle_id,{}).get('state')=='removed': raise ValueError('Checkpoint is recoverably removed; Restore it first')

    def rpc(self, request, compatible):
        operation=request['management_operation']
        if operation=='capabilities':
            return {'management_version':1,'inventory':True,'recoverable_removal':True,'restore_removed':True,'best_evaluation':False,
                    'external_deregistration':True,'permanent_deletion':False,'world_sizes':[1]}
        if operation=='reserve':
            pin=self.reserve(request['owner'],request['bundle_id'],request.get('reason','recovery'),request.get('status','draft'),request.get('require_existing',False))
            return {'protection':pin}
        if operation=='fork_pin':
            pin=self.reserve(request['owner'],request['bundle_id'],'fork_source','persistent')
            with self.locked(),self.database() as db:
                pin=self.get(db,'pin',request['owner'])
                pin['child_run_ids']=sorted(set(pin.get('child_run_ids',[]))|{request['child_run_id']})
                self.put(db,'pin',request['owner'],pin)
            return {'protection':pin}
        if operation=='release':
            self.release(request['owner'],draft_only=request.get('draft_only',False))
            return {'released':True}
        with self.locked(),self.database() as db:
            if operation=='policy':
                if 'policy' in request:
                    policy=management_policy(request['policy'])
                    if policy!=self.get(db,'policy','effective',management_policy()):self.put(db,'policy','revision',self.get(db,'policy','revision',0)+1)
                    self.put(db,'policy','effective',policy)
                return {'policy':self.get(db,'policy','effective',management_policy())}
            if operation=='reconcile': return {'operations':dict(list(self.records(db,'operation').items())[-200:]),'protections':list(self.records(db,'pin').values()),'automatic':self.get(db,'automatic','pending')}
            rows=self.inventory(db,compatible,validate=operation in {'validate','preview','apply','automatic'})
            policy=self.get(db,'policy','effective',management_policy())
            snapshot=management_digest([{'id':r['bundle_id'],'fp':r.get('fingerprint'),'availability':r['availability'],'lineage':r['lineage'],
                                         'validated':r['validated'],'pins':r['protections'],'revision':r.get('management_revision',0)} for r in rows])
            if operation in {'list','validate'}:
                offset=request.get('offset',0);limit=request.get('limit',100)
                if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=200: raise ValueError('Invalid checkpoint page')
                expected=request.get('snapshot')
                if expected and expected!=snapshot: raise ValueError('Inventory changed; refresh from first page')
                normal=[r for r in rows if r['availability']!='removed'];trash=[r for r in rows if r['availability']=='removed']
                normal_storage={k:v for r in normal for k,v in r.get('storage_members',{}).items()}
                removed_storage={k:v for r in trash for k,v in r.get('storage_members',{}).items() if k not in normal_storage}
                return {'rows':rows[offset:offset+limit],'next_offset':offset+limit if offset+limit<len(rows) else None,'total':len(rows),
                        'snapshot':snapshot,'complete':True,'confirmed_at':management_now(),'policy':policy,
                        'normal_bytes':sum(normal_storage.values()),'removed_bytes':sum(removed_storage.values()),
                        'unknown_sizes':sum(r['size_bytes'] is None for r in rows),'reclaimed_bytes':0}
            decisions=management_evaluate(rows,policy,active=request.get('active',True))
            plan={'snapshot':snapshot,'policy':policy,'policy_revision':self.get(db,'policy','revision',0),'decisions':decisions,'run_id':self.run_id}
            token=management_digest(plan)
            if operation=='preview': return {**plan,'token':token,'reclaimed_bytes':0}
            if operation=='automatic' and not policy['automatic']: return {'status':'disabled'}
            if operation not in {'apply','automatic','remove','restore','protect','unprotect'}: raise ValueError('Unsupported checkpoint management operation')
            key=request.get('operation_id')
            if not isinstance(key,str) or not 16<=len(key)<=128: raise ValueError('Durable operation identity required')
            request_hash=management_digest(request)
            old=self.get(db,'operation',key)
            if old:
                if old['request_hash']!=request_hash: raise ValueError('Operation key already used for another request')
                return old['result']
            if operation=='apply' and request.get('token')!=token: raise ValueError('Checkpoint policy or inventory changed; Preview again')
            selected=request.get('bundle_ids',[])
            if operation in {'apply','automatic'}: selected=[d['bundle_id'] for d in decisions if d['decision']=='remove']
            if not isinstance(selected,list) or len(selected)>4096 or len(set(selected))!=len(selected): raise ValueError('Invalid checkpoint selection')
            by_id={r['bundle_id']:r for r in rows}; outcomes=[]
            for bundle_id in selected:
                row=by_id.get(bundle_id)
                try:
                    if not row: raise ValueError('Checkpoint is unavailable')
                    if request.get('fingerprints',{}).get(bundle_id,row.get('fingerprint'))!=row.get('fingerprint'): raise ValueError('Checkpoint changed; refresh selection')
                    if operation=='protect':
                        owner='manual:'+bundle_id
                        self.put(db,'pin',owner,{'owner':owner,'bundle_id':bundle_id,'reason':'manual','status':'persistent','fingerprint':row.get('fingerprint'),'acquired_at':management_now()})
                    elif operation=='unprotect':
                        reason=request.get('reason')
                        if reason not in {'manual','fork_source'}: raise ValueError('Only manual or Fork-source reasons can be cleared')
                        if reason=='fork_source' and request.get('acknowledge_reproducibility') is not True: raise ValueError('Acknowledge loss of exact branch-point reproducibility; Child checkpoints remain independent')
                        for owner,pin in self.records(db,'pin').items():
                            if pin['bundle_id']==bundle_id and pin['reason']==reason:
                                db.execute('DELETE FROM records WHERE run=? AND kind=? AND key=?',(self.run_id,'pin',owner))
                    else:
                        if row['protections']: raise ValueError('Checkpoint has protection/reservation: '+', '.join(p['reason'] for p in row['protections']))
                        if row['ownership'] not in {'managed','external'}: raise ValueError('Ownership is unknown; removal is not authorised')
                        if operation!='restore' and request.get('active',True) and 'latest_recovery' in row['roles']: raise ValueError('Latest active recovery safeguard')
                        descriptor=self.descriptor(bundle_id,external=row['ownership']=='external');record=self.get(db,'bundle',bundle_id)
                        self.fresh(db,descriptor,record)
                        if operation=='restore':
                            removed=self.get(db,'removed',bundle_id)
                            if not removed or removed['state']!='removed': raise ValueError('Checkpoint is not removed')
                            if removed['fingerprint']!=management_digest(descriptor): raise ValueError('Restore conflicts with changed descriptor')
                            removed.update(state='restored',restored_at=management_now(),revision=removed['revision']+1)
                            self.put(db,'removed',bundle_id,removed)
                        else:
                            if row['availability'] not in {'available','incompatible','unvalidated'}: raise ValueError('Bundle is unavailable or already removed')
                            previous_revision=self.get(db,'removed',bundle_id,{}).get('revision',0)
                            removal={'state':'removed','revision':previous_revision+1,'identity':{'run_id':self.run_id,'bundle_id':bundle_id},'fingerprint':row['fingerprint'],
                                     'record':record,'reason':'automatic_retention' if operation=='automatic' else 'retention' if operation=='apply' else 'manual',
                                     'removed_at':management_now(),'prior_roles':row['roles'],'operation_id':key,'bytes_retained':True}
                            self.put(db,'removed',bundle_id,removal)
                    outcomes.append({'bundle_id':bundle_id,'status':'confirmed','revision':self.get(db,'removed',bundle_id,{}).get('revision',0)})
                except (ValueError,OSError,KeyError) as error: outcomes.append({'bundle_id':bundle_id,'status':'blocked','reason':str(error)})
            result={'operation_id':key,'outcomes':outcomes,'status':'partial' if any(o['status']!='confirmed' for o in outcomes) else 'confirmed','reclaimed_bytes':0}
            self.put(db,'operation',key,{'request_hash':request_hash,'result':result,'at':management_now()})
            return result


def management_queue(root, module, request):
    """Enqueue one durable generation and start a GPU-free detached/coalesced worker."""
    import subprocess
    import sys
    store=CheckpointStore(root,request['run_id'])
    with store.database() as db:
        if not store.get(db,'policy','effective',management_policy())['automatic']: return
        store.put(db,'automatic','pending',{'generation':uuid.uuid4().hex,'status':'pending','at':management_now(),'request':request})
    try:
        subprocess.Popen([sys.executable,'-m',module,'--instra-checkpoint-worker',store.run_id],cwd=root,
                         stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    except OSError:pass  # Preserve the pending record for explicit reconciliation.


def management_worker(root,run_id,compatible):
    store=CheckpointStore(root,run_id)
    with store.safe(store.home/(store.run_id+'.automatic.lock')).open('a') as worker_lock:
        try:fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for _ in range(4):
            with store.database() as db:job=store.get(db,'automatic','pending')
            if not job or job.get('status')!='pending':return
            request={**job['request'],'protocol':1,'operation':'management','management_operation':'automatic',
                     'active':True,'operation_id':'auto:'+job['generation']}
            result=store.rpc(request,lambda descriptor:compatible(request,descriptor))
            with store.database() as db:
                current=store.get(db,'automatic','pending')
                if current['generation']==job['generation']:
                    current['status']=result.get('status','pending')
                    store.put(db,'automatic','pending',current)
                    return
