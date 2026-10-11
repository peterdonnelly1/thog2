# vvv THOG microstep cursors, real export contracts and visible profiling defaults
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_thog2_local_dashboard import RunDashboardState, _processing_capture_snapshot
from run_thog2_lifecycle import build_parser
from run_thog2_owt_core import config_from_arguments
from sheet.local_chart_store import LocalChartReader, LocalChartStore, LocalPrematLiveWriter
from sheet.processing_ncu_compatibility import normalize_semantic_ncu_exports, select_representative_kernel_resources, write_outputs
from sheet.premat_processing import rewrite_processing_cli_for_core
from sheet import wandb_telemetry
from thog_grid_runner import CATALOGUE, command_for, expand, profiling_parameters


def snapshot(update, micro, sequence=1):
    return {"optimizer_update":update,"micro_step":micro,"pass_sequence":sequence,"pass_complete":True,"events":[{"event":"pass_end"}]}


def test_each_microstep_is_immutable_and_cursor_can_advance_within_one_step(tmp_path):
    path=tmp_path/'charts.sqlite3'
    store=LocalChartStore(path,run_name='microsteps',config={})
    writer=LocalPrematLiveWriter(path)
    try:
        writer.append(11,snapshot(11,0,1))
        writer.append(11,snapshot(11,1,2))
        writer.append(11,snapshot(11,2,3))
        writer.append(11,snapshot(11,1,99))
        reader=LocalChartReader(path)
        assert [s['pass_sequence'] for s in reader.premat_snapshots()]==[1,2,3]
        assert [s['micro_step'] for s in reader.premat_snapshots(after_update=11,after_micro_step=0)]==[1,2]
        state=RunDashboardState(path)
        delta=state.premat(after_update=11,after_micro_step=0)
        assert [s['micro_step'] for s in delta['snapshots']]==[1,2]
        assert delta['latest_micro_step']==2 and not delta['unchanged']
        assert state.premat(after_update=11,after_micro_step=2)['unchanged']
        # Old clients using only an optimiser-update cursor stay compatible.
        assert state.premat(after_update=11)['unchanged']
    finally:
        writer.close();store.close()


def test_legacy_snapshot_migrates_without_loss_and_retention_is_bounded(tmp_path):
    import zlib
    path=tmp_path/'charts.sqlite3'
    store=LocalChartStore(path,run_name='legacy',config={})
    legacy=snapshot(1,0);legacy.pop('micro_step')
    with store.connection:
        store.connection.execute('DROP TABLE premat_snapshots')
        store.connection.execute('CREATE TABLE premat_snapshots (optimizer_update INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        store.connection.execute('INSERT INTO premat_snapshots VALUES (?,?)',(1,zlib.compress(json.dumps(legacy).encode())))
    assert LocalChartReader(path).latest_premat_snapshot()['micro_step']==0
    writer=LocalPrematLiveWriter(path)
    try:
        writer.append(1,snapshot(1,1,2))
        assert len(LocalChartReader(path).premat_snapshots())==2
        for micro in range(130):writer.append(2,snapshot(2,micro,micro+3))
        retained=LocalChartReader(path).premat_snapshots()
        assert len(retained)==128
        assert [s['micro_step'] for s in retained]==list(range(2,130))
    finally:
        writer.close();store.close()


def test_live_sink_keeps_sibling_microsteps_when_training_publishes_a_burst(tmp_path):
    path=tmp_path/'charts.sqlite3'
    store=LocalChartStore(path,run_name='burst',config={})
    sink=wandb_telemetry._PrematLiveSink(path,retain_detailed_history=True)
    for micro in range(32):sink.publish(11,snapshot(11,micro,micro+1))
    sink.close();store.close()
    assert [s['micro_step'] for s in LocalChartReader(path).premat_snapshots()]==list(range(32))


@pytest.mark.parametrize('mode,expected',[('all',[True,True,True]),('first',[True,False,False])])
def test_telemetry_sampling_retains_all_or_first_original_forward_microsteps(monkeypatch,tmp_path,mode,expected):
    hooks={}
    runtime=SimpleNamespace(cpu_phase='original_forward')
    model=SimpleNamespace(_premat_runtime=runtime,set_premat_live_reporter=lambda publish,capture:hooks.update(publish=publish,capture=capture))
    config=SimpleNamespace(premat_instra='enabled',premat_instra_microsteps=mode,log_interval=10,max_updates=20)
    trainer=SimpleNamespace(config=config,raw_model=model,state=SimpleNamespace(completed_updates=0),distributed=SimpleNamespace(is_primary=True),_print_progress=lambda *a,**k:None)
    telemetry=SimpleNamespace(config={},name='fixture')
    received=[]
    monkeypatch.setattr(wandb_telemetry,'ensure_local_chart_store',lambda t:SimpleNamespace(path=tmp_path/'charts.sqlite3'))
    monkeypatch.setattr(wandb_telemetry,'_PrematLiveSink',lambda *a,**k:SimpleNamespace(publish=lambda update,payload:received.append((update,payload))))
    # Exercise the sampling function independently of unrelated installed timing wrappers.
    tree=ast.parse(Path(wandb_telemetry.__file__).read_text())
    definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='attach_telemetry')
    namespace=dict(vars(wandb_telemetry))
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(wandb_telemetry.__file__),'exec'),namespace)
    namespace['attach_telemetry'](trainer,telemetry)
    assert [hooks['capture'](seq) for seq in (1,2,3)]==expected
    runtime.cpu_phase='checkpoint_recompute';assert not hooks['capture'](4)
    runtime.cpu_phase='original_forward'
    trainer.state.completed_updates=1;assert not hooks['capture'](5)
    trainer.state.completed_updates=9;assert hooks['capture'](6)
    trainer.state.completed_updates=10
    hooks['publish'](snapshot(10,2,6))
    assert received[0][0]==10 and received[0][1]['micro_step']==2


def write_semantic(path,include_premat=False):
    headers=['ID','Process ID','Kernel Name','Context','Stream','Block Size','CC','NVTX Range','gpu__time_duration.sum','launch__registers_per_thread','launch__registers_per_thread_allocated','launch__shared_mem_per_block_allocated']
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(headers)
        for i,role in enumerate(['MAIN','PREMAT'] if include_premat else ['MAIN']):
            operation='consume' if role=='MAIN' else 'materialize'
            writer.writerow([i,111,role.lower()+'_gemm',1,7+i,'(128, 1, 1)','8.9',f'THOG2_PROCESSING|owner={role}|operation={operation}|family=O|layer=0',1000,32,4096,0])


def test_cpu_ncu_accepts_main_only_and_writes_explicit_na_without_fabricated_gpu_pairs(tmp_path):
    raw=tmp_path/'raw.csv';write_semantic(raw)
    rows=normalize_semantic_ncu_exports(raw,raw,materialisation_device='cpu_and_gpu')
    assert len(rows)==1 and rows[0]['role']=='MAIN'
    assert rows[0]['registers_per_thread']==32 and rows[0]['registers_per_block']==4096
    paths=write_outputs(select_representative_kernel_resources(rows),tmp_path/'processing',materialisation_device='cpu_and_gpu')
    compatibility=json.loads(paths['json'].read_text())
    assert compatibility['reason']=='N/A: CPU materialisation'
    assert not compatibility['available'] and compatibility['rows']==[]
    assert 'main_gemm' in paths['kernel_resources'].read_text()
    # The identical incomplete input must still fail in GPU materialisation mode.
    with pytest.raises(ValueError,match='both MAIN'):normalize_semantic_ncu_exports(raw,raw)


def test_gpu_ncu_still_exports_premat_compatibility(tmp_path):
    raw=tmp_path/'raw.csv';write_semantic(raw,include_premat=True)
    rows=normalize_semantic_ncu_exports(raw,raw)
    paths=write_outputs(select_representative_kernel_resources(rows),tmp_path/'processing')
    compatibility=json.loads(paths['json'].read_text())
    assert compatibility['available'] and compatibility['rows']


@pytest.mark.parametrize('mode', ['cpu_and_gpu', 'gpu'])
def test_active_ncu_launcher_uses_resolved_child_mode_and_finishes_real_exports(monkeypatch,tmp_path,mode):
    from sheet import premat_processing as processing
    from sheet import processing_ncu_2024_instra_patch as launcher
    monkeypatch.setenv(processing._PROCESSING_CHILD_ENV,'0')
    monkeypatch.setattr(launcher.tempfile,'mkdtemp',lambda **kwargs:str(tmp_path))
    monkeypatch.setattr(processing,'_find_ncu',lambda:'mock-ncu')
    run_dir=tmp_path/'run'
    (tmp_path/'handoff.json').write_text(json.dumps({'run_directory':str(run_dir),'config':{'premat_materialisation_device':mode}}))
    (tmp_path/'processing_ncu_trace.ncu-rep').write_text('captured report')
    raw=tmp_path/'raw.csv';write_semantic(raw,include_premat=mode=='gpu')
    exports=[]
    def export(*args,**kwargs):
        exports.append(kwargs['nvtx_rename'])
        return raw
    monkeypatch.setattr(processing,'_export_ncu_raw_csv',export)
    launches=[]
    monkeypatch.setattr(launcher.subprocess,'run',lambda command,**kwargs:launches.append(kwargs) or SimpleNamespace(returncode=0))
    # The CPU provider may come from a resumed checkpoint rather than CLI flags.
    argv=['--premat_processing_logging','enabled','--premat_processing_profiler','ncu','--ncu-probe-layer','0','--premat','enabled','--max-iters','20']
    assert processing.maybe_reexec_under_nsys(argv,entrypoint=Path('run_thog2_owt.py'))==0
    assert launches[0]['env']['THOG2_PREMAT_PROCESSING_PROFILER']=='ncu'
    assert exports==[False,True]
    artifacts=run_dir/'processing'
    compatibility=json.loads((artifacts/'processing_premat_compatibility.json').read_text())
    assert bool(compatibility['rows'])==(mode=='gpu')
    if mode=='cpu_and_gpu':assert compatibility['reason']=='N/A: CPU materialisation'
    assert (artifacts/'processing_ncu_trace.ncu-rep').read_text()=='captured report'
    assert (artifacts/'processing_ncu_raw.csv').is_file()
    assert (artifacts/'processing_ncu_semantic.csv').is_file()
    assert 'main_gemm' in (artifacts/'processing_ncu_kernel_resources.csv').read_text()


def recipe(**parameters):
    return {'label':'CPU pair','profilers':['nsys','ncu'],'parameters':{'--max-iters':20,'--warmup-iters':1,'--n-layer':[4],'DEPTH.order':[2],'--premat':['enabled'],'--premat_materialisation_device':['cpu_and_gpu'],**parameters}}


def test_runner_pair_defaults_are_visible_in_resolved_parameters_and_cli():
    runs=expand(recipe(),stable_preview=True)
    assert [run['profiler'] for run in runs]==['nsys','ncu']
    assert runs[0]['pairing_id']==runs[1]['pairing_id']
    for run in runs:
        values=run['parameters']
        assert values['--premat_processing_profiler']==run['profiler']
        assert values['--premat_processing_logging']=='enabled'
        assert values['--premat_processing_logging_capture_update']==10
        assert values['--premat_instra__full_step_timing_capture_and_chart']=='enable'
        assert values['--premat_instra__full_step_timing_capture_and_chart_capture']=='step 10'
        assert values['--premat_instra_microsteps']=='all'
        argv=command_for({**run,'grid_tag':'G-00000'}, {})[3:]
        parsed=build_parser().parse_args(rewrite_processing_cli_for_core(argv))
        assert parsed.premat_instra__full_step_timing_capture_and_chart_capture==10
        assert config_from_arguments(parsed).premat_instra_microsteps=='all'
        assert argv.count('--premat_processing_profiler')==1


def test_profiler_defaults_respect_explicit_settings_and_short_runs():
    settings=profiling_parameters({'--max-iters':3},'nsys')
    assert settings['--premat_processing_logging_capture_update']==3
    assert settings['--premat_instra__full_step_timing_capture_and_chart_capture']=='step 3'
    explicit={'--max-iters':20,'--premat_processing_logging_capture_update':2,'--premat_instra__full_step_timing_capture_and_chart':'disable','--premat_instra_microsteps':'first','--premat':'enabled'}
    result=profiling_parameters(explicit,'ncu')
    assert result['--premat_instra_microsteps']=='first'
    assert result['--premat_instra__full_step_timing_capture_and_chart']=='disable'
    assert '--premat_instra__full_step_timing_capture_and_chart_capture' not in result
    with pytest.raises(ValueError,match='matching'):
        profiling_parameters({'--max-iters':20,'--premat_materialisation_device':'cpu_and_gpu','--premat_instra__full_step_timing_capture_and_chart_capture':'step 2'},'nsys')


def test_cli_optional_first_microstep_reaches_training_config():
    parser=build_parser()
    for mode in ('all','first'):
        args=parser.parse_args(['--model-type','sheet','--max-iters','20','--premat_instra_microsteps',mode])
        config=config_from_arguments(args)
        assert config.premat_instra_microsteps==mode
        assert config.to_training_config(vocab_size=32,world_size=1,out_dir=Path('out-test')).premat_instra_microsteps==mode
    assert parser.parse_args([]).premat_instra_microsteps=='all'


def test_processing_lifecycle_matches_nsight_capture_microstep():
    rows=[snapshot(11,m,m+1) for m in range(3)]
    state=SimpleNamespace(reader=SimpleNamespace(premat_snapshots=lambda:rows))
    # Nsight capture metadata uses one-based microsteps, storage uses zero-based.
    assert _processing_capture_snapshot(state,{'metadata':{'capture':{'optimizer_update':11,'micro_step':1}}})['micro_step']==0
    assert _processing_capture_snapshot(state,{'metadata':{'capture':{'optimizer_update':11,'micro_step':3}}})['micro_step']==2
    assert len(_processing_capture_snapshot(state,{'metadata':{'capture':{'optimizer_update':11,'full_update':True}}})['events'])==3


def test_runner_catalogue_exposes_nsight_fields_and_moves_pure_cpu_controls():
    for key in ('--premat_processing_profiler','--premat_processing_logging_capture_update','--premat_instra__full_step_timing_capture_and_chart','--premat_instra__full_step_timing_capture_and_chart_capture'):
        assert CATALOGUE[key]['category']=='NSIGHT'
        assert not CATALOGUE[key].get('ui_hidden')
        assert CATALOGUE[key]['kind']=='fixed'
    assert CATALOGUE['--premat_instra_microsteps']['category']=='Premat'
    for key,spec in CATALOGUE.items():
        if key.startswith('--premat_cpu_'):assert spec['category']=='CPU Premat'
    for key in ('--premat','--premat_target_matrix','--premat_instra','--premat_processing_profiler'):
        assert 'CPU Premat' in CATALOGUE[key]['extra_categories']
# ^^^ THOG

# vvv THOG sampling is operational state on resumed and forked configurations

def test_microstep_selection_survives_resume_and_can_be_overridden(tmp_path):
    from run_thog2_lifecycle import _training_config_for_lifecycle
    from sheet.run_config import OwtRunConfig
    from sheet.training_config import TrainingConfig
    checkpoint=TrainingConfig(device='cpu',dtype='float32',premat_instra_microsteps='first')
    run=OwtRunConfig(model_type='dense',max_iters=50,warmup_iters=1,device='cpu',dtype='float32',premat_instra_microsteps='first')
    assert _training_config_for_lifecycle(checkpoint,run,target_updates=50,out_dir=tmp_path).premat_instra_microsteps=='first'
    from dataclasses import replace
    run=replace(run,premat_instra_microsteps='all')
    assert _training_config_for_lifecycle(checkpoint,run,target_updates=50,out_dir=tmp_path).premat_instra_microsteps=='all'
# ^^^ THOG
