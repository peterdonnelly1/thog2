# vvv THOG validate real public Runner commands and keep HTTP parse failures observable
import argparse
import ast
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

import instra_network
from thog_grid_runner import command_for, expand, COMMON
from run_thog2_lifecycle import build_parser
from run_thog2_owt_core import config_from_arguments, geometry_plan_from_arguments


def width_parameters(preset='width'):
    return {'--geometry-preset':preset, '--optimizer':'adamw', '--n-layer':4,
        '--n-embd':16, '--n-head':4, '--block-size':8, '--batch-size':1,
        '--max-iters':20, '--warmup-iters':0, '--learning-rate':0.0009,
        '--min-lr':0.00009, '--gradient-accumulation-steps':1,
        '--checkpoint-segment-size':2, 'WIDTH.order':6,
        'WIDTH.compressor':'dct', 'DEPTH.compressor':'chebyshev',
        'DEPTH.compressor_version':'auto'}


@pytest.mark.parametrize('preset', ['width', 'width-type-I'])
@pytest.mark.parametrize('orders,depth_orders,count', [([6],None,1),([4,6,8],None,3),([4,6,8],[1,2],6)])
def test_public_optimizer_commands_select_width_without_implicit_depth(preset,orders,depth_orders,count):
    parameters=width_parameters(preset)
    parameters['WIDTH.order']=orders
    parameters['--select-width']=False
    if depth_orders is not None:
        parameters['DEPTH.order']=depth_orders
        parameters['--select-depth']=True
    runs=expand({'label':'Width optimizer grid', 'parameters':parameters})
    assert len(runs)==count
    for run in runs:
        command=command_for({**run,'grid_tag':'G-TEST'}, {})
        assert command.count('--select-width')==1
        arguments=build_parser().parse_args(command[3:])
        config=config_from_arguments(arguments,geometry_plan=geometry_plan_from_arguments(arguments))
        assert arguments.optimizer=='adamw'
        assert config.geometry_preset=='width-type-I'
        assert config.width_enabled
        assert config.width_depth_enabled==(depth_orders is not None)
        assert ('--select-depth' in command)==(depth_orders is not None)
        if depth_orders is None:
            assert not any(value.startswith('DEPTH.') for value in command)


def test_mixed_presets_keep_automatic_width_flag_on_width_trials():
    parameters=width_parameters(['dense','width','depth'])
    parameters['DEPTH.order']=2
    runs=expand({'label':'Mixed geometry', 'parameters':parameters})
    assert len(runs)==3
    for run in runs:
        command=command_for({**run,'grid_tag':'G-TEST'}, {})
        arguments=build_parser().parse_args(command[3:])
        config=config_from_arguments(arguments,geometry_plan=geometry_plan_from_arguments(arguments))
        width=run['parameters'].get('--geometry-preset')=='width-type-I'
        assert config.width_enabled==width
        assert ('--select-width' in command)==width
        assert not config.width_depth_enabled


@pytest.mark.parametrize('preset', ['width', 'width-type-I'])
@pytest.mark.parametrize('explicit_depth', [None, False])
def test_width_preset_ignores_inherited_depth_order_in_preview_and_exports(preset, explicit_depth):
    parameters=width_parameters(preset)
    parameters['WIDTH.order']=[4,6]
    parameters['DEPTH.order']=[1,12,24]
    parameters['DEPTH.compressor_version']='not-a-width-setting'
    if explicit_depth is not None:
        parameters['--select-depth']=explicit_depth
    runs=expand({'label':'Inherited DEPTH defaults','parameters':parameters})
    assert len(runs)==2
    for run in runs:
        assert not any(key.startswith('DEPTH.') for key in run['parameters'])
        command=command_for({**run,'grid_tag':'G-TEST'}, {})
        assert '--select-depth' not in command
        arguments=build_parser().parse_args(command[3:])
        config=config_from_arguments(arguments,geometry_plan=geometry_plan_from_arguments(arguments))
        assert not config.width_depth_enabled
        assert config.compact_identity()['depth'] is None
        assert '_DEPTH_' not in config.compact_artifact_fragment()
    raw_command=command_for({'parameters':{**parameters,'WIDTH.order':4,'DEPTH.order':12},
                            'grid_tag':'G-TEST','run_id':'raw-width','profiler':'none'}, {})
    assert '--select-depth' not in raw_command
    assert not any(value.startswith('DEPTH.') for value in raw_command)


@pytest.mark.parametrize('order', [512, 1024])
def test_chebyshev_high_orders_are_accepted_and_match_dct(order):
    import torch
    from sheet.width import build_width_basis
    analysis, metadata = build_width_basis(1024,order,'chebyshev')
    reference, _ = build_width_basis(1024,order,'dct')
    signs = torch.where(torch.arange(order) % 2 == 0, 1., -1.).to(torch.float64)
    torch.testing.assert_close(analysis,reference * signs[:,None],atol=1e-11,rtol=0)
    assert metadata['raw_numerical_rank']==order


@pytest.mark.parametrize('enabled',[True,False])
def test_catalogue_boolean_optional_compressor_control_reaches_public_parser(enabled):
    recipe={'label':'Boolean option','parameters':{'--hyperblock':True,'--direct-factorised-hyperblock-mlp':enabled}}
    run=expand(recipe)[0]
    arguments=build_parser().parse_args(command_for({**run,'grid_tag':'G-TEST'}, {})[3:])
    assert arguments.direct_factorised_hyperblock_mlp is enabled


def test_captured_1024_width_recipe_resolves_both_requested_dct_orders_without_depth():
    parameters=width_parameters()
    parameters.update({'--n-embd':1024,'--n-head':16,'--n-layer':16,'--block-size':1024,
        '--batch-size':16,'--gradient-accumulation-steps':6,'--checkpoint-segment-size':4,
        'DEPTH.order':12,'WIDTH.order':[512,1024]})
    runs=expand({'label':'Captured width-only recipe','parameters':parameters})
    assert len(runs)==2
    for run in runs:
        arguments=build_parser().parse_args(command_for({**run,'grid_tag':'G-TEST'}, {})[3:])
        config=config_from_arguments(arguments,geometry_plan=geometry_plan_from_arguments(arguments))
        assert not config.width_depth_enabled
        assert config.compact_artifact_fragment() in ('WIDTH_dct_r512','WIDTH_dct_r1024')


def test_width_startup_report_is_compact_with_full_explain_metadata():
    from sheet.geometry_registry import format_geometry_plan
    arguments=build_parser().parse_args(['--geometry-preset','width','--select-width',
        '--n-embd','1024','--n-head','16','--option','WIDTH.order=512','--option','WIDTH.compressor=dct'])
    plan=geometry_plan_from_arguments(arguments)
    report=format_geometry_plan(plan)
    assert 'D=1024 / r=512' in report
    assert 'depth compression:' in report and 'disabled' in report
    assert len(report.splitlines())<30
    assert 'mode_order' not in report
    assert 'mode_order' in format_geometry_plan(plan,detailed=True)


def test_width_aliases_share_identity_and_do_not_duplicate_grid_trials():
    identities=[]
    for preset in ('width','width-type-I'):
        arguments=build_parser().parse_args(['--geometry-preset',preset,'--select-width',
            '--option','WIDTH.order=6','--option','WIDTH.compressor=dct',
            '--n-embd','16','--n-head','4','--n-layer','4','--max-iters','20'])
        config=config_from_arguments(arguments,geometry_plan=geometry_plan_from_arguments(arguments))
        identities.append(config.compact_identity())
    assert identities[0]==identities[1]
    assert len(expand({'label':'Aliases','parameters':width_parameters(['width','width-type-I'])}))==1
    assert 'WIDTH.order' in COMMON


def test_runner_parser_errors_are_values_with_readable_diagnostics(monkeypatch):
    import run_thog2_lifecycle
    monkeypatch.setattr(run_thog2_lifecycle,'build_parser',lambda:argparse.ArgumentParser())
    with pytest.raises(ValueError,match='Invalid WIDTH Recipe command: unrecognized arguments:'):
        expand({'label':'Rejected command','parameters':width_parameters()})


def test_runner_system_exit_returns_http_error_and_server_remains_usable():
    source=(Path(__file__).resolve().parents[1]/'run_thog2_local_dashboard.py').read_text()
    function=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='_runner_do_post')
    def preview(recipe):
        if recipe.get('fail'):
            raise SystemExit(2)
        return {'total_runs':1}
    namespace={'json':json,'HTTPStatus':HTTPStatus,'_instra_network':instra_network,
        '_runner_service':SimpleNamespace(preview=preview)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'runner_http', 'exec'),namespace)
    class Handler(BaseHTTPRequestHandler):
        do_POST=namespace['_runner_do_post']
        def _send_json(self,payload,status=HTTPStatus.OK):
            body=json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self,*args):
            pass
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        url=f'http://127.0.0.1:{server.server_port}/api/runner/action'
        try:
            request=Request(url,data=json.dumps({'action':'preview','recipe':{'fail':True}}).encode(),headers={'Content-Type':'application/json'})
            with pytest.raises(HTTPError) as rejected:
                urlopen(request,timeout=3)
            assert rejected.value.code==400
            assert 'parser exit 2' in json.load(rejected.value)['error']
            request=Request(url,data=json.dumps({'action':'preview','recipe':{}}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=3) as response:
                assert response.status==200
                assert json.load(response)['total_runs']==1
        finally:
            server.shutdown()
            worker.join(timeout=3)
# ^^^ THOG
