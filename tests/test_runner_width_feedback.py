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
