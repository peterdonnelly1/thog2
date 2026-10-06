# vvv THOG workflow, deterministic probes, checkpoint identity, and disabled-path regression acceptance
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from run_thog2_owt_core import build_parser, config_from_arguments, geometry_plan_from_arguments
from sheet.local_chart_store import LocalChartStore, LocalChartReader
from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig
from sheet.width import build_width_basis
from sheet.width_dashboard import width_figure
from sheet.width_instrumentation import capture_due


VALID = ['--geometry-preset', 'width-type-I', '--select-width', '--n-embd', '16', '--n-head', '4', '--n-layer', '4', '--option', 'WIDTH.order=6', '--option', 'WIDTH.compressor=dct']


def resolve(argv):
    args = build_parser().parse_args(argv)
    return config_from_arguments(args, geometry_plan=geometry_plan_from_arguments(args))


@pytest.mark.parametrize('suffix', [[], ['--select-depth', '--option', 'DEPTH.order=2', '--option', 'DEPTH.compressor=haar'],
    ['--select-depth', '--option', 'DEPTH.order=3', '--depth-compress-layer-norm-and-bias']])
def test_cli_width_selection_and_independent_depth(suffix):
    config = resolve(VALID + suffix)
    assert config.width_enabled
    assert config.width_order == 6
    assert config.width_depth_enabled == bool(suffix)
    identity = config.compact_identity()
    assert identity['reference_width'] == 16 and identity['residual_width'] == 6
    assert identity['attention_width'] == 16 and identity['mlp_hidden_width'] == 64
    training = config.to_training_config(vocab_size=23, world_size=1, out_dir=Path('out'))
    assert training.width_order == 6 and training.width_compressor == 'dct'
    assert training.model_arguments()['n_embd'] == 16
    assert 'WIDTH_dct_r6' in config.artifact_name


@pytest.mark.parametrize('extras', [
    ['--option', 'WIDTH.order=3'], ['--option', 'WIDTH.unknown=3'], ['--option', 'WIDTH.group_size=3'],
    ['--option', 'DEPTH.order=2'], ['--select-element', 'MLP_UP'], ['--plastic__enabled'], ['--hyperblock'],
    ['--basis-family', 'chebyshev'], ['--basis-version', 'auto'],
    ['--option', 'WIDTH.compressor_version=haar_balanced_binary_orthonormal_v1'],
    ['--premat', 'enabled'], ['--model-type', 'dense'],
    ['--select-depth', '--option', 'DEPTH.order=5'],
    ['--instrumentation__width_activation_curves__log_every_n_steps', '0'],
    ['--instrumentation__width_activation_curves__probe_orders', '2,2'],
    ['--instrumentation__width_activation_curves__probe_orders', '6'],
    ['--instrumentation__width_activation_curves__probe_orders', '0'],
    ['--instrumentation__width_activation_curves__history_length', '0'],
    ['--instrumentation__width_activation_curves__sample_tokens_per_layer', '0'],
    ['--instrumentation__width_activation_curves__feature_evaluation_points', '1'],
    ['--instrumentation__width_activation_curves__start_step', '5', '--instrumentation__width_activation_curves__end_step', '4'],
])
def test_configuration_conflicts_rejected_before_training(extras):
    with pytest.raises((ValueError, SystemExit)):
        resolve(VALID + extras)


@pytest.mark.parametrize('arguments', [
    ['--model-type', 'sheet', '--option', 'WIDTH.order=4'],
    ['--model-type', 'sheet', '--select-width', '--option', 'WIDTH.order=4'],
    ['--model-type', 'sheet', '--geometry-preset', 'width-type-I', '--option', 'WIDTH.order=4'],
    ['--model-type', 'dense', '--instrumentation__width_activation_curves__mode', 'basic'],
])
def test_missing_selection_and_inactive_width_options_rejected(arguments):
    with pytest.raises(ValueError): resolve(arguments)


@pytest.mark.parametrize('order', ['1', '0', '17', '5.5'])
def test_invalid_retained_width_rejected(order):
    arguments = VALID.copy();arguments[arguments.index('WIDTH.order=6')] = f'WIDTH.order={order}'
    with pytest.raises(ValueError): resolve(arguments)


def trainer_config(**changes):
    values = dict(model_type='thog2_sheet', n_embd=8, n_head=2, n_layer=2,
                  block_size=5, vocab_size=17, batch_size=2, max_updates=10,
                  gradient_accumulation_steps=2, checkpoint_segment_size=1,
                  width_enabled=True, width_order=3, width_compressor='dct', geometry_preset='width-type-I')
    values.update(changes)
    return TrainingConfig(**values)


def test_resume_restores_optimizer_probe_rng_and_training_trajectory(tmp_path):
    config = trainer_config(instrumentation__width_activation_curves__mode='probes',
        instrumentation__width_activation_curves__log_every_n_steps=1,
        instrumentation__width_activation_curves__probe_every_n_steps=2,
        instrumentation__width_activation_curves__history_length=2)
    tokens = torch.arange(140) % 17
    trainer = Stage6Trainer(config, tokens, tokens)
    trainer.train_one_update();trainer.train_one_update()
    checkpoint = trainer.save_checkpoint(tmp_path / 'width.pt')
    assert len(trainer._width_instrumentation.history) == 2
    resumed = Stage6Trainer.from_checkpoint(checkpoint, tokens, tokens, expected_config=config)
    assert resumed._width_instrumentation.sample_starts == trainer._width_instrumentation.sample_starts
    assert resumed._width_instrumentation.last_capture_step == 2
    trainer.train_one_update(); resumed.train_one_update()
    for name, parameter in trainer.raw_model.state_dict().items():
        torch.testing.assert_close(parameter, resumed.raw_model.state_dict()[name], atol=0, rtol=0)
    assert trainer.batch_source.trace[-1] == resumed.batch_source.trace[-1]
    trainer.close();resumed.close()


@pytest.mark.parametrize('change', [{'width_order': 4}, {'width_compressor': 'haar', 'width_compressor_version':'auto'},
                                   {'width_depth_enabled':True,'depth_order':1}, {'n_embd':10}])
def test_resume_rejects_incompatible_representation_before_loading(tmp_path, change):
    config = trainer_config()
    tokens = torch.arange(140) % 17
    trainer = Stage6Trainer(config,tokens,tokens)
    path = trainer.save_checkpoint(tmp_path/'width.pt')
    with pytest.raises(ValueError,match='incompatible width checkpoint'):
        Stage6Trainer.from_checkpoint(path,tokens,tokens,expected_config=replace(config,**change))
    trainer.close()


def test_checkpoint_rejects_corrupted_basis(tmp_path):
    config=trainer_config();tokens=torch.arange(100)%17
    trainer=Stage6Trainer(config,tokens,tokens);payload=trainer.checkpoint_payload()
    payload['model']['width_basis.analysis'] = payload['model']['width_basis.analysis'].clone()
    payload['model']['width_basis.analysis'][0,0]+=0.01
    path=tmp_path/'corrupt.pt';torch.save(payload,path)
    with pytest.raises(ValueError,match='fingerprint'): Stage6Trainer.from_checkpoint(path,tokens,tokens,expected_config=config)
    trainer.close()


def test_probes_preserve_training_rng_gradient_and_parameter_state():
    config=trainer_config(instrumentation__width_activation_curves__mode='probes')
    tokens=torch.arange(100)%17;trainer=Stage6Trainer(config,tokens,tokens)
    from sheet.width_instrumentation import width_instrumentation_for
    instrumentation=width_instrumentation_for(trainer)
    original_rng=torch.get_rng_state().clone()
    batches=copy.deepcopy(trainer.batch_source.state_dict())
    before={name:parameter.detach().clone() for name,parameter in trainer.raw_model.named_parameters()}
    snapshot=instrumentation._probes(1)
    assert trainer.raw_model.training
    assert torch.equal(torch.get_rng_state(),original_rng)
    assert trainer.raw_model._width_probe_order is None and trainer.raw_model._width_capture is None
    for name,parameter in trainer.raw_model.named_parameters():
        assert torch.equal(parameter,before[name]) and parameter.grad is None
    for name in ('train_generator_state','validation_generator_state'):
        if name in batches: assert torch.equal(batches[name],trainer.batch_source.state_dict()[name])
    assert snapshot['probe_token_count']==5
    assert snapshot['probes'][0]['retained_prefix']==1
    assert len(snapshot['probe_residual_sites'])==4
    assert len(snapshot['probe_normalization_sites'])==5
    assert {row['operation'] for row in snapshot['normalization_profile']}=={'width_norm_construct','width_norm_apply'}
    trainer.close()


def test_disabled_capture_has_no_hooks_buffers_or_instrumentation():
    config=trainer_config();tokens=torch.arange(100)%17;trainer=Stage6Trainer(config,tokens,tokens)
    trainer.train_one_update()
    assert '_width_instrumentation' not in vars(trainer)
    assert trainer.raw_model._width_capture is None
    assert all(not module._forward_hooks for module in trainer.raw_model.modules())
    assert 'width_instrumentation' not in trainer.checkpoint_payload()
    trainer.close()


def test_cadence_completed_updates_and_inclusive_window():
    assert [step for step in range(11) if capture_due(step,3,1,10)]==[1,4,7,10]
    assert [step for step in range(11) if capture_due(step,3,0,-1)]==[3,6,9]


def test_local_chart_bounds_old_database_and_width_figures(tmp_path):
    store=LocalChartStore(tmp_path/'charts.sqlite3',run_name='width',config={'width_enabled':True})
    assert LocalChartReader(store.path).width_snapshots()==()
    sample={'site':'embedding.residual','mode_energy':[1,2,3],'feature_indices':[0,2,4], 'feature_curves':[[2,3,4]]}
    for step in range(5):
        snapshot={'optimizer_update':step,'representation':{'reference_width':8,'residual_width':3},
                  'residual_sites':[sample],'probe_residual_sites':[sample],
                  'probes':[{'retained_prefix':1,'delta_loss':.2}]}
        store.append_width_snapshot(snapshot,history_length=2)
    records=LocalChartReader(store.path).width_snapshots()
    assert [row['optimizer_update'] for row in records]==[3,4]
    for chart in ('width_energy','width_probes','width_curves'):
        payload=width_figure(records,chart,step=3)
        assert payload['figure']['data'] and payload['selected_step']==3
    store.close()


def test_runner_scoped_width_and_depth_command():
    import thog_grid_runner as runner
    recipe={'label':'compact','parameters':{'--geometry-preset':'width-type-I','--select-width':True,
        'WIDTH.order':[3,4], 'WIDTH.compressor':'dct', 'DEPTH.order':[1], 'DEPTH.compressor':'haar',
        '--n-embd':[8],'--n-head':[2],'--n-layer':[2],'--warmup-iters':[0],'--max-iters':5}}
    runs=runner.expand(recipe)
    assert len(runs)==2
    run={**runs[0],'grid_tag':'G-00001'}
    command=runner.command_for(run, {})
    assert '--select-width' in command and 'WIDTH.order=3' in command and 'DEPTH.compressor=haar' in command
    config=resolve(command[3:]);assert config.width_depth_enabled and config.width_order==3


@pytest.mark.parametrize('orders,depth_orders,expected_count', [([4],None,1),([4,8,16],None,3),([4,8,16],[1,2],6)])
def test_runner_width_single_and_grid_without_implicit_depth(orders,depth_orders,expected_count):
    import thog_grid_runner as runner
    parameters = {'--geometry-preset':'width-type-I','--select-width':True,
        'WIDTH.order':orders,'WIDTH.compressor':'dct','--n-embd':[16],
        '--n-head':[4],'--n-layer':[4],'--warmup-iters':[0],'--max-iters':20,
        '--instrumentation__width_activation_curves__mode':'probes',
        '--instrumentation__width_activation_curves__end_step':-1}
    if depth_orders is not None:
        parameters.update({'DEPTH.order':depth_orders,'DEPTH.compressor':'chebyshev'})
    runs = runner.expand({'label':'width smoke grid','parameters':parameters})
    assert len(runs) == expected_count
    resolved_orders = set()
    for run in runs:
        command = runner.command_for({**run,'grid_tag':'G-00001'}, {})
        config = resolve(command[3:])
        assert config.width_enabled and config.width_depth_enabled == (depth_orders is not None)
        assert ('--select-depth' in command) == (depth_orders is not None)
        assert config.instrumentation__width_activation_curves__end_step == -1
        resolved_orders.add((config.width_order,config.o_depth if depth_orders is not None else None))
    assert len(resolved_orders) == expected_count


def test_baseline_checkpoint_dictionary_does_not_acquire_width_fields():
    config=TrainingConfig(model_type='dense')
    assert not any(name.startswith('width_') or 'width_activation' in name for name in config.persistent_dict())


def test_multiple_depth_modes_match_materialization_gradients():
    from sheet.model import SheetGPT, SheetGPTConfig
    model=SheetGPT(SheetGPTConfig(n_embd=8,n_head=2,n_layer=4,block_size=5,vocab_size=17,
        width_enabled=True,width_order=3,width_compressor='haar',width_depth_enabled=True,
        geometry_preset='width-type-I',depth_order=3,basis_family='dct',basis_version='dct_ii_orthonormal_v1')).double()
    bank=model.trajectory.coefficients['attention_input_weight']
    with torch.no_grad(): bank.normal_(std=.05)
    inputs=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
    direct=model.trajectory.linear(inputs,'attention_input_weight',2,'attention_input_bias')
    materialized=torch.einsum('oip,p->oi',bank,model.trajectory.depth_basis[2])
    reference=torch.nn.functional.linear(inputs,materialized,model.trajectory.vector('attention_input_bias',2))
    torch.testing.assert_close(direct,reference,atol=1e-9,rtol=0)
    for actual,expected in zip(torch.autograd.grad(direct.square().sum(),(inputs,bank),retain_graph=True),torch.autograd.grad(reference.square().sum(),(inputs,bank))):
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=0)


def test_complete_pilot_run_emits_width_diagnostics_without_materialization(tmp_path):
    config = trainer_config(max_updates=2, out_dir=str(tmp_path), width_depth_enabled=True,
                            depth_order=2, basis_family='haar', basis_version='haar_balanced_binary_orthonormal_v1')
    trainer = Stage6Trainer(config, torch.arange(140) % 17, torch.arange(140) % 17)
    result = trainer.run_pilot(run_id='width-final-report', protocol_sha256='fixture', dataset={'name':'fixture'},
                               result_path=tmp_path / 'result.json')
    assert result['status'] == 'completed'
    diagnostics = result['sheet_diagnostics']
    assert diagnostics['generated_weights'] == {}
    assert diagnostics['width']['width']['residual_width'] == 3
    assert all(row['depth_mode_axis'] == 2 for row in diagnostics['coefficient_utilization'].values())
    trainer.close()


@pytest.mark.parametrize('suffix', [[], ['--select-depth', '--option', 'DEPTH.order=1', '--option', 'DEPTH.compressor=haar']])
def test_public_shell_launch_preserves_width_options_and_capture_namespace(suffix):
    command = ['bash', 'train_OWT.sh', '-p', 'width-type-I', '--select-width', '--option', 'WIDTH.order=3',
               '--option', 'WIDTH.compressor=dct', '-L', '2', '-D', '8', '-H', '2', '-C', '8', '-P', '2',
               '-n', '5', '-w', '0', '-b', '1', '-A', '1', '-x', 'true', '-I', 'none',
               '--instrumentation__width_activation_curves__mode=basic', *suffix, '--', '--print-resolved-json']
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload, _ = json.JSONDecoder().raw_decode(result.stdout[result.stdout.index('{'):])
    config = payload['canonical_config']
    assert config['width_order'] == 3 and config['width_depth_enabled'] == bool(suffix)
    assert config['instrumentation__width_activation_curves__mode'] == 'basic'


@pytest.mark.parametrize('flag,value', [('-B','chebyshev'),('-v','auto')])
def test_shell_rejects_explicit_global_basis_defaults_in_width_mode(flag,value):
    result = subprocess.run(['bash','train_OWT_core.sh','-p','width-type-I','--select-width',
        '--option','WIDTH.order=3','-L','2','-D','8','-H','2','-C','8','-P','2','-n','5','-w','0',
        '-x','true',flag,value], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'scoped WIDTH/DEPTH' in result.stderr


def test_inline_version_and_explicit_auto_resolve_to_the_same_identity():
    arguments = VALID.copy()
    arguments[arguments.index('WIDTH.compressor=dct')] = 'WIDTH.compressor=dct@dct_ii_orthonormal_v1'
    assert resolve(arguments + ['--option','WIDTH.compressor_version=auto']).compact_identity() == resolve(VALID).compact_identity()


@pytest.mark.parametrize('joint_depth', [False, True])
def test_public_training_cli_fresh_resume_and_recovered_checkpoint(tmp_path, monkeypatch, joint_depth):
    import numpy as np
    import pickle
    from run_thog2_owt import main
    from run_thog2_lifecycle import build_parser as lifecycle_parser, explicit_destinations, prepare_context
    from sheet.checkpoints import load_payload

    data = tmp_path / 'data'; data.mkdir()
    for name in ('train', 'val'):
        (np.arange(180, dtype=np.uint16) % 17).tofile(data / (name + '.bin'))
    with (data / 'meta.pkl').open('wb') as handle:
        pickle.dump({'vocab_size':17}, handle)
    monkeypatch.setenv('THOG2_CURVE_ROOT', str(tmp_path / 'curves'))
    common = ['--data-dir',str(data),'--dataset','fixture','--checkpoint-root',str(tmp_path/'checkpoints'),
              '--log-root',str(tmp_path/'logs'),'--result-root',str(tmp_path/'results'),
              '--wandb-root',str(tmp_path/'wandb'),'--instrumentation','none','--device','cpu','--dtype','float32',
              '--max-iters','2','--warmup-iters','0','--eval-interval','1','--eval-iters','1',
              '--n-layer','2','--n-head','2','--n-embd','8','--block-size','5','--batch-size','1',
              '--gradient-accumulation-steps','1','--run-start-label','261006-0001',
              '--model-type','sheet','--geometry-preset','width-type-I','--select-width',
              '--option','WIDTH.order=3','--option','WIDTH.compressor=dct',
              '--instrumentation__width_activation_curves__mode','probes',
              '--instrumentation__width_activation_curves__log_every_n_steps','1',
              '--instrumentation__width_activation_curves__probe_every_n_steps','1']
    if joint_depth:
        common += ['--select-depth','--option','DEPTH.order=1','--option','DEPTH.compressor=haar']
    assert main(common) == 0
    checkpoint = next((tmp_path / 'checkpoints').glob('*/ckpt.pt'))
    before = load_payload(checkpoint)
    argv = ['--resume',str(checkpoint),'--max-iters','3','--instrumentation','none']
    assert main(argv) == 0
    after = load_payload(checkpoint)
    assert after['completed_updates'] == 3
    assert after['width_instrumentation']['sample_starts'] == before['width_instrumentation']['sample_starts']
    assert after['width_instrumentation']['last_capture_step'] == 3
    assert after['trainer_config']['width_depth_enabled'] == joint_depth
    assert len(after['optimizer']['state']) == len(before['optimizer']['state'])

    # A valid direct Stage6 checkpoint can be recovered without run-lifecycle metadata.
    after.pop('lifecycle')
    torch.save(after, checkpoint)
    recovery = ['--resume',str(checkpoint),'--max-iters','4','--instrumentation','none','--data-dir',str(data)]
    parser = lifecycle_parser(); arguments = parser.parse_args(recovery)
    context = prepare_context(arguments, explicit_destinations(parser, recovery))
    assert context['config'].width_enabled and context['config'].width_order == 3
    assert context['config'].width_depth_enabled == joint_depth
    assert context['config'].instrumentation__width_activation_curves__mode == 'probes'


def test_lifecycle_resume_rejects_width_geometry_reselection():
    from run_thog2_lifecycle import _assert_material_arguments, build_parser as lifecycle_parser
    parser = lifecycle_parser(); arguments = parser.parse_args(['--select-width'])
    with pytest.raises(ValueError, match='checkpoint geometry is authoritative'):
        _assert_material_arguments(arguments, {'select_width'}, resolve(VALID), 'resume')


@pytest.mark.parametrize('constant_first', [False, True])
def test_tiled_projection_discarded_energy_matches_explicit_affine_reference(constant_first):
    from sheet.width import WidthBasis, projected_layer_norm
    from sheet.width_instrumentation import WidthSampleCapture
    from torch.nn import functional as functional
    generator = torch.Generator().manual_seed(840)
    D, r = 600, 7
    if constant_first:
        analysis, metadata = build_width_basis(D, r, 'dct')
    else:
        analysis = torch.linalg.qr(torch.randn(D, r, generator=generator, dtype=torch.float64)).Q.T
        metadata = {'constant_first_mode':False}
    basis = WidthBasis(analysis, metadata)
    model = SimpleNamespace(width_basis=basis, config=SimpleNamespace(n_embd=D))
    options = {'sample_tokens_per_layer':2,'feature_evaluation_points':17}
    capture = WidthSampleCapture(model, options, probes=True)
    coefficients = torch.randn(3,4,r,generator=generator,dtype=torch.float64)
    gain = torch.randn(D,generator=generator,dtype=torch.float64)
    bias = torch.randn(D,generator=generator,dtype=torch.float64)
    output = projected_layer_norm(coefficients,basis,gain,bias)
    capture.normalization('ln_f',coefficients,gain,bias,output)
    capture.residual('ln_f.head_input',output)
    sample = coefficients.reshape(-1,r)[:2]
    affine = functional.layer_norm(sample @ analysis,(D,),gain,bias,1e-5)
    projected = affine @ analysis.T
    discarded = affine.square().sum(-1) - projected.square().sum(-1)
    record = capture.norms['ln_f']
    assert record['post_affine_discarded_energy'] == pytest.approx(float(discarded.mean()),abs=1e-9)
    assert record['post_affine_discarded_fraction'] == pytest.approx(float(discarded.sum()/affine.square().sum()),abs=1e-12)
    assert record['post_affine_discarded_energy'] > 1.0
    assert record['diagnostic_feature_tile'] == 256
    curves = capture.residuals['ln_f.head_input']
    assert curves['token_vectors_in_operation'] == 12
    assert curves['token_sample_count'] == 2
    assert len(curves['feature_indices']) <= 17 and len(curves['feature_curves']) == 2
# ^^^ THOG
