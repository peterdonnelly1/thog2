# vvv THOG CPU worker provenance must never change scheduler transitions or verified outcomes
from pathlib import Path
import json

import pytest

from run_thog2_owt_core import build_parser, config_from_arguments
from sheet.premat_cpu_config import CPU_CHOICES
from tests.test_cpu_materialisation_lifecycle import setup_runtime, wait_until


@pytest.mark.parametrize('ready', [False, True])
def test_real_scheduler_keeps_outcome_events_when_worker_metadata_has_event_and_clock(monkeypatch, ready):
    runtime, cuda, _, _ = setup_runtime(monkeypatch)
    original_lookup = runtime.cpu_provider.lookup
    def lookup(family, layer, policy):
        matrix, metadata = original_lookup(family, layer, policy)
        return matrix, {**metadata, 'event': 'cpu_matrix_ready', 'host_time_ns': 1,
                        'sequence': 987654, 'pass_sequence': -3, 'owner': 'worker',
                        'state': 'UNAVAILABLE', 'final_outcome': 'incorrect_cached_outcome'}
    monkeypatch.setattr(runtime.cpu_provider, 'lookup', lookup)
    cuda.complete_premat_on_record = ready
    try:
        runtime.layer_start(0)
        wait_until(lambda: any(upload.completion is not None for upload in runtime.cpu_uploads.values()))
        candidate = runtime._candidates[(0, 'O')]
        expected = 'FULL HIT' if ready else 'COMPLETE MISS'
        runtime.acquire('O', 0)
        runtime.consumed('O', 0)
        acquired = next(event for event in runtime._events if event['event'] == 'matrix_acquired')
        consumed = next(event for event in runtime._events if event['event'] == 'consumed' and event.get('family') == 'O')
        for event in (acquired, consumed):
            assert event['event'] == event['event_type']
            assert event['host_time_ns'] > 1
            assert event['sequence'] != 987654 and event['pass_sequence'] != -3
            assert event['owner'] != 'worker'
            assert event['final_outcome'] == expected
            assert event['cpu_preparation_event'] == 'cpu_matrix_ready'
            assert event['cpu_preparation_host_time_ns'] == 1
        assert runtime._candidate_payload(candidate)['state'] == 'CONSUMED'
        assert runtime._candidate_payload(candidate)['final_outcome'] == expected
        assert runtime._candidate_payload(candidate)['sequence'] == candidate.sequence
        if not ready:
            assert acquired['fallback_reason'] == 'upload_not_complete'
        for upload in runtime.cpu_uploads.values():
            if upload.completion is not None:
                upload.completion.complete = True
    finally:
        runtime.end()
        runtime.close()


@pytest.mark.parametrize('mode', ['complete_step', 'latest'])
def test_playback_preference_reaches_cli_runner_training_and_checkpoint_execution_overrides(mode, tmp_path):
    from thog_grid_runner import command_for, validate_recipe
    from sheet.training_config import EXECUTION_OVERRIDE_FIELDS
    parameters = {'--model-type': 'sheet', '--max-iters': 20, '--warmup-iters': 0,
                  '--premat_instra_playback': mode}
    validate_recipe({'label': 'playback', 'parameters': parameters, 'profilers': ['none'], 'gpu_pool': []})
    command = command_for({'parameters': parameters, 'profiler': 'none', 'run_id': 'test', 'grid_tag': 'G-00000'}, {})
    arguments = build_parser().parse_args(command[3:])
    config = config_from_arguments(arguments)
    assert config.premat_instra_playback == mode
    assert config.to_training_config(vocab_size=32, world_size=1, out_dir=tmp_path).premat_instra_playback == mode
    assert 'premat_instra_playback' in EXECUTION_OVERRIDE_FIELDS
    assert build_parser().parse_args([]).premat_instra_playback == 'complete_step'


def test_cpu_help_lists_canonical_options_and_batch_presets_allow_arbitrary_sizes():
    catalogue = json.loads((Path(__file__).parents[1] / 'instra_runner_catalogue.json').read_text())
    for name, choices in CPU_CHOICES.items():
        assert catalogue['--' + name]['choices'] == list(choices)
        assert all(choice in catalogue['--' + name]['options_help'] for choice in choices)
    batch = catalogue['--premat_cpu_layer_batch_size']
    assert 'single_layer' in batch['suggestions'] and 'all_layers' in batch['suggestions']
    assert 'choices' not in batch, 'preset suggestions must not forbid arbitrary integer batches'
    assert '>= 2' in batch['options_help']
# ^^^ THOG
