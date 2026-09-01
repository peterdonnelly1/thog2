# vvv THOG
"""Dependency-light behavioural checks for the September 1 Instra changes."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)


class InstraEnhancementTests(unittest.TestCase):
    def test_new_and_legacy_cli_spellings_and_effective_environment(self):
        depth = SimpleNamespace(_CLI_INSTALLED_ATTRIBUTE='_installed', _DEFAULT_SCALARS_PER_MATRIX=3,
                                _DEFAULT_DEPTH_POINTS=256, _DEFAULT_TIME_MODE='latest',
                                _DEFAULT_HISTORY_LENGTH=20, _DEFAULT_LOG_EVERY_N_STEPS=100,
                                _DEFAULT_DESTINATION='local')
        scope = {'argparse': argparse, '_depth': depth, 'os': os,
                 '_environment_name': lambda suffix: 'THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_' + suffix}
        load_functions('sheet/depth_weight_curves_cli_compat_patch.py',
                       {'_normalized_options', '_explicit_bool', '_ensure_cli_arguments_with_normalized_aliases'}, scope)
        load_functions('sheet/depth_weight_curves_and_observational_probes_patch.py', {'_publish_cli_environment'}, scope)
        for count_name, shared_name in (
            ('coupling_pairs_per_matrix', 'same_coupling_pairs_all_runs'),
            ('scalar_weights_per_matrix', 'same_coordinates_all_runs'),
        ):
            for spelling in (lambda value: value, lambda value: value.replace('_', '-'),
                             lambda value: value.replace('__', '_').replace('_', '-')):
                with self.subTest(count_name=count_name, spelling=spelling), patch.dict(os.environ, {}, clear=True):
                    parser = argparse.ArgumentParser()
                    scope['_ensure_cli_arguments_with_normalized_aliases'](parser)
                    options = ['--' + spelling('instrumentation__depth_weight_curves__' + count_name), '7',
                               '--' + spelling('instrumentation__depth_weight_curves__' + shared_name), 'false']
                    values = parser.parse_args(options)
                    scope['_publish_cli_environment'](values)
                    self.assertEqual(values.instrumentation__depth_weight_curves__coupling_pairs_per_matrix, 7)
                    self.assertEqual(values.instrumentation__depth_weight_curves__scalar_weights_per_matrix, 7)
                    self.assertFalse(values.instrumentation__depth_weight_curves__same_coupling_pairs_all_runs)
                    self.assertEqual(os.environ['THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_SCALAR_WEIGHTS_PER_MATRIX'], '7')
                    self.assertEqual(os.environ['THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_SAME_COORDINATES_ALL_RUNS'], 'false')
                    self.assertNotIn('--instrumentation__depth_weight_curves__scalar_weights_per_matrix', parser.format_help())
                    values = parser.parse_args([options[2], 'true'])
                    scope['_publish_cli_environment'](values)
                    self.assertTrue(values.instrumentation__depth_weight_curves__same_coupling_pairs_all_runs)

    def test_snapshot_details_roundtrip_and_hash_guard(self):
        spec = importlib.util.spec_from_file_location('snapshot_overview', ROOT / 'sheet/dense_snapshot_overview.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {'compatibility_hash': 'physical', 'tensor_payload_hash': 'weights',
                   'compatibility_payload': {'physical_layer_count': 16, 'model_width': 1024}}
        config = SimpleNamespace(n_layer=16, learning_rate=0.001, model_seed=42, nested={'x': [1, 2]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pt'
            path.write_bytes(b'immutable snapshot fixture')
            before = path.read_bytes()
            missing = module.snapshot_overview_metadata(path, payload)
            self.assertNotIn('source_hyperparameters', missing)
            module.write_snapshot_overview(path, payload, config)
            details = module.snapshot_overview_metadata(path, payload)
            self.assertEqual(details['source_hyperparameters']['learning_rate'], 0.001)
            self.assertEqual(details['source_hyperparameters']['nested'], {'x': [1, 2]})
            self.assertEqual(details['snapshot_hyperparameters'], payload['compatibility_payload'])
            self.assertEqual(path.read_bytes(), before)
            mismatched = module.snapshot_overview_metadata(path, {**payload, 'tensor_payload_hash': 'different'})
            self.assertNotIn('source_hyperparameters', mismatched)
            Path(str(path) + '.overview.json').write_text('malformed')
            self.assertEqual(module.snapshot_overview_metadata(path, payload), missing)
            module.write_snapshot_overview(Path(directory) / 'absent' / 'source.pt', payload, config)

    def test_canonical_names_persist_in_chart_configuration(self):
        captured = {}
        scope = {'Any': object, 'Optional': __import__('typing').Optional,
                 'Mapping': __import__('typing').Mapping, 'os': os,
                 '_DEPTH_WEIGHT_ENVIRONMENT_ROWS': (
                    ('instrumentation__depth_weight_curves__coupling_pairs_per_matrix', 'PAIR_COUNT', '3'),
                    ('instrumentation__depth_weight_curves__same_coupling_pairs_all_runs', 'PAIR_SHARED', 'false')),
                 '_START_DESTINATION': 'start', '_START_ENVIRONMENT': 'START',
                 '_END_DESTINATION': 'end', '_END_ENVIRONMENT': 'END', '_EXPLICIT_STEP_RANGE': {},
                 '_ORIGINAL_LOCAL_CHART_STORE_INIT': lambda *args, **kwargs: captured.update(kwargs)}
        load_functions('sheet/depth_weight_step_range_config_patch.py', {'_local_chart_store_init_with_weight_step_range'}, scope)
        with patch.dict(os.environ, {}, clear=True):
            scope['_local_chart_store_init_with_weight_step_range'](None, 'fixture', run_name='run', config={
                'instrumentation__depth_weight_curves__scalar_weights_per_matrix': 9,
                'instrumentation__depth_weight_curves__same_coordinates_all_runs': True})
        config = captured['config']
        self.assertEqual(config['instrumentation__depth_weight_curves__coupling_pairs_per_matrix'], 9)
        self.assertTrue(config['instrumentation__depth_weight_curves__same_coupling_pairs_all_runs'])
        self.assertNotIn('instrumentation__depth_weight_curves__scalar_weights_per_matrix', config)


if __name__ == '__main__':
    unittest.main()
# ^^^ THOG
