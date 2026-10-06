# vvv THOG bounded observational width capture in the existing training/chart lifecycle
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
try:
    import resource
except ImportError:  # CPU peak RSS is an explicitly optional platform counter.
    resource = None
import sys
import time

import torch

from .width import WIDTH_CAPTURE_DEFAULTS, WIDTH_CAPTURE_PREFIX


def validate_width_capture_configuration(config):
    if WIDTH_CAPTURE_PREFIX + 'mode' not in vars(config):
        return None
    values = {suffix: vars(config)[WIDTH_CAPTURE_PREFIX + suffix] for suffix in WIDTH_CAPTURE_DEFAULTS}
    if values['mode'] not in ('off', 'basic', 'probes'):
        raise ValueError('width capture mode must be off, basic, or probes')
    if values['mode'] != 'off' and not config.width_enabled:
        raise ValueError('width capture requires --select-width')
    for suffix in ('log_every_n_steps', 'probe_every_n_steps', 'history_length', 'sample_tokens_per_layer', 'feature_evaluation_points'):
        value = values[suffix]
        minimum = 2 if suffix == 'feature_evaluation_points' else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{WIDTH_CAPTURE_PREFIX}{suffix} must be an integer >= {minimum}')
    if isinstance(values['start_step'], bool) or not isinstance(values['start_step'], int) or values['start_step'] < 0:
        raise ValueError('width capture start_step must be a nonnegative integer')
    end = values['end_step']
    if isinstance(end, bool) or not isinstance(end, int) or (end != -1 and end < values['start_step']):
        raise ValueError('width capture end_step must be -1 or >= start_step')
    if config.width_enabled:
        if values['probe_orders'] == 'auto':
            orders = [max(1, config.width_order // 2)]
        else:
            try:
                orders = [int(value.strip()) for value in str(values['probe_orders']).split(',')]
            except ValueError as error:
                raise ValueError('width probe_orders must be auto or comma-separated integers') from error
            if len(set(orders)) != len(orders) or any(not 1 <= order < config.width_order for order in orders):
                raise ValueError('width probe_orders must be distinct prefixes satisfying 1 <= k < r')
        values['resolved_probe_orders'] = orders
        values['feature_evaluation_points'] = min(values['feature_evaluation_points'], config.n_embd)
    return values


def capture_due(step, cadence, start, end):
    return step > 0 and step >= start and (end == -1 or step <= end) and (step - start) % cadence == 0


def retained_python_bytes(value, seen=None):
    """Count the detached Python history buffers once per object identity."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(retained_python_bytes(key, seen) + retained_python_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, deque)):
        size += sum(retained_python_bytes(item, seen) for item in value)
    return size


class WidthSampleCapture:
    def __init__(self, model, options, *, probes=False):
        self.model = model
        self.options = options
        self.probes = probes
        self.residuals = {}
        self.norms = {}
        self.elapsed_seconds = 0.0
        self.sample_bytes = 0

    @torch.no_grad()
    def residual(self, site, coefficients):
        if site in self.residuals:
            return
        started = time.perf_counter()
        count = self.options['sample_tokens_per_layer']
        sample = coefficients.detach().reshape(-1, coefficients.shape[-1])[:count].to(torch.float64)
        analysis = self.model.width_basis.analysis.to(torch.float64)
        reference_width = self.model.config.n_embd
        mean = sample @ self.model.width_basis.constant_analysis.to(torch.float64) / reference_width
        mean_energy = reference_width * mean.square().mean()
        total_energy = sample.square().sum(dim=-1).mean()
        payload = {'site': site, 'token_sample_count': sample.shape[0],
                   'token_vectors_in_operation': coefficients.numel() // coefficients.shape[-1],
                   'residual_shape': list(coefficients.shape),
                   'mode_energy': sample.square().mean(dim=0).cpu().tolist(),
                   'reference_mean_energy': float(mean_energy),
                   'reference_centered_energy': float((total_energy - mean_energy).clamp_min(0)),
                   'constant_first_mode': self.model.width_basis.constant_first_mode}
        if self.probes:
            grid = torch.linspace(0, reference_width - 1, self.options['feature_evaluation_points'], device=sample.device).round().long().unique()
            payload['feature_indices'] = grid.cpu().tolist()
            payload['feature_curves'] = (sample @ analysis[:, grid]).cpu().tolist()
        self.residuals[site] = payload
        self.sample_bytes += sample.numel() * sample.element_size()
        self.elapsed_seconds += time.perf_counter() - started

    @torch.no_grad()
    def normalization(self, site, inputs, gain, bias, output):
        if site in self.norms:
            return
        started = time.perf_counter()
        gamma = gain.detach().double()
        analysis = self.model.width_basis.analysis.double()
        gain_operator = (analysis * gamma) @ analysis.T
        mean_gain = gamma.mean()
        baseline = torch.eye(gain_operator.shape[0], device=gamma.device, dtype=gamma.dtype) * mean_gain
        payload = {'site': site,
                   'gain_relative_variation': float(gamma.std(unbiased=False) / gamma.square().mean().sqrt().clamp_min(1e-12)),
                   'gain_relative_coupling': float((gain_operator - baseline).norm() / baseline.norm().clamp_min(1e-12))}
        if self.probes:
            sample = inputs.detach().reshape(-1, inputs.shape[-1])[:self.options['sample_tokens_per_layer']].double()
            reference_width = self.model.config.n_embd
            mean = sample @ self.model.width_basis.constant_analysis.double() / reference_width
            variance = (sample.square().sum(dim=-1) / reference_width - mean.square()).clamp_min(0)
            if self.model.width_basis.constant_first_mode:
                variance = sample[:, 1:].square().sum(dim=-1) / reference_width
            norm = torch.rsqrt(variance + 1e-5)
            affine_energy = sample.new_zeros(sample.shape[0])
            # Only bounded diagnostic tokens and feature tiles are ever synthesized.
            for start in range(0, reference_width, 256):
                stop = min(reference_width, start + 256)
                represented = sample @ analysis[:, start:stop]
                affine = (represented - mean[:, None]) * norm[:, None] * gamma[start:stop]
                if bias is not None:
                    affine = affine + bias.detach().double()[start:stop]
                affine_energy += affine.square().sum(dim=-1)
            projected = output.detach().reshape(-1, output.shape[-1])[:sample.shape[0]].double()
            retained_energy = projected.square().sum(dim=-1)
            discarded = (affine_energy - retained_energy).clamp_min(0)
            payload.update({'post_affine_discarded_energy': float(discarded.mean()),
                            'post_affine_discarded_fraction': float(discarded.sum() / affine_energy.sum().clamp_min(1e-12)),
                            'diagnostic_feature_tile': min(256, reference_width)})
        self.norms[site] = payload
        self.elapsed_seconds += time.perf_counter() - started


class WidthInstrumentation:
    def __init__(self, trainer):
        self.trainer = trainer
        self.options = validate_width_capture_configuration(trainer.config)
        self.history = deque(maxlen=self.options['history_length'])
        self.last_capture_step = 0
        self.sample_starts = None
        self.sample_seed = int(trainer.config.data_seed) + 37013

    def state_dict(self):
        return {'options': self.options, 'last_capture_step': self.last_capture_step,
                'sample_seed': self.sample_seed, 'sample_starts': self.sample_starts,
                'history': list(self.history)}

    def load_state_dict(self, state):
        if state['options'] != self.options:
            raise ValueError('width instrumentation configuration differs from checkpoint')
        self.last_capture_step = int(state['last_capture_step'])
        self.sample_seed = int(state['sample_seed'])
        self.sample_starts = state['sample_starts']
        self.history.extend(state.get('history', ())[-self.options['history_length']:])

    def _probe_tokens(self):
        source = self.trainer.batch_source
        if self.sample_starts is None:
            generator = torch.Generator(device='cpu').manual_seed(self.sample_seed)
            self.sample_starts = torch.randint(source._storage_length(source.validation_tokens) - source.block_size,
                                               (1,), generator=generator).tolist()
        inputs = torch.stack([source._slice(source.validation_tokens, start, start + source.block_size) for start in self.sample_starts])
        targets = torch.stack([source._slice(source.validation_tokens, start + 1, start + source.block_size + 1) for start in self.sample_starts])
        return inputs.to(self.trainer.device), targets.to(self.trainer.device)

    @torch.no_grad()
    def _probes(self, step):
        model = self.trainer.raw_model
        inputs, targets = self._probe_tokens()
        training = model.training
        module_training_states = [(module, module.training) for module in model.modules()]
        previous_mask = model._width_probe_order
        previous_capture = model._width_capture
        started = time.perf_counter()
        curve_capture = WidthSampleCapture(model, self.options, probes=True)
        try:
            model.eval()
            model._width_probe_order = None
            model._width_capture = curve_capture
            devices = [self.trainer.device.index or 0] if self.trainer.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices), self.trainer.autocast_context():
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.trainer.device.type == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities) as profiler:
                    _, loss = model(inputs, targets)
                normalization_profile = [{'operation': event.key,
                    'cpu_total_ms': event.cpu_time_total / 1000.0,
                    'device_total_ms': vars(event).get('device_time_total', 0.0) / 1000.0,
                    'calls': event.count} for event in profiler.key_averages()
                    if event.key in ('width_norm_construct', 'width_norm_apply')]
                baseline = float(loss)
                model._width_capture = None
                probes = []
                for order in self.options['resolved_probe_orders']:
                    model._width_probe_order = order
                    _, loss = model(inputs, targets)
                    value = float(loss)
                    probes.append({'retained_prefix': order, 'loss': value, 'delta_loss': value - baseline})
        finally:
            model._width_capture = previous_capture
            model._width_probe_order = previous_mask
            model.train(training)
            for module, state in module_training_states:
                module.training = state
        return {'baseline_validation_loss': baseline, 'baseline_perplexity': math.exp(min(baseline, 700)),
                'probe_token_count': int(targets.numel()), 'sample_starts': self.sample_starts,
                'sample_seed': self.sample_seed,
                'sample_token_digest': hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
                'normalization_profile': normalization_profile,
                'normalization_profile_window': 'one bounded baseline validation forward; instrumentation cost excluded from normal update timing',
                'probes': probes, 'probe_seconds': time.perf_counter() - started,
                'probe_interpretation': 'positive delta_loss means ablation harmed this trained model; not retrained smaller-model performance',
                'probe_residual_sites': list(curve_capture.residuals.values()),
                'probe_normalization_sites': list(curve_capture.norms.values())}

    def train_one_update(self, function):
        trainer = self.trainer
        step = trainer.state.completed_updates + 1
        options = self.options
        basic_due = capture_due(step, options['log_every_n_steps'], options['start_step'], options['end_step'])
        probe_due = options['mode'] == 'probes' and capture_due(step, options['probe_every_n_steps'], options['start_step'], options['end_step'])
        if not basic_due and not probe_due:
            return function()
        model = trainer.raw_model
        capture = WidthSampleCapture(model, options) if basic_due else None
        logical_saved = 0
        logical_activations = 0
        distinct_storages = {}
        parameter_storages = {value.untyped_storage().data_ptr() for value in (*model.parameters(), *model.buffers())}
        def pack(tensor):
            nonlocal logical_saved, logical_activations
            size = tensor.numel() * tensor.element_size()
            logical_saved += size
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_storages:
                logical_activations += size
                distinct_storages[storage.data_ptr()] = storage.nbytes()
            return tensor
        model._width_capture = capture
        if trainer.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.device)
        started = time.perf_counter()
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor) if basic_due else nullcontext():
                metrics = function()
            if trainer.device.type == 'cuda':
                torch.cuda.synchronize(trainer.device)
            training_seconds = time.perf_counter() - started
        finally:
            model._width_capture = None
        if trainer.state.completed_updates != step:
            return metrics
        diagnostic_started = time.perf_counter()
        cpu_peak = None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        snapshot = {'optimizer_update': step, 'representation': model._width_identity,
                    'residual_sites': [] if capture is None else list(capture.residuals.values()),
                    'normalization_sites': [] if capture is None else list(capture.norms.values()),
                    'training_loss': metrics.get('training_loss'),
                    'complete_update_ms_including_capture': 1000 * training_seconds,
                    'memory': {'gpu_peak_allocated_bytes': None if trainer.device.type != 'cuda' else torch.cuda.max_memory_allocated(trainer.device),
                               'gpu_peak_reserved_bytes': None if trainer.device.type != 'cuda' else torch.cuda.max_memory_reserved(trainer.device),
                               'gpu_peak_window': 'captured completed update; includes resident model, optimizer and basic capture; excludes startup and loss probes',
                               'cpu_peak_resident_bytes': cpu_peak, 'cpu_peak_window': 'process lifetime; includes startup, optimizer, and diagnostic work',
                               'saved_tensor_logical_bytes': logical_saved if basic_due else None,
                               'saved_activation_logical_bytes': logical_activations if basic_due else None,
                               'saved_activation_distinct_storage_bytes': sum(distinct_storages.values()) if basic_due else None,
                               'saved_activation_window': 'captured update; parameter and basis storages excluded; backward and checkpoint replay included',
                               'gradient_bytes': sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None),
                               'gradient_window': 'after completed optimizer update and existing zero_grad(set_to_none=True); usually zero allocated gradient bytes',
                               'nominal_gradient_parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                               'distinct_storage_policy': 'unique nonparameter storage addresses observed across this update; address reuse may coalesce records; not a peak-memory counter',
                               'optimizer_state_bytes': sum(value.numel() * value.element_size() for state in trainer.optimizer.state.values() for value in state.values() if torch.is_tensor(value)),
                               'unavailable': (['GPU counters on CPU'] if trainer.device.type != 'cuda' else []) + (['CPU peak RSS on this platform'] if resource is None else [])}}
        if probe_due:
            snapshot.update(self._probes(step))
        self.last_capture_step = step
        self.history.append(snapshot)
        snapshot['retained_buffer_measurement'] = 'recursive sys.getsizeof of detached rolling Python history, counted once per object identity; excludes temporary diagnostics and published copies'
        snapshot['retained_buffer_bytes'] = 0
        snapshot['retained_buffer_bytes'] = retained_python_bytes(self.history)
        snapshot['serialized_history_bytes'] = len(json.dumps(list(self.history)).encode('utf-8'))
        snapshot['basic_capture_seconds'] = 0.0 if capture is None else capture.elapsed_seconds
        overhead = time.perf_counter() - diagnostic_started + snapshot['basic_capture_seconds']
        snapshot['instrumentation_seconds'] = overhead
        snapshot['complete_update_ms_excluding_capture'] = 1000 * max(training_seconds - snapshot['basic_capture_seconds'], 0)
        snapshot['tokens_per_training_second'] = trainer.config.batch_size * trainer.config.block_size * trainer.config.gradient_accumulation_steps / max(training_seconds - snapshot['basic_capture_seconds'], 1e-12)
        publish_started = time.perf_counter()
        self._publish(snapshot)
        metrics['width_instrumentation_seconds'] = overhead + time.perf_counter() - publish_started
        return metrics

    def _publish(self, snapshot):
        telemetry = vars(self.trainer).get('_width_telemetry')
        if telemetry is None or not self.trainer.distributed.is_primary:
            return
        from .local_chart_store import ensure_local_chart_store
        ensure_local_chart_store(telemetry).append_width_snapshot(snapshot, history_length=self.options['history_length'])
        if telemetry.run is not None:
            payload = {'width/capture': snapshot, 'width/reference_width': self.trainer.config.n_embd,
                       'width/retained_width': self.trainer.config.width_order,
                       'width/training_loss': snapshot['training_loss']}
            telemetry.run.log(payload, step=snapshot['optimizer_update'])


def width_instrumentation_for(trainer):
    if not trainer.config.width_enabled or vars(trainer.config)[WIDTH_CAPTURE_PREFIX + 'mode'] == 'off':
        return None
    instrumentation = vars(trainer).get('_width_instrumentation')
    if instrumentation is None:
        instrumentation = WidthInstrumentation(trainer)
        trainer._width_instrumentation = instrumentation
    return instrumentation
# ^^^ THOG
