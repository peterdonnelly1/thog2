# vvv THOG reproducible CPU functional comparison required by CSA-VAL-007/008
"""Compare matched training conditions; each variant gets a fresh RSS-measurement process.

This short character next-token toy supports functional acceptance, not a claim
about language-model quality or GPU performance. The same-r conventional norm
is an isolated experiment control and is never a selectable product variant.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = ('dense', 'depth_only', 'width_only', 'width_depth', 'ordinary_same_r_norm_control')
TRAIN_TEXT = ('the river turns around the stone. the sun lights the hill. '
              'a small bird rests on the tree. the wind moves the leaves.\n') * 64
VALIDATION_TEXT = ('the bird turns around the tree. the river moves around the hill. '
                   'the sun lights the leaves and the wind moves the bird.\n') * 12


def worker(variant, seed, updates):
    import torch
    from torch.nn import functional as F
    from sheet.stage6_trainer import Stage6Trainer
    from sheet.training_config import TrainingConfig
    import sheet.width as width_module

    torch.set_num_threads(2)
    os.environ['THOG2_OPTIMIZER'] = 'adamw'
    os.environ['THOG2_FAST_DISCARD'] = 'true'
    characters = sorted(set(TRAIN_TEXT + VALIDATION_TEXT))
    mapping = {character: index for index, character in enumerate(characters)}
    train = torch.tensor([mapping[c] for c in TRAIN_TEXT])
    validation = torch.tensor([mapping[c] for c in VALIDATION_TEXT])
    width = variant in ('width_only', 'width_depth', 'ordinary_same_r_norm_control')
    depth = variant in ('depth_only', 'width_depth')
    config = TrainingConfig(
        model_type='dense' if variant == 'dense' else 'thog2_sheet',
        geometry_preset='width-type-I' if width else 'depth' if depth else None,
        n_embd=32, n_head=4, n_layer=4, block_size=16, vocab_size=len(characters),
        depth_order=2, base_row_order=8, basis_family=None if variant == 'dense' else 'dct',
        basis_version='chebyshev_first_kind_roots_v1' if variant == 'dense' else 'dct_ii_orthonormal_v1',
        width_enabled=width, width_order=12 if width else None,
        width_compressor='dct', width_depth_enabled=width and depth,
        batch_size=2, gradient_accumulation_steps=2, max_updates=updates,
        checkpoint_segment_size=2, dropout=0.0, bias=True,
        model_seed=seed, data_seed=seed + 5000, dtype='float32', device='cpu',
        learning_rate=0.002, min_learning_rate=0.002, decay_learning_rate=False,
        decay_updates=updates, residual_init_depth_source='true_layer_depth',
    )
    original_normalize = width_module.width_normalize
    trainer = None
    try:
        # Discard construction chatter without altering the trainer or timing boundary.
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Stage6Trainer(config, train, validation)
        model = trainer.raw_model
        if variant == 'ordinary_same_r_norm_control':
            for norms in model.transformer.width_norms:
                for name in ('ln_1', 'ln_2'):
                    norms[name] = torch.nn.LayerNorm(config.width_order, bias=config.bias)
            model.transformer.ln_f = torch.nn.LayerNorm(config.width_order, bias=config.bias)
            def conventional_normalize(model, inputs, layer_index, name):
                norm = model.transformer.ln_f if name == 'ln_f' else model.transformer.width_norms[layer_index][name]
                return F.layer_norm(inputs, (config.width_order,), norm.weight, norm.bias, 1e-5)
            width_module.width_normalize = conventional_normalize
            # Bind ordinary compact affine arrays to the same AdamW policy after replacement.
            from sheet.optimizer_factory import build_optimizer
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.optimizer = build_optimizer(model, weight_decay=config.weight_decay,
                    learning_rate=config.learning_rate, betas=(config.beta1, config.beta2), device_type='cpu')
        elapsed = []
        losses = []
        for _ in range(updates):
            started = time.perf_counter()
            metrics = trainer.train_one_update()
            elapsed.append(time.perf_counter() - started)
            losses.append(float(metrics['training_loss']))
        inputs = torch.stack([validation[start:start + config.block_size] for start in range(0, 128, config.block_size)])
        targets = torch.stack([validation[start + 1:start + config.block_size + 1] for start in range(0, 128, config.block_size)])
        model.eval()
        with torch.no_grad():
            _, held_loss = model(inputs, targets)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                model(inputs[:1], targets[:1])
        normalization_profile = [dict(operation=event.key, cpu_total_ms=event.cpu_time_total / 1000,
            calls=event.count) for event in profiler.key_averages() if event.key in ('width_norm_construct', 'width_norm_apply')]
        parameters = list(model.parameters())
        steady = elapsed[2:] if len(elapsed) > 2 else elapsed
        return {
            'variant': variant, 'model_seed': seed, 'data_seed': config.data_seed,
            'reference_width': 32, 'residual_width': 12 if width else 32, 'executed_layers': 4,
            'attention_internal_width': 32, 'mlp_hidden_width': 128, 'depth_order': 2 if depth else None,
            'sequence_length': 16, 'batch_size': 2, 'gradient_accumulation': 2,
            'completed_updates': trainer.state.completed_updates, 'training_tokens': updates * 64,
            'evaluation_token_count': targets.numel(), 'tokenizer': 'sorted characters in the embedded corpus',
            'vocabulary': characters, 'optimizer': 'AdamW', 'learning_rate': 0.002,
            'beta1': config.beta1, 'beta2': config.beta2, 'weight_decay': config.weight_decay,
            'gradient_clip': config.grad_clip, 'dropout': 0.0, 'dtype': 'float32', 'cpu_threads': 2,
            'checkpoint_segment_size': 2, 'fast_discard': True,
            'training_batch_trace_digest': hashlib.sha256(json.dumps(trainer.batch_source.trace, default=str).encode()).hexdigest(),
            'evaluation_token_digest': hashlib.sha256(inputs.numpy().tobytes() + targets.numpy().tobytes()).hexdigest(),
            'trained_parameters': sum(p.numel() for p in parameters),
            'learned_parameter_bytes': sum(p.numel() * p.element_size() for p in parameters),
            'fixed_buffer_bytes': sum(b.numel() * b.element_size() for b in model.buffers()),
            'nominal_gradient_parameter_bytes': sum(p.numel() * p.element_size() for p in parameters),
            'actual_optimizer_state_bytes': sum(v.numel() * v.element_size() for state in trainer.optimizer.state.values() for v in state.values() if torch.is_tensor(v)),
            'cpu_peak_resident_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            'cpu_peak_window': 'fresh child process lifetime; includes imports, construction, optimizer, training, validation and one profiler capture; not resettable',
            'gpu_peak_allocated_bytes': None, 'gpu_peak_reserved_bytes': None, 'gpu_counters_status': 'unavailable: CPU functional comparison',
            'complete_update_ms_mean_all_updates': statistics.mean(elapsed) * 1000,
            'complete_update_ms_mean_excluding_first_two_warmup_updates': statistics.mean(steady) * 1000,
            'tokens_per_training_second_all_updates': updates * 64 / sum(elapsed),
            'tokens_per_training_second_excluding_first_two_warmup_updates': 64 / statistics.mean(steady),
            'timing_window': 'full train_one_update: forward, loss, backward/replay, clipping, optimizer, schedule, zero-grad; instrumentation off; imports/startup/validation excluded',
            'last_training_cross_entropy': losses[-1], 'held_out_cross_entropy': float(held_loss),
            'held_out_perplexity': math.exp(float(held_loss)),
            'normalization_profile': normalization_profile,
            'normalization_profile_window': 'one post-training validation forward, batch=1, sequence=16; excludes profiler startup and training backward',
        }
    finally:
        width_module.width_normalize = original_normalize
        if trainer is not None:
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--seeds', default='77,1337')
    parser.add_argument('--updates', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('evidence/residual_width_cpu_comparison.json'))
    args = parser.parse_args()
    if args.updates < 3:
        parser.error('at least three updates are required')
    if args.variant:
        print(json.dumps(worker(args.variant, args.seed, args.updates)))
        return
    rows = []
    for seed in (int(value) for value in args.seeds.split(',')):
        for variant in VARIANTS:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--variant', variant,
                '--seed', str(seed), '--updates', str(args.updates)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'{variant} failed: {result.stderr}')
            row = json.loads(result.stdout)
            rows.append(row)
            print(f"{seed} {variant}: CE={row['held_out_cross_entropy']:.5f}, parameters={row['trained_parameters']}, "
                  f"update_ms={row['complete_update_ms_mean_excluding_first_two_warmup_updates']:.3f}", flush=True)
    for seed in sorted({row['model_seed'] for row in rows}):
        cohort = [row for row in rows if row['model_seed'] == seed]
        assert len({row['evaluation_token_digest'] for row in cohort}) == 1
        assert len({row['training_batch_trace_digest'] for row in cohort}) == 1
    import torch
    output = {'scope': 'reproducible character next-token CPU toy; no language-model-quality or GPU-speed claim',
              'torch_version': torch.__version__, 'python_version': sys.version, 'cpu_threads': 2,
              'training_corpus_sha256': hashlib.sha256(TRAIN_TEXT.encode()).hexdigest(),
              'validation_corpus_sha256': hashlib.sha256(VALIDATION_TEXT.encode()).hexdigest(), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
# ^^^ THOG
