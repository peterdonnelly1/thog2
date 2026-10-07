# vvv THOG numerical and architecture acceptance for residual reparameterisation
import copy
import math

import pytest
import torch
from torch.nn import functional as functional

from model import GPT, GPTConfig
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.width import WidthBasis, build_width_basis, projected_layer_norm


def make_model(retained_width=3, reference_width=8, depth=False, vectors=False, family='dct', bias=True):
    return SheetGPT(SheetGPTConfig(n_embd=reference_width, n_head=2, n_layer=2,
        vocab_size=17, block_size=5, dropout=0.0, bias=bias, geometry_preset='width-type-I',
        width_enabled=True, width_order=retained_width, width_compressor=family,
        width_depth_enabled=depth, depth_order=1, basis_family='haar',
        basis_version='haar_balanced_binary_orthonormal_v1',
        depth_compress_layer_norm_and_bias=vectors)).double()


@pytest.mark.parametrize('family', ['chebyshev', 'dct', 'haar', 'lapped_cosine', 'general'])
@pytest.mark.parametrize('dtype,tolerance', [(torch.float64, 1e-9), (torch.float32, 1e-4)])
@pytest.mark.parametrize('bias_enabled', [False, True])
@pytest.mark.parametrize('input_kind', ['ordinary', 'constant', 'near_constant'])
def test_projected_normalization_output_and_all_gradients(family, dtype, tolerance, bias_enabled, input_kind):
    generator = torch.Generator().manual_seed(1357)
    if family == 'general':
        columns = torch.linalg.qr(torch.randn(9, 4, generator=generator, dtype=torch.float64)).Q
        analysis, metadata = columns.T, {'constant_first_mode': False}
    else:
        analysis, metadata = build_width_basis(9, 4, family)
    basis = WidthBasis(analysis.to(dtype), metadata)
    coefficients = torch.randn(2, 3, 4, generator=generator, dtype=dtype)
    if input_kind != 'ordinary':
        coefficients.zero_()
        coefficients[..., 0] = 0.1
        if input_kind == 'near_constant':
            coefficients[..., 1:] = 1.0e-5
    gain = torch.randn(9, generator=generator, dtype=dtype)
    bias = torch.randn(9, generator=generator, dtype=dtype) if bias_enabled else None
    arguments = [coefficients.requires_grad_(), gain.requires_grad_()]
    if bias is not None:
        arguments.append(bias.requires_grad_())
    output = projected_layer_norm(coefficients, basis, gain, bias)
    explicit = functional.layer_norm(coefficients @ basis.analysis, (9,), gain, bias, 1e-5) @ basis.analysis.T
    assert (output - explicit).abs().max() <= tolerance
    multiplier = torch.randn(output.shape, generator=generator, dtype=dtype)
    folded_gradients = torch.autograd.grad((output * multiplier).sum(), arguments, retain_graph=True)
    explicit_gradients = torch.autograd.grad((explicit * multiplier).sum(), arguments)
    for folded, reference in zip(folded_gradients, explicit_gradients):
        # Near-constant float32 LayerNorm can amplify gradients into the hundreds;
        # permit two relative ulps while retaining the original absolute tolerance.
        torch.testing.assert_close(folded, reference, atol=tolerance,
                                   rtol=2 * torch.finfo(dtype).eps if dtype == torch.float32 else 0)


@pytest.mark.parametrize('family', ['chebyshev', 'dct', 'haar', 'lapped_cosine'])
def test_all_registered_basis_storage_and_causality(family):
    model = make_model(family=family)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    target = torch.tensor([[2, 3, 4, 5, 6]])
    logits, loss = model(tokens, target)
    changed = tokens.clone(); changed[:, 3:] = 9
    changed_logits, _ = model(changed, target)
    torch.testing.assert_close(logits[:, :3], changed_logits[:, :3], atol=1e-12, rtol=0)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.transformer.wte.weight.shape == (17, 3)
    assert model.transformer.wpe.weight.shape == (5, 3)
    assert model.transformer.wte.weight is model.lm_head.weight
    assert len([name for name, _ in model.named_buffers() if name.endswith('.analysis')]) == 1
    for norm_pair in model.transformer.width_norms:
        assert norm_pair.ln_1.width_basis is model.width_basis
        assert norm_pair.ln_2.width_basis is model.width_basis
    assert model.transformer.ln_f.width_basis is model.width_basis
    assert model.trajectory.coefficients['attention_input_weight'].shape == (24, 3, 2)
    assert model.trajectory.coefficients['mlp_expansion_weight'].shape == (32, 3, 2)
    assert not model.compact_state_violations()


@pytest.mark.parametrize('depth,vectors', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('bias', [False, True])
def test_truncated_model_matches_explicit_projected_reference(monkeypatch, depth, bias, vectors):
    torch.manual_seed(37)
    model = make_model(depth=depth, vectors=vectors, bias=bias)
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 6, 5, 7, 2]])
    targets = (tokens + 1) % 17
    logits, loss = model(tokens, targets)
    loss.backward()
    def explicit_norm(inputs, basis, gain, bias, epsilon=1e-5):
        return functional.layer_norm(inputs @ basis.analysis, (basis.reference_width,), gain, bias, epsilon) @ basis.analysis.T
    monkeypatch.setattr('sheet.width.projected_layer_norm', explicit_norm)
    reference_logits, reference_loss = reference(tokens, targets)
    reference_loss.backward()
    torch.testing.assert_close(logits, reference_logits, atol=1e-9, rtol=0)
    for (_, parameter), (_, reference_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-9, rtol=0)


def test_full_width_dense_equivalence_outputs_and_gradients():
    torch.manual_seed(7331)
    dense = GPT(GPTConfig(n_embd=8, n_head=2, n_layer=2, vocab_size=17, block_size=5, dropout=0, bias=True)).double()
    compact = make_model(retained_width=8)
    analysis = compact.width_basis.analysis
    mappings = [('attention_input_weight', 'attention_input_bias', 'attn', 'c_attn', False),
                ('attention_output_weight', 'attention_output_bias', 'attn', 'c_proj', True),
                ('mlp_expansion_weight', 'mlp_expansion_bias', 'mlp', 'c_fc', False),
                ('mlp_contraction_weight', 'mlp_contraction_bias', 'mlp', 'c_proj', True)]
    with torch.no_grad():
        compact.transformer.wte.weight.copy_(dense.transformer.wte.weight @ analysis.T)
        compact.transformer.wpe.weight.copy_(dense.transformer.wpe.weight @ analysis.T)
        compact.transformer.ln_f.weight.copy_(dense.transformer.ln_f.weight)
        compact.transformer.ln_f.bias.copy_(dense.transformer.ln_f.bias)
        for layer_index, block in enumerate(dense.transformer.h):
            for norm_name in ['ln_1', 'ln_2']:
                norm = vars(block)['_modules'][norm_name]
                compact_norm = compact.transformer.width_norms[layer_index][norm_name]
                compact_norm.weight.copy_(norm.weight)
                compact_norm.bias.copy_(norm.bias)
            for matrix_name, bias_name, branch, linear_name, output_facing in mappings:
                linear = vars(vars(block)['_modules'][branch])['_modules'][linear_name]
                compact.trajectory.coefficients[matrix_name][..., layer_index].copy_(analysis @ linear.weight if output_facing else linear.weight @ analysis.T)
                compact.trajectory.vector_parameters[bias_name][:, layer_index].copy_(analysis @ linear.bias if output_facing else linear.bias)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 6, 5, 7, 2]])
    targets = (tokens + 1) % 17
    dense_logits, dense_loss = dense(tokens, targets)
    compact_logits, compact_loss = compact(tokens, targets)
    dense_loss.backward(); compact_loss.backward()
    torch.testing.assert_close(dense_logits, compact_logits, atol=1e-9, rtol=0)
    torch.testing.assert_close(compact.transformer.wte.weight.grad, dense.transformer.wte.weight.grad @ analysis.T, atol=1e-9, rtol=0)
    torch.testing.assert_close(compact.transformer.wpe.weight.grad, dense.transformer.wpe.weight.grad @ analysis.T, atol=1e-9, rtol=0)
    for layer_index, block in enumerate(dense.transformer.h):
        for matrix_name, bias_name, branch, linear_name, output_facing in mappings:
            linear = vars(vars(block)['_modules'][branch])['_modules'][linear_name]
            expected = analysis @ linear.weight.grad if output_facing else linear.weight.grad @ analysis.T
            torch.testing.assert_close(compact.trajectory.coefficients[matrix_name].grad[..., layer_index], expected, atol=1e-9, rtol=0)
            expected_bias = analysis @ linear.bias.grad if output_facing else linear.bias.grad
            torch.testing.assert_close(compact.trajectory.vector_parameters[bias_name].grad[:, layer_index], expected_bias, atol=1e-9, rtol=0)
        for norm_name in ['ln_1', 'ln_2']:
            norm = vars(block)['_modules'][norm_name]
            compact_norm = compact.transformer.width_norms[layer_index][norm_name]
            torch.testing.assert_close(norm.weight.grad, compact_norm.weight.grad, atol=1e-9, rtol=0)
            torch.testing.assert_close(norm.bias.grad, compact_norm.bias.grad, atol=1e-9, rtol=0)
    torch.testing.assert_close(dense.transformer.ln_f.weight.grad, compact.transformer.ln_f.weight.grad, atol=1e-9, rtol=0)
    torch.testing.assert_close(dense.transformer.ln_f.bias.grad, compact.transformer.ln_f.bias.grad, atol=1e-9, rtol=0)


def test_direct_depth_contraction_outputs_and_gradients():
    model = make_model(depth=True, vectors=True)
    for item in model.trajectory.metadata:
        bank = model.trajectory.coefficients[item.name]
        inputs = torch.randn(2, 5, bank.shape[1], dtype=torch.float64, requires_grad=True)
        direct = model.trajectory.linear(inputs, item.name, 1, 'absent_bias')
        materialized = torch.einsum('oip,p->oi', bank, model.trajectory.depth_basis[1])
        reference = functional.linear(inputs, materialized)
        torch.testing.assert_close(direct, reference, atol=1e-9, rtol=0)
        for actual, expected in zip(torch.autograd.grad(direct.square().sum(), (inputs, bank), retain_graph=True), torch.autograd.grad(reference.square().sum(), (inputs, bank))):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


def test_no_stale_gain_and_minimum_precision():
    model = make_model()
    norm = model.transformer.ln_f
    inputs = torch.randn(2, 5, 3, dtype=torch.float64)
    before = norm(inputs).detach()
    with torch.no_grad(): norm.weight.mul_(2)
    torch.testing.assert_close(norm(inputs), before * 2, atol=1e-9, rtol=0)
    assert not any('gain_operator' in name for name in model.state_dict())
    model.half()
    assert model.width_basis.analysis.dtype == torch.float32


def test_high_order_chebyshev_is_full_rank():
    analysis, metadata = build_width_basis(128, 100, 'chebyshev')
    assert metadata['raw_numerical_rank'] == 100
    torch.testing.assert_close(analysis @ analysis.T, torch.eye(100,dtype=analysis.dtype), atol=1e-11, rtol=0)


def test_registered_general_basis_and_nonorthonormal_rejection(monkeypatch):
    from sheet.bases import registry
    from sheet.bases.protocol import BasisDefinition, BasisKernel
    class GeneralKernel(BasisKernel):
        def coordinates(self, sample_count, **kwargs):
            return torch.arange(sample_count, **{k:v for k,v in kwargs.items() if v is not None})
        def raw_basis(self, coordinates, order):
            return torch.linalg.qr(torch.randn(len(coordinates), order, dtype=torch.float64,
                                               generator=torch.Generator().manual_seed(814))).Q
        def stabilize(self, raw):
            return raw
    kernel = GeneralKernel('general_width_test', 'general_width_test_v1', 'fixture_integer', 'fixture_qr')
    definition = BasisDefinition(kernel.basis_family, ('general_width_alias',), kernel.basis_version,
                                 'GENERAL_WIDTH_TEST', True, False, kernel)
    monkeypatch.setattr(registry, 'BASIS_REGISTRY', registry.BasisRegistry((*registry.BASIS_REGISTRY.definitions(), definition)))
    analysis, metadata = build_width_basis(9, 4, 'general_width_alias')
    assert not metadata['constant_first_mode']
    torch.testing.assert_close(analysis @ analysis.T, torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=0)
    monkeypatch.setattr(GeneralKernel, 'stabilize', lambda self, raw: raw * 2)
    with pytest.raises(ValueError, match='stabilization failed'):
        build_width_basis(9, 4, 'general_width_test')


def test_parameterised_lapped_version_certifies_the_requested_span():
    _, metadata = build_width_basis(64, 20, 'lapped_cosine', 'lapped_cosine_dc_preserving_orthonormal_v1_w8_o500')
    assert metadata['raw_span_relative_error'] < 1e-12
    assert metadata['raw_condition_number'] == pytest.approx(1)


def test_bfloat16_autocast_keeps_folded_math_in_float32():
    analysis, metadata = build_width_basis(12, 5, 'dct')
    basis = WidthBasis(analysis, metadata)
    inputs = torch.randn(2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    gain = torch.randn(12, requires_grad=True)
    bias = torch.randn(12, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = projected_layer_norm(inputs, basis, gain, bias)
    explicit = functional.layer_norm(inputs.float() @ analysis.float(), (12,), gain, bias, 1e-5) @ analysis.float().T
    torch.testing.assert_close(actual, explicit.to(torch.bfloat16), atol=0, rtol=0)
    actual.float().square().mean().backward()
    assert inputs.grad is not None and gain.grad is not None and bias.grad is not None


def test_normal_execution_residual_sites_stay_compact_and_never_materialize(monkeypatch):
    import sheet.width as width
    model = make_model(depth=True, vectors=True)
    original_site = width.width_site
    sites = []
    def inspect_site(model, site, values):
        sites.append((site, tuple(values.shape)))
        assert values.shape[-1] == model.config.width_order
        return original_site(model, site, values)
    monkeypatch.setattr(width, 'width_site', inspect_site)
    monkeypatch.setattr(model.trajectory, 'materialize', lambda *args: pytest.fail('normal execution materialized weights'))
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    _, loss = model(tokens, (tokens + 1) % 17)
    loss.backward()
    assert len(sites) == model.config.n_layer + 2
# ^^^ THOG
