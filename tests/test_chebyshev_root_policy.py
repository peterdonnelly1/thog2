# vvv THOG independently certify root sampling, direct normalization and continuous evaluation
import math

import pytest
import torch

from sheet.basis import build_stabilized_basis, normalized_coordinates, stabilized_chebyshev_basis_at_coordinates
from sheet.bases import CHEBYSHEV_BASIS_VERSION, get_basis_kernel, normalize_basis_version


@pytest.mark.parametrize("samples,order", [(1, 1), (2, 2), (7, 7), (16, 12), (128, 100), (1024, 512), (1024, 1024)])
def test_root_basis_matches_independent_cosine_formula_without_qr(monkeypatch, samples, order):
    monkeypatch.setattr(torch.linalg, "qr", lambda *args, **kwargs: pytest.fail("root construction invoked QR"))
    basis = build_stabilized_basis(samples,order,runtime_dtype=torch.float64)
    index = torch.arange(samples,dtype=torch.float64)[:,None] + .5
    frequency = torch.arange(order,dtype=torch.float64)[None,:]
    reference = torch.cos(math.pi * index * frequency / samples) * (-1.) ** frequency
    reference *= math.sqrt(2. / samples);reference[:,0] /= math.sqrt(2.)
    torch.testing.assert_close(basis,reference,atol=2e-11,rtol=0)
    torch.testing.assert_close(basis.T@basis,torch.eye(order,dtype=basis.dtype),atol=2e-11,rtol=0)


def test_nodes_are_monotone_inside_endpoints_and_cluster_near_them():
    nodes = normalized_coordinates(32)
    assert -1 < nodes[0] < nodes[-1] < 1
    gaps = nodes[1:] - nodes[:-1]
    assert bool((gaps > 0).all())
    assert gaps[0] < gaps[len(gaps)//2] / 5
    torch.testing.assert_close(nodes,-nodes.flip(0),atol=1e-15,rtol=0)


def test_curve_evaluation_matches_executed_root_samples_and_has_finite_gradients():
    nodes = normalized_coordinates(16).requires_grad_()
    evaluated = stabilized_chebyshev_basis_at_coordinates(nodes,order=12,reference_sample_count=16)
    reference = build_stabilized_basis(16,12,runtime_dtype=torch.float64)
    torch.testing.assert_close(evaluated,reference,atol=1e-14,rtol=0)
    evaluated.square().sum().backward()
    assert bool(torch.isfinite(nodes.grad).all())


def test_metadata_identifies_new_policy_and_validation_rejects_corruption():
    kernel = get_basis_kernel("chebyshev")
    assert kernel.basis_version == CHEBYSHEV_BASIS_VERSION == "chebyshev_first_kind_roots_v1"
    assert "roots" in kernel.coordinate_policy and "no_qr" in kernel.stabilization_policy
    raw = kernel.raw_basis(kernel.coordinates(8),8)
    with pytest.raises(ValueError,match="orthogonality"):
        kernel.stabilize(raw * 2)
    with pytest.raises((ValueError,FloatingPointError),match="non-finite"):
        kernel.stabilize(raw.fill_(float("nan")))
    with pytest.raises(ValueError,match="exceed"):
        build_stabilized_basis(8,9)
    with pytest.raises(ValueError,match="basis_version"):
        normalize_basis_version("chebyshev","chebyshev_first_kind_qr_v1")


def test_optimizer_histories_use_the_shared_root_basis_without_qr(monkeypatch):
    from sheet.thogopt_math import history_basis
    monkeypatch.setattr(torch.linalg, "qr", lambda *args, **kwargs: pytest.fail("history construction invoked QR"))
    actual = history_basis(16, 12, dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(actual, build_stabilized_basis(16, 12), atol=0, rtol=0)
    torch.testing.assert_close(history_basis(16,16,dtype=torch.float64,device=torch.device("cpu")),
                               torch.eye(16,dtype=torch.float64),atol=0,rtol=0)


def test_optimizer_history_curves_pass_through_every_executed_root_sample():
    import numpy as np
    from sheet.thogopt_dashboard import _curve
    from sheet.thogopt_math import history_basis
    basis = history_basis(7,5,dtype=torch.float64,device=torch.device("cpu"))
    coefficients = torch.tensor([.3,.2,-.1,.05,.01],dtype=torch.float64)
    values = (basis @ coefficients).tolist()
    family = {"values":{"raw_momentum":[values]},"q_m":basis.tolist(),
              "momentum_coefficients":[coefficients.tolist()]}
    layer_axis, curve = _curve(family,"raw_momentum",0)
    assert layer_axis[0] == 1 and layer_axis[-1] == 7
    np.testing.assert_allclose([curve[layer_axis.index(layer)] for layer in range(1,8)], values, atol=1e-14,rtol=0)
# ^^^ THOG
