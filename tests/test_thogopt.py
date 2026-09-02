# vvv THOG
from copy import deepcopy
import numpy as np
import pytest
import torch
from scipy.optimize import LinearConstraint, minimize

from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.thogopt import Thogopt
from sheet.thogopt_math import fit_nonnegative, history_basis, resolve_history_count


torch.set_num_threads(1)


def make_optimizer(layers=4, order=4, history=None, dtype=torch.float64, **kwargs):
    geometry = SheetGeometryConfig(n_layer=layers, n_embd=2, n_head=1, depth_order=order,
        base_row_order=1, mlp_channel_order=1, o_attn_d_model=1, o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1, o_mlp_d_model=1, o_mlp_hidden=1, bias=True)
    trajectory = DepthTrajectory(geometry, runtime_dtype=dtype, depth_compress_layer_norm_and_bias=False)
    for name, parameter in trajectory.coefficients.items():
        parameter.requires_grad_(name == "attention_query_weight")
    parameter = trajectory.coefficients["attention_query_weight"]
    optimizer = Thogopt([parameter], trajectory=trajectory, lr=0.002, weight_decay=0.03,
        betas=(0.9, 0.95), momentum_history_coefficients=history or layers,
        scaling_history_coefficients=history or layers, tile_elements=8, **kwargs)
    return trajectory, parameter, optimizer


def supply(optimizer, gradient, scale=1.):
    optimizer.zero_grad()
    for layer, row in enumerate(gradient):
        optimizer.accumulate_layer_gradient("attention_query_weight", layer, row * scale)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_full_capacity_matches_adamw_500_updates(dtype):
    torch.manual_seed(54)
    trajectory, parameter, optimizer = make_optimizer(dtype=dtype)
    dense = torch.nn.Parameter(optimizer.weight_basis @ parameter.detach().reshape(-1,4).T)
    reference = torch.optim.AdamW([dense], lr=.002, betas=(.9,.95), weight_decay=.03, eps=1e-8)
    tolerance = dict(atol=1e-10, rtol=1e-9) if dtype==torch.float64 else dict(atol=2e-6, rtol=2e-5)
    for step in range(500):
        gradient = torch.randn(4,4,dtype=dtype) * (1e-9 if step < 10 else 1.)
        supply(optimizer, gradient)
        optimizer.step()
        dense.grad = gradient.clone()
        reference.step()
        actual = optimizer.weight_basis @ parameter.detach().reshape(-1,4).T
        torch.testing.assert_close(actual, dense, **tolerance)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[parameter][key], reference.state[dense][key], **tolerance)


def test_raw_capture_adds_microbatch_contributions_before_square():
    trajectory, parameter, optimizer = make_optimizer()
    for layer in range(4):
        trajectory.materialize("attention_query_weight", layer).sum().mul(8).backward()
        trajectory.materialize("attention_query_weight", layer).sum().mul(-4).backward()
    optimizer.prepare_gradients(loss_scale=4)
    optimizer.step()
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], torch.full((4,4), .05,dtype=torch.float64))


def test_compressed_fit_matches_independent_scipy_constrained_reference():
    q = history_basis(9, 4, dtype=torch.float64, device=torch.device("cpu"))
    target = torch.zeros(9, 3,dtype=torch.float64)
    target[0,0] = 1
    target[4,1] = .1
    target[8,2] = 1e-12
    actual, metrics = fit_nonnegative(q,target)
    assert metrics["constrained_columns"] > 0
    basis = q.numpy()
    for index in range(3):
        scale = float(target[:,index].max())
        y = target[:,index].numpy()/scale
        result = minimize(lambda a: .5*np.sum((basis@a-y)**2), basis.T@y,
            jac=lambda a: basis.T@(basis@a-y), constraints=LinearConstraint(basis,0,np.inf),
            method="SLSQP", options={"ftol":1e-14,"maxiter":2000})
        assert result.success
        np.testing.assert_allclose((q@actual[:,index]).numpy()/scale,basis@result.x,atol=1e-9,rtol=1e-9)
    assert float((q@actual).min()) >= -1e-14


def test_exact_histories_with_compressed_weights_match_projected_oracle():
    torch.manual_seed(33)
    trajectory, parameter, optimizer = make_optimizer(layers=6,order=2)
    weights = optimizer.weight_basis @ parameter.detach().reshape(-1,2).T
    m = torch.zeros_like(weights)
    v = torch.zeros_like(weights)
    for step in range(1,41):
        gradient = torch.randn_like(weights)
        supply(optimizer, gradient)
        optimizer.step()
        m = .9*m + .1*gradient
        v = .95*v + .05*gradient.square()
        update = -.002*(m/(1-.9**step))/((v/(1-.95**step)).sqrt()+1e-8)
        weights = (1-.002*.03)*weights + optimizer.weight_basis@optimizer.weight_analysis@update
        torch.testing.assert_close(optimizer.weight_basis@parameter.detach().reshape(-1,2).T,weights,atol=1e-10,rtol=1e-9)


def test_checkpoint_resume_and_rejected_size_change():
    trajectory, parameter, optimizer = make_optimizer(layers=6,order=2,history=3)
    supply(optimizer, torch.ones(6,4,dtype=torch.float64))
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    other_trajectory, other_parameter, other = make_optimizer(layers=6,order=2,history=3)
    with torch.no_grad(): other_parameter.copy_(parameter)
    other.load_state_dict(saved)
    gradient = torch.randn(6,4,dtype=torch.float64)
    supply(optimizer,gradient)
    supply(other,gradient)
    optimizer.step()
    other.step()
    torch.testing.assert_close(parameter,other_parameter,atol=0,rtol=0)
    _, _, incompatible = make_optimizer(layers=6,order=2,history=4)
    with pytest.raises(ValueError,match="incompatible thogopt"):
        incompatible.load_state_dict(saved)


def test_nonfinite_candidate_leaves_all_state_unchanged():
    _, parameter, optimizer = make_optimizer()
    supply(optimizer,torch.ones(4,4,dtype=torch.float64))
    optimizer.step()
    values = parameter.detach().clone()
    state = deepcopy(optimizer.state_dict())
    supply(optimizer,torch.full((4,4),1e200,dtype=torch.float64))
    with pytest.raises(FloatingPointError): optimizer.step()
    torch.testing.assert_close(values,parameter,atol=0,rtol=0)
    assert state["state"][0]["step"] == optimizer.state[parameter]["step"]
    torch.testing.assert_close(state["state"][0]["exp_avg"],optimizer.state[parameter]["exp_avg"],atol=0,rtol=0)


def test_reference_is_observational_and_resumable():
    _, parameter, optimizer = make_optimizer()
    optimizer.configure_reference({"attention_query_weight": [0,2]})
    supply(optimizer,torch.randn(4,4,dtype=torch.float64))
    optimizer.step()
    reference=optimizer.reference_histories["attention_query_weight"]
    torch.testing.assert_close(reference["exp_avg"],optimizer.state[parameter]["exp_avg"][:,[0,2]])
    assert "thogopt_reference" in optimizer.state_dict()


def test_independent_auto_counts():
    assert resolve_history_count("auto",nominal=31,layers=16)=={"requested":"auto","nominal":31,"effective":16}
    assert resolve_history_count(7,nominal=31,layers=16)["effective"]==7
    with pytest.raises(ValueError): resolve_history_count(17,nominal=31,layers=16)
# ^^^ THOG


@pytest.mark.parametrize("fast_discard", [True, False])
def test_production_training_accumulation_and_resume(monkeypatch, tmp_path, fast_discard):
    from sheet.stage4_trainer import Stage4Trainer
    from tests.stage4_test_support import stage4_training_config, stage4_tokens
    monkeypatch.setenv("THOG2_OPTIMIZER", "thogopt")
    monkeypatch.setenv("THOG2_FAST_DISCARD", str(fast_discard).lower())
    config = stage4_training_config(geometry_preset="depth", depth_order=3,
        gradient_accumulation_steps=2, max_updates=5)
    tokens = stage4_tokens()
    trainer = Stage4Trainer(config, *tokens)
    assert isinstance(trainer.optimizer, Thogopt)
    trainer.train_one_update()
    assert trainer.state.completed_updates == 1
    assert len(trainer.optimizer.state) > len(trainer.optimizer.families)
    path = tmp_path / "resume.pt"
    trainer.save_checkpoint(path)
    resumed = Stage4Trainer.from_checkpoint(path, *tokens)
    trainer.train_one_update()
    resumed.train_one_update()
    for left, right in zip(trainer.raw_model.parameters(), resumed.raw_model.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize('betas',[(0.,.95),(.9,0.),(0.,0.)])
def test_zero_beta_zero_gradient_and_absent_gradient_semantics(betas):
    _,parameter,optimizer=make_optimizer()
    optimizer.param_groups[0]['betas']=betas
    dense=torch.nn.Parameter(optimizer.weight_basis@parameter.detach().reshape(-1,4).T)
    reference=torch.optim.AdamW([dense],lr=.002,betas=betas,weight_decay=.03)
    for step in range(12):
        gradient=torch.zeros(4,4,dtype=torch.float64) if step%3 else torch.full((4,4),(-1.)**step,dtype=torch.float64)
        supply(optimizer,gradient);optimizer.step()
        dense.grad=gradient;reference.step()
        torch.testing.assert_close(optimizer.weight_basis@parameter.detach().reshape(-1,4).T,dense,atol=1e-10,rtol=1e-9)
    before=parameter.detach().clone();count=optimizer.state[parameter]['step']
    optimizer.zero_grad();optimizer.step()
    torch.testing.assert_close(parameter,before,atol=0,rtol=0)
    assert optimizer.state[parameter]['step']==count


def test_host_staging_budget_rejected_before_capture_install(monkeypatch):
    monkeypatch.setenv('THOG2_THOGOPT_HOST_BUDGET_MIB','0.000001')
    with pytest.raises(MemoryError,match='host staging'):make_optimizer()


def test_physical_gradient_clipping_excludes_coefficient_duplicate():
    _,parameter,optimizer=make_optimizer()
    supply(optimizer,torch.full((4,4),2.,dtype=torch.float64))
    parameter.grad=torch.full_like(parameter,1e6)
    assert optimizer.prepare_gradients(grad_clip=4.)==8.
    optimizer.step()
    torch.testing.assert_close(optimizer.state[parameter]['exp_avg_sq'],torch.full((4,4),.05*(2*4/(8+1e-6))**2,dtype=torch.float64))


def test_ordinary_half_parameter_retains_fp32_moments():
    _,_,optimizer=make_optimizer()
    parameter=torch.nn.Parameter(torch.ones(2,dtype=torch.float16))
    optimizer.add_param_group({'params':[parameter]})
    parameter.grad=torch.full_like(parameter,1e-4)
    optimizer.step()
    state=optimizer.state[parameter]
    assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
    assert bool((state['exp_avg_sq']>0).all())
