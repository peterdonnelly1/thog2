# vvv THOG real non-reentrant checkpoint replays mix native and cached paths without changing RNG, gradients or Adam updates
from contextlib import contextmanager
import copy
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from sheet.depth_numerical_policy import CpuDepthBinding, PrematBindingContext, materialize_cpu
from tests.test_cpu_materialisation import policy


@pytest.mark.parametrize("backend", ("matmul","einsum"))
@pytest.mark.parametrize("learned_depth", (False,True))
@pytest.mark.parametrize("original_cpu,replay_cpu", ((False,False),(True,False),(False,True),(True,True)))
def test_actual_checkpoint_segments_mixed_binding_matches_native_updates(backend,learned_depth,original_cpu,replay_cpu):
    torch.manual_seed(301)
    coefficients=torch.nn.Parameter(torch.randn(8,8,3)*0.08)
    rows=torch.nn.Parameter(torch.randn(4,3)*0.2,requires_grad=learned_depth)
    initial=(coefficients.detach().clone(),rows.detach().clone())
    def run(cpu_forward,cpu_replay):
        c=torch.nn.Parameter(initial[0].clone())
        b=torch.nn.Parameter(initial[1].clone(),requires_grad=learned_depth)
        optimizer=torch.optim.Adam([c]+([b] if learned_depth else []),lr=1e-3)
        phase=["original_forward"];calls=[];cache={};gradients=[];losses=[]
        @contextmanager
        def enter(name):
            old=phase[0];phase[0]=name
            try:yield
            finally:phase[0]=old
        def block(x,layer):
            cpu=cpu_forward if phase[0]=="original_forward" else cpu_replay
            calls.append((phase[0],layer,cpu))
            if cpu:
                key=(c._version,b._version,layer)
                if key not in cache:cache[key]=materialize_cpu(c,b[layer],policy(backend))
                w=CpuDepthBinding.apply(cache[key],PrematBindingContext(policy(backend)),c,b[layer])
            else:
                w=c.reshape(-1,3).matmul(b[layer]).reshape(8,8) if backend=="matmul" else torch.einsum("p,rcp->rc",b[layer],c)
            return x+torch.nn.functional.dropout(torch.nn.functional.linear(x,w).tanh(),p=0.2,training=True)
        torch.manual_seed(909)
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                x=torch.randn(2,4,8,requires_grad=True)
                for indices in ((0,1),(2,3)):
                    def segment(value,indices=indices):
                        for layer in indices:value=block(value,layer)
                        return value
                    x=checkpoint(segment,x,use_reentrant=False,context_fn=lambda:(enter("original_forward"),enter("checkpoint_recompute")))
                loss=x.square().mean()/2
                losses.append(loss.detach())
                loss.backward()
            gradients.append((c.grad.clone(),None if not learned_depth else b.grad.clone()))
            optimizer.step()
        return losses,gradients,(c.detach(),b.detach()),optimizer.state_dict(),torch.get_rng_state(),calls,len(cache)
    native=run(False,False);mixed=run(original_cpu,replay_cpu)
    for left,right in zip(native[0],mixed[0]):torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
    for left,right in zip(native[1],mixed[1]):
        torch.testing.assert_close(left[0],right[0],atol=3e-6,rtol=3e-5)
        if learned_depth:torch.testing.assert_close(left[1],right[1],atol=3e-6,rtol=3e-5)
    for left,right in zip(native[2],mixed[2]):torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
    for key,state in native[3]["state"].items():
        for name,value in state.items():torch.testing.assert_close(value,mixed[3]["state"][key][name],atol=3e-6,rtol=3e-5)
    assert torch.equal(native[4],mixed[4])
    assert any(phase=="checkpoint_recompute" for phase,_,_ in mixed[5])
    assert all(cpu==(original_cpu if phase=="original_forward" else replay_cpu) for phase,_,cpu in mixed[5])
    assert mixed[6]==(12 if original_cpu or replay_cpu else 0)
# ^^^ THOG
