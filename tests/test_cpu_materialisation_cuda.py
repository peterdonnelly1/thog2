# vvv THOG run on each supported CUDA/PyTorch host; CPU CI reports skips, never GPU qualification
import gc
import time
from contextlib import nullcontext
import pytest
import torch
from sheet.depth_numerical_policy import CpuDepthBinding, PrematBindingContext, effective_depth_policy, materialize_cpu
from sheet.premat_cpu import FAMILY_NAMES
from sheet.premat_cpu_config import CPU_DEFAULTS
from sheet.premat_cpu_runtime import CpuPrematRuntime
from tests.test_cpu_materialisation import tiny_trajectory
from tests.test_cpu_materialisation_checkpoint import assert_mixed_checkpoint

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA qualification requires an actual GPU")


@pytest.fixture(autouse=True)
def exact_math_flags():
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cuda.matmul.allow_tf32=old


@pytest.mark.parametrize("backend",("matmul","einsum"))
@pytest.mark.parametrize("dtype",("float32","float16","bfloat16"))
@pytest.mark.parametrize("family",tuple(FAMILY_NAMES))
@pytest.mark.parametrize("learned_depth",(False,True))
def test_cuda_native_values_saved_tensors_coefficient_basis_and_input_gradients(backend,dtype,family,learned_depth,recwarn):
    if dtype=="bfloat16" and not torch.cuda.is_bf16_supported():pytest.skip("GPU has no BF16 support")
    trajectory=tiny_trajectory()
    trajectory.coefficients={name:torch.nn.Parameter(value.detach().cuda()) for name,value in trajectory.coefficients.items()}
    trajectory.depth_basis=trajectory.depth_basis.cuda()
    trajectory.depth_materialisation_matmul=backend=="matmul"
    originals=[trajectory.coefficients[name] for name in FAMILY_NAMES[family]]
    copies=[torch.nn.Parameter(value.detach().clone()) for value in originals]
    row=trajectory.depth_basis[2].detach().requires_grad_(learned_depth)
    copy_row=row.detach().clone().requires_grad_(learned_depth)
    native_saved=[];bound_saved=[]
    def hooks(destination):
        def pack(value):
            destination.append((tuple(value.shape),value.dtype,value.device))
            return value
        return torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value)
    with torch.autocast("cuda",dtype=getattr(torch,dtype),enabled=dtype!="float32"):
        contract=effective_depth_policy(trajectory)
        def native(coefficient):
            if backend=="matmul":
                with torch.autocast("cuda",enabled=False):
                    return coefficient.reshape(-1,3).matmul(row).reshape(coefficient.shape[:2])
            return torch.einsum("p,rcp->rc",row,coefficient)
        with hooks(native_saved):expected=torch.cat([native(value) for value in originals])
        cached=torch.cat([materialize_cpu(value.detach().cpu(),copy_row.detach().cpu(),contract) for value in copies]).cuda()
        pairs=tuple(item for coefficient in copies for item in (coefficient,copy_row))
        with hooks(bound_saved):actual=CpuDepthBinding.apply(cached,PrematBindingContext(contract),*pairs)
        assert actual.dtype==expected.dtype and native_saved==bound_saved
        atol,rtol=(3e-6,3e-5) if dtype=="float32" else (2e-4,0.015) if dtype=="float16" else (2e-3,0.04)
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        x=torch.randn(2,4,expected.shape[1],device="cuda",requires_grad=True);x2=x.detach().clone().requires_grad_()
        loss=torch.nn.functional.linear(x,expected).float().square().mean()
        loss2=torch.nn.functional.linear(x2,actual).float().square().mean()
    loss.backward();loss2.backward()
    torch.testing.assert_close(x.grad,x2.grad,atol=atol,rtol=rtol)
    for left,right in zip(originals,copies):torch.testing.assert_close(left.grad,right.grad,atol=atol,rtol=rtol)
    if learned_depth:torch.testing.assert_close(row.grad,copy_row.grad,atol=atol,rtol=rtol)
    assert not any("stream" in str(w.message).lower() and ("grad" in str(w.message).lower() or "provenance" in str(w.message).lower()) for w in recwarn)


@pytest.mark.parametrize("backend",("matmul","einsum"))
@pytest.mark.parametrize("dtype",("float32","float16","bfloat16"))
@pytest.mark.parametrize("learned_depth",(False,True))
@pytest.mark.parametrize("original_cpu,replay_cpu",((False,False),(True,False),(False,True),(True,True)))
def test_cuda_actual_checkpoint_all_four_source_combinations_and_adam(backend,dtype,learned_depth,original_cpu,replay_cpu):
    if dtype=="bfloat16" and not torch.cuda.is_bf16_supported():pytest.skip("GPU has no BF16 support")
    assert_mixed_checkpoint(backend,learned_depth,original_cpu,replay_cpu,device="cuda",dtype=dtype)


def test_cuda_real_worker_snapshot_completed_upload_and_backward_storage():
    trajectory=tiny_trajectory()
    trajectory.coefficients={name:torch.nn.Parameter(value.detach().cuda()) for name,value in trajectory.coefficients.items()}
    trajectory.depth_basis=trajectory.depth_basis.cuda()
    trajectory.depth_materialisation_matmul=True
    def native(family,layer):
        coefficient=trajectory.coefficients[FAMILY_NAMES[family][0]]
        return coefficient.reshape(-1,3).matmul(trajectory.depth_basis[layer]).reshape(coefficient.shape[:2])
    def attach(family,layer,matrix,*,binding_context):
        pairs=tuple(item for name in FAMILY_NAMES[family] for item in (trajectory.coefficients[name],trajectory.depth_basis[layer]))
        return CpuDepthBinding.apply(matrix,binding_context,*pairs)
    runtime=CpuPrematRuntime(trajectory=trajectory,cpu_configuration={**CPU_DEFAULTS,"premat_materialisation_device":"cpu_and_gpu","premat_cpu_threads_per_worker":1,"premat_cpu_transfer_timing":"as_soon_as_ready"},materialize=native,attach=attach,n_embd=8,n_head=2,attention_mode="fused",target_matrix=2,stay_below_current_peak=False,gpu_memory_buffer_gb=0,target_layer=0,weight_matrix_target_order="r_to_l",cuda_stream_priority="normal",diagnostic_layer_delay_ms=0,enable_gpu_timing_diagnostic=False,logging_enabled=True,shadow_mode=False)
    try:
        runtime.optimizer_step=1;runtime.begin((0,1,2,3),reference=torch.randn(2,4,8,device="cuda"))
        runtime.layer_start(0)
        # Qualification forces the success path before readiness, outside any measured update.
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            runtime._wake()
            uploads=list(runtime.cpu_uploads.values())
            if uploads and any(item.metadata["layer_index"]==0 and item.completion is not None and item.completion.query() for item in uploads):break
            time.sleep(0.002)
        weight=runtime.acquire("O",0)
        assert runtime._candidates[(0,"O")].final_outcome=="FULL HIT"
        x=torch.randn(2,4,8,device="cuda",requires_grad=True)
        y=torch.nn.functional.linear(x,weight)
        runtime.consumed("O",0);runtime.end()
        assert runtime.report()["cpu_runtime"]["autograd_retained_bytes"]>=256
        snapshot=runtime.cpu_snapshot.snapshot_id
        runtime.begin((0,1,2,3),reference=x)
        assert runtime.cpu_snapshot.snapshot_id==snapshot
        source_events=[event for event in runtime.cpu_events if event["event"]=="snapshot_d2h_submitted"]
        assert len(source_events)==2 and {event["source_name"] for event in source_events}=={"depth_basis","attention_output_weight"}
        runtime.end();y.square().mean().backward()
        assert x.grad is not None and trajectory.coefficients["attention_output_weight"].grad.device.type=="cuda"
        del y,weight;gc.collect();torch.cuda.synchronize()
        runtime._refresh_available()
        assert not any(item.consumed for item in runtime.cpu_uploads.values())
        assert all(not event["cuda_initialized"] for event in runtime.cpu_provider.report()["worker_health"])
    finally:runtime.close()
# ^^^ THOG
