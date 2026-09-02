# vvv THOG
"""Ordering/lifetime tests plus CUDA tests runnable on the training machines."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from sheet.thogopt_transfers import HostGradientTransfers, candidate_memory_budget
from tests.test_thogopt import make_optimizer, supply


def test_transient_budget_respects_released_peak_physical_memory_and_reserves():
    assert candidate_memory_budget(allocated=400,reserved=900,peak=800,free=300,missing_state=0,workspace=100)==300
    assert candidate_memory_budget(allocated=400,reserved=400,peak=800,free=200,missing_state=0,workspace=100)==100
    assert candidate_memory_budget(allocated=400,reserved=900,peak=400,free=300,missing_state=0,workspace=100)==300
    assert candidate_memory_budget(allocated=400,reserved=900,peak=800,free=300,missing_state=500,workspace=100)==0
    # Even abundant aggregate headroom cannot justify placing candidates in
    # backward's cached blocks while permanent histories are being initialized.
    assert candidate_memory_budget(allocated=400,reserved=9000,peak=8000,free=3000,missing_state=1,workspace=100)==0


def test_deferred_drain_waits_before_consuming_in_fifo_order_and_discards():
    # CUDA-free test of the actual consumer/order path. Values appear only when
    # the copy's event is synchronized; non-associative summands expose reordering.
    queue = object.__new__(HostGradientTransfers)
    queue.buffers = [torch.zeros(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64)]
    destination = torch.tensor([1e16],dtype=torch.float64)
    queue.destinations = [destination,destination]
    queue.additions = [True,True]
    queue.next_slot = 1
    order = []
    def event(slot,value):
        def synchronize():
            order.append(slot);queue.buffers[slot].fill_(value)
        return SimpleNamespace(synchronize=synchronize)
    queue.events = [event(0,1.),event(1,-1e16)]
    queue.drain()
    assert order == [1,0]
    assert destination.item()==1.
    assert queue.destinations == [None,None]
    queue.destinations = [destination,None]
    queue.drain(discard=True)
    assert destination.item()==1.
    assert queue.destinations == [None,None]


def test_legacy_checkpoint_loads_with_new_execution_controls():
    _,parameter,optimizer=make_optimizer(transaction_storage="host",async_staging=False)
    supply(optimizer,torch.ones(4,4,dtype=torch.float64));optimizer.step()
    saved=deepcopy(optimizer.state_dict())
    for key in ('transaction_storage','async_gradient_staging','staging_fallback_reason'):
        saved['thogopt'].pop(key)
    saved['thogopt']['staging']='host_accumulation_and_transaction'
    _,other_parameter,other=make_optimizer()
    with torch.no_grad(): other_parameter.copy_(parameter)
    other.load_state_dict(saved)
    gradient=torch.randn(4,4,dtype=torch.float64)
    for current in (optimizer,other): supply(current,gradient);current.step()
    torch.testing.assert_close(parameter,other_parameter,rtol=0,atol=0)
    assert 'candidate=' in other.timing_summary()
    assert 'async_staging=no' in other.timing_summary()


def test_pending_gradients_drain_before_clipping_and_discard_before_next_update():
    class PendingCopies:
        def __init__(self): self.pending=[]
        def enqueue(self,source,destination,*,add=True): self.pending.append((source.clone().reshape(-1),destination,add))
        def drain(self,*,discard=False):
            if not discard:
                for source,destination,add in self.pending:
                    if add: destination.add_(source)
                    else: destination.copy_(source)
            self.pending.clear()
    _,parameter,optimizer=make_optimizer()
    _,reference_parameter,reference=make_optimizer()
    optimizer.gradient_transfers=PendingCopies()
    with torch.no_grad(): reference_parameter.copy_(parameter)
    gradient=torch.randn(4,4,dtype=torch.float64)
    supply(optimizer,gradient)
    assert len(optimizer.gradient_transfers.pending)==4
    optimizer.zero_grad()
    assert not optimizer.gradient_transfers.pending
    for current in (optimizer,reference):
        supply(current,gradient)
        current.prepare_gradients(loss_scale=3.,grad_clip=.5)
        current.step()
    torch.testing.assert_close(parameter,reference_parameter,rtol=0,atol=0)
    for key in ('exp_avg','exp_avg_sq'):
        torch.testing.assert_close(optimizer.state[parameter][key],reference.state[reference_parameter][key],rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA training-machine validation required")
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_cuda_async_staging_preserves_order_and_source_lifetime(dtype):
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported(): pytest.skip('BF16 unsupported')
    queue=HostGradientTransfers(capacity=4096,dtype=torch.float32,device='cuda:0')
    actual=torch.zeros(4096)
    expected=torch.zeros_like(actual)
    for index in range(19):
        source=(torch.arange(4096,device='cuda',dtype=torch.float32).reshape(64,64).T+index).to(dtype)
        expected.add_(source.cpu().float().reshape(-1))
        queue.enqueue(source,actual)
        del source
        # Make allocator reuse likely while the copy is in flight.
        temporary=torch.empty(4096,device='cuda',dtype=dtype);temporary.fill_(-123)
    queue.drain()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert queue.pinned_bytes==2*4096*4
    queue.enqueue(torch.ones(4096,device='cuda'),actual)
    queue.drain(discard=True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA training-machine validation required")
def test_cuda_device_candidates_match_host_path_and_reject_atomically(monkeypatch):
    from sheet.thogopt import Thogopt
    trajectory,_,_=make_optimizer(dtype=torch.float32)
    trajectory=trajectory.to('cuda')
    parameter=trajectory.coefficients['attention_query_weight']
    optimizer=Thogopt([parameter],trajectory=trajectory,lr=.002,weight_decay=.03,betas=(.9,.95))
    cache_releases=[]
    empty_cache=torch.cuda.empty_cache
    def release_cache():
        cache_releases.append(True)
        empty_cache()
    monkeypatch.setattr(torch.cuda,'empty_cache',release_cache)
    _,reference_parameter,reference=make_optimizer(dtype=torch.float32,async_staging=False,transaction_storage='host')
    with torch.no_grad():reference_parameter.copy_(parameter.cpu())
    pointers = None
    for index in range(8):
        # Leave a large cached allocation for candidate placement to reuse.
        backward_scratch=torch.empty(64*2**20,device='cuda',dtype=torch.uint8)
        backward_scratch.fill_(1)
        del backward_scratch
        gradient=torch.randn(4,4)
        for current,values in ((optimizer,gradient.cuda()),(reference,gradient)):
            supply(current,values);current.step()
        torch.testing.assert_close(parameter.cpu(),reference_parameter,atol=2e-6,rtol=2e-5)
        current_pointers = tuple(optimizer.state[parameter][key].data_ptr() for key in ('exp_avg','exp_avg_sq'))
        if index == 0:
            assert optimizer.last_step_metrics['transaction_device_candidate_bytes']==0
            pointers = current_pointers
            # Force later updates through device candidates independent of
            # unrelated peaks, after the real initialization policy has run.
            optimizer._transaction_budget=lambda: 2**20
        else:
            assert current_pointers == pointers
        assert len(cache_releases)==1
    assert optimizer.last_step_metrics['transaction_device_candidate_bytes']>0
    before=parameter.detach().clone();state=deepcopy(optimizer.state_dict())
    supply(optimizer,torch.full((4,4),1e30,device='cuda'))
    with pytest.raises(FloatingPointError):optimizer.step()
    torch.testing.assert_close(parameter,before,atol=0,rtol=0)
    for key in ('exp_avg','exp_avg_sq'):
        torch.testing.assert_close(optimizer.state[parameter][key],state['state'][0][key],atol=0,rtol=0)
# ^^^ THOG



def test_reused_gradient_storage_overwrites_first_contribution_without_stale_values():
    _,parameter,optimizer=make_optimizer()
    supply(optimizer,torch.full((4,4),7.,dtype=torch.float64));optimizer.step()
    address=optimizer.raw_gradients['attention_query_weight'].data_ptr()
    supply(optimizer,torch.full((4,4),2.,dtype=torch.float64))
    optimizer.accumulate_layer_gradient('attention_query_weight',0,torch.ones(4,dtype=torch.float64))
    assert optimizer.raw_gradients['attention_query_weight'].data_ptr()==address
    expected=torch.full((4,4),2.,dtype=torch.float64);expected[0].add_(1.)
    torch.testing.assert_close(optimizer.raw_gradients['attention_query_weight'],expected,atol=0,rtol=0)


def test_persistent_moments_keep_storage_and_release_compressed_candidates(monkeypatch):
    import weakref
    _,parameter,optimizer=make_optimizer()
    ordinary=torch.nn.Parameter(torch.ones(7,dtype=torch.float64))
    optimizer.add_param_group({'params':[ordinary]})
    candidates=[]
    original = optimizer._compressed_candidate
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        candidates.extend(weakref.ref(result[1][key]) for key in ('exp_avg','exp_avg_sq'))
        return result
    monkeypatch.setattr(optimizer,'_compressed_candidate',capture)
    pointers=None
    for _ in range(4):
        supply(optimizer,torch.randn(4,4,dtype=torch.float64))
        ordinary.grad=torch.randn_like(ordinary)
        optimizer.step()
        current=tuple(optimizer.state[p][key].data_ptr() for p in (parameter,ordinary) for key in ('exp_avg','exp_avg_sq'))
        if pointers is not None: assert current==pointers
        pointers=current
        # Same-device candidates must not silently become permanent histories.
        assert all(reference() is None for reference in candidates)
        candidates.clear()


def test_late_state_allocation_failure_leaves_existing_state_and_weights_unchanged(monkeypatch):
    _,parameter,optimizer=make_optimizer()
    ordinary=torch.nn.Parameter(torch.ones(7,dtype=torch.float64))
    optimizer.add_param_group({'params':[ordinary]})
    supply(optimizer,torch.ones(4,4,dtype=torch.float64));optimizer.step()
    before=deepcopy(optimizer.state_dict())
    weights=[p.detach().clone() for p in (parameter,ordinary)]
    pointers=tuple(optimizer.state[parameter][key].data_ptr() for key in ('exp_avg','exp_avg_sq'))
    supply(optimizer,torch.full((4,4),2.,dtype=torch.float64))
    ordinary.grad=torch.ones_like(ordinary)
    real_empty_like=torch.empty_like
    allocations=[]
    def fail_second_allocation(value,**kwargs):
        allocations.append(value.shape)
        if len(allocations)==2: raise torch.OutOfMemoryError('injected state allocation failure')
        return real_empty_like(value,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'empty_like',fail_second_allocation)
        with pytest.raises(torch.OutOfMemoryError,match='injected state allocation'):
            optimizer.step()
    assert len(allocations)==2
    assert not optimizer.state.get(ordinary)
    for p,weight in zip((parameter,ordinary),weights):
        torch.testing.assert_close(p,weight,rtol=0,atol=0)
    for key,pointer in zip(('exp_avg','exp_avg_sq'),pointers):
        assert optimizer.state[parameter][key].data_ptr()==pointer
        torch.testing.assert_close(optimizer.state[parameter][key],before['state'][0][key],rtol=0,atol=0)
    assert optimizer.state[parameter]['step']==1
