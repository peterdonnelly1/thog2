# vvv THOG
from __future__ import annotations

import torch
from torch import nn

from sheet.distributed import DistributedContext, DistributedIdentity


def test_manually_projected_gradient_is_meaned_across_ranks(
    monkeypatch,
) -> None:
    context = DistributedContext(
        identity=DistributedIdentity(
            active=True,
            backend="gloo",
            rank=0,
            local_rank=0,
            world_size=2,
            device="cpu",
        ),
        device=torch.device("cpu"),
        owns_process_group=False,
    )
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([4.0])

    def fake_all_reduce(tensor, *, op) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        tensor.add_(torch.tensor([5.0]))

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        fake_all_reduce,
    )
    context.mean_gradients_((parameter,))
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([4.5]),
        rtol=0.0,
        atol=0.0,
    )


def test_manual_gradient_mean_is_noop_without_distributed_training() -> None:
    context = DistributedContext(
        identity=DistributedIdentity(
            active=False,
            backend="none",
            rank=0,
            local_rank=0,
            world_size=1,
            device="cpu",
        ),
        device=torch.device("cpu"),
        owns_process_group=False,
    )
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([7.0])
    context.mean_gradients_((parameter,))
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([7.0]),
        rtol=0.0,
        atol=0.0,
    )
# ^^^ THOG
