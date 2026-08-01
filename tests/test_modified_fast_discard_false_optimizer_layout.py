# vvv THOG
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from sheet.update_retained_materializations import (
    attach_update_retained_materializations,
)


class _AutocastProjectionTrajectory(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(
            torch.arange(
                12,
                dtype=torch.float32,
                device=device,
            ).view(4, 3)
        )
        self.register_buffer(
            "basis",
            torch.tensor(
                [[1.0, 0.5, -0.25], [-0.5, 0.25, 1.0]],
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if name != "weight":
            raise KeyError(name)
        if layer_index != 0:
            raise IndexError(layer_index)
        return torch.einsum(
            "ij,kj->ki",
            self.coefficient,
            self.basis,
        )


def _project_one_bfloat16_gradient(
    trajectory: _AutocastProjectionTrajectory,
    *,
    device_type: str,
) -> None:
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=True,
    )
    assert controller.begin() is True
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
    ):
        operational = trajectory.materialize("weight", 0)
    assert operational.dtype == torch.bfloat16
    operational.float().square().sum().backward()
    projected = controller.finalize()
    assert any(
        parameter is trajectory.coefficient
        for parameter in projected
    )


def test_bfloat16_projection_matches_compact_parameter_layout() -> None:
    trajectory = _AutocastProjectionTrajectory(torch.device("cpu"))
    _project_one_bfloat16_gradient(
        trajectory,
        device_type="cpu",
    )
    parameter = trajectory.coefficient
    gradient = parameter.grad
    assert gradient is not None
    assert gradient.dtype == parameter.dtype
    assert gradient.device == parameter.device
    assert gradient.layout == parameter.layout
    assert gradient.stride() == parameter.stride()
    assert gradient.is_contiguous() == parameter.is_contiguous()

    optimizer = torch.optim.AdamW(
        trajectory.parameters(),
        lr=1.0e-3,
    )
    optimizer.step()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_bfloat16_projection_is_accepted_by_fused_adam() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    trajectory = _AutocastProjectionTrajectory(device)
    _project_one_bfloat16_gradient(
        trajectory,
        device_type="cuda",
    )
    parameter = trajectory.coefficient
    gradient = parameter.grad
    assert gradient is not None
    assert gradient.dtype == parameter.dtype
    assert gradient.device == parameter.device
    assert gradient.layout == parameter.layout
    assert gradient.stride() == parameter.stride()

    optimizer = torch.optim.AdamW(
        trajectory.parameters(),
        lr=1.0e-3,
        fused=True,
    )
    optimizer.step()
# ^^^ THOG
