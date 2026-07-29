# vvv THOG
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .block_trajectory import BlockTrajectory
from .depth_trajectory import DepthTrajectory
from .jpeg_like_v1_trajectory import JpegLikeV1Trajectory
from .mlp_block_trajectory import MlpBlockTrajectory
from .semantic_materializer import MLP_EXPANSION_WEIGHT
from .trajectory import SheetTrajectory


@dataclass(frozen=True)
class CoefficientBank:
    name: str
    parameter: nn.Parameter
    order_axis: int

    @property
    def order_count(self) -> int:
        return int(self.parameter.shape[self.order_axis])

    @property
    def coefficients_per_order(self) -> int:
        return int(self.parameter.numel() // self.order_count)

    def order_slice(self, order: int) -> Tensor:
        if order < 0 or order >= self.order_count:
            raise IndexError(f"coefficient order out of range for {self.name}: {order}")
        index = [slice(None)] * self.parameter.ndim
        index[self.order_axis] = order
        return self.parameter[tuple(index)]


def _depth_trajectory_axis(trajectory: DepthTrajectory, name: str) -> Optional[int]:
    metadata = trajectory.family_metadata(name)
    representation = trajectory._representation(metadata)
    if representation == "depth_coefficients":
        return 2
    if representation == "legacy_sheet_col":
        return 1
    if representation == "conventional_per_layer":
        return None
    raise RuntimeError(f"unknown DEPTH representation for {name}: {representation!r}")


def coefficient_depth_axis(trajectory: nn.Module, name: str) -> Optional[int]:
    """Return the axis representing DEPTH coefficient order, or None for uncompressed state."""
    if isinstance(trajectory, JpegLikeV1Trajectory):
        if name == MLP_EXPANSION_WEIGHT:
            return 3
        return _depth_trajectory_axis(trajectory.depth, name)

    if isinstance(trajectory, BlockTrajectory):
        if name in trajectory._block_metadata_by_name:
            metadata = trajectory._block_metadata_by_name[name]
            return 0 if metadata.attention_head_axis == "none" else 1
        return _depth_trajectory_axis(trajectory.depth, name)

    if isinstance(trajectory, MlpBlockTrajectory):
        if name in trajectory._mlp_metadata_by_name:
            return 0
        return _depth_trajectory_axis(trajectory.depth, name)

    if isinstance(trajectory, DepthTrajectory):
        return _depth_trajectory_axis(trajectory, name)

    if isinstance(trajectory, SheetTrajectory):
        return 1

    raise TypeError(f"unsupported trajectory type for salience analysis: {type(trajectory).__name__}")


def discover_depth_banks(model: nn.Module) -> Tuple[CoefficientBank, ...]:
    trajectory = getattr(model, "trajectory", None)
    if trajectory is None or not hasattr(trajectory, "coefficients") or not hasattr(trajectory, "metadata"):
        raise TypeError("coefficient salience requires a THOG model exposing trajectory coefficients")

    banks: List[CoefficientBank] = []
    for metadata in trajectory.metadata:
        name = str(metadata.name)
        parameter = trajectory.coefficients[name]
        axis = coefficient_depth_axis(trajectory, name)
        if axis is None:
            continue
        if axis < 0 or axis >= parameter.ndim:
            raise RuntimeError(f"invalid coefficient order axis for {name}: axis={axis}, shape={tuple(parameter.shape)}")
        banks.append(CoefficientBank(name=name, parameter=parameter, order_axis=axis))

    if not banks:
        raise ValueError("checkpoint exposes no DEPTH coefficient banks")

    order_counts = {bank.order_count for bank in banks}
    if len(order_counts) != 1:
        details = ", ".join(f"{bank.name}:{bank.order_count}" for bank in banks)
        raise ValueError(f"DEPTH coefficient banks disagree on order count: {details}")
    return tuple(banks)


def select_banks(banks: Sequence[CoefficientBank], scope: str) -> Tuple[CoefficientBank, ...]:
    normalized = scope.strip()
    if not normalized:
        raise ValueError("scope must not be empty")
    if normalized.lower() == "depth":
        return tuple(banks)

    requested = tuple(part.strip() for part in normalized.split(",") if part.strip())
    if not requested:
        raise ValueError("scope contains no coefficient family names")
    by_name = {bank.name: bank for bank in banks}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"unknown coefficient family/families {unknown}; available: {available}")
    return tuple(by_name[name] for name in requested)


@contextmanager
def zero_order_temporarily(banks: Sequence[CoefficientBank], order: int) -> Iterator[None]:
    originals: List[Tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for bank in banks:
            target = bank.order_slice(order)
            saved = target.detach().clone()
            originals.append((target, saved))
            target.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for target, saved in originals:
                target.copy_(saved)
            for target, saved in originals:
                if not torch.equal(target, saved):
                    raise RuntimeError("coefficient restoration failed after salience ablation")


def _rms_from_slices(slices: Iterable[Tensor]) -> Optional[float]:
    square_sum = 0.0
    count = 0
    for values in slices:
        detached = values.detach().to(dtype=torch.float64, device="cpu")
        square_sum += float(detached.square().sum().item())
        count += detached.numel()
    if count == 0:
        return None
    return math.sqrt(square_sum / count)


def coefficient_rms_by_order(banks: Sequence[CoefficientBank]) -> Tuple[float, ...]:
    order_count = banks[0].order_count
    return tuple(float(_rms_from_slices(bank.order_slice(order) for bank in banks)) for order in range(order_count))


def gradient_diagnostics_by_order(
    banks: Sequence[CoefficientBank],
) -> Tuple[Tuple[Optional[float], ...], Tuple[Optional[float], ...]]:
    order_count = banks[0].order_count
    gradient_rms: List[Optional[float]] = []
    first_order_proxy: List[Optional[float]] = []
    for order in range(order_count):
        gradients = []
        proxy = 0.0
        proxy_present = False
        for bank in banks:
            gradient = bank.parameter.grad
            if gradient is None:
                continue
            index = [slice(None)] * gradient.ndim
            index[bank.order_axis] = order
            gradient_slice = gradient[tuple(index)]
            coefficient_slice = bank.order_slice(order)
            gradients.append(gradient_slice)
            proxy -= float(
                (coefficient_slice.detach().to(dtype=torch.float64, device="cpu")
                 * gradient_slice.detach().to(dtype=torch.float64, device="cpu")).sum().item()
            )
            proxy_present = True
        gradient_rms.append(_rms_from_slices(gradients))
        first_order_proxy.append(proxy if proxy_present else None)
    return tuple(gradient_rms), tuple(first_order_proxy)


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        stop = position + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[position][1]:
            stop += 1
        average_rank = (position + 1 + stop) / 2.0
        for item_index in range(position, stop):
            ranks[indexed[item_index][0]] = average_rank
        position = stop
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have the same length")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def concentration_statistics(mean_delta_loss: Sequence[float]) -> Dict[str, Optional[float]]:
    positive = [max(float(value), 0.0) for value in mean_delta_loss]
    total = sum(positive)
    order_count = len(positive)
    if order_count == 0 or total <= 0.0:
        return {
            "effective_salience_dimension": None,
            "effective_salience_dimension_ratio": None,
            "top_quartile_positive_salience_fraction": None,
        }
    square_sum = sum(value * value for value in positive)
    effective = (total * total) / square_sum
    top_count = max(1, math.ceil(order_count / 4))
    top_fraction = sum(sorted(positive, reverse=True)[:top_count]) / total
    return {
        "effective_salience_dimension": effective,
        "effective_salience_dimension_ratio": effective / order_count,
        "top_quartile_positive_salience_fraction": top_fraction,
    }


__all__ = [
    "CoefficientBank",
    "coefficient_depth_axis",
    "coefficient_rms_by_order",
    "concentration_statistics",
    "discover_depth_banks",
    "gradient_diagnostics_by_order",
    "select_banks",
    "spearman_rho",
    "zero_order_temporarily",
]
# ^^^ THOG
