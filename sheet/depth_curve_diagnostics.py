# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from .curve_trajectory import CurveTrajectory
from .semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


DEPTH_CURVE_PLOT_MODES = ("none", "final", "eval")
DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS = 16384

_DEPTH_CURVE_FAMILY_LABELS: Dict[str, str] = {
    ATTENTION_QUERY_WEIGHT: "W_q",
    ATTENTION_KEY_WEIGHT: "W_k",
    ATTENTION_VALUE_WEIGHT: "W_v",
    ATTENTION_OUTPUT_WEIGHT: "W_o",
    MLP_EXPANSION_WEIGHT: "W_mlp_in",
    MLP_CONTRACTION_WEIGHT: "W_mlp_out",
}


@dataclass(frozen=True)
class DepthCurveSummary:
    tag: str
    label: str
    family_name: str
    layers: Tuple[int, ...]
    means: Tuple[float, ...]
    stds: Tuple[float, ...]
    sampled_elements: int
    total_elements: int


def normalize_depth_curve_plot_mode(value: str) -> str:
    selected = value.strip().lower()
    if selected not in DEPTH_CURVE_PLOT_MODES:
        raise ValueError(
            "THOG2_DEPTH_CURVE_PLOTS must be none, final, or eval; "
            f"got {value!r}"
        )
    return selected


def depth_curve_family_labels() -> Dict[str, str]:
    return dict(_DEPTH_CURVE_FAMILY_LABELS)


def _sample_indices(total_elements: int, sample_elements: int, device: torch.device) -> Tensor:
    if total_elements <= 0:
        raise ValueError("total_elements must be positive")
    if sample_elements <= 0:
        raise ValueError("sample_elements must be positive")
    count = min(total_elements, sample_elements)
    if count == total_elements:
        return torch.arange(total_elements, device=device, dtype=torch.long)
    offsets = torch.arange(count, device=device, dtype=torch.long)
    return torch.div(offsets * total_elements, count, rounding_mode="floor")


# vvv THOG
def _sampled_matrix_coefficients_to_cpu(coefficient: Tensor, sample_elements: int) -> Tuple[Tensor, int, int]:
    if coefficient.ndim != 3:
        raise ValueError("curve matrix coefficients must have shape [rows, cols, depth]")
    output_rows, row_width, _ = coefficient.shape
    total_elements = int(output_rows) * int(row_width)
    cpu_indices = _sample_indices(total_elements, sample_elements, torch.device("cpu"))
    row_indices = torch.div(cpu_indices, int(row_width), rounding_mode="floor").to(coefficient.device)
    col_indices = torch.remainder(cpu_indices, int(row_width)).to(coefficient.device)
    with torch.no_grad():
        sampled_gpu = coefficient.detach()[row_indices, col_indices, :].float()
        sampled_cpu = sampled_gpu.cpu()
    return sampled_cpu, int(cpu_indices.numel()), total_elements
# ^^^ THOG


def _sampled_layer_mean_std(trajectory: CurveTrajectory, family_name: str, sample_elements: int) -> Tuple[Tuple[float, ...], Tuple[float, ...], int, int]:
    coefficient = trajectory.coefficients[family_name]
    # vvv THOG
    sampled_coefficients, sampled_count, total_elements = _sampled_matrix_coefficients_to_cpu(
        coefficient,
        sample_elements,
    )
    depth_basis = trajectory.depth_basis.detach().float().cpu()
    # ^^^ THOG
    with torch.no_grad():
        generated_samples = depth_basis @ sampled_coefficients.transpose(0, 1)
        means = generated_samples.mean(dim=1)
        stds = generated_samples.std(dim=1, unbiased=False)
    return (
        tuple(float(value) for value in means.detach().cpu().tolist()),
        tuple(float(value) for value in stds.detach().cpu().tolist()),
        sampled_count,
        total_elements,
    )


def curve_depth_summaries(model: Any, sample_elements: int = DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS) -> Tuple[DepthCurveSummary, ...]:
    trajectory = getattr(model, "trajectory", None)
    if not isinstance(trajectory, CurveTrajectory):
        return ()
    layers = tuple(range(int(trajectory.config.n_layer)))
    rows = []
    for family_name, label in _DEPTH_CURVE_FAMILY_LABELS.items():
        means, stds, sampled_count, total_elements = _sampled_layer_mean_std(
            trajectory,
            family_name,
            sample_elements,
        )
        rows.append(
            DepthCurveSummary(
                tag=f"sheet/depth_curve/{label}",
                label=label,
                family_name=family_name,
                layers=layers,
                means=means,
                stds=stds,
                sampled_elements=sampled_count,
                total_elements=total_elements,
            )
        )
    return tuple(rows)


def depth_curve_figure(summary: DepthCurveSummary) -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.0))
    axes[0].plot(summary.layers, summary.means)
    axes[0].set_ylabel("mean")
    axes[0].set_title(f"{summary.label} generated depth curve")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(summary.layers, summary.stds)
    axes[1].set_xlabel("logical layer")
    axes[1].set_ylabel("std")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(
        f"{summary.label}: sampled {summary.sampled_elements}/{summary.total_elements} matrix elements",
        y=0.995,
    )
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS",
    "DEPTH_CURVE_PLOT_MODES",
    "DepthCurveSummary",
    "curve_depth_summaries",
    "depth_curve_family_labels",
    "depth_curve_figure",
    "normalize_depth_curve_plot_mode",
]
# ^^^ THOG