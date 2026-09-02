# vvv THOG
"""Numerical contract for compact histories of raw layer gradients."""
from __future__ import annotations

import torch
from torch import Tensor


def resolve_history_count(value: str | int, *, nominal: int, layers: int) -> dict:
    if value == "auto":
        return {"requested": "auto", "nominal": nominal, "effective": min(nominal, layers)}
    if isinstance(value, bool) or str(value).strip() != str(value) or not str(value).isdigit():
        raise ValueError("thogopt history count must be auto or a positive integer")
    count = int(value)
    if not 1 <= count <= layers:
        raise ValueError(f"thogopt history count must be in 1..{layers}; only {layers} layer samples exist")
    return {"requested": count, "nominal": count, "effective": count}


def history_basis(layers: int, count: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if not 1 <= count <= layers:
        raise ValueError("history basis requires 1 <= count <= layers")
    if count == layers:
        # A lossless sample chart avoids an ill-conditioned high-order Vandermonde.
        return torch.eye(layers, dtype=dtype, device=device)
    x = torch.linspace(-1, 1, layers, dtype=torch.float64, device="cpu")
    columns = [torch.ones_like(x)]
    if count > 1:
        columns.append(x)
    for _ in range(2, count):
        columns.append(2 * x * columns[-1] - columns[-2])
    raw = torch.stack(columns, dim=1)
    q, r = torch.linalg.qr(raw, mode="reduced")
    if torch.linalg.matrix_rank(raw) != count:
        raise ValueError("rank-deficient thogopt history basis")
    signs = torch.where(r.diag() < 0, -1., 1.)
    return (q * signs).to(device=device, dtype=dtype)


def fit_nonnegative(q: Tensor, target: Tensor, *, max_sweeps: int = 2048) -> tuple[Tensor, dict]:
    """Unweighted LS with Q a >= 0, using Hildreth dual coordinate descent.

    Q has orthonormal columns. Normalise each coupling independently so tiny
    second moments receive the same relative numerical treatment as large ones.
    This solves constraints on reconstructed samples, not on coefficients.
    """
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise FloatingPointError("thogopt second-moment target must be finite and nonnegative")
    scale = target.abs().amax(dim=0).clamp_min(torch.finfo(target.dtype).tiny)
    normal = target / scale
    coefficients = q.T @ normal
    values = q @ coefficients
    tolerance = 32 * torch.finfo(target.dtype).eps
    affected = values.amin(dim=0) < -tolerance
    count = int(affected.sum())
    sweeps = 0
    if count:
        selected = coefficients[:, affected].clone()
        samples = values[:, affected].clone()
        dual = torch.zeros_like(samples)
        gram = q @ q.T
        for sweep in range(max_sweeps):
            for layer in range(q.shape[0]):
                old = dual[layer].clone()
                dual[layer] = (old - samples[layer] / gram[layer, layer]).clamp_min(0)
                delta = dual[layer] - old
                samples.add_(gram[:, layer, None] * delta[None, :])
                selected.add_(q[layer, :, None] * delta[None, :])
            sweeps = sweep + 1
            if sweeps % 8 == 0:
                samples = q @ selected
                complementarity = torch.where(dual > tolerance, samples.abs(), (-samples).clamp_min(0))
                if float(complementarity.max()) <= tolerance:
                    break
        else:
            raise FloatingPointError(f"thogopt nonnegative history fit did not converge in {max_sweeps} sweeps")
        coefficients[:, affected] = selected
        values = q @ coefficients
    minimum = values.amin(dim=0)
    if bool((minimum < -2 * tolerance).any()):
        raise FloatingPointError("thogopt materially negative second-moment reconstruction")
    # A constant shift only corrects bounded floating-point feasibility error.
    correction = (-minimum).clamp_min(0)
    if bool((correction > 0).any()):
        coefficients.add_((q.T @ torch.ones(q.shape[0], device=q.device, dtype=q.dtype))[:, None] * correction)
    return coefficients * scale, {
        "constrained_columns": count,
        "fit_sweeps": sweeps,
        "roundoff_corrections": int((correction > 0).sum()),
        "maximum_roundoff_correction": float((correction * scale).max()),
    }


def comparison_errors(candidate: Tensor, reference: Tensor, *, floor: float = 1e-30) -> dict:
    difference = candidate.double() - reference.double()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "rms_error": float(difference.square().mean().sqrt()),
        "relative_l2_error": float(difference.norm() / reference.double().norm().clamp_min(floor)),
        "relative_denominator_floor": floor,
    }
# ^^^ THOG
