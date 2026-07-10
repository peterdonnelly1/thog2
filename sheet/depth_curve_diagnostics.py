# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

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
DEPTH_CURVE_RENDERERS = ("matplotlib", "plotly", "both")
DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS = 16384
DEFAULT_DEPTH_CURVE_HISTOGRAM_BINS = 64

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
    histogram_edges: Tuple[float, ...] = ()
    histogram_counts_by_layer: Tuple[Tuple[int, ...], ...] = ()


def normalize_depth_curve_plot_mode(value: str) -> str:
    selected = value.strip().lower()
    if selected not in DEPTH_CURVE_PLOT_MODES:
        raise ValueError(
            "THOG2_DEPTH_CURVE_PLOTS must be none, final, or eval; "
            f"got {value!r}"
        )
    return selected


# vvv THOG
def normalize_depth_curve_renderer(value: str) -> str:
    selected = value.strip().lower()
    if selected not in DEPTH_CURVE_RENDERERS:
        raise ValueError(
            "THOG2_DEPTH_CURVE_RENDERER must be matplotlib, plotly, or both; "
            f"got {value!r}"
        )
    return selected
# ^^^ THOG


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


def _histogram_edges(values: Tensor, histogram_bins: int) -> Tensor:
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    minimum = float(values.min().item())
    maximum = float(values.max().item())
    if minimum == maximum:
        padding = max(abs(minimum) * 1.0e-6, 1.0e-12)
        minimum -= padding
        maximum += padding
    return torch.linspace(minimum, maximum, histogram_bins + 1, dtype=torch.float32)


def _histogram_counts_by_layer(values: Tensor, edges: Tensor) -> Tuple[Tuple[int, ...], ...]:
    bins = int(edges.numel()) - 1
    minimum = float(edges[0].item())
    maximum = float(edges[-1].item())
    rows = []
    for layer_values in values:
        counts = torch.histc(layer_values.float(), bins=bins, min=minimum, max=maximum)
        rows.append(tuple(int(value) for value in counts.cpu().tolist()))
    return tuple(rows)
# ^^^ THOG


def _sampled_layer_mean_std(
    trajectory: CurveTrajectory,
    family_name: str,
    sample_elements: int,
    histogram_bins: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], int, int, Tuple[float, ...], Tuple[Tuple[int, ...], ...]]:
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
        edges = _histogram_edges(generated_samples, histogram_bins)
        counts = _histogram_counts_by_layer(generated_samples, edges)
    return (
        tuple(float(value) for value in means.detach().cpu().tolist()),
        tuple(float(value) for value in stds.detach().cpu().tolist()),
        sampled_count,
        total_elements,
        tuple(float(value) for value in edges.detach().cpu().tolist()),
        counts,
    )


def curve_depth_summaries(
    model: Any,
    sample_elements: int = DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS,
    histogram_bins: int = DEFAULT_DEPTH_CURVE_HISTOGRAM_BINS,
) -> Tuple[DepthCurveSummary, ...]:
    trajectory = getattr(model, "trajectory", None)
    if not isinstance(trajectory, CurveTrajectory):
        return ()
    layers = tuple(range(int(trajectory.config.n_layer)))
    rows = []
    for family_name, label in _DEPTH_CURVE_FAMILY_LABELS.items():
        means, stds, sampled_count, total_elements, histogram_edges, histogram_counts_by_layer = _sampled_layer_mean_std(
            trajectory,
            family_name,
            sample_elements,
            histogram_bins,
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
                histogram_edges=histogram_edges,
                histogram_counts_by_layer=histogram_counts_by_layer,
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


# vvv THOG
def _histogram_centers(edges: Sequence[float]) -> Tuple[float, ...]:
    return tuple((float(left) + float(right)) / 2.0 for left, right in zip(edges[:-1], edges[1:]))


def depth_curve_plotly_figure(summary: DepthCurveSummary) -> Any:
    if not summary.histogram_edges or not summary.histogram_counts_by_layer:
        raise ValueError("DepthCurveSummary must include histogram data for Plotly rendering")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    centers = _histogram_centers(summary.histogram_edges)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=("mean by logical layer", "std by logical layer", "selected-layer sampled weight histogram"),
        row_heights=(0.28, 0.28, 0.44),
    )
    fig.add_trace(
        go.Scatter(x=summary.layers, y=summary.means, mode="lines+markers", name="mean"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=summary.layers, y=summary.stds, mode="lines+markers", name="std"),
        row=2,
        col=1,
    )
    for index, layer in enumerate(summary.layers):
        fig.add_trace(
            go.Bar(
                x=centers,
                y=summary.histogram_counts_by_layer[index],
                name=f"layer {layer}",
                visible=index == 0,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    steps = []
    for index, layer in enumerate(summary.layers):
        visibility = [True, True] + [candidate == index for candidate in range(len(summary.layers))]
        steps.append(
            {
                "method": "update",
                "label": str(layer),
                "args": [
                    {"visible": visibility},
                    {"title": f"{summary.label} generated depth curve — selected layer {layer}"},
                ],
            }
        )
    fig.update_layout(
        title=f"{summary.label} generated depth curve — selected layer {summary.layers[0]}",
        height=900,
        hovermode="x unified",
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "logical layer: "},
                "pad": {"t": 35},
                "steps": steps,
            }
        ],
    )
    fig.update_xaxes(title_text="logical layer", row=1, col=1)
    fig.update_yaxes(title_text="mean", row=1, col=1)
    fig.update_xaxes(title_text="logical layer", row=2, col=1)
    fig.update_yaxes(title_text="std", row=2, col=1)
    fig.update_xaxes(title_text="generated sampled weight", row=3, col=1)
    fig.update_yaxes(title_text="sample count", row=3, col=1)
    return fig


def _summary_html_name(step: int, summary: DepthCurveSummary) -> str:
    safe_label = summary.label.replace("/", "_").replace(" ", "_")
    return f"step_{step:06d}_{safe_label}.html"


def _parse_summary_html_name(path: Path) -> Tuple[int, str] | None:
    name = path.name
    if not name.startswith("step_") or not name.endswith(".html"):
        return None
    pieces = name.removesuffix(".html").split("_", 2)
    if len(pieces) != 3:
        return None
    try:
        step = int(pieces[1])
    except ValueError:
        return None
    return step, pieces[2]


def write_depth_curve_plotly_html(summary: DepthCurveSummary, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = depth_curve_plotly_figure(summary)
    figure.write_html(str(output_path), include_plotlyjs="directory", full_html=True, auto_open=False)
    return output_path


def _depth_curve_snapshot_index(output_root: Path) -> Dict[str, Tuple[Tuple[int, str], ...]]:
    rows: Dict[str, list[Tuple[int, str]]] = {}
    for path in sorted(output_root.glob("step_*.html")):
        parsed = _parse_summary_html_name(path)
        if parsed is None:
            continue
        step, label = parsed
        rows.setdefault(label, []).append((step, path.name))
    return {label: tuple(sorted(values)) for label, values in sorted(rows.items())}


def _latest_by_family(snapshot_index: Dict[str, Tuple[Tuple[int, str], ...]]) -> Dict[str, Tuple[int, str]]:
    return {label: values[-1] for label, values in snapshot_index.items() if values}


def _family_cards(latest_by_family: Dict[str, Tuple[int, str]]) -> str:
    cards = []
    family_order = tuple(_DEPTH_CURVE_FAMILY_LABELS.values())
    for label in family_order:
        latest = latest_by_family.get(label)
        if latest is None:
            cards.append(
                "<section class=\"family-card missing\">"
                f"<h2>{escape(label)}</h2>"
                "<p>No snapshot yet.</p>"
                "</section>"
            )
            continue
        step, file_name = latest
        cards.append(
            "<section class=\"family-card\">"
            f"<h2>{escape(label)}</h2>"
            f"<p><b>latest step:</b> {step}</p>"
            f"<p><a class=\"button\" href=\"{escape(file_name)}\">Open full {escape(label)} chart</a></p>"
            f"<iframe src=\"{escape(file_name)}\" loading=\"lazy\" title=\"{escape(label)} latest depth curve\"></iframe>"
            "</section>"
        )
    return "\n".join(cards)


def _step_nav(snapshot_index: Dict[str, Tuple[Tuple[int, str], ...]]) -> str:
    steps = sorted({step for values in snapshot_index.values() for step, _ in values})
    if not steps:
        return "<p>No snapshots yet.</p>"
    return "\n".join(f"<a href=\"#step-{step:06d}\">{step}</a>" for step in steps)


def _snapshot_rows(snapshot_index: Dict[str, Tuple[Tuple[int, str], ...]]) -> str:
    family_order = tuple(_DEPTH_CURVE_FAMILY_LABELS.values())
    steps = sorted({step for values in snapshot_index.values() for step, _ in values})
    if not steps:
        return "<p>No snapshots yet.</p>"
    rows = []
    for step in steps:
        rows.append(f"<tr id=\"step-{step:06d}\"><th class=\"snapshot-step\">{step}</th><td class=\"snapshot-links\">")
        for label in family_order:
            matches = [file_name for candidate_step, file_name in snapshot_index.get(label, ()) if candidate_step == step]
            if matches:
                rows.append(f"<a class=\"button secondary\" href=\"{escape(matches[0])}\">{escape(label)}</a>")
            else:
                rows.append(f"<span class=\"button disabled\">{escape(label)}</span>")
        rows.append("</td></tr>")
    return (
        "<table class=\"snapshot-table\">"
        "<thead><tr><th>step</th><th>families</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def write_depth_curve_local_viewer(
    summaries: Sequence[DepthCurveSummary],
    output_root: Path,
    step: int,
    artifact_name: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        file_name = _summary_html_name(step, summary)
        write_depth_curve_plotly_html(summary, output_root / file_name)
    index_path = output_root / "index.html"
    snapshot_index = _depth_curve_snapshot_index(output_root)
    latest_by_family = _latest_by_family(snapshot_index)
    index_path.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html>",
                "<head>",
                "  <meta charset=\"utf-8\">",
                "  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
                f"  <title>THOG depth curves — {escape(artifact_name)}</title>",
                "  <style>",
                "    :root{color-scheme:light dark;--bg:#101114;--fg:#f4f4f5;--muted:#a1a1aa;--card:#181a20;--line:#303442;--button:#2563eb;--button2:#374151}",
                "    body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:var(--bg);color:var(--fg)}",
                "    header{position:sticky;top:0;z-index:10;background:rgba(16,17,20,.96);border-bottom:1px solid var(--line);padding:.8rem 1.25rem}",
                "    main{padding:1rem 1.25rem 1.5rem;max-width:1800px;margin:0 auto}",
                "    h1{margin:.05rem 0 .25rem;font-size:1.35rem}h2{margin:.2rem 0 .55rem}p{margin:.25rem 0}",
                "    .meta{color:var(--muted);font-size:.9rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:1rem}",
                "    .family-card{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:.85rem;box-shadow:0 2px 10px rgba(0,0,0,.2)}",
                "    .family-card.missing{opacity:.55}.family-card iframe{width:100%;height:520px;border:1px solid var(--line);border-radius:8px;background:white}",
                "    .button{display:inline-block;border-radius:999px;background:var(--button);color:white;text-decoration:none;padding:.32rem .58rem;margin:.08rem .12rem;font-size:.9rem;font-weight:650;line-height:1.15}",
                "    .button.secondary{background:var(--button2)}.button.disabled{display:inline-block;border-radius:999px;border:1px solid var(--line);color:var(--muted);padding:.32rem .58rem;margin:.08rem .12rem;font-size:.9rem;line-height:1.15}",
                "    code{background:#27272a;padding:.12rem .28rem;border-radius:4px}nav{display:flex;flex-wrap:wrap;gap:.25rem;margin-top:.35rem}nav a{color:#93c5fd}",
                "    .snapshot-table{width:100%;border-collapse:collapse;margin-top:.2rem}.snapshot-table th,.snapshot-table td{border-top:1px solid var(--line);padding:.28rem .35rem;vertical-align:middle}",
                "    .snapshot-table thead th{color:var(--muted);font-weight:700;text-align:left}.snapshot-step{width:5.5rem;text-align:right;white-space:nowrap;color:var(--muted);font-variant-numeric:tabular-nums}",
                "    .snapshot-links{display:flex;flex-wrap:wrap;align-items:center;gap:.12rem}.snapshot-section{margin-top:1.1rem}",
                "  </style>",
                "</head>",
                "<body>",
                "  <header>",
                "    <h1>THOG depth curves</h1>",
                f"    <div class=\"meta\"><b>artifact:</b> {escape(artifact_name)} &nbsp; <b>latest step:</b> {step}</div>",
                "    <div class=\"meta\">Open a card to drill into a weight family, then use the layer slider inside the Plotly chart.</div>",
                "    <nav>",
                _step_nav(snapshot_index),
                "    </nav>",
                "  </header>",
                "  <main>",
                "    <h2>Weight-family dashboard</h2>",
                "    <div class=\"grid\">",
                _family_cards(latest_by_family),
                "    </div>",
                "    <section class=\"snapshot-section\">",
                "      <h2>Snapshots by eval step</h2>",
                _snapshot_rows(snapshot_index),
                "    </section>",
                "  </main>",
                "</body>",
                "</html>",
            )
        ),
        encoding="utf-8",
    )
    return index_path
# ^^^ THOG


__all__ = [
    "DEFAULT_DEPTH_CURVE_HISTOGRAM_BINS",
    "DEFAULT_DEPTH_CURVE_SAMPLE_ELEMENTS",
    "DEPTH_CURVE_PLOT_MODES",
    "DEPTH_CURVE_RENDERERS",
    "DepthCurveSummary",
    "curve_depth_summaries",
    "depth_curve_family_labels",
    "depth_curve_figure",
    "depth_curve_plotly_figure",
    "normalize_depth_curve_plot_mode",
    "normalize_depth_curve_renderer",
    "write_depth_curve_local_viewer",
    "write_depth_curve_plotly_html",
]
# ^^^ THOG
