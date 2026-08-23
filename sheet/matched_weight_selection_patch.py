# vvv THOG
"""Matched scalar-weight selection for THOG/DENSE diagnostics and INSTRA."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from . import dense_weight_curves_patch as _dense
from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import depth_weight_curves_v2_patch as _v2
from .local_chart_store import local_chart_root


WEIGHT_SELECTION_PROTOCOL = "matched_six_v1"
_SELECTION_FILE = ".instra_weight_selection.json"
_DEFAULT_SELECTION = {
    "protocol": WEIGHT_SELECTION_PROTOCOL,
    "user_selected": False,
    "model_feature": 0,
    "intermediate_feature": 0,
}


def _selection_path(root: Optional[Path] = None) -> Path:
    return (local_chart_root() if root is None else Path(root)) / _SELECTION_FILE


def _normalise_selection(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    try:
        model_feature = int(source.get("model_feature", 0))
    except (TypeError, ValueError):
        model_feature = 0
    try:
        intermediate_feature = int(source.get("intermediate_feature", 0))
    except (TypeError, ValueError):
        intermediate_feature = 0
    return {
        "protocol": WEIGHT_SELECTION_PROTOCOL,
        "user_selected": source.get("user_selected") is True,
        "model_feature": model_feature,
        "intermediate_feature": intermediate_feature,
    }


def read_weight_selection(root: Optional[Path] = None) -> Dict[str, Any]:
    try:
        value = json.loads(_selection_path(root).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return dict(_DEFAULT_SELECTION)
    return _normalise_selection(value)


def write_weight_selection(
    value: Mapping[str, Any],
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    selection = _normalise_selection(value)
    path = _selection_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(selection, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return selection


def _logical_to_matrix(
    chart_name: str,
    model_feature: int,
    intermediate_feature: int,
) -> Tuple[int, int]:
    if chart_name in {"attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "mlp_up"}:
        return intermediate_feature, model_feature
    if chart_name in {"attn_out_head_N", "mlp_down"}:
        return model_feature, intermediate_feature
    raise KeyError(chart_name)


def _matrix_to_logical(
    chart_name: str,
    output_row: int,
    row_index: int,
) -> Tuple[int, int]:
    if chart_name in {"attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "mlp_up"}:
        return row_index, output_row
    if chart_name in {"attn_out_head_N", "mlp_down"}:
        return output_row, row_index
    raise KeyError(chart_name)


def _random_logical_coordinates(
    *,
    seed: int,
    width: int,
    n_head: int,
    count: int,
) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    if width < 1 or n_head < 1 or width % n_head != 0:
        raise ValueError("n_embd must be positive and divisible by n_head")
    head = random.Random(seed ^ 0xA771).randrange(n_head)
    head_dim = width // n_head
    head_start = head * head_dim
    population = width * head_dim
    generator = random.Random(seed ^ 0x4D41544348)
    indices = generator.sample(range(population), min(max(1, count), population))
    return head, tuple(
        (flat % width, head_start + flat // width)
        for flat in indices
    )


def _matched_selection(
    trainer: Any,
    telemetry: Any,
    *,
    width: int,
    n_head: int,
) -> Dict[str, Any]:
    seed = int(_depth._selection_seed(trainer, telemetry))
    head, random_logical = _random_logical_coordinates(
        seed=seed,
        width=width,
        n_head=n_head,
        count=int(_depth._scalar_weights_per_matrix()),
    )
    configured = read_weight_selection()
    model_feature = int(configured["model_feature"])
    intermediate_feature = int(configured["intermediate_feature"])
    user_valid = (
        configured["user_selected"] is True
        and 0 <= model_feature < width
        and 0 <= intermediate_feature < width
    )
    logical = list(random_logical)
    if user_valid and (model_feature, intermediate_feature) not in logical:
        logical.append((model_feature, intermediate_feature))
    selection: Dict[str, Any] = {
        "seed": seed,
        "attention_head": head,
        "matched_weight_protocol": WEIGHT_SELECTION_PROTOCOL,
        "matched_feature_count": width,
        "matched_random_logical_coordinates": random_logical,
        "matched_user_selection": {**configured, "valid_for_run": user_valid},
    }
    for chart_name in _v2._CHART_FAMILIES:
        selection[chart_name] = tuple(
            _logical_to_matrix(chart_name, model, intermediate)
            for model, intermediate in logical
        )
    return selection


def _selection_is_current(cached: Any) -> bool:
    if not isinstance(cached, Mapping):
        return False
    if cached.get("matched_weight_protocol") != WEIGHT_SELECTION_PROTOCOL:
        return False
    configured = read_weight_selection()
    user = cached.get("matched_user_selection")
    return (
        isinstance(user, Mapping)
        and user.get("user_selected") == configured["user_selected"]
        and int(user.get("model_feature", -1)) == configured["model_feature"]
        and int(user.get("intermediate_feature", -1)) == configured["intermediate_feature"]
    )


def _selected_scalar_coordinates_matched(trainer: Any, telemetry: Any) -> Dict[str, Any]:
    cached = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    if _selection_is_current(cached):
        return dict(cached)
    trajectory = _depth._depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return {}
    selection = _matched_selection(
        trainer,
        telemetry,
        width=int(trajectory.config.n_embd),
        n_head=int(trajectory.config.n_head),
    )
    setattr(telemetry, "_thog_depth_weight_curve_selection", selection)
    return selection


def _dense_selection_matched(trainer: Any, telemetry: Any) -> Dict[str, Any]:
    cached = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    if _selection_is_current(cached):
        return dict(cached)
    model = trainer.raw_model
    if not _dense._dense_model(model):
        return {}
    selection = _matched_selection(
        trainer,
        telemetry,
        width=int(model.config.n_embd),
        n_head=int(model.config.n_head),
    )
    setattr(telemetry, "_thog_depth_weight_curve_selection", selection)
    return selection


_v2._selected_scalar_coordinates_v2 = _selected_scalar_coordinates_matched
_depth._selected_scalar_coordinates = _selected_scalar_coordinates_matched
_dense._dense_selection = _dense_selection_matched

_ORIGINAL_SNAPSHOT = _depth._depth_weight_snapshot


def _snapshot_with_matched_selection(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    snapshot = _ORIGINAL_SNAPSHOT(
        trainer,
        telemetry,
        optimizer_update=optimizer_update,
    )
    if not snapshot:
        return snapshot
    selection = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    if not isinstance(selection, Mapping):
        return snapshot

    random_logical = {
        (int(pair[0]), int(pair[1]))
        for pair in selection.get("matched_random_logical_coordinates", ())
    }
    user = selection.get("matched_user_selection", {})
    user_active = (
        isinstance(user, Mapping)
        and user.get("user_selected") is True
        and user.get("valid_for_run") is True
    )
    user_model = int(user.get("model_feature", 0)) if isinstance(user, Mapping) else 0
    user_intermediate = (
        int(user.get("intermediate_feature", 0))
        if isinstance(user, Mapping)
        else 0
    )

    for chart_name, family in snapshot.get("families", {}).items():
        annotated = []
        for curve in family.get("curves", ()):
            current = dict(curve)
            model_feature, intermediate_feature = _matrix_to_logical(
                chart_name,
                int(current["output_row"]),
                int(current["row_index"]),
            )
            is_random = (model_feature, intermediate_feature) in random_logical
            is_user = (
                user_active
                and model_feature == user_model
                and intermediate_feature == user_intermediate
            )
            current["model_feature"] = model_feature
            current["intermediate_feature"] = intermediate_feature
            current["selection_kind"] = (
                "user_random"
                if is_user and is_random
                else ("user" if is_user else "random")
            )
            annotated.append(current)
        family["curves"] = tuple(annotated)

    snapshot["weight_selection"] = {
        "protocol": WEIGHT_SELECTION_PROTOCOL,
        "user_selected": bool(user.get("user_selected", False)) if isinstance(user, Mapping) else False,
        "model_feature": user_model,
        "intermediate_feature": user_intermediate,
        "feature_count": int(selection["matched_feature_count"]),
        "applied": user_active,
    }
    return snapshot


_depth._depth_weight_snapshot = _snapshot_with_matched_selection

_ORIGINAL_FIGURE_BUILDER = _v2._build_depth_plotly_figure


def _trace_scalar_id(trace: Any) -> Optional[str]:
    meta = getattr(trace, "meta", None)
    if not isinstance(meta, Mapping):
        return None
    for key in ("instra_thog_scalar_id", "instra_dense_scalar_id"):
        if meta.get(key):
            return str(meta[key])
    return None


def _trace_optimizer_update(trace: Any) -> Optional[int]:
    meta = getattr(trace, "meta", None)
    if not isinstance(meta, Mapping):
        return None
    for key in ("instra_thog_optimizer_update", "instra_dense_optimizer_update"):
        try:
            return int(meta[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _build_figure_with_matched_selection(
    snapshots: Sequence[Mapping[str, Any]],
    chart_name: str,
):
    figure = _ORIGINAL_FIGURE_BUILDER(snapshots, chart_name)
    retained = tuple(snapshots)
    if not retained:
        return figure
    selection = retained[-1].get("weight_selection")
    if not isinstance(selection, Mapping):
        return figure

    curves: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for snapshot in retained:
        optimizer_update = int(snapshot["optimizer_update"])
        for curve in snapshot.get("families", {}).get(chart_name, {}).get("curves", ()):
            curves[(optimizer_update, str(curve.get("scalar_id", "")))] = curve

    for trace in figure.data:
        scalar_id = _trace_scalar_id(trace)
        optimizer_update = _trace_optimizer_update(trace)
        curve = curves.get((optimizer_update, scalar_id or "")) if optimizer_update is not None else None
        if not isinstance(curve, Mapping):
            continue
        model_feature = int(curve["model_feature"])
        intermediate_feature = int(curve["intermediate_feature"])
        family_label = "attention feature" if chart_name.startswith("attn_") else "MLP feature"
        logical_label = f"model {model_feature} · {family_label} {intermediate_feature}"
        prior_meta = trace.meta if isinstance(trace.meta, Mapping) else {}
        trace.meta = {
            **prior_meta,
            "instra_weight_selection_protocol": WEIGHT_SELECTION_PROTOCOL,
            "instra_weight_selection_kind": str(curve.get("selection_kind", "random")),
            "instra_weight_model_feature": model_feature,
            "instra_weight_intermediate_feature": intermediate_feature,
            "instra_weight_feature_count": int(selection["feature_count"]),
        }
        if scalar_id:
            trace.name = str(trace.name or "").replace(scalar_id, logical_label)
            trace.hovertemplate = str(trace.hovertemplate or "").replace(
                scalar_id,
                logical_label,
            )

    titles = {
        "attn_q_head_N": "attention query",
        "attn_k_head_N": "attention key",
        "attn_v_head_N": "attention value",
        "attn_out_head_N": "attention output",
        "mlp_up": "MLP expansion",
        "mlp_down": "MLP contraction",
    }
    title_text = str(figure.layout.title.text or "")
    if " — " in title_text:
        prefix, remainder = title_text.split(" — ", 1)
        suffix = f"<br>{remainder.split('<br>', 1)[1]}" if "<br>" in remainder else ""
        figure.layout.title.text = f"{prefix} — {titles[chart_name]}{suffix}"
    return figure


_v2._build_depth_plotly_figure = _build_figure_with_matched_selection
_depth._build_depth_plotly_figure = _build_figure_with_matched_selection


def install_dashboard(dashboard: Any) -> None:
    if getattr(dashboard, "_thog2_matched_weight_selection_installed", False):
        return
    dashboard._thog2_matched_weight_selection_installed = True
    original_handler_for = dashboard._handler_for

    def handler_for(catalog: Any) -> Any:
        base_handler = original_handler_for(catalog)

        class Handler(base_handler):
            def do_GET(self) -> None:
                if urlparse(self.path).path != "/api/weight-selection":
                    return super().do_GET()
                self._send_json(read_weight_selection(catalog.root))

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/api/weight-selection":
                    self._send(
                        b"not found\n",
                        content_type="text/plain; charset=utf-8",
                        status=dashboard.HTTPStatus.NOT_FOUND,
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length < 1 or content_length > 65536:
                        raise ValueError("weight-selection request body is missing or too large")
                    payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                    if not isinstance(payload, Mapping):
                        raise ValueError("weight-selection request must be a JSON object")
                    model_feature = int(payload.get("model_feature", 0))
                    intermediate_feature = int(payload.get("intermediate_feature", 0))
                    if model_feature < 0 or intermediate_feature < 0:
                        raise ValueError("weight feature indices must be non-negative integers")
                    self._send_json(
                        write_weight_selection(
                            {
                                "user_selected": payload.get("user_selected") is True,
                                "model_feature": model_feature,
                                "intermediate_feature": intermediate_feature,
                            },
                            catalog.root,
                        )
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                except Exception as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        return Handler

    dashboard._handler_for = handler_for


__all__ = [
    "WEIGHT_SELECTION_PROTOCOL",
    "install_dashboard",
    "read_weight_selection",
    "write_weight_selection",
]
# ^^^ THOG
