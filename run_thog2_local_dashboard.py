# vvv THOG
"""Serve instra run charts, local files and W&B file manifests."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import stat
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from sheet import depth_weight_curves_v2_patch as depth_curves
from sheet import plastic_depth_wandb_probe_curves_patch as probe_curves
from sheet.local_chart_store import (
    LOCAL_CHART_DATABASE_NAME,
    LocalChartReader,
)


_ASSET_ROOT = Path(__file__).resolve().parent / "sheet" / "local_dashboard_assets"
# _ASSET_NAMES = frozenset(("dashboard.css", "dashboard.js"))
# vvv THOG serve the local heatmap/view-size patch after the established dashboard script
# _ASSET_NAMES = frozenset(("dashboard.css", "dashboard.js", "dashboard_heatmap_patch.js"))
_ASSET_NAMES = frozenset(("dashboard.css", "dashboard.js", "dashboard_heatmap_patch.js", "dashboard_ui_patch.css"))
# ^^^ THOG
_WANDB_FILE_CACHE_SECONDS = 30.0
_WANDB_FILE_LIMIT = 5000


def _active_run_state(value: Any) -> bool:
    return str(value) in {"preparing", "recording", "monitoring", "running"}


def _preset_from_configuration(configuration: Dict[str, Any]) -> str:
    model_type = str(configuration.get("model_type", "")).strip()
    if model_type.lower() == "dense":
        return "dense"
    geometry_preset = str(configuration.get("geometry_preset", "")).strip()
    return geometry_preset or model_type


def _optional_metadata_integer(metadata: Dict[str, str], key: str) -> Optional[int]:
    text = str(metadata.get(key, "")).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _target_update(configuration: Dict[str, Any]) -> Optional[int]:
    candidates = (
        configuration.get("max_iters"),
        configuration.get("max_updates"),
        configuration.get("lifecycle", {}).get("target_updates")
        if isinstance(configuration.get("lifecycle"), dict)
        else None,
    )
    for value in candidates:
        try:
            target = int(value)
        except (TypeError, ValueError):
            continue
        if target >= 0:
            return target
    return None


def _reported_run_state(
    stored_state: str,
    *,
    current_update: Optional[int],
    target_update: Optional[int],
) -> str:
    state = str(stored_state or "unknown")
    if state in {"finished", "stopped", "crashed"}:
        if target_update is None:
            return "stopped" if state == "crashed" else state
        return (
            "finished"
            if current_update is not None and current_update >= target_update
            else "stopped"
        )
    return state


def _weight_data_lost(
    metadata: Dict[str, str],
    *,
    current_update: Optional[int],
    snapshot_count: int,
) -> bool:
    if metadata.get("weight_capture_expected") != "true" or snapshot_count > 0:
        return False
    if current_update is None:
        return False
    start = _optional_metadata_integer(metadata, "weight_capture_start_step")
    end = _optional_metadata_integer(metadata, "weight_capture_end_step")
    first_due = max(1, start) if start is not None else 1
    if end is not None and end < first_due:
        return False
    return int(current_update) >= first_due


def _modified_time(path: Path) -> float:
    candidates = (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
    return max(
        (candidate.stat().st_mtime for candidate in candidates if candidate.exists()),
        default=0.0,
    )


def _timestamp_from_epoch(value: float) -> str:
    if value <= 0.0:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _normalise_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("backslashes are not permitted in file paths")
    path = PurePosixPath(value or ".")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("file path must remain inside the selected run")
    return PurePosixPath(*(part for part in path.parts if part not in {"", "."}))


def _relative_path_text(path: PurePosixPath) -> str:
    return "" if str(path) == "." else path.as_posix()


def _wandb_run_reference(wandb_url: str, wandb_run_id: str) -> str:
    parts = tuple(part for part in urlparse(wandb_url).path.split("/") if part)
    try:
        runs_index = parts.index("runs")
    except ValueError as error:
        raise ValueError("W&B run URL does not contain a runs path") from error
    if runs_index < 2:
        raise ValueError("W&B run URL does not identify an entity and project")
    run_id = wandb_run_id or (parts[runs_index + 1] if len(parts) > runs_index + 1 else "")
    if not run_id:
        raise ValueError("W&B run ID is unavailable")
    return f"{parts[runs_index - 2]}/{parts[runs_index - 1]}/{run_id}"


class RunDashboardState:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.reader = LocalChartReader(self.database_path)
        self.lock = threading.Lock()
        self.cached_revision: Optional[Tuple[Any, ...]] = None
        self.cached_figures: Dict[str, Any] = {
            "heatmap": None,
            "heatmap_dimensions": {"layers": 0, "probes": 0},
            "depth": {},
        }

    def status(self) -> Dict[str, Any]:
        status = self.reader.status()
        metadata = self.reader.metadata()
        data_updated_at = metadata.get(
            "data_updated_at",
            metadata.get("updated_at"),
        )
        revision = (
            status["heatmap_count"],
            status["heatmap_maximum_update"],
            status["depth_snapshot_count"],
            status["depth_maximum_update"],
            data_updated_at,
        )
        configuration = json.loads(metadata.get("config_json", "{}"))
        chart_maximum_update = max(
            (
                value
                for value in (
                    status["heatmap_maximum_update"],
                    status["depth_maximum_update"],
                )
                if value is not None
            ),
            default=None,
        )
        heartbeat_update = _optional_metadata_integer(metadata, "current_update")
        maximum_update = max(
            (
                value
                for value in (heartbeat_update, chart_maximum_update)
                if value is not None
            ),
            default=None,
        )
        target_update = _target_update(configuration)
        run_state = _reported_run_state(
            metadata.get("run_state", "unknown"),
            current_update=maximum_update,
            target_update=target_update,
        )
        modified_at = _timestamp_from_epoch(_modified_time(self.database_path))
        artifact_name = metadata.get(
            "artifact_name",
            metadata.get("run_name", self.database_path.parent.name),
        )
        local_run_id = metadata.get("local_run_id", self.database_path.parent.name)
        wandb_run_id = metadata.get("wandb_run_id", "")
        is_legacy_layout = bool(
            not wandb_run_id
            and local_run_id == artifact_name
            and self.database_path.parent.name == artifact_name
        )
        return {
            **status,
            "run_name": artifact_name,
            "artifact_name": artifact_name,
            "local_run_id": local_run_id,
            "dashboard_run_id": (
                f"legacy:{local_run_id}" if is_legacy_layout else local_run_id
            ),
            "wandb_run_id": wandb_run_id,
            "wandb_url": metadata.get("wandb_url", ""),
            "run_state": run_state,
            "target_update": target_update,
            "data_lost": _weight_data_lost(
                metadata,
                current_update=maximum_update,
                snapshot_count=int(status["depth_snapshot_count"]),
            ),
            "is_legacy_layout": is_legacy_layout,
            "created_at": metadata.get("created_at", modified_at),
            "updated_at": metadata.get(
                "heartbeat_at",
                metadata.get("updated_at", modified_at),
            ),
            "heartbeat_at": metadata.get(
                "heartbeat_at",
                metadata.get("updated_at", modified_at),
            ),
            "data_updated_at": data_updated_at or modified_at,
            "host_label": str(configuration.get("host_label", "")),
            "model_type": str(configuration.get("model_type", "")),
            "preset": _preset_from_configuration(configuration),
            # vvv THOG surface only capture/routing controls; history length is now a reversible per-chart Instra setting
            "heatmap_settings": {
                "mode": configuration.get("instrumentation__delta_loss_v_layer_heatmap"),
                "destination": configuration.get("instrumentation__delta_loss_v_layer_heatmap__destination", "local"),
                "abs_limit": configuration.get("instrumentation__delta_loss_v_layer_heatmap_abs_limit", 0.05),
            },
            # ^^^ THOG
            "maximum_update": maximum_update,
            "chart_maximum_update": chart_maximum_update,
            "database_bytes": int(self.database_path.stat().st_size),
            "run_directory": str(self.database_path.parent.resolve()),
            "revision": revision,
        }

    def figures(self) -> Dict[str, Any]:
        status = self.status()
        revision = tuple(status["revision"])
        with self.lock:
            if revision == self.cached_revision:
                return self.cached_figures
            heatmap_history = self.reader.heatmap_history()
            heatmap_figure = None
            maximum_layers = 0
            if heatmap_history:
                maximum_layers = max(len(record["values"]) for record in heatmap_history)
                metadata = self.reader.metadata()
                configuration = json.loads(metadata.get("config_json", "{}"))
                abs_limit = float(
                    configuration.get(
                        "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
                        0.05,
                    )
                )
                heatmap_figure = probe_curves._delta_loss_heatmap_figure(
                    heatmap_history,
                    maximum_layers=maximum_layers,
                    abs_limit=abs_limit,
                ).to_plotly_json()
            snapshots = self.reader.depth_weight_snapshots()
            depth_figures = {}
            if snapshots:
                available = snapshots[-1].get("families", {})
                for chart_name in depth_curves._CHART_FAMILIES:
                    if chart_name not in available:
                        continue
                    depth_figures[chart_name] = depth_curves._build_depth_plotly_figure(
                        snapshots,
                        chart_name,
                    ).to_plotly_json()
            self.cached_revision = revision
            self.cached_figures = {
                "heatmap": heatmap_figure,
                "heatmap_dimensions": {
                    "layers": maximum_layers,
                    "probes": min(len(heatmap_history), 512),
                },
                "depth": depth_figures,
            }
            return self.cached_figures


class DashboardCatalog:
    def __init__(
        self,
        *,
        root: Path,
        requested_run: Optional[str] = None,
        wandb_api_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.requested_run = requested_run
        self.wandb_api_factory = wandb_api_factory
        self.lock = threading.Lock()
        self.states: Dict[Path, RunDashboardState] = {}
        self.wandb_file_cache: Dict[str, Tuple[float, Tuple[Dict[str, Any], ...]]] = {}

    def _candidate_paths(self) -> Tuple[Path, ...]:
        if not self.root.is_dir():
            return ()
        candidates = tuple(
            sorted(
                self.root.glob(f"**/{LOCAL_CHART_DATABASE_NAME}"),
                key=_modified_time,
                reverse=True,
            )
        )
        if not self.requested_run:
            return candidates
        supplied = Path(self.requested_run)
        if supplied.is_file():
            return (supplied,)
        matches = []
        for path in candidates:
            if self.requested_run in {path.parent.name, path.parent.parent.name}:
                matches.append(path)
                continue
            try:
                metadata = LocalChartReader(path).metadata()
            except (OSError, sqlite3.DatabaseError):
                continue
            if self.requested_run in {
                metadata.get("local_run_id"),
                metadata.get("wandb_run_id"),
                metadata.get("artifact_name"),
                metadata.get("run_name"),
            }:
                matches.append(path)
        return tuple(matches)

    def _state_for_path(self, path: Path) -> RunDashboardState:
        resolved = path.resolve()
        with self.lock:
            state = self.states.get(resolved)
            if state is None:
                state = RunDashboardState(resolved)
                self.states[resolved] = state
            return state

    def runs(self) -> Dict[str, Any]:
        runs = []
        for path in self._candidate_paths():
            try:
                status = self._state_for_path(path).status()
            except (OSError, sqlite3.DatabaseError, ValueError, json.JSONDecodeError):
                continue
            runs.append(status)
        runs.sort(
            key=lambda run: (str(run["created_at"]), str(run["updated_at"])),
            reverse=True,
        )
        return {
            "runs": runs,
            "waiting": not bool(runs),
            "requested_run": self.requested_run,
            "recommended_run_id": self._recommended_run_id(runs),
            "root": str(self.root.resolve()),
        }

    def _recommended_run_id(self, runs: list[Dict[str, Any]]) -> Optional[str]:
        if not runs:
            return None
        candidates = list(runs)
        if self.requested_run:
            artifact_matches = [
                run
                for run in runs
                if self.requested_run
                in {str(run["artifact_name"]), str(run["run_name"])}
            ]
            exact = [
                run
                for run in runs
                if self.requested_run
                in {str(run["local_run_id"]), str(run["wandb_run_id"])}
                and not (artifact_matches and bool(run["is_legacy_layout"]))
            ]
            candidates = exact or artifact_matches or candidates

        preferred = max(
            candidates,
            key=lambda run: (
                _active_run_state(run["run_state"]),
                bool(run["wandb_run_id"]),
                not bool(run["is_legacy_layout"]),
                str(run["updated_at"]),
            ),
        )
        return str(preferred["dashboard_run_id"])

    def state_for_run(self, run_name: str) -> RunDashboardState:
        artifact_matches = []
        dashboard_matches = []
        exact_matches = []
        for path in self._candidate_paths():
            state = self._state_for_path(path)
            try:
                status = state.status()
            except (OSError, sqlite3.DatabaseError, ValueError, json.JSONDecodeError):
                continue
            if run_name == str(status["dashboard_run_id"]):
                dashboard_matches.append((state, status))
                continue
            if run_name in {
                str(status["local_run_id"]),
                str(status["wandb_run_id"]),
            }:
                exact_matches.append((state, status))
                continue
            if run_name in {
                str(status["run_name"]),
                str(status["artifact_name"]),
            }:
                artifact_matches.append((state, status))
        if dashboard_matches:
            return dashboard_matches[0][0]
        if exact_matches:
            if not (
                all(bool(item[1]["is_legacy_layout"]) for item in exact_matches)
                and any(not bool(item[1]["is_legacy_layout"]) for item in artifact_matches)
            ):
                return max(
                    exact_matches,
                    key=lambda item: (
                        _active_run_state(item[1]["run_state"]),
                        bool(item[1]["wandb_run_id"]),
                        not bool(item[1]["is_legacy_layout"]),
                        str(item[1]["updated_at"]),
                    ),
                )[0]
        if artifact_matches:
            return max(
                artifact_matches,
                key=lambda item: (
                    _active_run_state(item[1]["run_state"]),
                    bool(item[1]["wandb_run_id"]),
                    not bool(item[1]["is_legacy_layout"]),
                    str(item[1]["updated_at"]),
                ),
            )[0]
        raise KeyError(f"local chart run not found: {run_name}")

    def delete_run(self, run_name: str) -> Dict[str, Any]:
        state = self.state_for_run(run_name)
        database_path = state.database_path.resolve()
        root = self.root.resolve()
        try:
            database_path.relative_to(root)
        except ValueError as error:
            raise PermissionError(
                "refusing to delete a local chart database outside the configured root"
            ) from error
        if database_path.name != LOCAL_CHART_DATABASE_NAME:
            raise ValueError(f"unexpected local chart database name: {database_path.name}")

        status = state.status()
        deleted = []
        with state.lock:
            for candidate in (
                database_path,
                Path(f"{database_path}-wal"),
                Path(f"{database_path}-shm"),
            ):
                if not candidate.exists():
                    continue
                candidate.unlink()
                deleted.append(str(candidate))

        removed_directory = False
        if not bool(status["is_legacy_layout"]):
            try:
                database_path.parent.rmdir()
                removed_directory = True
            except OSError:
                pass
        with self.lock:
            self.states.pop(database_path, None)
        return {
            "deleted_run_id": status["dashboard_run_id"],
            "deleted_files": deleted,
            "removed_directory": removed_directory,
        }

    def _resolved_local_path(
        self,
        run_name: str,
        relative_path: str,
    ) -> Tuple[RunDashboardState, Path, PurePosixPath]:
        state = self.state_for_run(run_name)
        root = state.database_path.parent.resolve()
        relative = _normalise_relative_path(relative_path)
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise PermissionError("symbolic links are not accessible through instra")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PermissionError(
                "refusing to access a path outside the selected instra run"
            ) from error
        return state, resolved, relative

    def local_files(self, run_name: str, relative_path: str = "") -> Dict[str, Any]:
        state, directory, relative = self._resolved_local_path(run_name, relative_path)
        if not directory.is_dir():
            raise NotADirectoryError(relative_path)
        entries = []
        for child in directory.iterdir():
            child_stat = child.lstat()
            child_relative = relative / child.name
            if stat.S_ISLNK(child_stat.st_mode):
                kind = "symlink"
                size = None
            elif stat.S_ISDIR(child_stat.st_mode):
                kind = "folder"
                size = None
            elif stat.S_ISREG(child_stat.st_mode):
                kind = "file"
                size = int(child_stat.st_size)
            else:
                kind = "other"
                size = None
            entry = {
                "name": child.name,
                "path": _relative_path_text(child_relative),
                "kind": kind,
                "size": size,
                "modified_at": _timestamp_from_epoch(child_stat.st_mtime),
                "mime_type": mimetypes.guess_type(child.name)[0] or "",
            }
            if kind == "folder":
                folder_count = 0
                file_count = 0
                try:
                    for descendant in child.iterdir():
                        descendant_stat = descendant.lstat()
                        if stat.S_ISLNK(descendant_stat.st_mode):
                            continue
                        if stat.S_ISDIR(descendant_stat.st_mode):
                            folder_count += 1
                        elif stat.S_ISREG(descendant_stat.st_mode):
                            file_count += 1
                except OSError:
                    pass
                entry["folder_count"] = folder_count
                entry["file_count"] = file_count
            entries.append(entry)
        entries.sort(
            key=lambda entry: (
                entry["kind"] != "folder",
                str(entry["name"]).casefold(),
            )
        )
        status = state.status()
        parent = relative.parent
        return {
            "source": "instra",
            "available": True,
            "root_path": str(state.database_path.parent.resolve()),
            "current_path": _relative_path_text(relative),
            "parent_path": None if str(relative) == "." else _relative_path_text(parent),
            "entries": entries,
            "entry_count": len(entries),
            "run_id": status["dashboard_run_id"],
        }

    def local_file(self, run_name: str, relative_path: str) -> Path:
        _state, path, _relative = self._resolved_local_path(run_name, relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        return path

    def _wandb_file_manifest(
        self,
        status: Dict[str, Any],
        *,
        refresh: bool,
    ) -> Tuple[Dict[str, Any], ...]:
        reference = _wandb_run_reference(
            str(status["wandb_url"]),
            str(status["wandb_run_id"]),
        )
        cached = self.wandb_file_cache.get(reference)
        if cached and not refresh and time.monotonic() - cached[0] < _WANDB_FILE_CACHE_SECONDS:
            return cached[1]
        if self.wandb_api_factory is None:
            import wandb

            api = wandb.Api(timeout=10)
        else:
            api = self.wandb_api_factory()
        run = api.run(reference)
        manifest = []
        for index, remote_file in enumerate(run.files()):
            if index >= _WANDB_FILE_LIMIT:
                break
            name = str(remote_file.name)
            relative = _normalise_relative_path(name)
            if str(relative) == ".":
                continue
            try:
                size = int(remote_file.size)
            except (AttributeError, TypeError, ValueError):
                size = None
            try:
                mime_type = str(remote_file.mimetype or "")
            except AttributeError:
                mime_type = mimetypes.guess_type(name)[0] or ""
            try:
                digest = str(remote_file.md5 or "")
            except AttributeError:
                digest = ""
            try:
                modified_at = str(remote_file.updated_at or "")
            except AttributeError:
                modified_at = ""
            try:
                download_url = str(remote_file.direct_url or "")
            except AttributeError:
                download_url = ""
            if not download_url:
                try:
                    download_url = str(remote_file.url or "")
                except AttributeError:
                    download_url = ""
            if urlparse(download_url).scheme not in {"http", "https"}:
                download_url = ""
            manifest.append(
                {
                    "name": relative.name,
                    "path": _relative_path_text(relative),
                    "kind": "file",
                    "size": size,
                    "modified_at": modified_at,
                    "mime_type": mime_type,
                    "digest": digest,
                    "download_url": download_url,
                }
            )
        resolved = tuple(manifest)
        self.wandb_file_cache[reference] = (time.monotonic(), resolved)
        return resolved

    def wandb_files(
        self,
        run_name: str,
        relative_path: str = "",
        *,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        state = self.state_for_run(run_name)
        status = state.status()
        wandb_url = str(status["wandb_url"])
        if not wandb_url or not status["wandb_run_id"]:
            return {
                "source": "wandb",
                "available": False,
                "error": "This local run is not linked to a W&B run.",
                "entries": [],
                "current_path": "",
                "parent_path": None,
                "wandb_files_url": "",
            }
        relative = _normalise_relative_path(relative_path)
        try:
            manifest = self._wandb_file_manifest(status, refresh=refresh)
        except Exception as error:
            return {
                "source": "wandb",
                "available": False,
                "error": f"W&B files could not be loaded: {error}",
                "entries": [],
                "current_path": "",
                "parent_path": None,
                "wandb_files_url": f"{wandb_url.rstrip('/')}/files",
            }
        prefix_parts = () if str(relative) == "." else relative.parts
        folders: Dict[str, Dict[str, Any]] = {}
        folder_descendants: Dict[str, set[str]] = {}
        files = []
        for item in manifest:
            item_path = PurePosixPath(str(item["path"]))
            if item_path.parts[: len(prefix_parts)] != prefix_parts:
                continue
            remainder = item_path.parts[len(prefix_parts) :]
            if not remainder:
                continue
            if len(remainder) > 1:
                folder_name = remainder[0]
                folder_path = relative / folder_name
                folder = folders.setdefault(
                    folder_name,
                    {
                        "name": folder_name,
                        "path": _relative_path_text(folder_path),
                        "kind": "folder",
                        "size": 0,
                        "modified_at": "",
                        "mime_type": "",
                        "child_count": 0,
                        "file_count": 0,
                        "folder_count": 0,
                    },
                )
                folder["child_count"] += 1
                folder["file_count"] += 1
                descendants = folder_descendants.setdefault(folder_name, set())
                for depth in range(1, len(remainder) - 1):
                    descendants.add("/".join(remainder[1 : depth + 1]))
                if item["size"] is not None:
                    folder["size"] += int(item["size"])
                continue
            files.append(dict(item))
        for folder_name, descendants in folder_descendants.items():
            folders[folder_name]["folder_count"] = len(descendants)
        entries = sorted(folders.values(), key=lambda item: str(item["name"]).casefold())
        entries.extend(sorted(files, key=lambda item: str(item["name"]).casefold()))
        return {
            "source": "wandb",
            "available": True,
            "current_path": _relative_path_text(relative),
            "parent_path": None if str(relative) == "." else _relative_path_text(relative.parent),
            "entries": entries,
            "entry_count": len(entries),
            "manifest_count": len(manifest),
            "wandb_files_url": f"{wandb_url.rstrip('/')}/files",
            "run_id": status["dashboard_run_id"],
        }


def _handler_for(catalog: DashboardCatalog):
    plotly_javascript = get_plotlyjs().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            body: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache_control: str = "no-store",
        ) -> None:
            # vvv THOG browser navigation/refresh can close a response socket while this worker is writing; treat that as a normal client disconnect
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache_control)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            # ^^^ THOG

        def _send_json(
            self,
            value: Any,
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(
                value,
                cls=PlotlyJSONEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send(
                body,
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _send_file(self, path: Path, *, download: bool) -> None:
            file_size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            disposition = "attachment" if download else "inline"
            encoded_name = quote(path.name, safe="")
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Content-Disposition", f"{disposition}; filename*=UTF-8''{encoded_name}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "sandbox; default-src 'none'")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path in {"/", "/runs"} or path.startswith("/runs/"):
                    self._send(
                        (_ASSET_ROOT / "index.html").read_bytes(),
                        content_type="text/html; charset=utf-8",
                    )
                    return
                if path == "/plotly.min.js":
                    self._send(
                        plotly_javascript,
                        content_type="text/javascript; charset=utf-8",
                        cache_control="public, max-age=86400",
                    )
                    return
                if path.startswith("/assets/"):
                    asset_name = Path(path).name
                    if asset_name not in _ASSET_NAMES:
                        raise FileNotFoundError(asset_name)
                    asset_path = _ASSET_ROOT / asset_name
                    content_type = mimetypes.guess_type(asset_path.name)[0]
                    self._send(
                        asset_path.read_bytes(),
                        content_type=f"{content_type or 'application/octet-stream'}; charset=utf-8",
                    )
                    return
                if path == "/api/runs":
                    self._send_json(catalog.runs())
                    return
                if path in {"/api/local-files", "/api/wandb-files"}:
                    run_name = query.get("run", [""])[0]
                    if not run_name:
                        self._send_json(
                            {"error": "run query parameter is required"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    relative_path = query.get("path", [""])[0]
                    if path == "/api/local-files":
                        value = catalog.local_files(run_name, relative_path)
                    else:
                        refresh = query.get("refresh", ["0"])[0] == "1"
                        value = catalog.wandb_files(
                            run_name,
                            relative_path,
                            refresh=refresh,
                        )
                    self._send_json(value)
                    return
                if path == "/api/local-file":
                    run_name = query.get("run", [""])[0]
                    relative_path = query.get("path", [""])[0]
                    if not run_name or not relative_path:
                        self._send_json(
                            {"error": "run and path query parameters are required"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    local_file = catalog.local_file(run_name, relative_path)
                    self._send_file(
                        local_file,
                        download=query.get("download", ["0"])[0] == "1",
                    )
                    return
                if path in {"/api/status", "/api/figures"}:
                    run_name = query.get("run", [""])[0]
                    if not run_name:
                        self._send_json(
                            {"error": "run query parameter is required"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    state = catalog.state_for_run(run_name)
                    value = state.status() if path == "/api/status" else state.figures()
                    self._send_json(value)
                    return
                self._send(
                    b"not found\n",
                    content_type="text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                )
            except (FileNotFoundError, KeyError, NotADirectoryError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
            except PermissionError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.FORBIDDEN)
            except ValueError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path != "/api/run":
                    self._send(
                        b"not found\n",
                        content_type="text/plain; charset=utf-8",
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                run_name = query.get("run", [""])[0]
                if not run_name:
                    self._send_json(
                        {"error": "run query parameter is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_json(catalog.delete_run(run_name))
            except (FileNotFoundError, KeyError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
            except PermissionError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.FORBIDDEN)
            except ValueError as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, _format: str, *_arguments: Any) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve instra charts and files from compact local run data"
    )
    parser.add_argument(
        "--run",
        help="optional artifact name or charts.sqlite3 path to wait for and show",
    )
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    catalog = DashboardCatalog(root=arguments.root, requested_run=arguments.run)
    server = ThreadingHTTPServer(
        (str(arguments.host), int(arguments.port)),
        _handler_for(catalog),
    )
    url = f"http://{arguments.host}:{arguments.port}/"
    print(f"instra: {url}", flush=True)
    if arguments.run:
        print(f"waiting for local chart run: {arguments.run}", flush=True)
    else:
        print(f"watching for local chart runs below: {arguments.root.resolve()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
