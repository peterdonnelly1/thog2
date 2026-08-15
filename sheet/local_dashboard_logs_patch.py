# vvv THOG
"""Add a bounded incremental train.log endpoint to the local dashboard launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


_DEFAULT_INITIAL_BYTES = 1024 * 1024
_MAXIMUM_BYTES = 2 * 1024 * 1024


def _timestamped_artifact_suffix(directory_name: str) -> str:
    # YYMMDD_HHMM_<artifact>; return the preserved artifact prefix after timestamp.
    if len(directory_name) >= 13 and directory_name[6:7] == "_" and directory_name[11:12] == "_":
        stamp = directory_name[:11].replace("_", "")
        if stamp.isdigit():
            return directory_name[12:]
    return directory_name


def _matching_train_logs(catalog: Any, state: Any) -> tuple[Path, ...]:
    status = state.status()
    artifact_name = str(status.get("artifact_name") or status.get("run_name") or "").strip()
    if not artifact_name:
        return ()
    root = Path(catalog.root)
    candidates = []
    for candidate in root.glob("*/train.log"):
        try:
            if not candidate.is_file():
                continue
        except OSError:
            continue
        directory_name = candidate.parent.name
        suffix = _timestamped_artifact_suffix(directory_name)
        if (
            directory_name == artifact_name
            or directory_name.endswith(f"_{artifact_name}")
            or artifact_name.startswith(suffix)
        ):
            candidates.append(candidate)
    return tuple(candidates)


def _resolve_train_log(catalog: Any, state: Any, dashboard_module: Any) -> Optional[Path]:
    candidates = _matching_train_logs(catalog, state)
    if not candidates:
        return None
    target_mtime = float(dashboard_module._modified_time(state.database_path))

    def distance(candidate: Path) -> tuple[float, float]:
        try:
            modified = float(candidate.stat().st_mtime)
        except OSError:
            return (float("inf"), 0.0)
        return (abs(modified - target_mtime), -modified)

    return min(candidates, key=distance)


def _read_log_tail(
    catalog: Any,
    dashboard_module: Any,
    run_name: str,
    *,
    offset: Optional[int],
    maximum_bytes: int,
) -> dict[str, Any]:
    state = catalog.state_for_run(run_name)
    log_path = _resolve_train_log(catalog, state, dashboard_module)
    if log_path is None:
        return {
            "available": False,
            "path": "",
            "size": 0,
            "start": 0,
            "end": 0,
            "reset": True,
            "text": "",
        }

    size = int(log_path.stat().st_size)
    limit = max(4096, min(_MAXIMUM_BYTES, int(maximum_bytes)))
    reset = offset is None or offset < 0 or offset > size
    if reset:
        start = max(0, size - min(limit, _DEFAULT_INITIAL_BYTES))
    else:
        start = int(offset)
        if size - start > limit:
            start = max(0, size - limit)
            reset = True

    with log_path.open("rb") as handle:
        handle.seek(start)
        if reset and start > 0:
            handle.readline()  # discard the partial first line from a bounded tail read
            start = int(handle.tell())
        payload = handle.read(limit)
        end = int(handle.tell())

    return {
        "available": True,
        "path": str(log_path.resolve()),
        "size": size,
        "start": start,
        "end": end,
        "reset": reset,
        "text": payload.decode("utf-8", errors="replace"),
    }


def install(dashboard_module: Any) -> None:
    original_handler_for = dashboard_module._handler_for

    def handler_for(catalog: Any):
        handler = original_handler_for(catalog)
        original_do_get = handler.do_GET

        def do_get(self: Any) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/log":
                original_do_get(self)
                return
            query = parse_qs(parsed.query)
            run_name = query.get("run", [""])[0]
            if not run_name:
                self._send_json(
                    {"error": "run query parameter is required"},
                    status=dashboard_module.HTTPStatus.BAD_REQUEST,
                )
                return
            raw_offset = query.get("offset", [""])[0]
            raw_maximum = query.get("max_bytes", [str(_MAXIMUM_BYTES)])[0]
            try:
                offset = None if raw_offset == "" else int(raw_offset)
                maximum_bytes = int(raw_maximum)
                self._send_json(
                    _read_log_tail(
                        catalog,
                        dashboard_module,
                        run_name,
                        offset=offset,
                        maximum_bytes=maximum_bytes,
                    )
                )
            except (FileNotFoundError, KeyError) as error:
                self._send_json({"error": str(error)}, status=dashboard_module.HTTPStatus.NOT_FOUND)
            except (OSError, ValueError) as error:
                self._send_json({"error": str(error)}, status=dashboard_module.HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_json(
                    {"error": str(error)},
                    status=dashboard_module.HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        handler.do_GET = do_get
        return handler

    dashboard_module._handler_for = handler_for
# ^^^ THOG
