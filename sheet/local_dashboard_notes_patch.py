# vvv THOG
"""Add durable user notes to the local INSTRA run database and HTTP API."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


_MAXIMUM_NOTE_CHARACTERS = 4000
_MAXIMUM_REQUEST_BYTES = 16 * 1024


def _normalise_notes(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("notes must be text")
    notes = value.replace("\r\n", "\n").replace("\r", "\n")
    if len(notes) > _MAXIMUM_NOTE_CHARACTERS:
        raise ValueError(
            f"notes must not exceed {_MAXIMUM_NOTE_CHARACTERS} characters"
        )
    return notes


def _write_run_notes(database_path: Path, notes: str) -> str:
    resolved = _normalise_notes(notes)
    connection = sqlite3.connect(Path(database_path), timeout=30.0)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("user_notes", resolved),
                (
                    "user_notes_updated_at",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return resolved


def install(dashboard: Any) -> None:
    if getattr(dashboard, "_thog2_dashboard_notes_patch_installed", False):
        return
    dashboard._thog2_dashboard_notes_patch_installed = True

    original_status = dashboard.RunDashboardState.status

    def status_with_notes(self: Any) -> dict[str, Any]:
        status = original_status(self)
        metadata = self.reader.metadata()
        return {
            **status,
            "notes": metadata.get("user_notes", ""),
            "notes_updated_at": metadata.get("user_notes_updated_at", ""),
        }

    dashboard.RunDashboardState.status = status_with_notes

    original_handler_for = dashboard._handler_for

    def handler_for_notes(catalog: Any) -> Any:
        base_handler = original_handler_for(catalog)

        class NotesHandler(base_handler):
            def do_PUT(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/run-notes":
                    self._send(
                        b"not found\n",
                        content_type="text/plain; charset=utf-8",
                        status=dashboard.HTTPStatus.NOT_FOUND,
                    )
                    return
                query = parse_qs(parsed.query)
                run_name = query.get("run", [""])[0]
                if not run_name:
                    self._send_json(
                        {"error": "run query parameter is required"},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length < 0 or content_length > _MAXIMUM_REQUEST_BYTES:
                        raise ValueError("notes request is too large")
                    payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("notes request must be a JSON object")
                    state = catalog.state_for_run(run_name)
                    notes = _write_run_notes(state.database_path, payload.get("notes"))
                    self._send_json({"run": run_name, "notes": notes})
                except (FileNotFoundError, KeyError) as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.NOT_FOUND,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                except (OSError, sqlite3.Error) as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        return NotesHandler

    dashboard._handler_for = handler_for_notes


__all__ = ["install"]
# ^^^ THOG
