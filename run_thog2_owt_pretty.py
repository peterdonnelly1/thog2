# vvv THOG
from __future__ import annotations

from typing import Any, Dict

import run_thog2_owt as _base


_CONSOLE_EVENT_PREFIXES = {
    "optimizer_progress": "T",
    "evaluation_completed": "V",
}


def _format_field(label: str, value: object, width: int) -> str:
    text = str(value)
    return f"{label} {text:>{width}}"


def _format_step_line(prefix: str, run_id: str, payload: Dict[str, Any]) -> str:
    cumulative_training_seconds = payload.get("cumulative_training_seconds", "-")
    tokens_per_second = payload.get("tok/s", "-")
    fields = [
        prefix,
        _format_field("cum time", cumulative_training_seconds, 6),
        _format_field("tok/s", tokens_per_second, 12),
        _format_field("step", payload.get("completed_updates", "-"), 6),
        _format_field("cum tokens", payload.get("consumed_tokens", "-"), 14),
    ]
    if prefix == "T":
        fields.extend(
            (
                _format_field("loss", payload.get("training_loss", "-"), 9),
                _format_field("learning rate", payload.get("learning_rate", "-"), 10),
                _format_field("gradient norm", payload.get("gradient_norm", "-"), 8),
            )
        )
    else:
        fields.extend(
            (
                _format_field("train loss", payload.get("training_loss", "-"), 9),
                _format_field("val loss", payload.get("validation_loss", "-"), 9),
            )
        )
    fields.append(f"run_id {run_id}")
    return " | ".join(fields)


def format_console_progress_line(run_id: str, event: str, payload: Dict[str, Any]) -> str:
    prefix = _CONSOLE_EVENT_PREFIXES.get(event)
    if prefix is not None:
        return _format_step_line(prefix, run_id, payload)
    fields = [event]
    fields.extend(f"{key.replace('_', ' ')} {value}" for key, value in payload.items())
    fields.append(f"run_id {run_id}")
    return " | ".join(fields)


class PrettyOwtTrainer(_base.OwtTrainer):
    """Canonical OWT trainer with compact aligned console progress."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._console_cumulative_training_seconds = 0.0

    def _print_progress(self, run_id: str, event: str, **payload: Any) -> None:
        if not self.distributed.is_primary:
            return
        values = dict(payload)
        if "consumed_tokens" in values:
            values["consumed_tokens"] = int(values["consumed_tokens"]) * int(self.distributed.world_size)
        if "cumulative_training_seconds" in values:
            self._console_cumulative_training_seconds = float(values["cumulative_training_seconds"])
        elif event == "evaluation_completed":
            values["cumulative_training_seconds"] = self._console_cumulative_training_seconds
        formatted = _base.format_console_progress_payload(_base.add_console_tokens_per_second(values))
        print(format_console_progress_line(run_id, event, formatted), flush=True)


def _config_from_arguments(arguments):
    arguments.experiment_prefix = arguments.run_name or "NO_PREFIX"
    return _original_config_from_arguments(arguments)


_original_config_from_arguments = _base.config_from_arguments
_base.config_from_arguments = _config_from_arguments
_base.OwtTrainer = PrettyOwtTrainer


if __name__ == "__main__":
    raise SystemExit(_base.main())
# ^^^ THOG
