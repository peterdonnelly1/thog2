from pathlib import Path

# vvv THOG apply stable resume naming and session-correct throughput accounting


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    Path("sheet/run_config.py"),
    '''        fields = [
            f"n_{self.max_iters}",
            f"b_{self.batch_size}",
''',
    '''        fields = [
            # vvv THOG stable artifact identity must not depend on the mutable target update count
            # f"n_{self.max_iters}",
            # ^^^ THOG
            f"b_{self.batch_size}",
''',
)

replace_once(
    Path("sheet/run_naming.py"),
    '''    fields = [
        f"n_{max_iters}",
        f"b_{batch_size}",
''',
    '''    fields = [
        # vvv THOG stable artifact identity must not depend on the mutable target update count
        # f"n_{max_iters}",
        # ^^^ THOG
        f"b_{batch_size}",
''',
)

replace_once(
    Path("run_thog2_owt.py"),
    '''def add_console_tokens_per_second(payload: Dict[str, Any]) -> Dict[str, Any]:
    values = dict(payload)
    elapsed = values.get("cumulative_training_seconds", values.get("training_seconds"))
    consumed_tokens = values.get("consumed_tokens")
    if elapsed is None or consumed_tokens is None:
        return values
    elapsed_value = float(elapsed)
    if elapsed_value <= 0.0:
        return values
    values["tok/s"] = float(consumed_tokens) / elapsed_value
    return values
''',
    '''# vvv THOG resumed throughput uses only tokens processed by the current process session
# Lifetime consumed_tokens remains available for progress and accounting.
def add_console_tokens_per_second(payload: Dict[str, Any]) -> Dict[str, Any]:
    values = dict(payload)
    elapsed = values.get("cumulative_training_seconds", values.get("training_seconds"))
    throughput_tokens = values.pop("session_consumed_tokens", None)
    if throughput_tokens is None:
        throughput_tokens = values.get("consumed_tokens")
    if elapsed is None or throughput_tokens is None:
        return values
    elapsed_value = float(elapsed)
    if elapsed_value <= 0.0:
        return values
    values["tok/s"] = float(throughput_tokens) / elapsed_value
    return values
# ^^^ THOG
''',
)

replace_once(
    Path("run_thog2_owt.py"),
    '''        if "consumed_tokens" in values:
            values["consumed_tokens"] = int(values["consumed_tokens"]) * int(self.distributed.world_size)
        super()._print_progress(run_id, event, **format_console_progress_payload(add_console_tokens_per_second(values)))  # <<< THOG console progress now includes right-aligned tok/s and stable numeric widths.
''',
    '''        if "consumed_tokens" in values:
            values["consumed_tokens"] = int(values["consumed_tokens"]) * int(self.distributed.world_size)
        # vvv THOG apply the same global-token multiplier to session throughput accounting
        if "session_consumed_tokens" in values:
            values["session_consumed_tokens"] = int(values["session_consumed_tokens"]) * int(self.distributed.world_size)
        # ^^^ THOG
        super()._print_progress(run_id, event, **format_console_progress_payload(add_console_tokens_per_second(values)))  # <<< THOG console progress now includes right-aligned tok/s and stable numeric widths.
''',
)

replace_once(
    Path("run_thog2_owt.py"),
    '''        result["budget"]["tokens_per_update"] *= multiplier
        result["budget"]["consumed_tokens"] *= multiplier
        for row in result["updates"]:
            row["consumed_tokens"] *= multiplier
        for row in result["evaluations"]:
            row["consumed_tokens"] *= multiplier
        result["timing"]["tokens_per_training_second"] *= multiplier
''',
    '''        result["budget"]["tokens_per_update"] *= multiplier
        result["budget"]["consumed_tokens"] *= multiplier
        # vvv THOG preserve global-token accounting for resumed-session throughput fields
        result["budget"]["session_consumed_tokens"] *= multiplier
        # ^^^ THOG
        for row in result["updates"]:
            row["consumed_tokens"] *= multiplier
            # vvv THOG resumed-session token counts are global under DDP
            row["session_consumed_tokens"] *= multiplier
            # ^^^ THOG
        for row in result["evaluations"]:
            row["consumed_tokens"] *= multiplier
            # vvv THOG resumed-session token counts are global under DDP
            row["session_consumed_tokens"] *= multiplier
            # ^^^ THOG
        result["timing"]["tokens_per_training_second"] *= multiplier
''',
)

path = Path("sheet/stage6_trainer.py")
text = path.read_text(encoding="utf-8")
anchor = '''def trace_digest(trace) -> str:
    payload = json.dumps(trace, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


'''
replacement = '''def trace_digest(trace) -> str:
    payload = json.dumps(trace, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# vvv THOG resumed runs retain lifetime token totals but measure throughput from this process session only
def _session_consumed_tokens(
    starting_completed_updates: int,
    completed_updates: int,
    tokens_per_update: int,
) -> int:
    session_completed_updates = completed_updates - starting_completed_updates
    if session_completed_updates < 0:
        raise ValueError("completed updates moved backwards during the training session")
    return session_completed_updates * tokens_per_update
# ^^^ THOG


'''
if anchor not in text:
    raise SystemExit("stage6 trace anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''        tokens_per_update = (
            self.config.batch_size
            * self.config.gradient_accumulation_steps
            * self.config.block_size
        )
        training_seconds = 0.0
'''
replacement = '''        tokens_per_update = (
            self.config.batch_size
            * self.config.gradient_accumulation_steps
            * self.config.block_size
        )
        # vvv THOG anchor resumed-session timing before the first update in this process
        starting_completed_updates = self.state.completed_updates
        # ^^^ THOG
        training_seconds = 0.0
'''
if anchor not in text:
    raise SystemExit("stage6 run start anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''            completed_updates = self.state.completed_updates
            update_rows.append(
                {
                    **metrics,
                    "update_seconds": elapsed,
                    "cumulative_training_seconds": training_seconds,
                    "cumulative_wall_seconds": time.perf_counter() - wall_started,
                    "consumed_tokens": completed_updates * tokens_per_update,
                }
            )
'''
replacement = '''            completed_updates = self.state.completed_updates
            # vvv THOG separate lifetime token accounting from tokens processed since this resume/start
            current_session_consumed_tokens = _session_consumed_tokens(
                starting_completed_updates,
                completed_updates,
                tokens_per_update,
            )
            # ^^^ THOG
            update_rows.append(
                {
                    **metrics,
                    "update_seconds": elapsed,
                    "cumulative_training_seconds": training_seconds,
                    "cumulative_wall_seconds": time.perf_counter() - wall_started,
                    "consumed_tokens": completed_updates * tokens_per_update,
                    "session_consumed_tokens": current_session_consumed_tokens,
                }
            )
'''
if anchor not in text:
    raise SystemExit("stage6 update row anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                    completed_updates=completed_updates,
                    consumed_tokens=completed_updates * tokens_per_update,
                    training_loss=metrics["training_loss"],
'''
replacement = '''                    completed_updates=completed_updates,
                    consumed_tokens=completed_updates * tokens_per_update,
                    session_consumed_tokens=current_session_consumed_tokens,
                    training_loss=metrics["training_loss"],
'''
if anchor not in text:
    raise SystemExit("stage6 optimizer progress anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                        "completed_updates": completed_updates,
                        "consumed_tokens": completed_updates * tokens_per_update,
                        "training_seconds": training_seconds,
'''
replacement = '''                        "completed_updates": completed_updates,
                        "consumed_tokens": completed_updates * tokens_per_update,
                        "session_consumed_tokens": current_session_consumed_tokens,
                        "training_seconds": training_seconds,
'''
if anchor not in text:
    raise SystemExit("stage6 periodic evaluation row anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                    completed_updates=completed_updates,
                    consumed_tokens=completed_updates * tokens_per_update,
                    cumulative_training_seconds=training_seconds,                                                                                       # <<< THOG expose cumulative training time and tok/s on validation rows
'''
replacement = '''                    completed_updates=completed_updates,
                    consumed_tokens=completed_updates * tokens_per_update,
                    session_consumed_tokens=current_session_consumed_tokens,
                    cumulative_training_seconds=training_seconds,                                                                                       # <<< THOG expose cumulative training time and tok/s on validation rows
'''
if anchor not in text:
    raise SystemExit("stage6 periodic evaluation progress anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''        if (
            self.config.eval_interval > 0
            and (
'''
replacement = '''        # vvv THOG final session token count is independent of lifetime completed updates
        final_session_consumed_tokens = _session_consumed_tokens(
            starting_completed_updates,
            self.state.completed_updates,
            tokens_per_update,
        )
        # ^^^ THOG
        if (
            self.config.eval_interval > 0
            and (
'''
if anchor not in text:
    raise SystemExit("stage6 final evaluation anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                    "completed_updates": self.state.completed_updates,
                    "consumed_tokens": self.state.completed_updates * tokens_per_update,
                    "training_seconds": training_seconds,
'''
replacement = '''                    "completed_updates": self.state.completed_updates,
                    "consumed_tokens": self.state.completed_updates * tokens_per_update,
                    "session_consumed_tokens": final_session_consumed_tokens,
                    "training_seconds": training_seconds,
'''
if anchor not in text:
    raise SystemExit("stage6 final evaluation row anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                completed_updates=self.state.completed_updates,
                consumed_tokens=self.state.completed_updates * tokens_per_update,
                cumulative_training_seconds=training_seconds,                                                                                           # <<< THOG expose cumulative training time and tok/s on final validation row
'''
replacement = '''                completed_updates=self.state.completed_updates,
                consumed_tokens=self.state.completed_updates * tokens_per_update,
                session_consumed_tokens=final_session_consumed_tokens,
                cumulative_training_seconds=training_seconds,                                                                                           # <<< THOG expose cumulative training time and tok/s on final validation row
'''
if anchor not in text:
    raise SystemExit("stage6 final evaluation progress anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''            "budget": {
                "completed_updates": self.state.completed_updates,
                "tokens_per_update": tokens_per_update,
                "consumed_tokens": self.state.completed_updates * tokens_per_update,
            },
'''
replacement = '''            "budget": {
                "completed_updates": self.state.completed_updates,
                "tokens_per_update": tokens_per_update,
                "consumed_tokens": self.state.completed_updates * tokens_per_update,
                # vvv THOG expose this process session separately from lifetime progress
                "session_completed_updates": self.state.completed_updates - starting_completed_updates,
                "session_consumed_tokens": final_session_consumed_tokens,
                # ^^^ THOG
            },
'''
if anchor not in text:
    raise SystemExit("stage6 budget anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''                "tokens_per_training_second": (
                    self.state.completed_updates * tokens_per_update / training_seconds
                    if training_seconds > 0.0
                    else 0.0
                ),
'''
replacement = '''                "tokens_per_training_second": (
                    final_session_consumed_tokens / training_seconds
                    if training_seconds > 0.0
                    else 0.0
                ),
'''
if anchor not in text:
    raise SystemExit("stage6 final throughput anchor not found")
text = text.replace(anchor, replacement, 1)

anchor = '''            completed_updates=self.state.completed_updates,
            consumed_tokens=self.state.completed_updates * tokens_per_update,
            final_validation_loss=evaluation_rows[-1]["val"],
'''
replacement = '''            completed_updates=self.state.completed_updates,
            consumed_tokens=self.state.completed_updates * tokens_per_update,
            session_consumed_tokens=final_session_consumed_tokens,
            final_validation_loss=evaluation_rows[-1]["val"],
'''
if anchor not in text:
    raise SystemExit("stage6 run completed anchor not found")
text = text.replace(anchor, replacement, 1)

path.write_text(text, encoding="utf-8")

Path("tests/test_resume_identity_and_throughput.py").write_text(
    '''# vvv THOG
from __future__ import annotations

import math

from run_thog2_owt import add_console_tokens_per_second
from sheet.run_config import OwtRunConfig
from sheet.run_naming import build_artifact_name
from sheet.stage6_trainer import _session_consumed_tokens


def test_owt_artifact_identity_does_not_change_when_only_target_update_count_changes() -> None:
    short = OwtRunConfig(
        model_type="sheet",
        run_start_label="260714-0800",
        max_iters=10_000,
        warmup_iters=100,
    )
    long = OwtRunConfig(
        model_type="sheet",
        run_start_label="260714-0800",
        max_iters=50_000,
        warmup_iters=100,
    )
    assert short.artifact_name == long.artifact_name
    assert "_n_10000_" not in f"_{short.artifact_name}_"
    assert "_n_50000_" not in f"_{long.artifact_name}_"


def test_legacy_artifact_builder_excludes_mutable_target_update_count() -> None:
    common = dict(
        model_type="thog2_sheet",
        host_label="dreedle",
        run_name="BALLAN",
        dataset_name="openwebtext",
        n_layer=64,
        n_head=16,
        n_embd=1024,
        block_size=256,
        batch_size=32,
        gradient_accumulation_steps=4,
        warmup_iters=100,
        checkpoint_interval=500,
        checkpoint_segment_size=12,
        depth_order=16,
        base_row_order=256,
    )
    assert build_artifact_name(max_iters=10_000, **common) == build_artifact_name(max_iters=50_000, **common)


def test_console_throughput_prefers_session_tokens_but_keeps_lifetime_total() -> None:
    payload = add_console_tokens_per_second(
        {
            "consumed_tokens": 330_000_000,
            "session_consumed_tokens": 327_680,
            "cumulative_training_seconds": 35.0,
        }
    )
    assert payload["consumed_tokens"] == 330_000_000
    assert "session_consumed_tokens" not in payload
    assert math.isclose(payload["tok/s"], 327_680 / 35.0)


def test_console_throughput_retains_fresh_run_fallback() -> None:
    payload = add_console_tokens_per_second(
        {
            "consumed_tokens": 655_360,
            "cumulative_training_seconds": 70.0,
        }
    )
    assert math.isclose(payload["tok/s"], 655_360 / 70.0)


def test_session_token_accounting_starts_at_resume_boundary() -> None:
    assert _session_consumed_tokens(10_000, 10_010, 32_768) == 327_680
    assert _session_consumed_tokens(0, 10, 32_768) == 327_680
# ^^^ THOG
''',
    encoding="utf-8",
)

# ^^^ THOG
