#!/usr/bin/env python3
# vvv THOG
"""Finish PLASTIC CLI/console refinements after the primary applicator."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    if new in content:
        return
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    write(path, content.replace(old, new, 1))


def fix_coarse_progress_clock() -> None:
    path = "sheet/plastic_depth_coarse_runner.py"
    replace_once(
        path,
        "    clock: Clock = time.perf_counter,\n    progress_sink: Optional[ProgressSink] = None,\n",
        "    clock: Clock = time.perf_counter,\n"
        "    progress_clock: Clock = time.perf_counter,\n"
        "    progress_sink: Optional[ProgressSink] = None,\n",
    )
    replace_once(
        path,
        "    started = clock()\n    try:\n",
        "    started = clock()\n"
        "    progress_started = progress_clock()\n"
        "    try:\n",
    )
    replace_once(
        path,
        '''                elapsed_seconds = (
                    float(prior_training_elapsed_seconds)
                    + max(0.0, float(clock() - started))
                )''',
        '''                elapsed_seconds = (
                    float(prior_training_elapsed_seconds)
                    + max(0.0, float(progress_clock() - progress_started))
                )''',
    )


def rewrite_coarse_report() -> None:
    path = "sheet/plastic_depth_coarse.py"
    content = read(path)
    start = content.index("def render_plastic_coarse_report(")
    end = content.index("\n\ndef coarse_results_payload(", start)
    replacement = '''def render_plastic_coarse_report(
    scored_trials: Sequence[ScoredPlasticCoarseTrial],
    winner: ScoredPlasticCoarseTrial,
    *,
    training_steps: int,
    evaluation_steps_count: int,
    ansi: bool,
) -> str:
    if not scored_trials:
        raise ValueError("scored_trials must not be empty")
    objective = scored_trials[0].objective
    heading = scored_trials[0].objective_heading
    for row in scored_trials:
        if row.objective != objective or row.objective_heading != heading:
            raise ValueError("all scored trials must use one objective")

    headers = [
        "PLASTIC COARSE RESULTS",
        f"{len(scored_trials)} trials x {training_steps} training steps",
        f"validation mean over final {evaluation_steps_count} batches",
        f"goal: {objective}",
    ]
    if objective == "relative_training_wall_time":
        reference = winner.reference_training_elapsed_seconds
        headers.append(f"reference training elapsed_s: {float(reference):.6f}")

    columns = (
        f"{'trial':>5} {'layers':>6} {'elapsed_s':>10} {'sec/step':>10} "
        f"{'tok/s':>9} {'mean_val':>10} {'val_std':>9} {'peak_GiB':>9} "
        + (f"{'within_budget':>13} " if objective == "memory_budget" else "")
        + f"{heading:>18} status"
    )
    lines = headers + [columns]
    for row in scored_trials:
        result = row.result
        marker = (
            f" {_WINNER_STYLE_START}<<< WINNER{_STYLE_END}"
            if row is winner and ansi
            else (" <<< WINNER" if row is winner else "")
        )
        within = ""
        if objective == "memory_budget":
            within = f"{('yes' if row.within_budget else 'no'):>13} "
        if result.status == "failed":
            reason = " ".join(str(result.error_message or "no reason recorded").split())
            status = (
                f"failed - because {result.error_class or 'Exception'}: {reason}"
            )
        else:
            status = "ok" if row.selectable else "unselectable"
        lines.append(
            f"{result.trial_index:5d} {result.layers:6d} "
            f"{_format_optional(result.training_elapsed_seconds, 10, 2)} "
            f"{_format_optional(result.seconds_per_step, 10, 5)} "
            f"{_format_optional(result.tokens_per_second, 9, 0)} "
            f"{_format_optional(result.mean_validation_loss, 10, 4)} "
            f"{_format_optional(result.validation_loss_std, 9, 4)} "
            f"{_format_optional(result.peak_allocated_gib, 9, 2)} "
            f"{within}{_format_optional(row.score, 18, 6)} {status}{marker}"
        )
    return "\\n".join(lines)
'''
    write(path, content[:start] + replacement + content[end:])


def wire_wrapper_values_and_startup() -> None:
    path = "train_OWT_core.sh"
    content = read(path)
    if 'optional_args+=(--plastic__log_interval_coarse' not in content:
        anchor = '    optional_args+=(--plastic__layer_count_update_brake "$PLASTIC_LAYER_COUNT_UPDATE_BRAKE")\n'
        addition = (
            '    optional_args+=(--plastic__log_interval_coarse "$PLASTIC_LOG_INTERVAL_COARSE")\n'
            '    if [[ "$PLASTIC_COARSE_PHASE_ROLL_THROUGH" == true ]]; then\n'
            '      optional_args+=(--plastic__coarse_phase_roll_through)\n'
            '    else\n'
            '      optional_args+=(--no-plastic__coarse_phase_roll_through)\n'
            '    fi\n'
        )
        if anchor not in content:
            raise RuntimeError("PLASTIC optional-argument anchor missing")
        content = content.replace(anchor, anchor + addition, 1)
    if '--max-nonfinite-update-skips "$MAX_NONFINITE_UPDATE_SKIPS"' not in content:
        anchor = '    --eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" --checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS" --learning-rate "$learning_rate_value" --min-lr "$min_lr_value"\n'
        replacement = (
            '    --eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" '
            '--checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS" '
            '--learning-rate "$learning_rate_value" --min-lr "$min_lr_value" '
            '--max-nonfinite-update-skips "$MAX_NONFINITE_UPDATE_SKIPS"\n'
        )
        if anchor not in content:
            raise RuntimeError("non-finite runner-argument anchor missing")
        content = content.replace(anchor, replacement, 1)
    if "  non-finite updates:" not in content:
        anchor = "  instrumentation:    $INSTRUMENTATION\n"
        addition = "  non-finite updates: policy=skip max_skips=$MAX_NONFINITE_UPDATE_SKIPS\n"
        if anchor not in content:
            raise RuntimeError("startup instrumentation anchor missing")
        content = content.replace(anchor, anchor + addition, 1)
    write(path, content)


def update_superseded_tests() -> None:
    path = "tests/test_plastic_depth_coarse_runtime_recovery.py"
    content = read(path)
    content = content.replace(
        '    assert "PLASTIC COARSE FAILURES" in report\n'
        '    assert "trial 2 layers=8 completed=1/2" in report\n'
        '    assert "OutOfMemoryError: CUDA out of memory" in report\n',
        '    assert "PLASTIC COARSE FAILURES" not in report\n'
        '    assert "failed - because OutOfMemoryError: CUDA out of memory" in report\n',
    )
    write(path, content)

    path = "tests/test_plastic_cli_console_refinements.py"
    content = read(path)
    content = content.replace(
        '    assert winner_line.startswith("    1")\n'
        '    assert "\\x1b[1;92m<<< WINNER\\x1b[0m" in winner_line\n',
        '    assert winner_line.startswith("    1")\n'
        '    assert not winner_line.startswith("\\x1b[")\n'
        '    assert "\\x1b[1;92m<<< WINNER\\x1b[0m" in winner_line\n',
    )
    if "test_wrapper_help_covers_every_registered_plastic_option" not in content:
        content += '''\n\ndef test_wrapper_help_covers_every_registered_plastic_option() -> None:
    parser = core.build_parser()
    completed = subprocess.run(
        ("bash", "./train_OWT.sh", "-h"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    registered = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--plastic__") or option.startswith("--no-plastic__")
    }
    missing = sorted(option for option in registered if option not in completed.stdout)
    assert missing == []
    assert "--max-nonfinite-update-skips" in completed.stdout
'''
    write(path, content)


def main() -> None:
    fix_coarse_progress_clock()
    rewrite_coarse_report()
    wire_wrapper_values_and_startup()
    update_superseded_tests()


if __name__ == "__main__":
    main()
# ^^^ THOG
