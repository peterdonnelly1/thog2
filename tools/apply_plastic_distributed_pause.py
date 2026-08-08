from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one distributed-pause anchor, found {count}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    path = "sheet/plastic_depth_lifecycle.py"
    replace_once(
        path,
        '    PlasticCoarsePauseResult,\n'
        '    run_plastic_coarse_review_pause,\n',
        '    PlasticCoarsePauseResult,\n'
        '    run_distributed_plastic_coarse_review_pause,\n'
        '    run_plastic_coarse_review_pause,\n',
    )
    replace_once(
        path,
        'def _distributed_pause(\n'
        '    coordinator: Any,\n'
        '    *,\n'
        '    pause_runner: PauseRunner,\n'
        '    pause_duration_seconds: float,\n'
        '    console_stream: TextIO,\n'
        ') -> PlasticCoarsePauseResult:\n'
        '    local_result: Optional[PlasticCoarsePauseResult] = None\n'
        '    if coordinator.is_primary:\n'
        '        local_result = pause_runner(\n'
        '            duration_seconds=pause_duration_seconds,\n'
        '            output=console_stream,\n'
        '            checkpoint_callback=None,\n'
        '        )\n'
        '    gathered = coordinator.all_gather_object(local_result)\n'
        '    result = gathered[0]\n'
        '    if not isinstance(result, PlasticCoarsePauseResult):\n'
        '        raise RuntimeError("rank 0 did not provide a PLASTIC COARSE pause disposition")\n'
        '    coordinator.barrier()\n'
        '    return result\n'
        '\n'
        '\n',
        '',
    )
    replace_once(
        path,
        '        pause_result = _distributed_pause(\n'
        '            coordinator,\n'
        '            pause_runner=pause_runner,\n'
        '            pause_duration_seconds=pause_duration_seconds,\n'
        '            console_stream=console_stream,\n'
        '        )\n',
        '        pause_result = run_distributed_plastic_coarse_review_pause(\n'
        '            coordinator,\n'
        '            duration_seconds=pause_duration_seconds,\n'
        '            output=console_stream,\n'
        '            pause_runner=pause_runner,\n'
        '        )\n',
    )


if __name__ == "__main__":
    main()
