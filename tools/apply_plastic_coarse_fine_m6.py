from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one M6 anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def update_module_contract() -> None:
    path = "sheet/plastic_depth_lookahead_patch.py"
    replace_once(
        path,
        '"""Exact-radius PLASTIC DEPTH lookahead and console reporting.\n\n'
        'This patch keeps the existing inline-probe execution path but separates three\n'
        'ideas that were previously collapsed into adjacent N-1/N/N+1 probing:\n\n'
        '* decision probes are exactly L-radius, L, L+radius where valid;\n'
        '* bridge candidates L±max_step are checkpointed only so the selected one-step\n'
        '  training prefix exists;\n'
        '* console statistics report the exact decision probes, not the bridge points.\n'
        '"""\n',
        '"""Full-radius PLASTIC DEPTH FINE probing and bounded count movement.\n\n'
        'Every valid integer count in the inclusive configured radius is measured on\n'
        'one shared first-microstep chain.  The robust winner records the desired probe\n'
        'count, while max_step independently limits the committed prefix transition.\n'
        '"""\n',
    )


def enumerate_full_radius() -> None:
    path = "sheet/plastic_depth_lookahead_patch.py"
    replace_once(
        path,
        'def _lookahead_counts(current: int, maximum: int, radius: int, max_step: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:\n'
        '    decision_counts = {current}\n'
        '    execution_counts = {current}\n'
        '    if current - radius >= 1:\n'
        '        decision_counts.add(current - radius)\n'
        '        execution_counts.add(max(1, current - max_step))\n'
        '    if current + radius <= maximum:\n'
        '        decision_counts.add(current + radius)\n'
        '        execution_counts.add(min(maximum, current + max_step))\n'
        '    execution_counts.update(decision_counts)\n'
        '    return tuple(sorted(decision_counts)), tuple(sorted(execution_counts))\n',
        'def _lookahead_counts(current: int, maximum: int, radius: int, max_step: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:\n'
        '    del max_step\n'
        '    lower = max(1, current - radius)\n'
        '    upper = min(maximum, current + radius)\n'
        '    decision_counts = tuple(range(lower, upper + 1))\n'
        '    return decision_counts, decision_counts\n',
    )


def strengthen_robust_qualification() -> None:
    path = "sheet/plastic_depth_lookahead_patch.py"
    replace_once(
        path,
        '            median, mad, sigma = _robust_scale(values, paired_difference)\n'
        '            standardized = -paired_difference / sigma\n'
        '            significant = len(values) >= minimum_observations and paired_difference < -noise_lambda * sigma\n'
        '            if significant and not brake_active:\n'
        '                passing.append((standardized, offset, candidate_count))\n',
        '            median, mad, sigma = _robust_scale(values, paired_difference)\n'
        '            standardized = -median / sigma\n'
        '            favourable_count = sum(value < 0.0 for value in values)\n'
        '            significant = (\n'
        '                len(values) >= minimum_observations\n'
        '                and median < -noise_lambda * sigma\n'
        '                and paired_difference < 0.0\n'
        '                and favourable_count * 2 > len(values)\n'
        '            )\n'
        '            if significant and not brake_active:\n'
        '                passing.append((standardized, offset, candidate_count))\n',
    )
    replace_once(
        path,
        '    if passing:\n'
        '        _, selected_offset, _ = max(passing, key=lambda item: (item[0], -item[2]))\n'
        '        step = max(-max_step, min(max_step, selected_offset))\n'
        '        selected_count = current_count + step\n'
        '        for offset in candidate_offsets:\n'
        '            updated_histories.pop(_history_key(selected_count, offset), None)\n',
        '    if passing:\n'
        '        _, selected_offset, _ = max(passing, key=lambda item: (item[0], -item[2]))\n'
        '        step = max(-max_step, min(max_step, selected_offset))\n'
        '        selected_count = current_count + step\n'
        '        updated_histories = {}\n',
    )


def main() -> None:
    update_module_contract()
    enumerate_full_radius()
    strengthen_robust_qualification()


if __name__ == "__main__":
    main()
