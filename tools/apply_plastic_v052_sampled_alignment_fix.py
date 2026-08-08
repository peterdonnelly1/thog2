#!/usr/bin/env python3
# vvv THOG
"""Keep sampled/probe suffix columns stable after moving sampled beside layers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "sheet/plastic_depth_directional_coherence_patch.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    constant_anchor = '_SAMPLED_BY_RUN_ID: Dict[str, Tuple[str, ...]] = {}\n'
    constant_replacement = (
        constant_anchor
        + '_MIN_FINAL_SAMPLED_COLUMN = 328\n'
    )
    if '_MIN_FINAL_SAMPLED_COLUMN = 328\n' not in text:
        if constant_anchor not in text:
            raise RuntimeError("missing sampled alignment constant anchor")
        text = text.replace(constant_anchor, constant_replacement, 1)

    helper_anchor = '''def _highlight_changed_sampled_values(run_id: str, event: str, line: str) -> str:\n'''
    helper = '''# vvv THOG retain one real tab before sampled while forcing T/V sampled to the same visible terminal column\ndef _align_sampled_to_minimum_tab_column(line: str) -> str:\n    sampled = _SAMPLED_ARRAY.search(line)\n    if sampled is None:\n        return line\n    prefix = line[: sampled.start()].rstrip(" \\t")\n    suffix = line[sampled.start() :]\n    prefix += "\\t"\n    while len(_ANSI_ESCAPE.sub("", prefix).expandtabs(8)) < _MIN_FINAL_SAMPLED_COLUMN:\n        prefix += "\\t"\n    return prefix + suffix\n# ^^^ THOG\n\n\n'''
    if 'def _align_sampled_to_minimum_tab_column(line: str) -> str:\n' not in text:
        if helper_anchor not in text:
            raise RuntimeError("missing sampled alignment helper anchor")
        text = text.replace(helper_anchor, helper + helper_anchor, 1)

    call_anchor = '    line = _move_sampled_after_layers(line)\n    line = _highlight_changed_sampled_values(run_id, event, line)\n'
    call_replacement = (
        '    line = _move_sampled_after_layers(line)\n'
        '    line = _align_sampled_to_minimum_tab_column(line)\n'
        '    line = _highlight_changed_sampled_values(run_id, event, line)\n'
    )
    if call_replacement not in text:
        if call_anchor not in text:
            raise RuntimeError("missing sampled alignment call anchor")
        text = text.replace(call_anchor, call_replacement, 1)

    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
# ^^^ THOG
