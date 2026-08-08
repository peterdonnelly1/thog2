from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tools/apply_plastic_full_radius_oom.py"
    text = path.read_text(encoding="utf-8")
    old = (
        '    marker = "    # ^^^ THOG\\n\\n    def forward(\\n"\n'
        '    if text.count(marker) != 1:\n'
        '        raise RuntimeError("training_model.py forward insertion marker was not found exactly once")\n'
    )
    new = (
        '    marker = "    # ^^^ THOG\\n\\n    def forward(\\n"\n'
        '    class_start = text.index("class TrainingSheetGPT")\n'
        '    marker_index = text.index(marker, class_start)\n'
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("full-radius OOM tool target-selection anchor was not found")
    old = (
        '    text = text.replace(marker, "    # ^^^ THOG\\n\\n" + method + "    def forward(\\n", 1)\n'
    )
    new = (
        '    replacement = "    # ^^^ THOG\\n\\n" + method + "    def forward(\\n"\n'
        '    text = text[:marker_index] + replacement + text[marker_index + len(marker):]\n'
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("full-radius OOM tool insertion anchor was not found")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
