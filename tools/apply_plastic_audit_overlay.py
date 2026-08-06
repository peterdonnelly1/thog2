from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "sheet/__init__.py"
    text = path.read_text(encoding="utf-8")
    if "plastic_depth_audit_patch" in text:
        return
    anchor = (
        '# vvv THOG final COARSE/FINE overlay wins after all compatibility patches and probes every valid integer count in the configured radius\n'
        'from . import plastic_depth_coarse_fine_patch as _plastic_depth_coarse_fine_patch\n'
        '# ^^^ THOG\n'
    )
    addition = anchor + (
        '\n# vvv THOG install replayable FINE count-decision audit after final selector and commit semantics are fixed\n'
        'from . import plastic_depth_audit_patch as _plastic_depth_audit_patch\n'
        '# ^^^ THOG\n'
    )
    if text.count(anchor) != 1:
        raise RuntimeError("final COARSE/FINE overlay anchor was not found exactly once")
    path.write_text(text.replace(anchor, addition, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
