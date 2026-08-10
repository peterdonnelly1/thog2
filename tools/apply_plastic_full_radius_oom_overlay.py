from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "sheet/__init__.py"
    text = path.read_text(encoding="utf-8")
    if "plastic_depth_full_radius_oom_patch" in text:
        return
    anchor = (
        '# vvv THOG final COARSE/FINE overlay wins after all compatibility patches and probes every valid integer count in the configured radius\n'
        'from . import plastic_depth_coarse_fine_patch as _plastic_depth_coarse_fine_patch\n'
        '# ^^^ THOG\n'
    )
    replacement = anchor + (
        '\n# vvv THOG recover candidate-local CUDA OOMs across the contiguous full-radius upward suffix\n'
        'from . import plastic_depth_full_radius_oom_patch as _plastic_depth_full_radius_oom_patch\n'
        '# ^^^ THOG\n'
    )
    if text.count(anchor) != 1:
        raise RuntimeError("final COARSE/FINE overlay anchor was not found exactly once")
    path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
