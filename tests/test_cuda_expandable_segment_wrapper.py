from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_wrapper(tmp_path: Path, *arguments: str, allocator_config: str | None = None) -> subprocess.CompletedProcess[str]:
    (tmp_path / "train_OWT.sh").write_text(
        (REPOSITORY_ROOT / "train_OWT.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "plastic_depth_lookahead_wrapper_options.sh").write_text(":\n", encoding="utf-8")
    (tmp_path / "train_OWT_core.sh").write_text(
        "printf 'allocator=%s\\n' \"$PYTORCH_CUDA_ALLOC_CONF\"\n"
        "printf 'mode=%s\\n' \"$THOG2_CUDA_EXPANDABLE_SEGMENT\"\n"
        "printf 'user_supplied=%s\\n' \"$THOG2_CUDA_ALLOC_CONF_USER_SUPPLIED\"\n"
        "printf 'args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    if allocator_config is None:
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        environment["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config
    return subprocess.run(
        ["bash", "train_OWT.sh", *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cuda_expandable_segment_is_enabled_by_default(tmp_path: Path) -> None:
    completed = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "allocator=expandable_segments:True" in completed.stdout
    assert "mode=enabled" in completed.stdout
    assert "user_supplied=false" in completed.stdout
    assert "WARNING:" not in completed.stderr


def test_cuda_expandable_segment_can_be_disabled_with_heavy_red_warning(tmp_path: Path) -> None:
    completed = _run_wrapper(tmp_path, "--cuda-expandable-segment", "disabled", "sentinel")

    assert completed.returncode == 0, completed.stderr
    assert "allocator=expandable_segments:False" in completed.stdout
    assert "mode=disabled" in completed.stdout
    assert "user_supplied=false" in completed.stdout
    assert "args=sentinel" in completed.stdout
    assert "\033[1;31m" in completed.stderr
    assert "WARNING: CUDA EXPANDABLE SEGMENTS ARE DISABLED" in completed.stderr
    assert completed.stderr.count("WARNING:") == 1


def test_cuda_expandable_segment_preserves_other_allocator_settings(tmp_path: Path) -> None:
    completed = _run_wrapper(
        tmp_path,
        "--cuda-expandable-segment=enabled",
        allocator_config="garbage_collection_threshold:0.8, expandable_segments:False,max_split_size_mb:256",
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "allocator=garbage_collection_threshold:0.8,max_split_size_mb:256,expandable_segments:True"
        in completed.stdout
    )
    assert "user_supplied=true" in completed.stdout


def test_cuda_expandable_segment_rejects_invalid_value(tmp_path: Path) -> None:
    completed = _run_wrapper(tmp_path, "--cuda-expandable-segment", "maybe")

    assert completed.returncode == 2
    assert "requires enabled or disabled; got: maybe" in completed.stderr
