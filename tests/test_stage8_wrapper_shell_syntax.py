from __future__ import annotations

import subprocess
from pathlib import Path


def test_stage8_top_level_wrappers_exist_and_are_shell_syntax_valid() -> None:
    wrappers = [
        "current_scruffy_train_OWT.sh",
        "current_scruffy_inference_OWT.sh",
        "current_dreedle_train_OWT.sh",
        "current_dreedle_inference_OWT.sh",
    ]
    for wrapper in wrappers:
        assert Path(wrapper).exists(), wrapper
        completed = subprocess.run(["bash", "-n", wrapper], text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
