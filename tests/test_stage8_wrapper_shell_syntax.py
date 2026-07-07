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


def test_stage8_training_wrappers_make_logging_backend_and_schedule_controls_explicit() -> None:
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "INSTRUMENTATION=\"tensorboard\"" in text
        assert "THOG2_INSTRUMENTATION" in text
        assert "-I INSTRUMENTATION" in text
        assert "EVAL_INTERVAL=20" in text
        assert "LOG_INTERVAL=1" in text
        assert "--eval-interval" in text
        assert "--log-interval" in text
        assert "--checkpoint-interval" in text
        assert "--warmup-iters" in text
