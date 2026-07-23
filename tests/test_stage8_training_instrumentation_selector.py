# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path


TRAINING_WRAPPERS = (
    "current_scruffy_train_OWT.sh",
    "current_dreedle_train_OWT.sh",
)


def test_stage8_training_wrappers_expose_one_unambiguous_instrumentation_selector() -> None:
    for wrapper in TRAINING_WRAPPERS:
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "tensorboard | wandb | wandb_offline | none" in text
        assert "-M WANDB_MODE" not in text
        assert "-W WANDB_ENABLED" not in text
        assert "M:" not in text.split("getopts", 1)[1].split(" option", 1)[0]
        assert "-W LAPPED_COSINE_WINDOW_LENGTH=" in text
        assert "W:" in text.split("getopts", 1)[1].split(" option", 1)[0]


def test_stage8_training_wrappers_derive_consistent_wandb_flags_and_modes_from_instrumentation() -> None:
    expected_fragments = (
        'tensorboard) INSTRUMENTATION_BACKEND="tensorboard"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled"',
        'wandb) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="online"',
        'wandb_offline) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="offline"',
        'none) INSTRUMENTATION_BACKEND="none"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled"',
    )
    for wrapper in TRAINING_WRAPPERS:
        text = Path(wrapper).read_text(encoding="utf-8")
        for fragment in expected_fragments:
            assert fragment in text
        assert 'export THOG2_INSTRUMENTATION="$INSTRUMENTATION_BACKEND"' in text
        assert '"$WANDB_FLAG" --wandb-mode "$WANDB_MODE"' in text


def test_stage8_training_wrapper_help_lists_only_the_single_instrumentation_option() -> None:
    for wrapper in TRAINING_WRAPPERS:
        completed = subprocess.run(["bash", wrapper, "-h"], text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        assert "-I INSTRUMENTATION=" in completed.stdout
        assert "wandb_offline" in completed.stdout
        assert "-M WANDB_MODE" not in completed.stdout
        assert "-W WANDB_ENABLED" not in completed.stdout
# ^^^ THOG
