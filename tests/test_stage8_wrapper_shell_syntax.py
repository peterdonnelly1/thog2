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


def test_stage8_training_wrappers_expose_all_six_semantic_order_controls() -> None:
    expected = (
        ("O_DEPTH=32", "-P O_DEPTH", "--o-depth"),
        ("O_ATTN_D_MODEL=64", "-Q O_ATTN_D_MODEL", "--o-attn-d-model"),
        ("O_ATTN_QKV_PER_CHANNEL=6", "-J O_ATTN_QKV_PER_CHANNEL", "--o-attn-qkv-per-channel"),
        ("O_ATTN_OUT_PER_CHANNEL=6", "-O O_ATTN_OUT_PER_CHANNEL", "--o-attn-out-per-channel"),
        ("O_MLP_D_MODEL=64", "-X O_MLP_D_MODEL", "--o-mlp-d-model"),
        ("O_MLP_HIDDEN=256", "-Y O_MLP_HIDDEN", "--o-mlp-hidden"),
    )
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        for variable, short_help, long_option in expected:
            assert variable in text
            assert short_help in text
            assert long_option in text
        assert "THOG2_MLP_CHANNEL_ORDER" in text
        assert "LOG_TIMESTAMP" in text
        assert "--log-timestamp" in text


def test_stage8_training_wrappers_auto_correct_dense_residual_source() -> None:
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "true_layer_depth" in text
        assert "dof_implied_depth" in text


def test_stage8_training_wrappers_keep_thog_marker_blocks() -> None:
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "# vvv THOG" in text
        assert "# ^^^ THOG" in text
