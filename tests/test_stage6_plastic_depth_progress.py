# vvv THOG
from sheet.stage6_trainer import format_progress_line


def _optimizer_payload(**overrides):
    payload = {
        "completed_updates": "     7",
        "timestamp": "260805:0926",
        "cumulative_training_seconds": "       42",
        "mean_step_seconds": "  0.1234",
        "tok/s": "       12345",
        "consumed_tokens": "       86016",
        "training_loss": "  9.8765",
        "training_loss_delta": "  -0.123",
        "learning_rate": " 1.000e-04",
        "gradient_norm": "   2.345",
        "current_layer_count": 4,
        "sampled_values": (0.00123, 2.5),
        "depth_sample_points": (1.25, 4.56, 8.94, 14.0),
    }
    payload.update(overrides)
    return payload


def test_plastic_depth_sample_points_are_final_optimizer_progress_field() -> None:
    line = format_progress_line("RUN", "optimizer_progress", _optimizer_payload())

    assert "current_layer_count = 4" in line
    assert "sampled_values = [1.23e-03, 2.50]" in line
    assert line.endswith("depth_samples = [1.3, 4.6, 8.9, 14.0]")


def test_optimizer_progress_omits_depth_samples_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["depth_sample_points"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "depth_samples" not in line
    assert line.endswith("sampled_values = [1.23e-03, 2.50]")
# ^^^ THOG
