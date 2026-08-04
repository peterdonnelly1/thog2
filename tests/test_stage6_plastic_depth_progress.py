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
        "plastic_probe_losses": (10.12345, 10.23456, 10.34567),
        "depth_sample_points": (1.25, 4.56, 8.94, 14.0),
    }
    payload.update(overrides)
    return payload


def test_plastic_probe_losses_precede_final_layer_indices_field() -> None:
    line = format_progress_line("RUN", "optimizer_progress", _optimizer_payload())

    assert "current_layer_count = 4" in line
    assert "sampled_values = [1.23e-03, 2.50]" in line
    assert "probe_losses = [10.123, 10.235, 10.346]" in line
    assert line.endswith("layer indices = [1.3, 4.6, 8.9, 14.0]")
    assert line.index("probe_losses = [10.123, 10.235, 10.346]") < line.index("layer indices = [1.3, 4.6, 8.9, 14.0]")


def test_missing_probe_loss_position_is_rendered_as_dash() -> None:
    line = format_progress_line(
        "RUN",
        "optimizer_progress",
        _optimizer_payload(plastic_probe_losses=(None, 10.23456, None)),
    )

    assert "probe_losses = [-, 10.235, -]" in line


def test_optimizer_progress_omits_layer_indices_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["depth_sample_points"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "layer indices" not in line
    assert line.endswith("probe_losses = [10.123, 10.235, 10.346]")
# ^^^ THOG
