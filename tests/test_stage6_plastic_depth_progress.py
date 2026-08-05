# vvv THOG
import run_thog2_owt  # noqa: F401  # <<< THOG activate public train_OWT console monkey patches before importing the formatter
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
        "plastic_probe_losses": (10.1234, 9.2345, 100.3456),
        "depth_sample_points": (1.26, 4.56, 8.94, 100.0),
    }
    payload.update(overrides)
    return payload


def test_plastic_probe_losses_precede_final_layer_indices_field() -> None:
    line = format_progress_line("RUN", "optimizer_progress", _optimizer_payload())

    assert "current_layer_count = 4" in line
    assert "sampled_values = [1.23e-03, 2.50]" in line
    assert "\tprobe_losses = [ 10.123,   9.235, 100.346]" in line
    assert line.endswith("\tlayer indices = [  1.3,   4.6,   8.9, 100.0]")
    assert line.index("\tprobe_losses =") < line.index("\tlayer indices =")


def test_optimizer_progress_omits_depth_samples_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["depth_sample_points"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "layer indices" not in line
    assert line.endswith("\tprobe_losses = [ 10.123,   9.235, 100.346]")


def test_optimizer_progress_omits_probe_losses_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["plastic_probe_losses"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "probe_losses" not in line
    assert line.endswith("\tlayer indices = [  1.3,   4.6,   8.9, 100.0]")


def test_missing_edge_probe_loss_uses_dash_placeholder() -> None:
    line = format_progress_line(
        "RUN",
        "optimizer_progress",
        _optimizer_payload(plastic_probe_losses=(None, 9.5, 9.25)),
    )

    assert "\tprobe_losses = [      -,   9.500,   9.250]" in line
# ^^^ THOG
