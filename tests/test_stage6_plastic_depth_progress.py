# vvv THOG
import run_thog2_owt  # noqa: F401  # <<< THOG activate public train_OWT console monkey patches before importing the formatter
from sheet.stage6_trainer import format_progress_line


_GREEN = "\033[1;38;2;0;255;0m"
_RESET = "\033[0m"


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
        "plastic_probe_offsets": (-3, 0, 3),
        "plastic_probe_edge_offsets": (-3, 3),
        "plastic_probe_losses": (10.1234, 9.2345, 100.3456),
        "plastic_loss_gain": (-0.8889, -91.1111),
        "plastic_score_z": (-2.5, 1.25),
        "depth_sample_points": (1.26, 4.56, 8.94, 100.0),
    }
    payload.update(overrides)
    return payload


def test_plastic_probe_losses_score_z_sample_layer_and_samples_are_ordered() -> None:
    line = format_progress_line("RUN", "optimizer_progress", _optimizer_payload())

    assert "layers = 4" in line
    assert "sampled_values" not in line
    assert "\tprobe_losses [L-3, L, L+3] = [ 10.123,   9.235, 100.346]" in line
    assert "loss_gain" not in line
    assert "score_z [L-3, L+3] = [    -2.50,     +1.25]" in line
    assert line.endswith("\tsample_layer = [  1.3,   4.6,   8.9, 100.0]")
    assert line.index("\tprobe_losses [L-3, L, L+3] =") < line.index("score_z [L-3, L+3] =")
    assert line.index("score_z [L-3, L+3] =") < line.index("\tsample_layer =")


def test_huge_score_z_uses_fixed_width_scientific_not_row_blowout() -> None:
    line = format_progress_line(
        "RUN",
        "optimizer_progress",
        _optimizer_payload(plastic_score_z=(43_945_312_500.0, -39_794_921_875.0)),
    )

    assert "score_z [L-3, L+3] = [+4.39e+10, -3.98e+10]" in line
    assert "sampled_values" not in line
    assert line.endswith("\tsample_layer = [  1.3,   4.6,   8.9, 100.0]")


def test_optimizer_progress_omits_depth_samples_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["depth_sample_points"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "sample_layer" not in line
    assert "sampled_values" not in line
    assert line.endswith("score_z [L-3, L+3] = [    -2.50,     +1.25]")


def test_optimizer_progress_omits_probe_losses_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["plastic_probe_offsets"]
    del payload["plastic_probe_edge_offsets"]
    del payload["plastic_probe_losses"]
    del payload["plastic_loss_gain"]
    del payload["plastic_score_z"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "probe_losses" not in line
    assert "loss_gain" not in line
    assert "score_z" not in line
    assert "sampled_values" not in line
    assert line.endswith("\tsample_layer = [  1.3,   4.6,   8.9, 100.0]")


def test_optimizer_progress_omits_score_z_when_absent() -> None:
    payload = _optimizer_payload()
    del payload["plastic_score_z"]

    line = format_progress_line("RUN", "optimizer_progress", payload)

    assert "score_z" not in line
    assert "loss_gain" not in line
    assert line.index("\tprobe_losses [L-3, L, L+3] =") < line.index("\tsample_layer =")
    assert "sampled_values" not in line


def test_missing_edge_probe_loss_and_score_z_use_dash_placeholders_and_green_only() -> None:
    line = format_progress_line(
        "RUN",
        "optimizer_progress",
        _optimizer_payload(
            plastic_probe_offsets=(-1, 0, 1),
            plastic_probe_edge_offsets=(-1, 1),
            plastic_probe_losses=(None, 9.5, 9.25),
            plastic_loss_gain=(None, 0.25),
            plastic_score_z=(None, 0.125),
        ),
    )

    assert f"\tprobe_losses [L-1, L, L+1] = [      -,   9.500,{_GREEN}   9.250{_RESET}]" in line
    assert "loss_gain" not in line
    assert "score_z [L-1, L+1] = [        -,     +0.12]" in line
    assert "\033[1;31m" not in line
    assert "sampled_values" not in line
# ^^^ THOG
