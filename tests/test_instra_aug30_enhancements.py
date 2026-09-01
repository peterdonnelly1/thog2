# vvv THOG
from pathlib import Path
from types import SimpleNamespace

import run_thog2_owt_core


def _header_config(**overrides):
    canonical = {
        "instrumentation__depth_weight_curves__destination": "local",
        "instrumentation__depth_weight_curves__time_mode": "accumulate",
        "instrumentation__depth_weight_curves__start_step": 1,
        "instrumentation__depth_weight_curves__end_step": 200,
        "instrumentation__depth_weight_curves__log_every_n_steps": 1,
        "instrumentation__depth_weight_curves__history_length": 200,
        "instrumentation__depth_weight_curves__scalar_weights_per_matrix": 3,
        "instrumentation__depth_weight_curves__depth_evaluation_points": 256,
        "instrumentation__depth_weight_curves__same_coordinates_all_runs": True,
    }
    values = {
        "model_type": "dense",
        "save_dense_initialisation_snapshot": False,
        "initialise_from_dense_snapshot": None,
        "dense_snapshot_chebyshev_order": None,
        "dense_snapshot_chebyshev_version": None,
    }
    values.update(overrides)
    return SimpleNamespace(
        **values,
        canonical_dict=lambda *, world_size: dict(canonical, world_size=world_size),
    )


def test_cli_header_summarises_weight_instrumentation_and_snapshot_roles():
    saved = run_thog2_owt_core._console_header_summary(
        _header_config(save_dense_initialisation_snapshot=True),
        world_size=1,
    )
    assert saved["weight_curves"] == (
        "destination=local mode=accumulate capture=1..200 every=1 history=200 "
        "coupling_pairs/matrix=3 depth_points=256 same_coupling_pairs=true"
    )
    assert saved["dense_snapshot"] == (
        "A save step-zero tensors directory=dense_baseline_snapshots"
    )

    loaded = run_thog2_owt_core._console_header_summary(
        _header_config(
            initialise_from_dense_snapshot="dense_baseline_snapshots/baseline.pt",
            dense_snapshot_chebyshev_order=16,
            dense_snapshot_chebyshev_version="chebyshev_qr_v1",
        ),
        world_size=1,
    )
    assert loaded["dense_snapshot"].startswith(
        "B compressor-baselined DENSE load=dense_baseline_snapshots/baseline.pt "
    )
    assert "compressor=chebyshev order=16 version=chebyshev_qr_v1" in loaded["dense_snapshot"]

    compact = run_thog2_owt_core._console_header_summary(
        _header_config(
            model_type="sheet",
            initialise_from_dense_snapshot="baseline.pt",
            dense_snapshot_chebyshev_order=16,
            dense_snapshot_chebyshev_version="chebyshev_qr_v1",
        ),
        world_size=1,
    )
    assert compact["dense_snapshot"].startswith("C Compact Run load=baseline.pt")


def test_shell_header_and_overview_command_capture_are_wired():
    repository_root = Path(__file__).parents[1]
    shell = (repository_root / "train_OWT_core.sh").read_text()
    core = (repository_root / "run_thog2_owt_core.py").read_text()
    lifecycle = (repository_root / "run_thog2_lifecycle.py").read_text()
    assert "weight curves:      $weight_curves_console" in shell
    assert "DENSE snapshot:     $dense_snapshot_console" in shell
    assert '.get("console_header", {})' in shell
    assert "runtime_overview_metadata()" in core
    assert "**runtime_metadata" in core
    assert "THOG2_RUNSTRING" in shell
    assert '"console_header": core._console_header_summary(' in lifecycle
# ^^^ THOG
