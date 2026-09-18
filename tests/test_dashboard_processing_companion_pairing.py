# vvv THOG
import json
from pathlib import Path

import run_thog2_local_dashboard as dashboard


class _Reader:
    def __init__(self, artifact: str, host: str = "scruffy") -> None:
        self._metadata = {
            "artifact_name": artifact,
            "config_json": json.dumps({"host_label": host}),
        }

    def metadata(self):
        return dict(self._metadata)


class _State:
    def __init__(self, artifact: str, host: str = "scruffy", database_path: Path | None = None) -> None:
        self.reader = _Reader(artifact, host)
        self.database_path = database_path or (Path("/tmp") / artifact / "charts.sqlite3")
        self._artifact = artifact
        self._host = host

    def status(self):
        return {
            "dashboard_run_id": self.database_path.parent.name,
            "artifact_name": self._artifact,
            "created_at": "",
            "host_label": self._host,
        }


class _Catalog:
    def __init__(self, root: Path, states: dict[Path, _State]) -> None:
        self.root = root
        self._states = {path.resolve(): state for path, state in states.items()}

    def _state_for_path(self, path: Path):
        return self._states[path.resolve()]


def test_nsys_and_ncu_prefixes_pair_when_encoded_configuration_matches() -> None:
    encoded = "G0_chebyshev__d_owt_A_6_b_16__C_1024_D_1024_H_16_L_16__P_12"
    nsys = _State(f"260916-1612_scruffy_NSYS_PREMAT___{encoded}")
    ncu = _State(f"260916-1807_scruffy_NCU_PREMAT___{encoded}")
    assert dashboard._processing_pair_key(nsys) == dashboard._processing_pair_key(ncu)


def test_pairing_rejects_different_host_or_encoded_configuration() -> None:
    left = _State("260916-1612_scruffy_NSYS_PREMAT___CONFIG_A", "scruffy")
    different_config = _State("260916-1807_scruffy_NCU_PREMAT___CONFIG_B", "scruffy")
    different_host = _State("260916-1807_dreedle_NCU_PREMAT___CONFIG_A", "dreedle")
    assert dashboard._processing_pair_key(left) != dashboard._processing_pair_key(different_config)
    assert dashboard._processing_pair_key(left) != dashboard._processing_pair_key(different_host)


def test_artifact_time_and_profiler_kind_are_explicit() -> None:
    early = "260916-1405_scruffy_NSYS_PREMAT___CONFIG_A"
    late = "260916-1807_scruffy_NCU_PREMAT___CONFIG_A"
    assert dashboard._artifact_timestamp_seconds(early) < dashboard._artifact_timestamp_seconds(late)
    assert dashboard._is_ncu_artifact(late)
    assert not dashboard._is_ncu_artifact(early)


def test_nearest_viable_ncu_skips_closer_invalid_capture(tmp_path: Path) -> None:
    encoded = "G0_chebyshev__d_owt_A_6_b_16__C_1024_D_1024_H_16_L_16__P_12"

    def make_state(artifact: str) -> tuple[Path, _State]:
        run_dir = tmp_path / artifact / artifact.split("_")[0]
        run_dir.mkdir(parents=True)
        database_path = run_dir / "charts.sqlite3"
        database_path.write_text("")
        return database_path, _State(artifact, database_path=database_path)

    selected_path, selected = make_state(f"260916-1405_scruffy_NSYS_PREMAT___{encoded}")
    invalid_path, _invalid = make_state(f"260916-1440_scruffy_NCU_PREMAT___{encoded}")
    nearest_valid_path, nearest_valid = make_state(f"260916-1616_scruffy_NCU_PREMAT___{encoded}")
    later_valid_path, _later_valid = make_state(f"260916-1807_scruffy_NCU_PREMAT___{encoded}")
    decoy_nsys_path, _decoy_nsys = make_state(f"260916-1410_scruffy_NSYS_PREMAT___{encoded}")

    valid_payload = {"rows": [{"compatibility_class": "RED"}]}
    for path in (nearest_valid_path, later_valid_path, decoy_nsys_path):
        processing = path.parent / "processing"
        processing.mkdir()
        (processing / "processing_premat_compatibility.json").write_text(json.dumps(valid_payload))

    states = {
        selected_path: selected,
        invalid_path: _invalid,
        nearest_valid_path: nearest_valid,
        later_valid_path: _later_valid,
        decoy_nsys_path: _decoy_nsys,
    }
    catalog = _Catalog(tmp_path, states)
    selected._instra_dashboard_catalog = catalog

    match = dashboard._matching_ncu_companion(selected)
    assert match is not None
    assert match[2] is nearest_valid
    diagnostics = selected._instra_ncu_companion_diagnostics
    assert diagnostics["skipped_closer_invalid_count"] == 1
    assert diagnostics["skipped_closer_invalid_artifacts"] == [
        f"260916-1440_scruffy_NCU_PREMAT___{encoded}"
    ]


def test_candidate_kind_and_pairing_use_metadata_not_directory_names(tmp_path: Path) -> None:
    encoded = "G0_chebyshev__d_owt_A_6_b_16__C_1024_D_1024_H_16_L_16__P_12"
    selected_path = tmp_path / "opaque-a" / "local" / "charts.sqlite3"
    candidate_path = tmp_path / "opaque-b" / "local" / "charts.sqlite3"
    selected_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    selected_path.write_text("")
    candidate_path.write_text("")
    selected = _State(
        f"260916-1405_scruffy_NSYS_PREMAT___{encoded}",
        database_path=selected_path,
    )
    candidate = _State(
        f"260916-1440_scruffy_NCU_PREMAT___{encoded}",
        database_path=candidate_path,
    )
    processing = candidate_path.parent / "processing"
    processing.mkdir()
    (processing / "processing_premat_compatibility.json").write_text(
        json.dumps({"rows": [{"compatibility_class": "GREEN"}]})
    )
    selected._instra_dashboard_catalog = _Catalog(
        tmp_path,
        {selected_path: selected, candidate_path: candidate},
    )

    match = dashboard._matching_ncu_companion(selected)
    assert match is not None
    assert match[2] is candidate


def test_claimed_ncu_is_skipped_and_persisted_pair_is_preferred(tmp_path: Path) -> None:
    encoded = "G0_chebyshev__d_owt_A_6_b_16__C_1024_D_1024_H_16_L_16__P_12"

    def make_state(artifact: str) -> tuple[Path, _State]:
        run_dir = tmp_path / artifact / artifact.split("_")[0]
        run_dir.mkdir(parents=True)
        database_path = run_dir / "charts.sqlite3"
        database_path.write_text("")
        state = _State(artifact, database_path=database_path)
        if "_NCU_" in artifact:
            processing = run_dir / "processing"
            processing.mkdir()
            (processing / "processing_premat_compatibility.json").write_text(
                json.dumps({"rows": [{"compatibility_class": "GREEN"}]})
            )
        return database_path, state

    selected_path, selected = make_state(f"260916-1405_scruffy_NSYS_PREMAT___{encoded}")
    nearest_path, nearest = make_state(f"260916-1410_scruffy_NCU_PREMAT___{encoded}")
    fallback_path, fallback = make_state(f"260916-1420_scruffy_NCU_PREMAT___{encoded}")
    selected._instra_dashboard_catalog = _Catalog(
        tmp_path,
        {selected_path: selected, nearest_path: nearest, fallback_path: fallback},
    )

    match = dashboard._matching_ncu_companion(
        selected,
        excluded_ncu_run_ids={nearest.database_path.parent.name},
    )
    assert match is not None
    assert match[2] is fallback

    persisted = dashboard._matching_ncu_companion(
        selected,
        excluded_ncu_run_ids={fallback.database_path.parent.name},
        preferred_ncu_run_id=nearest.database_path.parent.name,
    )
    assert persisted is not None
    assert persisted[2] is nearest


def test_ncu_download_manifest_includes_original_report_and_csv_exports(tmp_path: Path) -> None:
    database_path = tmp_path / "run" / "charts.sqlite3"
    database_path.parent.mkdir()
    database_path.write_text("")
    processing = database_path.parent / "processing"
    processing.mkdir()
    for filename in (
        "processing_ncu_trace.ncu-rep",
        "processing_ncu_raw.csv",
        "processing_ncu_semantic.csv",
    ):
        (processing / filename).write_text("data")
    state = _State("260916-1807_scruffy_NCU_PREMAT___CONFIG", database_path=database_path)

    assert dashboard._ncu_processing_download_files(state) == {
        "raw_ncu": "processing_ncu_trace.ncu-rep",
        "ncu_raw_csv": "processing_ncu_raw.csv",
        "ncu_semantic_csv": "processing_ncu_semantic.csv",
    }


def test_lifecycle_rows_prefer_capture_relative_time_and_retain_legacy_fallback() -> None:
    snapshot = {
        "pass_sequence": 7,
        "events": [
            {
                "sequence": 1,
                "candidate_sequence": 3,
                "event": "materialising",
                "processing_capture_elapsed_ms": 2.5,
                "elapsed_ms": 2.0,
                "layer_index": 4,
                "family": "DOWN",
                "owner": "premat",
            },
            {
                "sequence": 2,
                "candidate_sequence": 3,
                "event": "deadline_down",
                "elapsed_ms": 3.0,
                "layer_index": 4,
                "family": "DOWN",
            },
        ],
    }
    rows = dashboard._processing_lifecycle_rows(snapshot)
    assert rows[0]["capture_time_ms"] == 2.5
    assert rows[0]["timing_basis"] == "capture_relative_host"
    assert rows[0]["job_id"] == "p7:c3"
    assert rows[1]["timing_basis"] == "pass_relative_legacy"
    summary = dashboard._processing_lifecycle_summary(rows)
    assert summary[0]["submitted_ms"] == 2.5
    assert summary[0]["deadline_ms"] == 3.0


def test_lifecycle_rows_reconstruct_capture_time_from_host_clock() -> None:
    snapshot = {
        "pass_sequence": 8,
        "events": [
            {
                "sequence": 1,
                "event": "materialising",
                "processing_capture_elapsed_ms": None,
                "host_time_ns": 1_004_250_000,
                "elapsed_ms": 8.0,
                "layer_index": 2,
                "family": "DOWN",
            },
        ],
    }

    rows = dashboard._processing_lifecycle_rows(snapshot, 1_000_000_000)

    assert rows[0]["capture_time_ms"] == 4.25
    assert rows[0]["timing_basis"] == "capture_relative_host"
# ^^^ THOG
