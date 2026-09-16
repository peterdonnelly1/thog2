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
    def __init__(self, artifact: str, host: str = "scruffy") -> None:
        self.reader = _Reader(artifact, host)
        self.database_path = Path("/tmp") / artifact / "charts.sqlite3"


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
# ^^^ THOG
