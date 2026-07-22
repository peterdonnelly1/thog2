# vvv THOG
from __future__ import annotations

import os
from pathlib import Path

# Allow plain unittest discovery to exercise Stage 1 tests without running the heavyweight
# calibration wrapper. The dedicated Stage 1 runner overwrites this with real evidence.
_discovery_calibration = Path(__file__).resolve().parent / "fixtures" / "stage1_discovery_calibration.json"
os.environ.setdefault("THOG2_STAGE1_CALIBRATION", str(_discovery_calibration))
# ^^^ THOG
