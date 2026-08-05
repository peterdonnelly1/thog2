# vvv THOG
from __future__ import annotations

import os


def pytest_runtest_setup(item):
    os.environ.setdefault("THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL", "1")
# ^^^ THOG
