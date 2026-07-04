# vvv THOG
from __future__ import annotations

import run_thog2_owt
from sheet.residual_run_config import OwtRunConfig


run_thog2_owt.OwtRunConfig = OwtRunConfig


if __name__ == "__main__":
    raise SystemExit(run_thog2_owt.main())
# ^^^ THOG
