# vvv THOG
from __future__ import annotations

from typing import Optional

import run_thog2_owt
from sheet.residual_run_config import OwtRunConfig


run_thog2_owt.OwtRunConfig = OwtRunConfig


def resolve_residual_instrumentation(
    instrumentation: Optional[str],
    legacy_wandb: Optional[bool],
    wandb_mode: str,
) -> str:
    if instrumentation is None:
        if legacy_wandb is False:
            return "none"
        return "tensorboard"
    if instrumentation == "wandb" and wandb_mode == "disabled":
        return "none"
    return instrumentation


run_thog2_owt.resolve_instrumentation = resolve_residual_instrumentation


if __name__ == "__main__":
    raise SystemExit(run_thog2_owt.main())
# ^^^ THOG
