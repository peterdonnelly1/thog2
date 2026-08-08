from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from sheet.plastic_depth_pause import PlasticCoarsePauseResult
from sheet.plastic_depth_resume import (
    PLASTIC_RESUME_CHECKPOINT_EXIT,
    PLASTIC_RESUME_CONTINUE_FINE,
    PLASTIC_RESUME_NOT_APPLICABLE,
    resume_plastic_coarse_fine_boundary,
)


class _Distributed:
    is_primary = True

    def __init__(self) -> None:
        self.barriers = 0

    def all_gather_object(self, value):
        return [value]

    def barrier(self) -> None:
        self.barriers += 1


def _trainer(state):
    checkpoints = []
    trainer = SimpleNamespace(
        distributed=_Distributed(),
        plastic_coarse_fine_state=state,
        plastic_coarse_provenance=state,
    )
    trainer.save_checkpoint = lambda path: checkpoints.append(Path(path))
    return trainer, checkpoints


def _review_pause_state(remaining: float = 700.0):
    return {
        "phase": "review_pause",
        "selected_layers": 8,
        "trials": [{"trial_index": 1, "layers": 8}],
        "pause": {
            "disposition": "checkpoint_exit",
            "elapsed_seconds": 200.0,
            "remaining_seconds": remaining,
        },
    }


def test_ordinary_fine_resume_is_untouched() -> None:
    trainer, checkpoints = _trainer({"phase": "fine"})

    result = resume_plastic_coarse_fine_boundary(
        trainer,
        "checkpoint.pt",
        output=io.StringIO(),
        pause_runner=lambda **_: (_ for _ in ()).throw(
            AssertionError("ordinary FINE resume must not enter pause")
        ),
    )

    assert result == PLASTIC_RESUME_NOT_APPLICABLE
    assert checkpoints == []


def test_timeout_resumes_fine_with_updated_pause_state() -> None:
    trainer, checkpoints = _trainer(_review_pause_state())
    observed = []

    def pause_runner(**kwargs):
        observed.append(kwargs["duration_seconds"])
        return PlasticCoarsePauseResult("timeout", 700.0, 0.0)

    result = resume_plastic_coarse_fine_boundary(
        trainer,
        "checkpoint.pt",
        output=io.StringIO(),
        pause_runner=pause_runner,
    )

    assert result == PLASTIC_RESUME_CONTINUE_FINE
    assert observed == [700.0]
    assert trainer.plastic_coarse_fine_state["phase"] == "fine"
    assert trainer.plastic_coarse_fine_state["selected_layers"] == 8
    assert trainer.plastic_coarse_fine_state["pause"]["remaining_seconds"] == 0.0
    assert checkpoints == []


def test_ctrl_f_resumes_fine_immediately() -> None:
    trainer, _ = _trainer(_review_pause_state())

    result = resume_plastic_coarse_fine_boundary(
        trainer,
        "checkpoint.pt",
        output=io.StringIO(),
        pause_runner=lambda **_: PlasticCoarsePauseResult("ctrl_f", 5.0, 695.0),
    )

    assert result == PLASTIC_RESUME_CONTINUE_FINE
    assert trainer.plastic_coarse_fine_state["phase"] == "fine"
    assert trainer.plastic_coarse_fine_state["pause"]["disposition"] == "ctrl_f"


def test_ctrl_g_recheckpoints_with_new_remaining_time_and_exits() -> None:
    trainer, checkpoints = _trainer(_review_pause_state())

    result = resume_plastic_coarse_fine_boundary(
        trainer,
        "checkpoint.pt",
        output=io.StringIO(),
        pause_runner=lambda **_: PlasticCoarsePauseResult(
            "checkpoint_exit",
            25.0,
            675.0,
        ),
    )

    assert result == PLASTIC_RESUME_CHECKPOINT_EXIT
    assert checkpoints == [Path("checkpoint.pt")]
    assert trainer.plastic_coarse_fine_state["phase"] == "review_pause"
    assert trainer.plastic_coarse_fine_state["pause"]["remaining_seconds"] == 675.0
    assert trainer.distributed.barriers == 1
