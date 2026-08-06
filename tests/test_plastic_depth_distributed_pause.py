from __future__ import annotations

import io

import pytest

from sheet.plastic_depth_pause import (
    PlasticCoarsePauseResult,
    run_distributed_plastic_coarse_review_pause,
)


class _Distributed:
    def __init__(
        self,
        *,
        is_primary: bool,
        primary_result: PlasticCoarsePauseResult | None = None,
    ) -> None:
        self.is_primary = is_primary
        self.primary_result = primary_result
        self.gathered_values = []
        self.barriers = 0

    def all_gather_object(self, value):
        self.gathered_values.append(value)
        return [value if self.is_primary else self.primary_result]

    def barrier(self) -> None:
        self.barriers += 1


def test_primary_pause_result_is_shared() -> None:
    distributed = _Distributed(is_primary=True)
    expected = PlasticCoarsePauseResult("ctrl_f", 12.0, 888.0)

    result = run_distributed_plastic_coarse_review_pause(
        distributed,
        duration_seconds=900.0,
        output=io.StringIO(),
        pause_runner=lambda **_: expected,
    )

    assert result == expected
    assert distributed.gathered_values == [expected]
    assert distributed.barriers == 1


def test_non_primary_does_not_read_terminal() -> None:
    expected = PlasticCoarsePauseResult("timeout", 900.0, 0.0)
    distributed = _Distributed(
        is_primary=False,
        primary_result=expected,
    )

    result = run_distributed_plastic_coarse_review_pause(
        distributed,
        output=io.StringIO(),
        pause_runner=lambda **_: pytest.fail("non-primary rank must not read /dev/tty"),
    )

    assert result == expected
    assert distributed.gathered_values == [None]
    assert distributed.barriers == 1


def test_primary_keyboard_interrupt_is_propagated_after_collective() -> None:
    distributed = _Distributed(is_primary=True)

    with pytest.raises(KeyboardInterrupt):
        run_distributed_plastic_coarse_review_pause(
            distributed,
            duration_seconds=900.0,
            output=io.StringIO(),
            pause_runner=lambda **_: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    gathered = distributed.gathered_values[0]
    assert gathered.disposition == "interrupt"
    assert distributed.barriers == 1


def test_secondary_rank_raises_shared_interrupt() -> None:
    distributed = _Distributed(
        is_primary=False,
        primary_result=PlasticCoarsePauseResult("interrupt", 1.0, 899.0),
    )

    with pytest.raises(KeyboardInterrupt):
        run_distributed_plastic_coarse_review_pause(
            distributed,
            output=io.StringIO(),
            pause_runner=lambda **_: pytest.fail("secondary rank must not run pause"),
        )

    assert distributed.barriers == 1
