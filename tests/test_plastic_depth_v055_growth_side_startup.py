from __future__ import annotations

import pytest

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v055_growth_side_discount_patch as growth
from sheet import plastic_depth_v055_growth_side_startup_patch as startup


def test_stratified_startup_rows_show_growth_side_discount(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.6")

    rows = startup._gradient_header_rows_with_growth_side_discount()

    assert rows[0] == ("plastic__layer_count_decision_algorithm:", v055.STRATIFIED_ALGORITHM)
    assert rows[-1] == (
        "plastic__layer_count_decision_algorithm__growth_side_discount:",
        "0.6",
    )


def test_legacy_startup_rows_do_not_show_growth_side_discount(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, tsk.LEGACY_DIRECTIONAL_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.6")

    rows = startup._gradient_header_rows_with_growth_side_discount()

    assert all("growth_side_discount" not in label for label, _value in rows)
