from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


premat_path = Path("sheet/premat.py")
source = premat_path.read_text()

source = replace_once(
    source,
    '''    wait_end_event: Optional[torch.cuda.Event] = None
    consumed: bool = False
''',
    '''    wait_end_event: Optional[torch.cuda.Event] = None
    consumed: bool = False
    duplicate_main_materialisations: int = 0
''',
    "interval duplicate field",
)
source = replace_once(
    source,
    '''    real_premat_consumed: bool = False
    forensic_interval: Optional[_ForensicCandidateInterval] = None
''',
    '''    real_premat_consumed: bool = False
    forensic_interval: Optional[_ForensicCandidateInterval] = None
    duplicate_main_materialisations: int = 0
''',
    "candidate duplicate field",
)

source = replace_once(
    source,
    '''            "forensic_dependency_before_premat_start_count": 0,
            # ^^^ THOG
''',
    '''            "forensic_dependency_before_premat_start_count": 0,
            "forensic_premature_wait_count": 0,
            "forensic_premature_wait_gap_ms_total": 0.0,
            # ^^^ THOG
''',
    "aggregate premature wait fields",
)
source = replace_once(
    source,
    '''            "dependency_before_premat_start_count": 0,
        }
''',
    '''            "dependency_before_premat_start_count": 0,
            "premature_wait_count": 0,
            "premature_wait_gap_ms_total": 0.0,
        }
''',
    "family premature wait fields",
)

source = replace_once(
    source,
    '''        main_consume_count = int(values.get("main_consume_count", 0))
        main_consume_ms = float(values.get("main_consume_gpu_ms_total", 0.0))
        return {
''',
    '''        main_consume_count = int(values.get("main_consume_count", 0))
        main_consume_ms = float(values.get("main_consume_gpu_ms_total", 0.0))
        premature_wait_count = int(values.get("premature_wait_count", 0))
        return {
''',
    "derived premature wait local",
)
source = replace_once(
    source,
    '''            "premat_consumption_fraction": (
                int(values.get("premat_consumed", 0)) / launched
                if launched > 0
                else None
            ),
''',
    '''            "premat_consumption_fraction": (
                int(values.get("premat_consumed", 0)) / launched
                if launched > 0
                else None
            ),
            "premat_materialisation_ms_mean": (
                premat_ms / launched
                if launched > 0
                else None
            ),
            "dependency_lead_window_ms_mean": (
                float(values.get("dependency_lead_window_ms_total", 0.0)) / dependency_count
                if dependency_count > 0
                else None
            ),
            "dependency_shortfall_ms_mean": (
                float(values.get("dependency_shortfall_ms_total", 0.0)) / dependency_count
                if dependency_count > 0
                else None
            ),
            "dependency_slack_ms_mean": (
                float(values.get("dependency_slack_ms_total", 0.0)) / dependency_count
                if dependency_count > 0
                else None
            ),
            "premature_wait_gap_ms_mean": (
                float(values.get("premature_wait_gap_ms_total", 0.0)) / premature_wait_count
                if premature_wait_count > 0
                else None
            ),
''',
    "derived diagnostic means",
)

source = replace_once(
    source,
    '''            "dependency_before_premat_start_count": int(aggregate["forensic_dependency_before_premat_start_count"]),
        }
''',
    '''            "dependency_before_premat_start_count": int(aggregate["forensic_dependency_before_premat_start_count"]),
            "premature_wait_count": int(aggregate["forensic_premature_wait_count"]),
            "premature_wait_gap_ms_total": float(aggregate["forensic_premature_wait_gap_ms_total"]),
        }
''',
    "cumulative premature wait report",
)
source = replace_once(
    source,
    '''                "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
            }
''',
    '''                "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
                "premature_wait_count": int(row["premature_wait_count"]),
                "premature_wait_gap_ms_total": float(row["premature_wait_gap_ms_total"]),
            }
''',
    "family premature wait report",
)

source = replace_once(
    source,
    '''                dependency_before_start_count = 0

                for candidate in pending.candidate_intervals:
''',
    '''                dependency_before_start_count = 0
                premature_wait_count = 0
                premature_wait_gap_ms_total = 0.0
                main_work_by_key = {
                    (int(row["layer_index"]), str(row["family"])): row
                    for row in main_work_rows
                }

                for candidate in pending.candidate_intervals:
''',
    "resolver premature wait locals",
)
source = replace_once(
    source,
    '''                    dependency_before_start = False
                    if candidate.dependency_event is not None:
''',
    '''                    dependency_before_start = False
                    premature_wait_gap_ms = None
                    if candidate.dependency_event is not None:
''',
    "candidate premature gap local",
)
source = replace_once(
    source,
    '''                        if shortfall_ms > 0.0:
                            shortfall_count += 1

                    row = {
''',
    '''                        if shortfall_ms > 0.0:
                            shortfall_count += 1
                        if candidate.wait_end_event is not None:
                            matching_main_work = main_work_by_key.get(
                                (candidate.layer_index, candidate.family)
                            )
                            if matching_main_work is not None:
                                wait_end_ms = at_ms(candidate.wait_end_event)
                                premature_wait_gap_ms = max(
                                    0.0,
                                    float(matching_main_work["start_ms"]) - wait_end_ms,
                                )
                                premature_wait_count += 1
                                premature_wait_gap_ms_total += premature_wait_gap_ms

                    row = {
''',
    "compute premature wait gap",
)
source = replace_once(
    source,
    '''                        "dependency_before_premat_start": dependency_before_start,
                        "temporal_overlap_ms": temporal_overlap_ms,
''',
    '''                        "dependency_before_premat_start": dependency_before_start,
                        "premature_wait_gap_ms": premature_wait_gap_ms,
                        "temporal_overlap_ms": temporal_overlap_ms,
''',
    "candidate premature gap row",
)
source = replace_once(
    source,
    '''                    family_row["unused"] += int(not candidate.consumed)
                    family_row["premat_gpu_ms_total"] += materialisation_ms
''',
    '''                    family_row["unused"] += int(not candidate.consumed)
                    family_row["duplicates"] += int(candidate.duplicate_main_materialisations)
                    family_row["premat_gpu_ms_total"] += materialisation_ms
''',
    "pass family duplicate count",
)
source = replace_once(
    source,
    '''                        family_row["dependency_before_premat_start_count"] += int(dependency_before_start)

                    premat_ms_total += materialisation_ms
''',
    '''                        family_row["dependency_before_premat_start_count"] += int(dependency_before_start)
                        if candidate.wait_end_event is not None and premature_wait_gap_ms is not None:
                            family_row["premature_wait_count"] += 1
                            family_row["premature_wait_gap_ms_total"] += premature_wait_gap_ms

                    premat_ms_total += materialisation_ms
''',
    "family premature gap aggregate",
)
source = replace_once(
    source,
    '''                    "duplicate_main_materialisations": 0,
                    "main_layer_count": len(layer_rows),
''',
    '''                    "duplicate_main_materialisations": sum(
                        int(item.duplicate_main_materialisations)
                        for item in pending.candidate_intervals
                    ),
                    "main_layer_count": len(layer_rows),
''',
    "pass duplicate summary",
)
source = replace_once(
    source,
    '''                    "dependency_before_premat_start_count": dependency_before_start_count,
                }
''',
    '''                    "dependency_before_premat_start_count": dependency_before_start_count,
                    "premature_wait_count": premature_wait_count,
                    "premature_wait_gap_ms_total": premature_wait_gap_ms_total,
                }
''',
    "pass premature wait summary",
)
source = replace_once(
    source,
    '''                            "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
                        })
''',
    '''                            "dependency_before_premat_start_count": int(row["dependency_before_premat_start_count"]),
                            "premature_wait_count": int(row["premature_wait_count"]),
                            "premature_wait_gap_ms_total": float(row["premature_wait_gap_ms_total"]),
                        })
''',
    "pass family premature wait summary",
)
source = replace_once(
    source,
    '''                "forensic_dependency_before_premat_start_count": dependency_before_start_count,
            }
''',
    '''                "forensic_dependency_before_premat_start_count": dependency_before_start_count,
                "forensic_premature_wait_count": premature_wait_count,
                "forensic_premature_wait_gap_ms_total": premature_wait_gap_ms_total,
            }
''',
    "aggregate premature wait deltas",
)
source = replace_once(
    source,
    '''                    "dependency_before_premat_start_count",
                ):
''',
    '''                    "dependency_before_premat_start_count",
                    "premature_wait_count",
                    "premature_wait_gap_ms_total",
                ):
''',
    "cumulative family premature fields",
)

# Track per-pass duplicate evidence in the same immutable candidate interval used for timing.
source = replace_once(
    source,
    '''            if candidate.real_premat_launched:
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
''',
    '''            if candidate.real_premat_launched:
                candidate.duplicate_main_materialisations += 1
                if candidate.forensic_interval is not None:
                    candidate.forensic_interval.duplicate_main_materialisations += 1
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
''',
    "fallback per-pass duplicate",
)
source = replace_once(
    source,
    '''            if candidate is not None and candidate.real_premat_launched:
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
''',
    '''            if candidate is not None and candidate.real_premat_launched:
                candidate.duplicate_main_materialisations += 1
                if candidate.forensic_interval is not None:
                    candidate.forensic_interval.duplicate_main_materialisations += 1
                self._aggregate["duplicate_main_materialisations_after_premat"] += 1
                self._forensic_by_family[candidate.family]["duplicates"] += 1
''',
    "replay per-pass duplicate",
)

premat_path.write_text(source)


js_path = Path("sheet/local_dashboard_assets/dashboard_premat.js")
js = js_path.read_text()
js = replace_once(
    js,
    '''      ["dependency shortfall", `${premat_ms(forensic.dependency_shortfall_ms_total)} · ${Number(forensic.dependency_shortfall_count || 0)}/${Number(forensic.dependency_count || 0)}`],
      ["PREMAT launched / used / unused", `${Number(forensic.premat_launched || 0)} / ${Number(forensic.premat_consumed || 0)} / ${Number(forensic.premat_unused || 0)}`],
''',
    '''      ["PREMAT materialisation mean", premat_ms(forensic.premat_materialisation_ms_mean)],
      ["lead to dependency mean", premat_ms(forensic.dependency_lead_window_ms_mean)],
      ["dependency shortfall", `${premat_ms(forensic.dependency_shortfall_ms_total)} · mean ${premat_ms(forensic.dependency_shortfall_ms_mean)} · ${Number(forensic.dependency_shortfall_count || 0)}/${Number(forensic.dependency_count || 0)}`],
      ["dependency slack mean", premat_ms(forensic.dependency_slack_ms_mean)],
      ["wait→consume independent gap", `${premat_ms(forensic.premature_wait_gap_ms_total)} · mean ${premat_ms(forensic.premature_wait_gap_ms_mean)}`],
      ["PREMAT launched / used / unused", `${Number(forensic.premat_launched || 0)} / ${Number(forensic.premat_consumed || 0)} / ${Number(forensic.premat_unused || 0)}`],
''',
    "Instra slack and wait specificity",
)
js_path.write_text(js)


test_path = Path("tests/test_premat.py")
tests = test_path.read_text()
tests = replace_once(
    tests,
    '''    assert "dependency shortfall" in javascript
    assert "PREMAT launched / used / unused" in javascript
''',
    '''    assert "PREMAT materialisation mean" in javascript
    assert "lead to dependency mean" in javascript
    assert "dependency shortfall" in javascript
    assert "dependency slack mean" in javascript
    assert "wait→consume independent gap" in javascript
    assert "PREMAT launched / used / unused" in javascript
''',
    "Instra specificity labels test",
)
# Synthetic pass: wait ends at 9 ms while the consuming QKV GEMM starts at 10 ms.
tests = replace_once(
    tests,
    '''                    family="QKV",
                    start_event=event(3.0),
                    end_event=event(7.0),
''',
    '''                    family="DOWN",
                    start_event=event(10.0),
                    end_event=event(14.0),
''',
    "synthetic matching consuming GEMM",
)
tests = replace_once(
    tests,
    '''    assert forensic["main_consume_gpu_ms_total"] == pytest.approx(4.0)
    assert forensic["premat_overlap_during_main_consume_ms_total"] == pytest.approx(4.0)
''',
    '''    assert forensic["main_consume_gpu_ms_total"] == pytest.approx(4.0)
    assert forensic["premat_overlap_during_main_consume_ms_total"] == pytest.approx(0.0)
    assert forensic["premature_wait_count"] == 1
    assert forensic["premature_wait_gap_ms_total"] == pytest.approx(1.0)
''',
    "synthetic premature wait assertions",
)
tests = replace_once(
    tests,
    '''    assert latest["main_consume_gpu_ms_mean"] == pytest.approx(4.0)
    assert latest["main_consume_covered_by_premat_fraction"] == pytest.approx(1.0)
''',
    '''    assert latest["main_consume_gpu_ms_mean"] == pytest.approx(4.0)
    assert latest["main_consume_covered_by_premat_fraction"] == pytest.approx(0.0)
    assert latest["premat_materialisation_ms_mean"] == pytest.approx(7.0)
    assert latest["dependency_lead_window_ms_mean"] == pytest.approx(6.0)
    assert latest["dependency_shortfall_ms_mean"] == pytest.approx(1.0)
    assert latest["premature_wait_gap_ms_mean"] == pytest.approx(1.0)
''',
    "synthetic derived specificity assertions",
)
test_path.write_text(tests)
