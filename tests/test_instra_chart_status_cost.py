# vvv THOG keep catalogue status accurate while bounding work for mature snapshot stores
import sqlite3

import pytest

from sheet.local_chart_store import LocalChartReader


@pytest.mark.parametrize("count", [0, 1, 10000])
def test_record_summary_uses_fast_count_and_index_endpoints(count):
    connection = sqlite3.connect(":memory:");connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE depth_weight_snapshots (optimizer_update INTEGER PRIMARY KEY, payload BLOB)")
    connection.executemany("INSERT INTO depth_weight_snapshots VALUES (?, ?)", ((2*i+3,b'x'*100) for i in range(count)))
    instructions = 0
    def progress():
        nonlocal instructions
        instructions += 1
        return 0
    connection.set_progress_handler(progress,1)
    summary = LocalChartReader._record_summary(connection,"depth_weight_snapshots")
    assert dict(summary) == {"count":count,"minimum_update":3 if count else None,
                             "maximum_update":2*(count-1)+3 if count else None}
    assert instructions < 100
    with pytest.raises(ValueError,match="unsupported"):
        LocalChartReader._record_summary(connection,"metadata; DROP TABLE metadata")
    connection.close()
# ^^^ THOG
