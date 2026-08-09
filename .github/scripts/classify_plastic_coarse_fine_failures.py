# vvv THOG
"""Classify broad PLASTIC regression failures against the recorded inherited baseline."""

from __future__ import annotations

import json
from pathlib import Path


cache_path = Path(".pytest_cache/v/cache/lastfailed")
branch_failed = set(
    json.loads(cache_path.read_text(encoding="utf-8"))
    if cache_path.exists()
    else {}
)
inherited = {
    line.strip()
    for line in Path("docs/THOG2_COARSE_FINE_INHERITED_FAILURES.txt")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
branch_only = sorted(branch_failed - inherited)
inherited_observed = sorted(branch_failed & inherited)
inherited_not_observed = sorted(inherited - branch_failed)

print("INHERITED_FAILURES_OBSERVED_BEGIN")
for node_id in inherited_observed:
    print(node_id)
print("INHERITED_FAILURES_OBSERVED_END")
print("BRANCH_ONLY_FAILURES_BEGIN")
for node_id in branch_only:
    print(node_id)
print("BRANCH_ONLY_FAILURES_END")
print(
    "Regression classification: "
    f"{len(branch_failed)} total failing node IDs; "
    f"{len(inherited_observed)} inherited; "
    f"{len(branch_only)} branch-only; "
    f"{len(inherited_not_observed)} recorded inherited failures now passing."
)
Path("branch_only_failures.txt").write_text(
    "\n".join(branch_only) + ("\n" if branch_only else ""),
    encoding="utf-8",
)
if branch_only:
    raise SystemExit("branch-only CPU regression failures remain")
# ^^^ THOG
