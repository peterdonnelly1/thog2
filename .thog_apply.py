from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREAMBLE = "#!/bin/bash\nset -euo pipefail\n\n"

for filename in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    path = ROOT / filename
    source = path.read_text(encoding="utf-8")
    source = source.removeprefix("#!/bin/bash\n").removeprefix("set -euo pipefail\n").lstrip("\n")
    path.write_text(PREAMBLE + source, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)

for filename in ("tests/test_optimizer_wrapper.py",):
    path = ROOT / filename
    source = path.read_text(encoding="utf-8")
    old = '            source = path.read_text(encoding="utf-8")\n            self.assertGreater(len(source.splitlines()), 400)'
    new = '            source = path.read_text(encoding="utf-8")\n            self.assertTrue(source.startswith("#!/bin/bash\\nset -euo pipefail\\n"))\n            self.assertGreater(len(source.splitlines()), 400)'
    if old not in source:
        raise RuntimeError(f"expected wrapper assertion marker missing from {filename}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
