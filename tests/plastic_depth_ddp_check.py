# vvv THOG
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        evidence_path = Path(directory) / "plastic_depth_ddp.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(ROOT / "tests" / "plastic_depth_ddp_worker.py"),
                "--evidence",
                str(evidence_path),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "PLASTIC DEPTH DDP check failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["completed_updates"] == 2
    assert evidence["count_decisions"] >= 1
    assert 1 <= evidence["active_layers"] <= 4
    assert evidence["model_state_max_delta"] == 0.0
    assert evidence["optimizer_state_max_delta"] == 0.0
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
# ^^^ THOG
