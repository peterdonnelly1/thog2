# vvv THOG
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sheet.stage4_runtime_check import compare_runtime_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        evidence = {
            "test": "S4-07",
            "satisfied": False,
            "reason": "CUDA is not available in this environment",
        }
    else:
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        evidence = compare_runtime_memory(device="cuda", dtype=dtype)
        evidence["device"] = torch.cuda.get_device_name(0)
        evidence["dtype"] = dtype
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if torch.cuda.is_available() and not evidence["satisfied"]:
        raise SystemExit("Stage 4 CUDA memory gate failed")


if __name__ == "__main__":
    main()
# ^^^ THOG
