#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python run_thog2_owt_inference.py "$@"
