#!/bin/bash
set -euo pipefail

# vvv THOG
# DEPTH convenience entry point. The canonical scruffy wrapper owns the common getopts contract.
# ^^^ THOG

cd "$(dirname "$0")"
exec bash ./current_scruffy_train_OWT.sh -p depth "$@"
