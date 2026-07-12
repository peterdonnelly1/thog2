#!/bin/bash
set -euo pipefail

# vvv THOG
# Dense convenience entry point. The canonical scruffy wrapper owns the common getopts contract.
# ^^^ THOG

cd "$(dirname "$0")"
exec ./current_scruffy_train_OWT.sh -p dense "$@"
