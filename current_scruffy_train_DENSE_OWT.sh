#!/bin/bash
set -euo pipefail

# vvv THOG
# Dense convenience entry point. The canonical scruffy wrapper owns the common
# options and passes through optimizer controls unchanged:
#   -y / --optimizer: adamw | sgd | sgd_nesterov | adafactor | rmsprop
#   --optimizer-momentum: momentum for SGD, Nesterov SGD, and RMSprop
# Optimizer-specific -c/-f defaults are documented by current_scruffy_train_OWT.sh -h.
# ^^^ THOG

cd "$(dirname "$0")"
exec bash ./current_scruffy_train_OWT.sh -p dense "$@"
