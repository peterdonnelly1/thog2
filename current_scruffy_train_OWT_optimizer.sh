#!/bin/bash
set -euo pipefail

# vvv THOG
# Optimizer-aware front end for current_scruffy_train_OWT.sh.
#
# Additional controls:
#   -y / --optimizer: adamw | sgd | sgd_nesterov | adafactor | rmsprop
#   --optimizer-momentum: momentum for SGD, Nesterov SGD, and RMSprop
#
# The authoritative defaults and examples are documented in
# optimizer_train_OWT_wrapper.sh. In particular, -c is the learning-rate code
# and capital -C remains the context length.
# ^^^ THOG

cd "$(dirname "$0")"
exec ./optimizer_train_OWT_wrapper.sh ./current_scruffy_train_OWT.sh "$@"
