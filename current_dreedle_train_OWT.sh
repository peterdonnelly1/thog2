#!/bin/bash
set -euo pipefail

# vvv THOG
# Current dreedle OpenWebText training wrapper for the PICTON compact-geometry contract.
# Dreedle runtime defaults: float16, sdpa. Dense baseline is available as -p dense.
#
# Optimizer selection:
#   -y NAME, --optimizer NAME
#       adamw | sgd | sgd_nesterov | adafactor | rmsprop
#       Aliases: adam; nesterov; sgd-nesterov.
#   --optimizer-momentum VALUE
#       Momentum for sgd, sgd_nesterov, and rmsprop. Default: 0.9.
#
# Optimizer-specific learning-rate defaults apply only when -c and/or -f are omitted:
#   optimizer       -c / maximum LR       -f / minimum LR
#   adamw             60 / 6.0e-4           06 / 6.0e-5
#   sgd             1000 / 1.0e-2          100 / 1.0e-3
#   sgd_nesterov    1000 / 1.0e-2          100 / 1.0e-3
#   adafactor       1000 / 1.0e-2          100 / 1.0e-3
#   rmsprop          100 / 1.0e-3           10 / 1.0e-4
#
# Explicit -c and -f values override those defaults independently. Lowercase -c is
# the learning-rate code; capital -C remains the context length. Non-AdamW runs add
# OPT_<OPTIMIZER> to the artifact suffix to prevent otherwise identical collisions.
# ^^^ THOG

cd "$(dirname "$0")"

for argument in "$@"; do
  if [[ "$argument" == "-h" ]]; then
    cat <<'EOF_OPTIMIZER_HELP'
Optimizer selection:
  -y NAME, --optimizer NAME
      adamw | sgd | sgd_nesterov | adafactor | rmsprop
  --optimizer-momentum VALUE
      Momentum for SGD, Nesterov SGD, and RMSprop; default 0.9.

Optimizer defaults when -c and/or -f are omitted:
  optimizer       -c / maximum LR       -f / minimum LR
  adamw             60 / 6.0e-4           06 / 6.0e-5
  sgd             1000 / 1.0e-2          100 / 1.0e-3
  sgd_nesterov    1000 / 1.0e-2          100 / 1.0e-3
  adafactor       1000 / 1.0e-2          100 / 1.0e-3
  rmsprop          100 / 1.0e-3           10 / 1.0e-4

  -c is the learning-rate code; capital -C is context length.
  Explicit -c/-f values override the optimizer defaults independently.

EOF_OPTIMIZER_HELP
    break
  fi
done

export THOG2_SOURCE_OPTIMIZER_TARGET=true
source ./optimizer_train_OWT_wrapper.sh ./.current_dreedle_train_OWT_impl "$@"
