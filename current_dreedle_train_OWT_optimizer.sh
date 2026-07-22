#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./optimizer_train_OWT_wrapper.sh ./current_dreedle_train_OWT.sh "$@"
