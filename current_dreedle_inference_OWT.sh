#!/bin/bash
set -euo pipefail
exec "$(dirname "$0")/scripts/current_dreedle_inference_OWT.sh" "$@"
