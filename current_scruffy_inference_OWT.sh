#!/bin/bash
set -euo pipefail
exec "$(dirname "$0")/scripts/current_scruffy_inference_OWT.sh" "$@"
