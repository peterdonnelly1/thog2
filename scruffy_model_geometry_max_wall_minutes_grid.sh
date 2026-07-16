#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG equal wall-clock compact-geometry screen; defaults are MELTON-like except for the geometry preset axis
MAX_WALL_MINUTES="${THOG2_GEOM_GRID_MAX_WALL_MINUTES:-360}"
MAX_ITERS="${THOG2_GEOM_GRID_MAX_ITERS:-1000000}"
RUN_NAME="${THOG2_GEOM_GRID_NAME:-GEOM_WALL_6H}"
GEOMETRIES="${THOG2_GEOM_GRID_GEOMETRIES:-legacy_sheet_col depth head_aware_block mlp_block full_block}"
EVAL_ITERS="${THOG2_GEOM_GRID_EVAL_ITERS:-50}"
EVAL_INTERVAL="${THOG2_GEOM_GRID_EVAL_INTERVAL:-1000000}"
# ^^^ THOG

exec ./current_scruffy_train_OWT.sh \
  -g "$RUN_NAME" \
  -p "$GEOMETRIES" \
  -n "$MAX_ITERS" \
  -b 32 \
  -c 60 \
  -f 06 \
  -A 4 \
  -u "$EVAL_ITERS" \
  -e "$EVAL_INTERVAL" \
  -l 10 \
  -w 20 \
  -k 0 \
  -I wandb \
  -F none \
  -B chebyshev \
  -L 64 \
  -H 16 \
  -D 1024 \
  -C 256 \
  -P 32 \
  -Q 256 \
  -J 6 \
  -O 6 \
  -X 64 \
  -Y 256 \
  -S 12 \
  -E true \
  -T bfloat16 \
  -K flash2 \
  -r depth_scaled \
  -z dof_implied_depth \
  -- --max-wall-minutes "$MAX_WALL_MINUTES" "$@"
# ^^^ THOG
