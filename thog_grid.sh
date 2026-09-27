#!/usr/bin/env bash
# vvv THOG GNU getopt public Grid wrapper; the shared catalogue defines the long-option surface
set -euo pipefail
cd "$(dirname "$0")"
long_options="$(python thog_grid_runner.py --getopt-options)"
parsed="$(getopt --options 'q:g:n:b:c:f:y:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:W:i:a:m:L:s:M:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h' --longoptions "$long_options" --name thog_grid.sh -- "$@")" || exit 2
eval "set -- $parsed"
exec python thog_grid_runner.py "$@"
# ^^^ THOG
