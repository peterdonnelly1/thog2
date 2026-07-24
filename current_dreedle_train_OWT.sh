#!/bin/bash
set -euo pipefail

# vvv THOG
# Runtime shim over the previous full wrapper body. This keeps the large working wrapper recoverable from Git history while applying small display/default fixes safely.
SOURCE_COMMIT="45cfe426e9caa24e9a7f4e23ad0fb5741a03e984"
SCRIPT_NAME="$(basename "$0")"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_SCRIPT="$(mktemp "$REPO_DIR/.${SCRIPT_NAME}.runtime.XXXXXX")"
cleanup_runtime_script() { rm -f "$RUNTIME_SCRIPT"; }
trap cleanup_runtime_script EXIT

git -C "$REPO_DIR" show "${SOURCE_COMMIT}:${SCRIPT_NAME}" > "$RUNTIME_SCRIPT"
python - "$RUNTIME_SCRIPT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = 'DEPTH_CURVE_PLOTS="${THOG2_DEPTH_CURVE_PLOTS:-eval}"'
new = 'DEPTH_CURVE_PLOTS="${THOG2_DEPTH_CURVE_PLOTS:-none}"'
if old not in text:
    raise SystemExit(f"missing wrapper default pattern: {old}")
text = text.replace(old, new, 1)

lines = text.splitlines()
out = []
in_report = False
for line in lines:
    if line == "  cat <<EOF_RUN":
        in_report = True
        out.append(line)
        continue
    if in_report and line == "EOF_RUN":
        in_report = False
        out.append(line)
        continue
    if in_report and line.startswith("  ") and ":" in line:
        label, value = line.split(":", 1)
        out.append(f"{label + ':':<42}{value.lstrip()}")
        continue
    out.append(line)
path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
PY

chmod +x "$RUNTIME_SCRIPT"
set +e
bash "$RUNTIME_SCRIPT" "$@"
RUN_STATUS=$?
set -e
exit "$RUN_STATUS"
# ^^^ THOG
