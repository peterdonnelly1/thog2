#!/bin/bash
# vvv THOG one-time persistent power capability setup on each executing host
set -euo pipefail
if [[ $EUID -ne 0 || -z ${SUDO_USER:-} || ! $SUDO_USER =~ ^[a-z_][a-z0-9_-]*\$?$ ]]; then
  echo 'Run this installer with sudo from the Instra host user account.' >&2
  exit 1
fi
source_file="$(cd -- "$(dirname -- "$0")" && pwd)/instra_power_control.py"
helper=/usr/local/libexec/instra-power-control
rule="/etc/sudoers.d/instra-power-control-${SUDO_USER}"
install -d -o root -g root -m 0755 /usr/local/libexec
install -o root -g root -m 0755 "$source_file" "$helper"
printf '%s ALL=(root) NOPASSWD: %s\n' "$SUDO_USER" "$helper" > "$rule"
chmod 0440 "$rule"
if ! visudo -cf "$rule"; then rm -f "$rule"; exit 1; fi
sudo -u "$SUDO_USER" sudo -n "$helper" --check
echo 'Instra GPU power control installed. Refresh discovery in Networks.'
# ^^^ THOG
