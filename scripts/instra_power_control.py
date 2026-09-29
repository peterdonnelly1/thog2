#!/usr/bin/python3
# vvv THOG root-owned, narrowly scoped GPU power helper; installed by Network host setup
"""Apply a validated NVIDIA GPU power limit and verify the resulting readback."""

import json
import re
import subprocess
import sys

NVIDIA_SMI = "/usr/bin/nvidia-smi"


def query(gpu_uuid):
    result = subprocess.run([NVIDIA_SMI, "-i", gpu_uuid,
                             "--query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=12, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected exactly one GPU")
    fields = [value.strip() for value in lines[0].split(",")]
    if len(fields) != 5 or fields[0] != gpu_uuid:
        raise ValueError("GPU identity or power-limit query changed")
    return dict(zip(("uuid", "current_w", "default_w", "minimum_w", "maximum_w"),
                    [gpu_uuid, *[float(value) for value in fields[1:]]]))


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        if __import__("os").geteuid() != 0:
            raise PermissionError("Power helper must run as root")
        print(json.dumps({"ready": True}))
        return
    if len(sys.argv) != 3 or not re.fullmatch(r"GPU-[A-Za-z0-9-]{8,80}", sys.argv[1]):
        raise ValueError("Usage: instra-power-control GPU-UUID default|watts")
    gpu_uuid, requested = sys.argv[1:]
    if requested != "default" and not re.fullmatch(r"[1-9][0-9]{1,2}", requested):
        raise ValueError("Power request must be 'default' or integer watts")
    before = query(gpu_uuid)
    target = before["default_w"] if requested == "default" else float(requested)
    if not before["minimum_w"] <= target <= before["maximum_w"]:
        raise ValueError(f"{target:g} W outside GPU range {before['minimum_w']:g}–{before['maximum_w']:g} W")
    if abs(before["current_w"] - target) > 0.5:
        changed = subprocess.run([NVIDIA_SMI, "-i", gpu_uuid, "-pl", f"{target:g}"],
                                 capture_output=True, text=True, timeout=12)
        if changed.returncode:
            raise RuntimeError((changed.stderr or changed.stdout).strip() or f"nvidia-smi exit {changed.returncode}")
    after = query(gpu_uuid)
    if abs(after["current_w"] - target) > 0.5:
        raise RuntimeError(f"Power readback {after['current_w']:g} W differs from requested {target:g} W")
    print(json.dumps({"before": before, "after": after, "target_w": target,
                      "changed": abs(before["current_w"] - target) > 0.5}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
# ^^^ THOG
