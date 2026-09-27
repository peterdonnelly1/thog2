# vvv THOG detached attempt supervisor records a durable result after the training process exits
"""Private Node Agent child; accepts only a local 0600 metadata file."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu_ordinal"])
    environment["THOG2_RUNNER_METADATA"] = json.dumps(config["runner_metadata"], separators=(",", ":"))                                   # <<< THOG carry identity into local charts and W&B telemetry
    environment.update(config.get("runner_environment", {}))                                                                                               # <<< THOG apply wrapper-only allocator and instrumentation policy before torch import
    process = None

    def stop(_signal, _frame):
        if process is not None and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    with open(config["log_path"], "ab", buffering=0) as output:
        try:
            process = subprocess.Popen(config["argv"], cwd=config["cwd"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            status = process.wait()
        except (OSError, ValueError) as error:
            output.write(f"Runner launch failed: {type(error).__name__}: {error}\n".encode())
            status = 127
    result = Path(config["exit_path"])
    temporary = result.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(f"{status}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, result)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
