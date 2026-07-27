#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


train_row_pattern = re.compile(
    r"^T\s+(?P<step>\d+)\s+(?P<seconds>\d+)s\s+tok/s=\s*(?P<toks>\d+)"
    r"\s+cum tokens=\s*(?P<tokens>\d+)\s+training loss\s+=\s*(?P<loss>[0-9.]+)"
)


def find_latest_log(log_glob: str) -> Path:
    candidates = sorted(
        Path(".").glob(log_glob),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no logs matched: {log_glob}")
    return candidates[0]


def parse_train_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    hours: list[float] = []
    losses: list[float] = []

    for line in log_path.read_text(errors="replace").splitlines():
        match = train_row_pattern.search(line)
        if match is None:
            continue

        steps.append(int(match.group("step")))
        hours.append(int(match.group("seconds")) / 3600.0)
        losses.append(float(match.group("loss")))

    if not steps:
        raise ValueError(f"no training rows found in {log_path}")

    return steps, hours, losses


def plot_train_log(
    log_path: Path,
    output_path: Path,
    y_min: float,
    y_max: float,
    y_minor_tick: float,
) -> None:
    steps, hours, losses = parse_train_log(log_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(hours, losses, linewidth=1.0, alpha=0.75, label="train loss")

    ax.set_xlabel("wall time, hours")
    ax.set_ylabel("training loss")
    ax.set_title(log_path.parent.name)

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_tick))

    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", axis="y", alpha=0.15)

    ax.legend()
    fig.tight_layout()

    tmp_path = output_path.with_name(output_path.name + ".tmp.png")
    fig.savefig(tmp_path, dpi=140)
    tmp_path.replace(output_path)
    plt.close(fig)

    print(
        f"wrote {output_path} from {len(steps)} points; "
        f"latest T{steps[-1]} loss={losses[-1]:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot THOG2 training loss from train.log")
    parser.add_argument(
        "--log-glob",
        default="logs/*/train.log",
        help="glob used to find the newest train.log, relative to current directory",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="specific train.log path; overrides --log-glob",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/thog_train_loss_y3_to_y6.png"),
        help="output PNG path",
    )
    parser.add_argument("--y-min", type=float, default=3.0)
    parser.add_argument("--y-max", type=float, default=6.0)
    parser.add_argument("--y-minor-tick", type=float, default=0.2)
    parser.add_argument("--watch", action="store_true", help="refresh repeatedly")
    parser.add_argument("--interval", type=float, default=30.0, help="refresh interval in seconds")
    args = parser.parse_args()

    while True:
        log_path = args.log_path if args.log_path is not None else find_latest_log(args.log_glob)

        plot_train_log(
            log_path=log_path,
            output_path=args.output,
            y_min=args.y_min,
            y_max=args.y_max,
            y_minor_tick=args.y_minor_tick,
        )

        if not args.watch:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()