# vvv THOG estimate comparable workloads, including iteration count, power policy and actual queue scheduling
"""Empirical Runner duration estimates; incompatible histories stay unknown."""
import heapq
import math
from collections import deque
from statistics import median

NON_TIMING_FIELDS = {"--max-iters", "--learning-rate", "--min-lr", "--warmup-iters", "--lr-decay-iters",
                     "--run-name", "--host-label", "--experiment-prefix", "--run-start-label", "--artifact-suffix",
                     "--log-timestamp", "--checkpoint-root", "--log-root", "--result-root", "--wandb-root",
                     "--wandb-project", "--wandb-entity"}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _canonical(value):
    if isinstance(value, str):
        if value in ("width", "width-type-I"):
            return "width"
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return float(value)
        except ValueError:
            pass
    if isinstance(value, list):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _canonical(item)) for key, item in value.items()))
    return value


def _workload_key(run):
    parameters = tuple(sorted((key, _canonical(value)) for key, value in run.get("parameters", {}).items()
                              if key not in NON_TIMING_FIELDS and value is not None))
    power = run.get("requested_power_w") or run.get("default_power_w") or run.get("gpu", {}).get("default_power_w")
    return (parameters, run.get("profiler"), run.get("dtype"), run.get("attention_backend"),
            run.get("gpu", {}).get("model"), _canonical(power))


def _predict(run, candidates):
    target = _number(run.get("parameters", {}).get("--max-iters"))
    exact = [float(prior["duration_seconds"]) for prior in candidates
             if _number(prior.get("parameters", {}).get("--max-iters")) == target]
    if exact:
        return median(exact[:8]), exact[:8], "exact", len(exact[:8])
    if target is None or any(run.get("parameters", {}).get(key) for key in ("--resume", "--fork", "--resume-from")):
        return None, [], "unknown", 0
    points = [(_number(prior.get("parameters", {}).get("--max-iters")), float(prior["duration_seconds"]))
              for prior in candidates]
    points = [(steps, seconds) for steps, seconds in points if steps is not None]
    slopes = [(right[1] - left[1]) / (right[0] - left[0])
              for index, left in enumerate(points[:16]) for right in points[index+1:16]
              if right[0] != left[0] and (right[1] - left[1]) / (right[0] - left[0]) > 0]
    if slopes and target <= 4 * max(steps for steps, _ in points):
        rate = median(slopes)
        startup = max(0.0, median(seconds - rate * steps for steps, seconds in points))
        return startup + rate * target, [startup + slope * target for slope in slopes], "fitted", len(points)
    scaled = [seconds * target / steps for steps, seconds in points
              if steps >= 50 and .25 <= target / steps <= 4]
    if scaled:
        return median(scaled[:8]), scaled[:8], "scaled", len(scaled[:8])
    return None, [], "unknown", 0


def estimate(runs, history, parallelism=1, *, dynamic=False):
    exemplars = [prior for grid in reversed(history) for prior in reversed(grid.get("runs", []))
                 if prior.get("state") == "completed" and _number(prior.get("duration_seconds")) is not None]
    # Index completed history once. A large Grid must not compare every trial
    # with every historical run, or refit identical learning-rate sweep points.
    workloads = {}
    for prior in exemplars:
        workloads.setdefault(_workload_key(prior), []).append(prior)
    retained_predictions = {}
    predictions = []
    for run in runs:
        workload = _workload_key(run)
        key = (workload, _number(run.get("parameters", {}).get("--max-iters")))
        if key not in retained_predictions:
            retained_predictions[key] = _predict(run, workloads.get(workload, []))
        predictions.append(retained_predictions[key])
    durations = [value[0] for value in predictions]
    matched = sum(value[3] for value in predictions)
    if not runs or any(duration is None for duration in durations):
        return {"seconds": None, "run_seconds": durations, "interval_seconds": None,
                "confidence": "insufficient", "exemplars": matched,
                "explanation": "No comparable completed workload for one or more runs; unrelated configurations and large smoke-run extrapolations are excluded"}
    def schedule(values):
        lanes = {(run.get("host_id"), run.get("gpu", {}).get("gpu_key")): 0.0 for run in runs}
        capacity = max(1, min(int(parallelism), len(lanes)))
        homogeneous = all(item.get("gpu", {}).get("model") == runs[0].get("gpu", {}).get("model") and
                          (item.get("requested_power_w") or item.get("default_power_w")) ==
                          (runs[0].get("requested_power_w") or runs[0].get("default_power_w")) and
                          item.get("dtype") == runs[0].get("dtype") and
                          item.get("attention_backend") == runs[0].get("attention_backend") for item in runs)
        if dynamic and homogeneous:
            slots = [0.0] * capacity
            for duration in values:
                heapq.heappush(slots, heapq.heappop(slots) + duration)
            return max(slots)
        # A queued run on a busy GPU cannot reserve a global slot and postpone
        # work on another idle GPU. Dispatch eligible lane heads in run order.
        queues = {key: deque() for key in lanes}
        for index, (run, duration) in enumerate(zip(runs, values)):
            key = (run.get("host_id"), run.get("gpu", {}).get("gpu_key"))
            queues[key].append((index, duration))
        available = [(queue[0][0], key) for key, queue in queues.items()]
        heapq.heapify(available)
        active = []
        clock = 0.0
        while available or active:
            while available and len(active) < capacity:
                index, key = heapq.heappop(available)
                _, duration = queues[key].popleft()
                heapq.heappush(active, (clock + duration, index, key))
            clock = active[0][0]
            while active and active[0][0] == clock:
                _, _, key = heapq.heappop(active)
                if queues[key]:
                    heapq.heappush(available, (queues[key][0][0], key))
        return clock
    lower, upper = [], []
    high = all(kind == "exact" and count >= 3 for _, _, kind, count in predictions)
    for central, samples, kind, count in predictions:
        margin = .1 if kind == "exact" and count >= 3 else .3 if kind == "exact" else .5
        lower.append(min(min(samples), central * (1-margin)))
        upper.append(max(max(samples), central * (1+margin)))
    kinds = {kind for _, _, kind, _ in predictions}
    return {"seconds": round(schedule(durations)), "run_seconds": durations,
            "interval_seconds": [round(schedule(lower)), round(schedule(upper))],
            "confidence": "high" if high else "low", "exemplars": matched,
            "explanation": f"Comparable configuration, GPU, power and profiling; {', '.join(sorted(kinds))} workload estimates; startup and evaluation are included in observed runtimes; external GPU queue waits are excluded"}
# ^^^ THOG
