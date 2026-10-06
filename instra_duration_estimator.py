# vvv THOG derive transparent Grid estimates from completed comparable Runner attempts
"""Conservative exemplar-based estimates, with no invented precision when empty."""

from statistics import median


FIELDS = ("--n-layer", "DEPTH.order", "--n-embd", "--block-size", "--batch-size",
          "--gradient-accumulation-steps", "--checkpoint-segment-size", "--optimizer",
          "--eval-interval", "--torch-compile", "--geometry-preset")


def estimate(runs, history, parallelism=1):
    exemplars = []
    for old_grid in history:
        for old_run in old_grid.get("runs", []):
            if old_run.get("state") != "completed" or not old_run.get("duration_seconds"):
                continue
            exemplars.append(old_run)
    if not exemplars:
        return {"seconds": None, "interval_seconds": None, "confidence": "insufficient",
                "exemplars": 0, "explanation": "No completed comparable Runner runs"}
    durations = []
    exact = 0
    for run in runs:
        scores = []
        for prior in exemplars:
            mismatch = sum(run["parameters"].get(field) != prior["parameters"].get(field) for field in FIELDS)
            mismatch += 2 * (run.get("profiler") != prior.get("profiler"))
            mismatch += 2 * (run.get("gpu", {}).get("model") != prior.get("gpu", {}).get("model"))
            mismatch += 2 * (run.get("dtype") != prior.get("dtype") or run.get("attention_backend") != prior.get("attention_backend"))
            scores.append((mismatch, prior["duration_seconds"]))
        best = min(score for score, _ in scores)
        if best == 0:
            exact += 1
        candidates = [duration for score, duration in scores if score == best]
        durations.append(median(candidates[:8]))
    # Simulate assigned GPUs and per-Grid parallelism, not a serial sum.
    timelines = {}
    for run, duration in zip(runs, durations):
        key = (run.get("host_id"), run.get("gpu", {}).get("gpu_key"))
        timelines[key] = timelines.get(key, 0) + duration
    if not timelines:
        return {"seconds": None, "interval_seconds": None, "confidence": "insufficient", "exemplars": 0}
    central = max(max(timelines.values()), sum(durations) / max(1, min(parallelism, len(timelines))))
    confidence = "high" if exact == len(runs) and len(exemplars) >= 3 else "low"
    spread = .2 if confidence == "high" else .6
    return {"seconds": round(central), "run_seconds": durations, "interval_seconds": [round(central * (1-spread)), round(central * (1+spread))],
            "confidence": confidence, "exemplars": len(exemplars),
            "explanation": f"{exact}/{len(runs)} exact matches; nearest runs used for remaining estimates"}
# ^^^ THOG
