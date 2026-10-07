# vvv THOG guard duration extrapolation, configuration matching and GPU queue scheduling
from copy import deepcopy
import time

import pytest

from instra_duration_estimator import estimate


def run(steps=100, seconds=None, gpu="gpu-0", **parameters):
    value = {"parameters": {"--max-iters": steps, "--n-embd": 1024, "WIDTH.order": 512, **parameters},
             "gpu": {"gpu_key": gpu, "model": "test GPU"}, "host_id": "host", "profiler": "none",
             "dtype": "bfloat16", "attention_backend": "sdpa", "default_power_w": 300}
    if seconds is not None:
        value.update(state="completed", duration_seconds=seconds)
    return value


def test_iteration_count_changes_prediction_and_smoke_extrapolation_stays_unknown():
    history = [{"runs": [run(100, 120)]}]
    assert estimate([run(200)], history)["seconds"] == 240
    assert estimate([run(10000)], history)["seconds"] is None
    assert estimate([run(200)], [{"runs": [run(1, 12)]}])["seconds"] is None


def test_multiple_lengths_fit_startup_cost_instead_of_scaling_it():
    history = [{"runs": [run(100, 140), run(200, 240)]}]
    result = estimate([run(300)], history)
    assert result["seconds"] == 340
    assert "fitted" in result["explanation"]
    assert result["confidence"] == "low"


@pytest.mark.parametrize("change", [
    {"WIDTH.order": 1024}, {"--basis-family": "dct"}, {"--eval-iters": 20},
    {"--premat": "enabled"}, {"--instrumentation__depth_weight_curves__log_every_n_steps": 1},
    {"--hyperblock-d-model-order": 12}, {"--layer-dropout-active-per-stratum": 2},
])
def test_incompatible_workloads_are_not_nearest_neighbours(change):
    assert estimate([run(**change)], [{"runs": [run(seconds=120)]}])["seconds"] is None


@pytest.mark.parametrize("key,value", [("requested_power_w", 200), ("dtype", "float32"),
                                      ("profiler", "ncu"), ("attention_backend", "eager")])
def test_hardware_execution_policy_must_match(key, value):
    candidate = run();candidate[key] = value
    assert estimate([candidate], [{"runs": [run(seconds=120)]}])["seconds"] is None


def test_learning_rate_sweep_and_numeric_strings_keep_comparable_samples():
    candidate = run(**{"--learning-rate": .001});candidate["parameters"]["--n-embd"] = "1024"
    result = estimate([candidate], [{"runs": [run(seconds=120, **{"--learning-rate": .0001})]}])
    assert result["seconds"] == 120


def test_queue_obeys_fixed_gpu_lanes_and_global_parallelism():
    history = [{"runs": [run(seconds=100)]}]
    runs = [run(gpu="gpu-0"), run(gpu="gpu-0"), run(gpu="gpu-1")]
    assert estimate(runs, history, 1)["seconds"] == 300
    assert estimate(runs, history, 2)["seconds"] == 200
    varied = [run(100,gpu="gpu-0"), run(100,gpu="gpu-1"), run(200,gpu="gpu-0")]
    assert estimate(varied, history, 2)["seconds"] == 300


def test_busy_fixed_lane_does_not_delay_another_idle_gpu():
    history = [{"runs": [run(seconds=100)]}]
    runs = [run(gpu="gpu-0"), run(gpu="gpu-0"), run(200, gpu="gpu-1")]
    assert estimate(runs, history, 2)["seconds"] == 200
    assert estimate(runs, history, 1)["seconds"] == 400


def test_high_confidence_requires_three_exact_samples_for_each_workload():
    history = [{"runs": [run(seconds=value) for value in (100, 110, 120)]}]
    assert estimate([run()], history)["confidence"] == "high"
    assert estimate([run(200)], history)["confidence"] == "low"


def test_unknown_resume_offset_prevents_iteration_scaling():
    resume = {"--resume": "checkpoint.pt"}
    assert estimate([run(200, **resume)], [{"runs": [run(100, 120, **resume)]}])["seconds"] is None


def test_large_grid_history_estimation_remains_responsive():
    history = [{"runs":[run(seconds=100) for _ in range(1000)]}]
    trials = [run(gpu=f"gpu-{index%2}", **{"--learning-rate": .001+index/1000000}) for index in range(1000)]
    started = time.monotonic()
    result = estimate(trials,history,2,dynamic=True)
    assert result["seconds"] == 50000
    assert time.monotonic()-started < 2
# ^^^ THOG
