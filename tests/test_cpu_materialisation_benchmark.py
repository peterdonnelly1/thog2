# vvv THOG whole-trainer RNG reproducibility and field-evidence failure regressions
import copy
import json
import os
import sys

import pytest
import torch

from tools import benchmark_cpu_materialisation as field


@pytest.fixture
def isolated_materializer(monkeypatch):
    from sheet import depth_materialisation_runtime as runtime
    from sheet.depth_trajectory import DepthTrajectory
    for name in ("_INSTALLED", "_PROFILING_INSTALLED"):
        monkeypatch.setattr(runtime, name, getattr(runtime, name))
    for name in ("__init__", "_materialize_depth_parameter"):
        monkeypatch.setattr(DepthTrajectory, name, getattr(DepthTrajectory, name))
    monkeypatch.setenv("THOG2_MATERIALISATION_PROFILING", "false")
    return runtime


@pytest.mark.parametrize("warmup", (0, 1))
@pytest.mark.parametrize("checkpoint", (0, 2))
def test_actual_trainer_dropout_is_matched_despite_different_ambient_rng(monkeypatch, warmup, checkpoint):
    # Exercise the real benchmark run and trainer on CPU; mock field memory meters.
    # The public benchmark still requires real CUDA for field measurements.
    options = field.parser().parse_args(["--device", "cpu", "--updates", "3", "--warmup", str(warmup), "--checkpoint", str(checkpoint)])
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(field, "memory", lambda runtime: {"processes": [], "total_pss_bytes": None})
    torch.manual_seed(11)
    reference = field.run(options, "off", 0)
    torch.manual_seed(999)
    candidate = field.run(options, "off", 0)
    rows = field.compare_run_results(reference, candidate, repeat=0, mode="off", atol=0, rtol=0)
    assert all(row["passed"] for row in rows), rows
    assert reference[0]["initial_conditions"]["training_seed"] == 373
    assert reference[0]["final_conditions"]["recorded_microbatches"] == (warmup + 3) * 2
    assert reference[2] and reference[0]["gradient_validation"]["gradient_tensors"] == len(reference[2])
    assert reference[0]["completed_updates"] == warmup + 3
    assert reference[0]["gradient_validation"]["completed_update"] == warmup + 4
    different_repeat = field.run(options, "off", 1)
    assert different_repeat[0]["initial_conditions"]["training_seed"] == 374
    assert different_repeat[0]["initial_conditions"]["model_state_sha256"] == reference[0]["initial_conditions"]["model_state_sha256"]
    assert different_repeat[0]["initial_conditions"]["cpu_rng_sha256"] != reference[0]["initial_conditions"]["cpu_rng_sha256"]
    assert different_repeat[0]["losses"] != reference[0]["losses"]


def fixture_result(options, mode, repeat):
    record = {"mode": mode, "repeat": repeat, "completed_updates": options.warmup + options.updates,
              "initial_conditions": {"training_seed": options.training_seed + repeat, "model_state_sha256": "model", "cpu_rng_sha256": "cpu", "cuda_rng_sha256": "cuda", "batch_rng_sha256": "data"},
              "final_conditions": {"cpu_rng_sha256": "final-cpu", "cuda_rng_sha256": "final-cuda", "batch_rng_sha256": "final-data", "batch_trace_sha256": "trace", "recorded_microbatches": 6},
              "losses": [1.0] * options.updates, "gradient_validation": {"loss": 1.0, "gradient_tensors": 1}, "milliseconds_per_update": 2.0, "tokens_per_second": 10.0, "gpu_peak_allocated_bytes": 64}
    return record, {"coefficient": torch.ones(2)}, {"coefficient": torch.ones(2)}, {"coefficient.step": torch.tensor(3.0)}


def mock_field_main(monkeypatch, tmp_path, *, smoke=True):
    output = tmp_path / "nested" / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", *( ["--smoke"] if smoke else ["--updates", "1", "--warmup", "0", "--repeats", "3"]), "--output", str(output)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU metadata")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setenv("THOG2_DEPTH_MATERIALISATION_MATMUL", os.environ.get("THOG2_DEPTH_MATERIALISATION_MATMUL", "false"))
    from sheet import depth_materialisation_runtime
    monkeypatch.setattr(depth_materialisation_runtime, "install_depth_materialisation_runtime", lambda: None)
    return output


def test_failed_parity_preserves_losses_provider_and_all_component_diagnostics(monkeypatch, tmp_path):
    output = mock_field_main(monkeypatch, tmp_path)
    def mismatched(options, mode, repeat):
        result = fixture_result(options, mode, repeat)
        if mode == "gpu":
            result[0]["losses"][0] += 0.01
            result[1]["coefficient"][0] += 0.1
            result[2].clear()
        return result
    monkeypatch.setattr(field, "run", mismatched)
    with pytest.raises(SystemExit, match="gpu/losses.*gpu/parameters.*gpu/gradients"):
        field.main()
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "parity_failed"
    assert len(evidence["runs"]) == 3 and "summary" not in evidence
    failed = {row["kind"]: row for row in evidence["parity"] if not row["passed"]}
    assert failed["losses"]["max_absolute_difference"] == pytest.approx(0.01)
    assert failed["losses"]["atol"] == 3e-5 and failed["losses"]["rtol"] == 3e-4
    assert failed["parameters"]["errors"][0]["tensor"] == "coefficient"
    assert failed["gradients"]["missing_tensors"] == ["coefficient"]
    assert all(row["passed"] for row in evidence["parity"] if row["mode"] == "cpu_and_gpu")
    assert not output.with_suffix(".json.tmp").exists()


def test_rng_mismatch_fails_even_when_numerical_results_match():
    options = field.parser().parse_args([])
    reference = fixture_result(options, "off", 0)
    candidate = copy.deepcopy(reference)
    candidate[0]["initial_conditions"]["cuda_rng_sha256"] = "different"
    candidate[0]["final_conditions"]["batch_rng_sha256"] = "different"
    rows = field.compare_run_results(reference, candidate, repeat=0, mode="gpu", atol=3e-5, rtol=3e-4)
    assert [row["kind"] for row in rows if not row["passed"]] == ["initial_conditions", "final_conditions"]


def test_preliminary_provider_rotation_and_success_evidence(monkeypatch, tmp_path):
    output = mock_field_main(monkeypatch, tmp_path, smoke=False)
    calls = []
    def matched(options, mode, repeat):
        calls.append((repeat, mode))
        return fixture_result(options, mode, repeat)
    monkeypatch.setattr(field, "run", matched)
    field.main()
    assert calls == [(0, "off"), (0, "gpu"), (0, "cpu_and_gpu"), (1, "gpu"), (1, "cpu_and_gpu"), (1, "off"), (2, "cpu_and_gpu"), (2, "off"), (2, "gpu")]
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "passed" and all(row["passed"] for row in evidence["parity"])
    assert len(evidence["parity"]) == 42
    assert set(evidence["summary"]) == {"off", "gpu", "cpu_and_gpu"}


def test_runtime_failure_preserves_completed_provider_evidence(monkeypatch, tmp_path):
    output = mock_field_main(monkeypatch, tmp_path)
    def failed(options, mode, repeat):
        if mode == "gpu":
            raise RuntimeError("test provider failure")
        return fixture_result(options, mode, repeat)
    monkeypatch.setattr(field, "run", failed)
    with pytest.raises(RuntimeError, match="test provider failure"):
        field.main()
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "runtime_failed"
    assert evidence["error"] == {"repeat": 0, "mode": "gpu", "type": "RuntimeError", "message": "test provider failure"}
    assert [row["mode"] for row in evidence["runs"]] == ["off"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA whole-trainer qualification")
@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_actual_cuda_whole_trainer_all_providers_match_with_dropout_and_rotated_order(monkeypatch, isolated_materializer, backend, dtype):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    options = field.parser().parse_args(["--backend", backend, "--dtype", dtype, "--threads", "1", "--warmup", "1", "--updates", "3"])
    monkeypatch.setenv("THOG2_DEPTH_MATERIALISATION_MATMUL", "true" if backend == "matmul" else "false")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    from sheet.depth_materialisation_runtime import install_depth_materialisation_runtime
    install_depth_materialisation_runtime()
    atol, rtol = (3e-5, 3e-4) if dtype == "float32" else (4e-4, 0.02) if dtype == "float16" else (3e-3, 0.06)
    failures = []
    for repeat, modes in enumerate((("off", "gpu", "cpu_and_gpu"), ("cpu_and_gpu", "off", "gpu"))):
        results = {mode: field.run(options, mode, repeat) for mode in modes}
        rows = [row for mode in ("gpu", "cpu_and_gpu") for row in field.compare_run_results(results["off"], results[mode], repeat=repeat, mode=mode, atol=atol, rtol=rtol)]
        failed = [row for row in rows if not row["passed"]]
        if failed:
            # A fresh ordinary run distinguishes provider drift from CUDA
            # variation already present without PREMAT. Diagnostic only: the
            # original provider failure still fails with unchanged tolerances.
            control = field.run(options, "off", repeat)
            control_rows = field.compare_run_results(results["off"], control, repeat=repeat, mode="off_repeat", atol=atol, rtol=rtol)
            control_failed = [row for row in control_rows if not row["passed"]]
            failures.append({"repeat": repeat, "failed_comparisons": failed,
                             "ordinary_control_passed": not control_failed, "ordinary_control_failed_comparisons": control_failed})
    if failures:
        pytest.fail(json.dumps({"backend": backend, "dtype": dtype, "failed_repeats": failures}, indent=2), pytrace=False)


def test_empty_gradient_comparison_cannot_pass():
    options = field.parser().parse_args([])
    reference = fixture_result(options, "off", 0)
    candidate = copy.deepcopy(reference)
    reference[2].clear()
    candidate[2].clear()
    rows = field.compare_run_results(reference, candidate, repeat=0, mode="cpu_and_gpu", atol=0, rtol=0)
    assert [row["kind"] for row in rows if not row["passed"]] == ["gradients"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA 30-update field regression")
@pytest.mark.parametrize("backend", ("matmul", "einsum"))
def test_actual_cuda_30_updates_with_warmup_match_all_providers(monkeypatch, isolated_materializer, backend):
    options = field.parser().parse_args(["--backend", backend, "--threads", "1", "--warmup", "5", "--updates", "30"])
    monkeypatch.setenv("THOG2_DEPTH_MATERIALISATION_MATMUL", "true" if backend == "matmul" else "false")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    isolated_materializer.install_depth_materialisation_runtime()
    for repeat, modes in enumerate((("off", "gpu", "cpu_and_gpu"), ("gpu", "cpu_and_gpu", "off"), ("cpu_and_gpu", "off", "gpu"))):
        results = {mode: field.run(options, mode, repeat) for mode in modes}
        rows = [row for mode in ("gpu", "cpu_and_gpu") for row in field.compare_run_results(results["off"], results[mode], repeat=repeat, mode=mode, atol=3e-5, rtol=3e-4)]
        failed = [row for row in rows if not row["passed"]]
        if failed:
            pytest.fail(json.dumps({"backend": backend, "dtype": "float32", "repeat": repeat, "failed_comparisons": failed}, indent=2), pytrace=False)
# ^^^ THOG
