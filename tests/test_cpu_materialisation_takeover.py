# vvv THOG regression coverage for shared binding, qualified predictions, censored evidence and CPU host identity
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import instra_node_agent as node_agent
from instra_duration_estimator import estimate
from sheet.depth_numerical_policy import CpuDepthBinding, PrematBindingContext, materialize_cpu
from sheet.premat_cpu import GemmPredictor
from sheet.processing_cpu_evidence import cpu_task_rows, extend_processing, update_timing_overlay
from tests.test_cpu_materialisation import policy
from tests.test_cpu_materialisation_lifecycle import setup_runtime, wait_until
from tests.test_instra_duration_workloads import run


@pytest.mark.parametrize("backend", ("matmul", "einsum"))
@pytest.mark.parametrize("learn_coefficients,learn_depth", ((True, False), (False, True), (True, True)))
def test_shared_premat_node_preserves_fused_dependencies_and_fresh_graphs(backend, learn_coefficients, learn_depth):
    torch.manual_seed(417)
    coefficients = [torch.nn.Parameter(torch.randn(4, 4, 3), requires_grad=learn_coefficients) for _ in range(3)]
    row = torch.nn.Parameter(torch.randn(3), requires_grad=learn_depth)
    cached = torch.cat([materialize_cpu(value, row, policy(backend)) for value in coefficients])
    reference_coefficients = [value.detach().clone().requires_grad_(learn_coefficients) for value in coefficients]
    reference_row = row.detach().clone().requires_grad_(learn_depth)
    for _ in range(2):
        pairs = tuple(item for coefficient in coefficients for item in (coefficient, row))
        result = CpuDepthBinding.apply(cached, PrematBindingContext(policy(backend)), *pairs)
        expected = torch.cat([value.reshape(-1, 3).matmul(reference_row).reshape(4, 4) if backend == "matmul" else torch.einsum("p,rcp->rc", reference_row, value) for value in reference_coefficients])
        torch.testing.assert_close(result, expected)
        gradient = torch.randn_like(result)
        result.backward(gradient)
        expected.backward(gradient)
    if learn_coefficients:
        for left, right in zip(coefficients, reference_coefficients):
            torch.testing.assert_close(left.grad, right.grad)
    if learn_depth:
        torch.testing.assert_close(row.grad, reference_row.grad)


def test_predictions_require_matching_upload_samples_and_keep_contexts_bounded():
    predictor = GemmPredictor(limit=2)
    contexts = [("original_forward", layer, "O", 2, 16, "matmul") for layer in range(3)]
    for context in contexts:
        for _ in range(3):
            predictor.observe(context, 20, 0.1)
    context = contexts[-1]
    predictor.observe_upload(contexts[-2], 10)
    missing = predictor.predict(context, 100_000_000, 1, 0.1)
    assert not missing["prediction_available"] and missing["prediction_fallback_reason"] == "upload_duration_unavailable"
    predictor.observe_upload(context, 2.1)
    prediction = predictor.predict(context, 100_000_000, 1, 0.1)
    assert prediction["intended_upload_submission_ns"] == 116_900_000
    assert prediction["estimated_upload_ms"] == 2.1
    predictor.observe_upload(contexts[0], 1)
    assert len(predictor.samples) == len(predictor.uploads) == 2
    assert not predictor.predict(context, 100_000_000, 1, float("nan"))["prediction_available"]
    predictor.reset("backend_changed")
    assert not predictor.samples and not predictor.uploads


def test_delayed_leading_edge_notifications_keep_each_copy_gate(monkeypatch):
    runtime, cuda, _, _ = setup_runtime(monkeypatch, transfer="previous_gemm_leading_edge")
    try:
        runtime.layer_start(0)
        first_gate, second_gate = torch.cuda.Event(), torch.cuda.Event()
        runtime.leading_opportunities.update({(0, "O"): first_gate, (1, "O"): second_gate})
        runtime._wake()
        wait_until(lambda: len(runtime.cpu_uploads) == 2 and all(item.completion is not None for item in runtime.cpu_uploads.values()))
        assert cuda.premat_stream.waited_events == [first_gate, second_gate]
        for upload in runtime.cpu_uploads.values():
            upload.completion.complete = True
    finally:
        runtime.end()
        runtime.close()


def data():
    return {"metadata": {"capture": {"optimizer_update": 1, "host_nvtx_lower_ns": 10_000, "host_nvtx_upper_ns": 10_000}, "run": {}, "warnings": [], "files": {}}, "intervals": [], "stream_resources": []}


def test_running_cpu_task_is_censored_and_never_reported_as_zero_service(tmp_path):
    event = {"event": "cpu_task_start", "cpu_matrix_id": "matrix", "cpu_task_id": "batch", "snapshot_id": "snapshot", "optimizer_step": 1, "start_ns": 12_000}
    rows = cpu_task_rows([event], data()["metadata"]["capture"], 1000, 11_000)
    assert rows[0]["clock_alignment_known"] and rows[0]["right_censored"]
    assert rows[0]["start_us"] == 2 and rows[0]["end_us"] == 10
    assert rows[0]["cpu_service_us"] is None and not rows[0]["completed"]
    payload = data()
    extend_processing(payload, tmp_path, "run_", {"cpu_lifecycle": [event]}, [], True, 1000, 11_000)
    summary = payload["cpu_summary"]
    assert summary["cpu_service_ms"] is None and summary["unique_cpu_matrices"] == 0
    assert summary["unique_cpu_jobs"] == summary["incomplete_cpu_matrices"] == 1
    assert payload["cpu_tasks"][0].get("end_ns") is None


def test_absent_cpu_evidence_keeps_all_numeric_lifecycle_totals_unknown(tmp_path):
    payload = data()
    extend_processing(payload, tmp_path, "run_", None, [], False, 1000, 11_000)
    for field in ("eligible_matrix_uses", "full_hits", "complete_misses", "unique_cpu_matrices", "cpu_service_ms", "late_upload_count", "late_upload_bytes", "cpu_cache_reuse_count", "prediction_qualified_uses", "prediction_unavailable_uses", "lifecycle_dropped_events"):
        assert payload["cpu_summary"][field] is None, field


def test_prediction_counts_use_only_eligible_matrix_uses(tmp_path):
    events = [
        {"event": "matrix_use", "optimizer_step": 1, "targeted": True, "prediction_available": True, "final_outcome": "FULL HIT"},
        {"event": "matrix_use", "optimizer_step": 1, "targeted": True, "prediction_available": False, "final_outcome": "COMPLETE MISS"},
        {"event": "matrix_use", "optimizer_step": 1, "targeted": False, "prediction_available": False, "final_outcome": "NOT TARGETED"},
    ]
    payload = data()
    extend_processing(payload, tmp_path, "run_", {"cpu_lifecycle": events}, [], True, 1000, 11_000)
    summary = payload["cpu_summary"]
    assert summary["eligible_matrix_uses"] == 2 and summary["not_targeted_uses"] == 1
    assert summary["prediction_qualified_uses"] == summary["prediction_unavailable_uses"] == 1


def test_full_update_keeps_shared_snapshot_precursors_and_censors_running_tasks():
    events = [
        {"event": "cpu_matrix_ready", "optimizer_step": 1, "snapshot_id": "shared", "cpu_matrix_id": "ready", "start_ns": 7000, "end_ns": 9000},
        {"event": "cpu_task_start", "optimizer_step": 1, "snapshot_id": "shared", "cpu_matrix_id": "running", "start_ns": 8000},
        {"event": "cpu_matrix_ready", "optimizer_step": 1, "snapshot_id": "unused", "cpu_matrix_id": "unrelated", "start_ns": 7000, "end_ns": 9000},
        {"event": "matrix_use", "optimizer_step": 2, "snapshot_id": "shared", "cpu_matrix_id": "ready"},
        {"event": "matrix_use", "optimizer_step": 2, "snapshot_id": "shared", "cpu_matrix_id": "running"},
    ]
    runtime = SimpleNamespace(report=lambda: {"cpu_lifecycle": events})
    overlay = update_timing_overlay(runtime, 2, 10_000, 20_000)["cpu_overlay"]
    assert not overlay["additive"] and len(overlay["lifecycle"]) == 4
    ready, running = overlay["tasks"]
    assert ready["precursor"] and ready["completed"] and ready["host_end_ms"] == -0.001
    assert running["right_censored"] and not running["completed"]
    assert running.get("end_ns") is None and running["host_end_ms"] == 0.01


def test_cpu_duration_histories_require_the_same_host_and_resolved_thread_budget():
    completed = run(seconds=120, **{"--premat_materialisation_device": "cpu_and_gpu", "--premat_cpu_threads_per_worker": 0})
    completed["cpu_execution"] = {"affinity_thread_budget": 7, "affinity_cores": list(range(8))}
    candidate = deepcopy(completed)
    candidate.pop("duration_seconds")
    assert estimate([candidate], [{"runs": [completed]}])["seconds"] == 120
    candidate["host_id"] = "other_host"
    assert estimate([candidate], [{"runs": [completed]}])["seconds"] is None
    candidate["host_id"] = completed["host_id"]
    candidate["cpu_execution"]["affinity_thread_budget"] = 3
    assert estimate([candidate], [{"runs": [completed]}])["seconds"] is None
    candidate = run()
    candidate["host_id"] = "other_host"
    assert estimate([candidate], [{"runs": [run(seconds=120)]}])["seconds"] == 120


def test_node_agent_discloses_affinity_without_importing_the_training_runtime(monkeypatch):
    monkeypatch.setattr(node_agent.os, "sched_getaffinity", lambda process_id: {2, 4, 5, 8})
    monkeypatch.setattr(node_agent, "_gpu_information", lambda: [])
    result = node_agent._runner_operation({}, "runner_reconcile", {})
    assert result["protocol"] == 3 and result["attempts"] == {}
    assert result["cpu_execution"] == {"affinity_cores": [2, 4, 5, 8], "affinity_thread_budget": 3}
# ^^^ THOG
