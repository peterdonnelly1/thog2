# vvv THOG dynamic pre-materialisation policy, configuration, and topology regression tests
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from run_thog2_owt_core import build_parser
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.run_config import OwtRunConfig
from sheet.training_model import TrainingSheetGPT
from sheet.premat import (
    CandidateEnvelope,
    CandidateState,
    PrematMemoryObservation,
    PrematRuntime,
    decide_candidate_admission,
    plastic_memory_budget_gib,
    validate_premat_configuration,
)


class _FakeTensor:
    def __init__(self, *, shape=(2, 4, 8), element_bytes: int = 2, numel: int | None = None) -> None:
        self.device = torch.device("cuda")
        self.shape = shape
        self.ndim = len(shape)
        self._element_bytes = element_bytes
        self._numel = numel if numel is not None else int(torch.tensor(shape).prod().item())
        self.recorded_streams = []

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_bytes

    def record_stream(self, stream) -> None:
        self.recorded_streams.append(stream)


class _FakeStream:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.waited_events = []
        self.waited_streams = []

    def wait_event(self, event) -> None:
        self.waited_events.append(event)
        event.complete = True

    def wait_stream(self, stream) -> None:
        self.waited_streams.append(stream)


class _FakeEvent:
    def __init__(self, cuda, *, enable_timing: bool) -> None:
        self.cuda = cuda
        self.enable_timing = enable_timing
        self.complete = False

    def record(self, stream) -> None:
        self.complete = stream.kind == "main" or self.cuda.complete_premat_on_record

    def query(self) -> bool:
        return self.complete

    def elapsed_time(self, _other) -> float:
        return 0.25


class _FakeCuda:
    def __init__(self) -> None:
        self.main_stream = _FakeStream("main")
        self.premat_stream = _FakeStream("premat")
        self.complete_premat_on_record = False
        self.stream_creations = 0
        self.allocated = 100
        self.reserved = 120
        self.free = 10_000_000
        self.total = 10_000_100

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: self.allocated)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: self.reserved)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (self.free, self.total))
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: self.main_stream)
        monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

        def create_stream(*, device=None):
            self.stream_creations += 1
            return self.premat_stream

        monkeypatch.setattr(torch.cuda, "Stream", create_stream)
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda *, enable_timing=False: _FakeEvent(
                self,
                enable_timing=enable_timing,
            ),
        )


def _runtime(
    monkeypatch,
    *,
    stay_below_current_peak: bool,
    attention_mode: str = "fused",
):
    fake_cuda = _FakeCuda()
    fake_cuda.install(monkeypatch)
    calls = []

    def materialize(family: str, layer_index: int):
        calls.append((family, layer_index))
        return _FakeTensor(numel=64)

    runtime = PrematRuntime(
        materialize=materialize,
        n_embd=8,
        n_head=2,
        attention_mode=attention_mode,
        stay_below_current_peak=stay_below_current_peak,
        gpu_memory_buffer_gb=0.0,
        logging_enabled=True,
    )
    runtime.begin((3, 5, 7), reference=_FakeTensor())
    return runtime, fake_cuda, calls


def _observation(*, allocated: int = 100, reserved: int = 120, peak: int = 200) -> PrematMemoryObservation:
    return PrematMemoryObservation(
        process_allocated_bytes=allocated,
        process_reserved_bytes=reserved,
        process_ordinary_peak_bytes=peak,
        device_free_bytes=500,
        device_total_bytes=1000,
    )


def test_peak_free_admission_enforces_process_and_device_guards_separately() -> None:
    admitted = decide_candidate_admission(
        observation=_observation(),
        envelope=CandidateEnvelope(30, 50, 20),
        stay_below_current_peak=True,
        gpu_memory_buffer_bytes=400,
    )
    assert admitted.admitted
    assert admitted.process_guard_passed
    assert admitted.device_guard_passed

    process_blocked = decide_candidate_admission(
        observation=_observation(peak=130),
        envelope=CandidateEnvelope(30, 50, 20),
        stay_below_current_peak=True,
        gpu_memory_buffer_bytes=0,
    )
    assert not process_blocked.admitted
    assert process_blocked.reason == "ordinary_process_peak"
    assert process_blocked.device_guard_passed


def test_live_reporter_publishes_pass_boundaries_and_total_event_count(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    snapshots = []
    runtime.set_live_reporter(snapshots.append)
    runtime.end()

    assert snapshots
    latest = snapshots[-1]
    assert latest["latest_event_sequence"] == latest["event_count"]
    assert latest["event_count"] >= len(latest["events"])
    assert latest["events"][-1]["event"] == "pass_end"
    assert latest["event_window_limit"] == 256


def test_global_admission_does_not_assume_fragmented_allocator_cache_is_reusable() -> None:
    decision = decide_candidate_admission(
        observation=_observation(allocated=100, reserved=180),
        envelope=CandidateEnvelope(50, 75, 0),
        stay_below_current_peak=False,
        gpu_memory_buffer_bytes=480,
    )
    assert decision.predicted_device_used_bytes == 575
    assert not decision.admitted
    assert decision.reason == "global_device_buffer"


def test_premat_headroom_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_premat_configuration(
            premat="enabled",
            attention_mode="fused",
            stay_below_current_peak=True,
            stay_within_global_buffer=True,
            gpu_memory_buffer_gb=1.0,
            logging="disabled",
            instra="disabled",
        )


def test_retired_plastic_memory_budget_cli_names_replacement(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--plastic__layer_count__memory_budget_gib", "12"])
    assert "--premat_gpu_memory_buffer_gb" in capsys.readouterr().err


def test_public_cli_exposes_exactly_the_seven_premat_options() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--premat")
    }
    assert option_strings == {
        "--premat",
        "--premat_attention_mode",
        "--premat_headroom_stay_below_current_peak",
        "--premat_headroom_stay_within_global_buffer",
        "--premat_gpu_memory_buffer_gb",
        "--premat_logging",
        "--premat_instra",
    }


def test_underscore_alias_normaliser_preserves_exact_premat_names() -> None:
    import sitecustomize

    assert sitecustomize._normalise_long_option(  # noqa: SLF001 - public CLI compatibility contract
        "--premat_gpu_memory_buffer_gb=4"
    ) == "--premat_gpu_memory_buffer_gb=4"


def test_enabled_default_headroom_and_canonical_provenance_are_resolved(tmp_path) -> None:
    run_config = OwtRunConfig(model_type="sheet", premat="enabled", device="cuda")
    training_config = run_config.to_training_config(
        vocab_size=32,
        world_size=1,
        out_dir=tmp_path,
    )
    assert training_config.premat_headroom_stay_below_current_peak is True
    canonical = run_config.canonical_dict(world_size=1)
    assert canonical["premat_resolved_headroom_mode"] == "stay_below_current_peak"
    assert canonical["premat_effective_fast_discard"] is True
    assert canonical["premat_schema_version"] == 2
    assert canonical["premat_lookahead_layer_limit"] == 1


def test_unfused_attention_matches_fused_math_on_cpu() -> None:
    common = dict(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=1,
        geometry_preset="depth",
        basis_family="chebyshev",
        direct_factorised_mlp=False,
    )
    torch.manual_seed(7)
    fused = SheetGPT(SheetGPTConfig(**common, premat_attention_mode="fused"))
    unfused = SheetGPT(SheetGPTConfig(**common, premat_attention_mode="unfused"))
    unfused.load_state_dict(fused.state_dict())
    fused.eval()
    unfused.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    fused_logits, _ = fused(tokens)
    unfused_logits, _ = unfused(tokens)
    torch.testing.assert_close(unfused_logits, fused_logits, rtol=1.0e-5, atol=1.0e-6)


def test_deadline_unavailable_is_claimed_by_main_without_late_premat_launch(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    assert calls == []
    weight = runtime.acquire("QKV", 3)
    assert calls == [("QKV", 3)]
    assert weight.recorded_streams
    candidate = runtime._candidates[(3, "QKV")]
    assert candidate.owner == "main"
    assert candidate.state == CandidateState.CONSUMING
    report = runtime.report()
    assert report["aggregate"]["ordinary_deadline_materialisations"] == 1
    assert any(
        event.get("decision") == "main_claim"
        and event.get("reason") == "unavailable_at_deadline"
        for event in report["events"]
    )


def test_materialising_deadline_waits_once_and_never_duplicates(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    assert calls == [("QKV", 3)]
    assert fake_cuda.stream_creations == 1
    weight = runtime.acquire("QKV", 3)
    assert calls == [("QKV", 3)]
    assert weight.recorded_streams[-1] is fake_cuda.main_stream
    assert len(fake_cuda.main_stream.waited_events) == 1
    candidate = runtime._candidates[(3, "QKV")]
    assert candidate.owner == "premat"
    assert candidate.state == CandidateState.CONSUMING
    assert candidate.critical_path_miss
    report = runtime.report()
    assert report["aggregate"]["waited_hits"] == 1
    assert report["aggregate"]["main_stream_wait_ms_total"] == pytest.approx(0.25)


def test_strict_head_defer_does_not_bypass_and_window_never_contains_l_plus_2(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    assert calls == []
    report = runtime.report()
    assert report["queue_head"]["family"] == "QKV"
    assert report["queue_head"]["deferred"] is True
    assert {candidate["layer_index"] for candidate in report["candidates"]} == {3, 5}
    assert all(candidate["owner"] == "none" for candidate in report["candidates"])


def test_lifecycle_rejects_illegal_transition(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    candidate = runtime._candidates[(3, "QKV")]
    with pytest.raises(RuntimeError, match="UNAVAILABLE->CONSUMED"):
        runtime._transition(candidate, CandidateState.CONSUMED, "invalid")


def test_unfused_envelope_includes_score_probability_and_mask_tensors(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
        attention_mode="unfused",
    )
    runtime.layer_start(3)
    qk = runtime._candidates[(3, "QK")]
    activation_bytes = 2 * 4 * 8 * 2
    score_bytes = 2 * 2 * 4 * 4 * 4
    mask_bytes = 4 * 4
    assert qk.envelope.foreground_overlap_bytes >= (
        4 * activation_bytes + 2 * score_bytes + mask_bytes
    )


def test_plastic_budget_accounts_for_other_gpu_users(monkeypatch) -> None:
    fake_cuda = _FakeCuda()
    fake_cuda.free = 3 * 1024 ** 3
    fake_cuda.reserved = 2 * 1024 ** 3
    fake_cuda.total = 16 * 1024 ** 3
    fake_cuda.install(monkeypatch)
    budget = plastic_memory_budget_gib(
        device=torch.device("cuda"),
        gpu_memory_buffer_gb=1.0,
    )
    assert budget == pytest.approx(4.0)


class _CpuCheckpointPrematRuntime:
    """CPU lifecycle double for checkpoint boundary and gradient regression coverage."""

    def __init__(self, model: TrainingSheetGPT) -> None:
        self.model = model
        self.active = False
        self.passes = []

    def begin(self, layer_indices, *, reference) -> None:
        assert not self.active
        self.active = True
        self.passes.append(
            {
                "layers": tuple(layer_indices),
                "grad_enabled": torch.is_grad_enabled(),
                "ended": False,
            }
        )

    def end(self) -> None:
        assert self.active
        self.passes[-1]["ended"] = True
        self.active = False

    def layer_start(self, _layer_index: int) -> None:
        assert self.active

    def event(self, _name: str, *, layer_index: int) -> None:
        assert self.active

    def acquire(self, family: str, layer_index: int):
        assert self.active
        return self.model._premat_materialize_candidate(family, layer_index)

    def consumed(self, _family: str, _layer_index: int) -> None:
        assert self.active


def test_checkpointed_premat_uses_fresh_reentrant_segment_passes_on_cpu() -> None:
    common = dict(
        block_size=4,
        vocab_size=16,
        n_layer=4,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=1,
        geometry_preset="depth",
        basis_family="chebyshev",
        direct_factorised_mlp=False,
        fast_discard=True,
        premat="disabled",
    )
    torch.manual_seed(23)
    reference = TrainingSheetGPT(SheetGPTConfig(**common))
    checkpointed = TrainingSheetGPT(SheetGPTConfig(**common))
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.set_checkpoint_segment_size(2)
    fake_runtime = _CpuCheckpointPrematRuntime(checkpointed)
    checkpointed._premat_runtime = fake_runtime
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    def run(model):
        model.zero_grad(set_to_none=True)
        logits, loss = model(tokens, targets)
        assert loss is not None
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return logits.detach(), loss.detach(), gradients

    reference_result = run(reference)
    checkpointed_result = run(checkpointed)
    torch.testing.assert_close(checkpointed_result[0], reference_result[0])
    torch.testing.assert_close(checkpointed_result[1], reference_result[1])
    assert checkpointed_result[2].keys() == reference_result[2].keys()
    for name in checkpointed_result[2]:
        torch.testing.assert_close(
            checkpointed_result[2][name],
            reference_result[2][name],
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    assert checkpointed.last_execution_report.segment_size == 1
    assert [entry["layers"] for entry in fake_runtime.passes] == [
        (0, 1, 2, 3),
        (3,),
        (2,),
        (1,),
        (0,),
    ]
    assert [entry["grad_enabled"] for entry in fake_runtime.passes] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert all(entry["ended"] for entry in fake_runtime.passes)
    assert not fake_runtime.active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance host required")
@pytest.mark.parametrize("attention_mode", ("fused", "unfused"))
@pytest.mark.parametrize("checkpoint_segment_size", (0, 2))
def test_cuda_premat_forward_backward_matches_same_mode_disabled(
    attention_mode: str,
    checkpoint_segment_size: int,
) -> None:
    common = dict(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=3,
        base_row_order=8,
        geometry_preset="depth",
        basis_family="chebyshev",
        direct_factorised_mlp=False,
        premat_attention_mode=attention_mode,
    )
    torch.manual_seed(17)
    disabled = TrainingSheetGPT(
        SheetGPTConfig(**common, premat="disabled")
    ).cuda()
    enabled = TrainingSheetGPT(
        SheetGPTConfig(
            **common,
            premat="enabled",
            premat_headroom_stay_within_global_buffer=True,
            premat_gpu_memory_buffer_gb=0.0,
            premat_logging="enabled",
        )
    ).cuda()
    enabled.load_state_dict(disabled.state_dict())
    disabled.set_checkpoint_segment_size(checkpoint_segment_size)
    enabled.set_checkpoint_segment_size(checkpoint_segment_size)
    tokens = torch.arange(16, device="cuda", dtype=torch.long).view(2, 8) % 32
    targets = torch.roll(tokens, shifts=-1, dims=1)

    def run(model):
        model.zero_grad(set_to_none=True)
        logits, loss = model(tokens, targets)
        assert loss is not None
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("trajectory.") and parameter.grad is not None
        }
        return logits.detach(), loss.detach(), gradients

    disabled_logits, disabled_loss, disabled_gradients = run(disabled)
    enabled_logits, enabled_loss, enabled_gradients = run(enabled)
    torch.testing.assert_close(enabled_logits, disabled_logits, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(enabled_loss, disabled_loss, rtol=1.0e-6, atol=1.0e-7)
    assert enabled_gradients.keys() == disabled_gradients.keys()
    for name in enabled_gradients:
        torch.testing.assert_close(
            enabled_gradients[name],
            disabled_gradients[name],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    report = enabled.premat_report()
    assert report is not None
    assert report["aggregate"]["admitted"] > 0
# ^^^ THOG
