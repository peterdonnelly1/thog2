# vvv THOG dynamic pre-materialisation policy, configuration, and topology regression tests
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import warnings

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
    PREMAT_HIGHEST_CUDA_STREAM_PRIORITY_REQUEST,
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
        self.complete = (
            (stream.kind == "main" and self.cuda.complete_main_on_record)
            or self.cuda.complete_premat_on_record
        )

    def query(self) -> bool:
        return self.complete

    def elapsed_time(self, _other) -> float:
        return 0.25


class _FakeCuda:
    def __init__(self) -> None:
        self.main_stream = _FakeStream("main")
        self.premat_stream = _FakeStream("premat")
        self.complete_premat_on_record = False
        self.complete_main_on_record = True
        self.stream_creations = 0
        self.stream_priorities = []
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

        def create_stream(*, device=None, priority=0):
            self.stream_creations += 1
            self.stream_priorities.append(priority)
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
    target_layer: int = 1,
    weight_matrix_target_order: str = "r_to_l",
    cuda_stream_priority: str = "normal",
    diagnostic_layer_delay_ms: float = 0.0,
    reference_element_bytes: int = 2,
    attach=None,
):
    fake_cuda = _FakeCuda()
    fake_cuda.install(monkeypatch)
    calls = []

    def materialize(family: str, layer_index: int):
        calls.append((family, layer_index))
        return _FakeTensor(numel=64)

    runtime = PrematRuntime(
        materialize=materialize,
        attach=attach,
        n_embd=8,
        n_head=2,
        attention_mode=attention_mode,
        target_layer=target_layer,
        weight_matrix_target_order=weight_matrix_target_order,
        stay_below_current_peak=stay_below_current_peak,
        gpu_memory_buffer_gb=0.0,
        cuda_stream_priority=cuda_stream_priority,
        diagnostic_layer_delay_ms=diagnostic_layer_delay_ms,
        logging_enabled=True,
    )
    runtime.begin(
        (3, 5, 7),
        reference=_FakeTensor(element_bytes=reference_element_bytes),
    )
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


def test_live_reporter_publishes_only_the_completed_microstep(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    snapshots = []
    runtime.set_live_reporter(snapshots.append)
    runtime.end()

    assert len(snapshots) == 1
    latest = snapshots[0]
    assert latest["pass_complete"] is True
    assert latest["pass_sequence"] == 1
    assert latest["latest_event_sequence"] == latest["event_count"]
    assert latest["event_count"] == len(latest["events"])
    assert latest["events"][-1]["event"] == "pass_end"
    assert latest["event_window_limit"] == "complete_pass"
    assert {event["pass_sequence"] for event in latest["events"]} == {1}


def test_live_reporter_skips_uncaptured_updates(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    snapshots = []
    runtime.set_live_reporter(snapshots.append, lambda _pass_sequence: False)
    runtime.end()
    assert snapshots == []


def test_live_reporter_never_combines_adjacent_microsteps(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    snapshots = []
    runtime.set_live_reporter(snapshots.append)
    runtime.end()
    runtime.begin((3, 5, 7), reference=_FakeTensor())
    runtime.end()

    assert [snapshot["pass_sequence"] for snapshot in snapshots] == [1, 2]
    assert {event["pass_sequence"] for event in snapshots[0]["events"]} == {1}
    assert {event["pass_sequence"] for event in snapshots[1]["events"]} == {2}


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
            target_layer=1,
            weight_matrix_target_order="r_to_l",
            stay_below_current_peak=True,
            stay_within_global_buffer=True,
            gpu_memory_buffer_gb=1.0,
            cuda_stream_priority="normal",
            diagnostic_layer_delay_ms=0.0,
            logging="disabled",
            instra="disabled",
        )


def test_retired_plastic_memory_budget_cli_names_replacement(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--plastic__layer_count__memory_budget_gib", "12"])
    assert "--premat_gpu_memory_buffer_gb" in capsys.readouterr().err


def test_public_cli_exposes_exactly_the_eleven_premat_options() -> None:
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
        "--premat_target_layer",
        "--premat_weight_matrix_target_order",
        "--premat_headroom_stay_below_current_peak",
        "--premat_headroom_stay_within_global_buffer",
        "--premat_gpu_memory_buffer_gb",
        "--premat_cuda_stream_priority",
        "--premat_diagnostic_layer_delay_ms",
        "--premat_logging",
        "--premat_instra",
    }


def test_wrapper_reclaims_unused_allocator_cache_for_premat_by_default() -> None:
    wrapper = Path("train_OWT_core.sh").read_text(encoding="utf-8")
    assert (
        'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,'
        'garbage_collection_threshold:0.8"'
    ) in wrapper


def test_premat_cuda_stream_priority_is_opt_in(monkeypatch) -> None:
    normal_runtime, normal_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    assert normal_cuda.stream_priorities == [0]
    assert normal_runtime.report()["cuda_stream_priority"] == "normal"

    high_runtime, high_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        cuda_stream_priority="high",
    )
    assert high_cuda.stream_priorities == [PREMAT_HIGHEST_CUDA_STREAM_PRIORITY_REQUEST]
    assert high_runtime.report()["cuda_stream_priority"] == "high"


def test_diagnostic_layer_delay_retains_only_the_fixed_reconsideration_loop(monkeypatch) -> None:
    clock_ns = [0]

    def perf_counter_ns() -> int:
        clock_ns[0] += 100_000
        return clock_ns[0]

    monkeypatch.setattr("sheet.premat.time.perf_counter_ns", perf_counter_ns)
    monkeypatch.setattr("sheet.premat.time.sleep", lambda _seconds: None)
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        diagnostic_layer_delay_ms=1.0,
    )
    runtime.layer_start(3)
    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    runtime.layer_complete(3)

    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    report = runtime.report()
    assert report["diagnostic_layer_delay_ms"] == 1.0
    assert report["aggregate"]["diagnostic_layer_delay_count"] == 1
    assert report["aggregate"]["diagnostic_layer_delay_ms_requested_total"] == 1.0
    assert report["aggregate"]["diagnostic_layer_delay_ms_actual_total"] >= 1.0
    assert any(
        event.get("event") == "advance_begin"
        and event.get("detail", {}).get("trigger") == "diagnostic_layer_delay_poll"
        for event in report["events"]
    )


def test_negative_diagnostic_layer_delay_is_rejected() -> None:
    with pytest.raises(ValueError, match="diagnostic_layer_delay"):
        SheetGPTConfig(premat_diagnostic_layer_delay_ms=-0.1)


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
    assert training_config.premat_cuda_stream_priority == "normal"
    assert training_config.premat_target_layer == 1
    assert training_config.premat_weight_matrix_target_order == "r_to_l"
    assert training_config.premat_diagnostic_layer_delay_ms == 0.0
    canonical = run_config.canonical_dict(world_size=1)
    assert canonical["premat_resolved_headroom_mode"] == "stay_below_current_peak"
    assert canonical["premat_effective_fast_discard"] is True
    assert canonical["premat_schema_version"] == 3
    assert canonical["premat_lookahead_layer_limit"] == 1
    assert canonical["premat_target_scope"] == "relative_layer_1"
    assert canonical["premat_target_order"] == "r_to_l"
    assert canonical["premat_cuda_stream_priority"] == "normal"
    assert canonical["premat_diagnostic_layer_delay_ms"] == 0.0


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


def test_advance_queues_every_admissible_fused_candidate_without_completion_gate(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    assert all(
        runtime._candidates[(5, family)].state == CandidateState.MATERIALISING
        for family in ("QKV", "O", "UP", "DOWN")
    )
    assert all(
        runtime._candidates[(3, family)].owner == "none"
        for family in ("QKV", "O", "UP", "DOWN")
    )
    assert all(
        runtime._candidates[(5, family)].owner == "premat"
        for family in ("QKV", "O", "UP", "DOWN")
    )

    report = runtime.report()
    assert report["target_order"] == "r_to_l"
    assert report["aggregate"]["queue_depth_high_water"] == 4
    # All four retained outputs and QKV's intrinsic concatenation workspace
    # are charged.  The shared foreground safety envelope is applied by each
    # admission decision, but is not multiplied once per queued matrix.
    assert report["memory"]["premat_cumulative_charged_bytes"] == 1_920
    assert any(
        event.get("event") == "advance_return"
        and event.get("detail", {}).get("submitted_count") == 4
        and event.get("detail", {}).get("return_reason") == "target_exhausted"
        for event in report["events"]
    )


def test_event_telemetry_exposes_raw_and_charged_allocator_memory(monkeypatch) -> None:
    runtime, fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    report = runtime.report()

    assert report["memory"]["reusable_allocator_bytes"] == 20
    assert report["memory"]["device_free_minus_buffer_bytes"] == fake_cuda.free
    considered = next(
        event
        for event in report["events"]
        if event.get("event") == "admission_considered"
    )
    assert considered["reusable_allocator_bytes"] == 20
    assert considered["device_free_minus_buffer_bytes"] == fake_cuda.free
    assert considered["detail"]["raw_memory"] == {
        "process_allocated_bytes": 100,
        "process_reserved_bytes": 120,
        "process_ordinary_peak_bytes": 100,
        "device_free_bytes": fake_cuda.free,
        "device_total_bytes": fake_cuda.total,
        "reusable_allocator_bytes": 20,
        "device_used_bytes": 100,
        "device_free_minus_buffer_bytes": fake_cuda.free,
    }
    charged = considered["detail"]["charged_memory"]
    assert charged["process_allocated_bytes"] == 100
    assert charged["process_reserved_bytes"] == 120
    assert charged["reusable_allocator_bytes"] == 20
    assert charged["device_free_minus_buffer_bytes"] == fake_cuda.free


def test_premat_submission_does_not_publish_auxiliary_stream_autocast_casts(monkeypatch) -> None:
    cache_enabled = [True]
    cache_transitions = []

    monkeypatch.setattr(
        torch,
        "is_autocast_cache_enabled",
        lambda: cache_enabled[0],
    )

    def set_cache_enabled(enabled: bool) -> None:
        cache_enabled[0] = bool(enabled)
        cache_transitions.append(bool(enabled))

    monkeypatch.setattr(torch, "set_autocast_cache_enabled", set_cache_enabled)
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)

    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    assert cache_transitions == [False, True] * 4
    assert cache_enabled[0] is True


@pytest.mark.parametrize(
    ("attention_mode", "target_order", "expected_families"),
    (
        ("fused", "l_to_r", ("QKV", "O", "UP", "DOWN")),
        ("fused", "r_to_l", ("DOWN", "UP", "O", "QKV")),
        ("unfused", "l_to_r", ("QK", "V", "O", "UP", "DOWN")),
        ("unfused", "r_to_l", ("DOWN", "UP", "O", "V", "QK")),
    ),
)
def test_fused_and_unfused_matrix_target_orders(
    monkeypatch,
    attention_mode,
    target_order,
    expected_families,
) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        attention_mode=attention_mode,
        weight_matrix_target_order=target_order,
    )
    runtime.layer_start(3)
    assert calls == [(family, 5) for family in expected_families]


@pytest.mark.parametrize(
    ("target_layer", "expected_layer"),
    ((0, 3), (1, 5), (2, 7)),
)
def test_exact_relative_target_layer_offsets(monkeypatch, target_layer, expected_layer) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        target_layer=target_layer,
    )
    runtime.layer_start(3)
    assert calls == [
        ("DOWN", expected_layer),
        ("UP", expected_layer),
        ("O", expected_layer),
        ("QKV", expected_layer),
    ]
    assert runtime.report()["target_offset"] == target_layer


def test_target_layer_boundary_is_safe_and_reported(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
        target_layer=2,
    )
    runtime.layer_start(7)
    assert calls == []
    assert any(
        event.get("event") == "advance_return"
        and event.get("reason") == "target_out_of_range"
        for event in runtime.report()["events"]
    )


def test_main_fallback_and_checkpoint_replay_keep_the_ordinary_autograd_path(monkeypatch) -> None:
    attached = []

    def attach(family, layer_index, tensor):
        attached.append((family, layer_index))
        return tensor

    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
        attach=attach,
    )
    runtime.layer_start(3)
    runtime.acquire("QKV", 3)
    assert attached == []
    runtime.end()
    runtime.materialize_for_consumption("O", 3)
    assert calls[-1] == ("O", 3)
    assert attached == []


def test_materialising_deadline_waits_once_and_never_duplicates(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    first_target_calls = [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    assert calls == first_target_calls
    assert fake_cuda.stream_creations == 1
    runtime.layer_start(5)
    weight = runtime.acquire("DOWN", 5)
    assert calls.count(("DOWN", 5)) == 1
    assert calls[:4] == first_target_calls
    assert weight.recorded_streams[-1] is fake_cuda.main_stream
    assert len(fake_cuda.main_stream.waited_events) == 1
    candidate = runtime._candidates[(5, "DOWN")]
    assert candidate.owner == "premat"
    assert candidate.state == CandidateState.CONSUMING
    assert candidate.critical_path_miss
    report = runtime.report()
    assert report["aggregate"]["waited_hits"] == 1
    assert report["aggregate"]["main_stream_wait_ms_total"] == pytest.approx(0.25)


def test_pending_premat_release_stays_charged_without_blocking_next_launch(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    runtime.layer_start(3)
    assert calls == [("DOWN", 5), ("UP", 5), ("O", 5), ("QKV", 5)]
    runtime.layer_start(5)
    weight = runtime.acquire("DOWN", 5)
    fake_cuda.complete_main_on_record = False
    del weight
    runtime.consumed("DOWN", 5)

    assert runtime._pending_releases
    release = runtime._pending_releases[0]
    assert release.retained_bytes > 0
    assert release.tensor is not None
    assert calls.count(("DOWN", 5)) == 1
    assert calls.count(("DOWN", 7)) == 1
    retained_before_release = runtime._retained_bytes

    release.end_event.complete = True
    runtime.event("after_qkv_complete", layer_index=5)
    assert runtime._pending_releases == []
    assert runtime._retained_bytes == retained_before_release - release.retained_bytes


def test_main_fallback_creates_no_zero_byte_release_gate(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    assert calls == []
    weight = runtime.acquire("QKV", 3)
    assert calls == [("QKV", 3)]
    fake_cuda.complete_main_on_record = False
    runtime._stay_below_current_peak = False
    del weight
    runtime.consumed("QKV", 3)

    assert runtime._pending_releases == []
    assert calls == [
        ("QKV", 3),
        ("DOWN", 5),
        ("UP", 5),
        ("O", 5),
        ("QKV", 5),
    ]


def test_strict_head_defer_does_not_bypass_and_window_never_contains_l_plus_2(monkeypatch) -> None:
    runtime, _fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    assert calls == []
    report = runtime.report()
    assert report["queue_head"]["family"] == "DOWN"
    assert report["queue_head"]["layer_index"] == 5
    assert report["queue_head"]["deferred"] is True
    assert {candidate["layer_index"] for candidate in report["candidates"]} == {3, 5}
    assert all(candidate["owner"] == "none" for candidate in report["candidates"])


def test_cumulative_charge_blocks_second_candidate_without_bypass(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    fake_cuda.free = 1_300
    fake_cuda.total = 1_400
    runtime.layer_start(3)

    assert calls == [("DOWN", 5)]
    report = runtime.report()
    assert report["queue_head"]["family"] == "DOWN"
    assert report["memory"]["premat_cumulative_charged_bytes"] == 512
    assert any(
        event.get("event") == "advance_return"
        and event.get("detail", {}).get("blocking_candidate", {}).get("family") == "UP"
        and event.get("detail", {}).get("submitted_count") == 1
        and event.get("detail", {}).get("cumulative_charged_bytes") == 512
        for event in report["events"]
    )


def test_lifecycle_rejects_illegal_transition(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    candidate = runtime._candidates[(5, "QKV")]
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
    score_bytes = 2 * 2 * 4 * 4 * 2
    mask_bytes = 4 * 4
    assert qk.envelope.foreground_overlap_bytes >= (
        4 * activation_bytes + 2 * score_bytes + mask_bytes
    )


def test_fused_envelope_does_not_charge_quadratic_attention_tensors(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
        attention_mode="fused",
    )
    runtime.layer_start(3)
    qkv = runtime._candidates[(3, "QKV")]
    activation_bytes = 2 * 4 * 8 * 2
    assert qkv.envelope.foreground_overlap_bytes == 4 * activation_bytes


def test_retained_matrix_sizes_follow_effective_materialisation_dtype(monkeypatch) -> None:
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
    )
    runtime.layer_start(3)
    assert runtime._candidates[(3, "QKV")].envelope.retained_bytes == 3 * 8 * 8 * 2
    assert runtime._candidates[(3, "O")].envelope.retained_bytes == 8 * 8 * 2
    assert runtime._candidates[(3, "UP")].envelope.retained_bytes == 4 * 8 * 8 * 2
    assert runtime._candidates[(3, "DOWN")].envelope.retained_bytes == 4 * 8 * 8 * 2


def test_cuda_autocast_dtype_overrides_fp32_pass_reference_for_envelopes(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "is_autocast_enabled",
        lambda device_type=None: device_type == "cuda",
    )
    monkeypatch.setattr(
        torch,
        "get_autocast_dtype",
        lambda device_type: torch.bfloat16,
    )
    runtime, _fake_cuda, _calls = _runtime(
        monkeypatch,
        stay_below_current_peak=True,
        reference_element_bytes=4,
    )
    runtime.layer_start(3)

    report = runtime.report()
    assert report["materialisation_element_bytes"] == 2
    assert report["memory"]["materialisation_element_bytes"] == 2
    assert report["memory"]["foreground_activation_bytes"] == 2 * 4 * 8 * 2
    assert runtime._candidates[(3, "DOWN")].envelope.retained_bytes == 4 * 8 * 8 * 2
    assert runtime._candidates[(3, "DOWN")].envelope.foreground_overlap_bytes == (
        4 * 2 * 4 * 8 * 2
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

    def layer_complete(self, _layer_index: int) -> None:
        assert self.active

    def event(self, _name: str, *, layer_index: int) -> None:
        assert self.active

    def acquire(self, family: str, layer_index: int):
        assert self.active
        return self.model._premat_materialize_candidate(family, layer_index)

    def consumed(self, _family: str, _layer_index: int) -> None:
        assert self.active

    def materialize_for_consumption(self, family: str, layer_index: int):
        assert not self.active
        return self.model._premat_materialize_candidate(family, layer_index)


def test_checkpointed_premat_preserves_configured_segments_and_gradients_on_cpu() -> None:
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss.backward()
        assert not any(
            "AccumulateGrad node's stream does not match" in str(item.message)
            for item in caught
        )
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

    assert checkpointed.last_execution_report.segment_size == 2
    assert [entry["layers"] for entry in fake_runtime.passes] == [(0, 1, 2, 3)]
    assert all(entry["grad_enabled"] for entry in fake_runtime.passes)
    assert all(entry["ended"] for entry in fake_runtime.passes)
    assert not fake_runtime.active


@pytest.mark.parametrize(
    "family_names",
    (
        ("attention_output_weight",),
        (
            "attention_query_weight",
            "attention_key_weight",
            "attention_value_weight",
        ),
    ),
)
def test_prematerialized_depth_binding_matches_ordinary_autograd_on_cpu(
    family_names,
) -> None:
    model = SheetGPT(
        SheetGPTConfig(
            block_size=4,
            vocab_size=16,
            n_layer=3,
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
    )
    trajectory = model.trajectory
    layer_index = 1
    ordinary_saved = []

    def pack_ordinary(tensor):
        ordinary_saved.append((tuple(tensor.shape), tensor.dtype, tensor.device))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_ordinary, lambda tensor: tensor):
        ordinary = torch.cat(
            tuple(trajectory.materialize(name, layer_index) for name in family_names),
            dim=0,
        )
    multiplier = torch.linspace(0.25, 1.25, ordinary.numel()).reshape_as(ordinary)
    (ordinary * multiplier).sum().backward()
    expected_gradients = {
        name: trajectory.coefficients[name].grad.detach().clone()
        for name in family_names
    }
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        cached = torch.cat(
            tuple(trajectory.materialize(name, layer_index) for name in family_names),
            dim=0,
        )
    attached_saved = []

    def pack_attached(tensor):
        attached_saved.append((tuple(tensor.shape), tensor.dtype, tensor.device))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_attached, lambda tensor: tensor):
        attached = trajectory.attach_prematerialized(
            tuple(family_names),
            layer_index,
            cached,
        )
    assert attached_saved == ordinary_saved
    torch.testing.assert_close(attached, ordinary.detach())
    (attached * multiplier).sum().backward()
    for name in family_names:
        torch.testing.assert_close(
            trajectory.coefficients[name].grad,
            expected_gradients[name],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance host required")
@pytest.mark.parametrize("attention_mode", ("fused", "unfused"))
@pytest.mark.parametrize("checkpoint_segment_size", (0, 2))
@pytest.mark.parametrize("autocast_dtype", (None, torch.bfloat16))
def test_cuda_premat_forward_backward_matches_same_mode_disabled(
    attention_mode: str,
    checkpoint_segment_size: int,
    autocast_dtype,
) -> None:
    if autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16 autocast")
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
        forward_context = (
            nullcontext()
            if autocast_dtype is None
            else torch.autocast(device_type="cuda", dtype=autocast_dtype)
        )
        with forward_context:
            logits, loss = model(tokens, targets)
        assert loss is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss.backward()
        assert not any(
            "AccumulateGrad node's stream does not match" in str(item.message)
            for item in caught
        )
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
    gradient_rtol, gradient_atol = (
        (1.0e-5, 1.0e-6)
        if autocast_dtype is None
        else (2.0e-2, 2.0e-4)
    )
    for name in enabled_gradients:
        # The attached analytic backward and ordinary BF16 einsum backward are
        # mathematically identical but use different contraction order.  Keep
        # FP32 strict and bound only the observed BF16 rounding difference.
        torch.testing.assert_close(
            enabled_gradients[name],
            disabled_gradients[name],
            rtol=gradient_rtol,
            atol=gradient_atol,
        )
    report = enabled.premat_report()
    assert report is not None
    assert report["aggregate"]["admitted"] > 0
# ^^^ THOG

