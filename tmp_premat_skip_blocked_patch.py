from pathlib import Path

path = Path("sheet/premat.py")
text = path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    text = text.replace(old, new, 1)


start = text.index("def _largest_same_stream_inactive_block_bytes(\n")
end = text.index("\ndef _memory_stat_nonnegative_int(\n", start)
new_helper = '''def _same_stream_inactive_block_summary(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> Dict[str, int]:
    """Summarise compatible inactive blocks for one CUDA stream/pool."""
    required_pool = _allocator_pool_for_request(request_bytes)
    sizes: List[int] = []
    for segment in snapshot:
        if not isinstance(segment, Mapping):
            raise ValueError("CUDA allocator snapshot contains a non-mapping segment")
        segment_device = segment.get("device")
        if isinstance(segment_device, bool) or not isinstance(segment_device, int):
            raise ValueError("CUDA allocator snapshot contains an invalid device index")
        if int(segment_device) != int(device_index):
            continue
        if int(segment.get("stream", -1)) != int(stream_id):
            continue
        if segment.get("segment_type") != required_pool:
            continue
        blocks = segment.get("blocks")
        if not isinstance(blocks, Sequence):
            raise ValueError("CUDA allocator snapshot segment has no block sequence")
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("CUDA allocator snapshot contains a non-mapping block")
            if block.get("state") != "inactive":
                continue
            size = block.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("CUDA allocator snapshot contains an invalid block size")
            sizes.append(int(size))
    sizes.sort(reverse=True)
    padded = sizes[:3] + [0] * max(0, 3 - len(sizes))
    return {
        "block_count": len(sizes),
        "sufficient_block_count": sum(
            1 for size in sizes if size >= int(request_bytes)
        ),
        "total_bytes": sum(sizes),
        "largest_block_bytes": padded[0],
        "second_largest_block_bytes": padded[1],
        "third_largest_block_bytes": padded[2],
    }


def _largest_same_stream_inactive_block_bytes(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> int:
    return _same_stream_inactive_block_summary(
        snapshot,
        device_index=device_index,
        stream_id=stream_id,
        request_bytes=request_bytes,
    )["largest_block_bytes"]

'''
text = text[:start] + new_helper + text[end + 1 :]

marker = "    def _allocator_certificate_available_bytes(self, pool: str) -> int:\n"
insert = '''    def _pending_release_diagnostics(self) -> Dict[str, int]:
        return {
            "pending_release_count": len(self._pending_releases),
            "pending_release_bytes": sum(
                int(release.retained_bytes) + int(release.transient_bytes)
                for release in self._pending_releases
            ),
            "certificate_bytes_waiting_for_release": sum(
                int(release.allocator_certificate_charge_bytes)
                for release in self._pending_releases
            ),
        }

'''
replace_once(marker, insert + marker, "pending-release helper insertion")

replace_once(
    '''            "certificate_refresh_deferred_live_charge": False,
            "largest_eligible_inactive_block_bytes": 0,
''',
    '''            "certificate_refresh_deferred_live_charge": False,
            "snapshot_eligible_inactive_block_count": 0,
            "snapshot_eligible_sufficient_block_count": 0,
            "snapshot_eligible_inactive_total_bytes": 0,
            "snapshot_eligible_largest_block_bytes": 0,
            "snapshot_eligible_second_largest_block_bytes": 0,
            "snapshot_eligible_third_largest_block_bytes": 0,
            **self._pending_release_diagnostics(),
            "largest_eligible_inactive_block_bytes": 0,
''',
    "cautious diagnostics fields",
)

replace_once(
    '''                        largest = _largest_same_stream_inactive_block_bytes(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
''',
    '''                        block_summary = _same_stream_inactive_block_summary(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
                        largest = int(block_summary["largest_block_bytes"])
                        detail["snapshot_eligible_inactive_block_count"] = int(
                            block_summary["block_count"]
                        )
                        detail["snapshot_eligible_sufficient_block_count"] = int(
                            block_summary["sufficient_block_count"]
                        )
                        detail["snapshot_eligible_inactive_total_bytes"] = int(
                            block_summary["total_bytes"]
                        )
                        detail["snapshot_eligible_largest_block_bytes"] = int(
                            block_summary["largest_block_bytes"]
                        )
                        detail["snapshot_eligible_second_largest_block_bytes"] = int(
                            block_summary["second_largest_block_bytes"]
                        )
                        detail["snapshot_eligible_third_largest_block_bytes"] = int(
                            block_summary["third_largest_block_bytes"]
                        )
''',
    "snapshot summary substitution",
)

replace_once(
    '''                "cumulative_charged_bytes": self._cumulative_charged_bytes(),
                "raw_memory": raw_memory,
''',
    '''                "cumulative_charged_bytes": self._cumulative_charged_bytes(),
                **self._pending_release_diagnostics(),
                "raw_memory": raw_memory,
''',
    "admission decision release diagnostics",
)

replace_once(
    '''        submitted: List[Dict[str, object]] = []
        target_position = self._position + self._target_layer
''',
    '''        submitted: List[Dict[str, object]] = []
        deferred_sequences: set[int] = set()
        target_position = self._position + self._target_layer
''',
    "advance deferred set",
)

replace_once(
    '''            candidate = self._next_premat_candidate()
            if candidate is None:
''',
    '''            candidate = self._next_premat_candidate(
                excluded_sequences=deferred_sequences
            )
            if candidate is None:
''',
    "advance candidate selection",
)

replace_once(
    '''                self._record_advance_return(
                    invocation=invocation,
                    trigger=trigger,
                    submitted=submitted,
                    reason="admission_rejected",
                    blocking_candidate=candidate,
                    blocking_reason=decision.reason,
                )
                return
''',
    '''                deferred_sequences.add(candidate.sequence)
                # A rejection is local to this matrix for this scheduler
                # invocation. Later matrices in the configured order still get
                # their own admission test. This candidate remains UNAVAILABLE
                # and will be reconsidered on the next invocation.
                continue
''',
    "rejection head-of-line return",
)

replace_once(
    '''    def _next_premat_candidate(self) -> Optional[_Candidate]:
        target_position = self._position + self._target_layer
        if self._position < 0 or target_position >= len(self._layer_indices):
            return None
        target_layer_index = self._layer_indices[target_position]
        return next(
            (
                item
                for item in sorted(
                    self._candidates.values(),
                    key=lambda candidate: candidate.sequence,
                )
                if item.state == CandidateState.UNAVAILABLE
                and item.layer_index == target_layer_index
            ),
            None,
        )
''',
    '''    def _next_premat_candidate(
        self,
        *,
        excluded_sequences: Optional[set[int]] = None,
    ) -> Optional[_Candidate]:
        target_position = self._position + self._target_layer
        if self._position < 0 or target_position >= len(self._layer_indices):
            return None
        target_layer_index = self._layer_indices[target_position]
        excluded = excluded_sequences or set()
        return next(
            (
                item
                for item in sorted(
                    self._candidates.values(),
                    key=lambda candidate: candidate.sequence,
                )
                if item.state == CandidateState.UNAVAILABLE
                and item.layer_index == target_layer_index
                and item.sequence not in excluded
            ),
            None,
        )
''',
    "next candidate exclusion support",
)

path.write_text(text)

test_path = Path("tests/test_premat_allocator_aware_admission.py")
tests = test_path.read_text()
replace_import = "    _largest_same_stream_inactive_block_bytes,\n"
if tests.count(replace_import) != 1:
    raise RuntimeError("test import marker mismatch")
tests = tests.replace(
    replace_import,
    replace_import + "    _same_stream_inactive_block_summary,\n",
    1,
)
tests += '''


def test_snapshot_summary_reports_multiple_compatible_blocks() -> None:
    snapshot = [
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 64 * MIB},
            {"state": "inactive", "size": 32 * MIB},
            {"state": "inactive", "size": 16 * MIB},
            {"state": "inactive", "size": 4 * MIB},
            {"state": "active_allocated", "size": 999 * MIB},
        ]},
        {"device": 0, "stream": 12, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 800 * MIB},
        ]},
    ]
    summary = _same_stream_inactive_block_summary(
        snapshot, device_index=0, stream_id=11, request_bytes=8 * MIB
    )
    assert summary == {
        "block_count": 4,
        "sufficient_block_count": 3,
        "total_bytes": 116 * MIB,
        "largest_block_bytes": 64 * MIB,
        "second_largest_block_bytes": 32 * MIB,
        "third_largest_block_bytes": 16 * MIB,
    }


def test_advance_rejection_does_not_head_of_line_block_later_matrices(monkeypatch) -> None:
    runtime = _runtime()
    runtime._allocator_aware_admission = "disabled"
    runtime._layer_indices = (0, 1)
    runtime._position = 0
    runtime._ensure_layer(1)
    observation, _ = _blocked_case()
    monkeypatch.setattr(runtime, "_refresh_available", lambda: None)
    monkeypatch.setattr(runtime, "_observe_memory", lambda: observation)
    monkeypatch.setattr(runtime, "_update_memory_aggregates", lambda _observation: None)
    monkeypatch.setattr(runtime, "_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_record_advance_return", lambda **kwargs: None)
    runtime._advance(trigger="test")
    layer_candidates = [
        candidate
        for candidate in runtime._candidates.values()
        if candidate.layer_index == 1
    ]
    assert len(layer_candidates) == 4
    assert all(candidate.first_considered_ns is not None for candidate in layer_candidates)
    assert runtime._aggregate["deferred"] == 4


def test_cautious_detail_includes_pending_release_diagnostics(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()

    class Pending:
        retained_bytes = 8 * MIB
        transient_bytes = 2 * MIB
        allocator_certificate_charge_bytes = 16 * MIB

    runtime._pending_releases = [Pending()]
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device=None: _stats())
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: _snapshot())
    _, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation,
        envelope=envelope,
        original_decision=original,
    )
    assert detail["pending_release_count"] == 1
    assert detail["pending_release_bytes"] == 10 * MIB
    assert detail["certificate_bytes_waiting_for_release"] == 16 * MIB
'''
test_path.write_text(tests)
