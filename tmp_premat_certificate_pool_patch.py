from __future__ import annotations

from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


premat_path = Path("sheet/premat.py")
premat = premat_path.read_text()

premat = replace_regex(
    premat,
    r"def _same_stream_inactive_block_summary\(.*?(?=\ndef _largest_same_stream_inactive_block_bytes\()",
    '''def _same_stream_inactive_block_sizes(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> Tuple[int, ...]:
    """Return every certified inactive block for one CUDA stream/pool."""
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
    return tuple(sorted(sizes, reverse=True))


def _same_stream_inactive_block_summary(
    snapshot: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    stream_id: int,
    request_bytes: int,
) -> Dict[str, int]:
    """Summarise compatible inactive blocks for one CUDA stream/pool."""
    sizes = _same_stream_inactive_block_sizes(
        snapshot,
        device_index=device_index,
        stream_id=stream_id,
        request_bytes=request_bytes,
    )
    padded = list(sizes[:3]) + [0] * max(0, 3 - len(sizes))
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

''',
    "inactive block helpers",
)

premat = replace_once(
    premat,
    '''    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_charge_bytes: int = 0
''',
    '''    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_block_id: Optional[int] = None
    allocator_certificate_charge_bytes: int = 0
''',
    "candidate certificate block id",
)
premat = replace_once(
    premat,
    '''    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_charge_bytes: int = 0


MaterializeCandidate''',
    '''    allocator_certificate_generation: Optional[int] = None
    allocator_certificate_pool: Optional[str] = None
    allocator_certificate_block_id: Optional[int] = None
    allocator_certificate_charge_bytes: int = 0


MaterializeCandidate''',
    "pending release certificate block id",
)

premat = replace_once(
    premat,
    '''        # Cautious allocator admission snapshots only to certify one contiguous
        # inactive block in each Premat-stream allocator pool.  The certificate
        # models a reusable working reserve: live Premat allocations consume it
        # and safe final release returns the charge.  It never grows above the
        # block actually observed by memory_snapshot().
        self._allocator_certificate_valid = {"small": False, "large": False}
        self._allocator_certificate_generation = {"small": 0, "large": 0}
        self._allocator_certificate_capacity_bytes = {"small": 0, "large": 0}
        self._allocator_certificate_live_charge_bytes = {"small": 0, "large": 0}
''',
    '''        # Cautious allocator admission certifies the complete set of inactive
        # blocks observed in each exact Premat-stream allocator pool.  The native
        # allocator is best-fit, so reservations mirror that policy against the
        # smallest certified block that can satisfy each conservatively rounded
        # request.  Per-block live charges prevent fragmented cache from being
        # treated as one fictitious contiguous allocation and prevent double use.
        self._allocator_certificate_valid = {"small": False, "large": False}
        self._allocator_certificate_generation = {"small": 0, "large": 0}
        self._allocator_certificate_blocks: Dict[str, List[Dict[str, int]]] = {
            "small": [],
            "large": [],
        }
        self._allocator_certificate_capacity_bytes = {"small": 0, "large": 0}
        self._allocator_certificate_live_charge_bytes = {"small": 0, "large": 0}
''',
    "certificate runtime state",
)

certificate_methods = '''    def _sync_allocator_certificate_totals(self, pool: str) -> None:
        blocks = self._allocator_certificate_blocks.get(pool, ())
        self._allocator_certificate_capacity_bytes[pool] = sum(
            int(block["capacity_bytes"]) for block in blocks
        )
        self._allocator_certificate_live_charge_bytes[pool] = sum(
            int(block["live_charge_bytes"]) for block in blocks
        )

    def _allocator_certificate_available_bytes(self, pool: str) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        return sum(
            max(0, int(block["capacity_bytes"]) - int(block["live_charge_bytes"]))
            for block in self._allocator_certificate_blocks.get(pool, ())
        )

    def _allocator_certificate_largest_available_block_bytes(self, pool: str) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        return max(
            (
                max(0, int(block["capacity_bytes"]) - int(block["live_charge_bytes"]))
                for block in self._allocator_certificate_blocks.get(pool, ())
            ),
            default=0,
        )

    def _allocator_certificate_sufficient_block_count(
        self,
        pool: str,
        charge_bytes: int,
    ) -> int:
        if not self._allocator_certificate_valid.get(pool, False):
            return 0
        charge = max(0, int(charge_bytes))
        return sum(
            1
            for block in self._allocator_certificate_blocks.get(pool, ())
            if int(block["capacity_bytes"]) - int(block["live_charge_bytes"]) >= charge
        )

    def _allocator_certificate_best_fit_block(
        self,
        pool: str,
        charge_bytes: int,
    ) -> Optional[Dict[str, int]]:
        if not self._allocator_certificate_valid.get(pool, False):
            return None
        charge = max(0, int(charge_bytes))
        eligible = [
            block
            for block in self._allocator_certificate_blocks.get(pool, ())
            if int(block["capacity_bytes"]) - int(block["live_charge_bytes"]) >= charge
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda block: (
                int(block["capacity_bytes"]) - int(block["live_charge_bytes"]),
                int(block["capacity_bytes"]),
                int(block["block_id"]),
            ),
        )

    def _debit_allocator_certificate_for_pool_shrink(
        self,
        *,
        pool: str,
        current_reserved_bytes: int,
    ) -> int:
        """Invalidate future use if the certified allocator pool has shrunk.

        A pool-level shrink does not identify which certified block disappeared.
        Guessing would make per-block accounting unsafe, so retain the old block
        records only until outstanding charges can be released and fail closed
        for new admissions.  With no live charge the caller may immediately take
        a fresh snapshot and build a new generation.
        """
        floor = self._allocator_certificate_pool_reserved_floor_bytes.get(pool)
        if floor is None:
            return 0
        current = max(0, int(current_reserved_bytes))
        if current >= int(floor):
            return 0
        shrink = int(floor) - current
        self._allocator_certificate_valid[pool] = False
        self._allocator_certificate_pool_reserved_floor_bytes[pool] = current
        return shrink

    def _reserve_allocator_certificate(
        self,
        *,
        request_bytes: int,
        required: bool,
    ) -> Dict[str, object]:
        pool = _allocator_pool_for_request(request_bytes)
        charge = _conservative_certificate_charge_bytes(request_bytes)
        generation = int(self._allocator_certificate_generation.get(pool, 0))
        before_live = int(self._allocator_certificate_live_charge_bytes.get(pool, 0))
        before_available = self._allocator_certificate_available_bytes(pool)
        block = self._allocator_certificate_best_fit_block(pool, charge)
        tracked = block is not None
        block_id: Optional[int] = None
        block_capacity = 0
        block_available_before = 0
        block_available_after = 0
        if required and not tracked:
            raise RuntimeError(
                "allocator-aware admission lost its certified Premat-stream reserve "
                "between decision and launch"
            )
        if tracked:
            block_id = int(block["block_id"])
            block_capacity = int(block["capacity_bytes"])
            block_available_before = max(
                0, block_capacity - int(block["live_charge_bytes"])
            )
            block["live_charge_bytes"] = int(block["live_charge_bytes"]) + charge
            block_available_after = max(
                0, block_capacity - int(block["live_charge_bytes"])
            )
            self._sync_allocator_certificate_totals(pool)
            after_available = self._allocator_certificate_available_bytes(pool)
        else:
            after_available = 0
            # An ordinary admission does not need allocator evidence, but it may
            # consume the same Premat-stream cache.  If the request cannot be
            # represented by one certified block, stop using the certificate
            # until all tracked charges drain and a fresh snapshot proves state.
            if self._allocator_certificate_valid.get(pool, False):
                self._allocator_certificate_valid[pool] = False
        return {
            "tracked": tracked,
            "pool": pool,
            "generation": generation if tracked else None,
            "block_id": block_id,
            "block_capacity_bytes": block_capacity,
            "block_available_before_bytes": block_available_before,
            "block_available_after_bytes": block_available_after,
            "charge_bytes": charge if tracked else 0,
            "before_live_charge_bytes": before_live,
            "after_live_charge_bytes": int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            ),
            "before_available_bytes": before_available,
            "after_available_bytes": after_available,
            "largest_available_block_bytes": (
                self._allocator_certificate_largest_available_block_bytes(pool)
            ),
        }

    def _release_allocator_certificate_charge(
        self,
        *,
        generation: Optional[int],
        pool: Optional[str],
        block_id: Optional[int],
        charge_bytes: int,
    ) -> bool:
        if (
            generation is None
            or pool not in ("small", "large")
            or block_id is None
            or charge_bytes <= 0
        ):
            return False
        if int(generation) != int(self._allocator_certificate_generation.get(pool, -1)):
            return False
        block = next(
            (
                item
                for item in self._allocator_certificate_blocks.get(pool, ())
                if int(item["block_id"]) == int(block_id)
            ),
            None,
        )
        if block is None:
            return False
        before = int(block["live_charge_bytes"])
        block["live_charge_bytes"] = max(0, before - int(charge_bytes))
        self._sync_allocator_certificate_totals(pool)
        return True

    def _release_allocator_certificate_candidate(self, candidate: _Candidate) -> bool:
        released = self._release_allocator_certificate_charge(
            generation=candidate.allocator_certificate_generation,
            pool=candidate.allocator_certificate_pool,
            block_id=candidate.allocator_certificate_block_id,
            charge_bytes=candidate.allocator_certificate_charge_bytes,
        )
        candidate.allocator_certificate_generation = None
        candidate.allocator_certificate_pool = None
        candidate.allocator_certificate_block_id = None
        candidate.allocator_certificate_charge_bytes = 0
        return released

'''
premat = replace_regex(
    premat,
    r"    def _allocator_certificate_available_bytes\(.*?(?=    def _cautious_allocator_aware_rescue\()",
    certificate_methods,
    "certificate methods",
)

premat = replace_once(
    premat,
    '''            "certificate_capacity_bytes": int(
                self._allocator_certificate_capacity_bytes.get(pool, 0)
            ),
            "certificate_live_charge_bytes": int(
''',
    '''            "certificate_capacity_bytes": int(
                self._allocator_certificate_capacity_bytes.get(pool, 0)
            ),
            "certificate_block_count": len(
                self._allocator_certificate_blocks.get(pool, ())
            ),
            "certificate_sufficient_block_count": self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            ),
            "certificate_largest_available_block_bytes": self._allocator_certificate_largest_available_block_bytes(pool),
            "certificate_live_charge_bytes": int(
''',
    "certificate detail defaults",
)

premat = replace_once(
    premat,
    '''            certificate_available = self._allocator_certificate_available_bytes(pool)
            certificate_valid = bool(self._allocator_certificate_valid.get(pool, False))
            live_charge = int(self._allocator_certificate_live_charge_bytes.get(pool, 0))
            if (not certificate_valid) or certificate_available < required_certificate_charge:
''',
    '''            certificate_available = self._allocator_certificate_available_bytes(pool)
            certificate_valid = bool(self._allocator_certificate_valid.get(pool, False))
            certificate_sufficient = self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            )
            live_charge = int(self._allocator_certificate_live_charge_bytes.get(pool, 0))
            if (not certificate_valid) or certificate_sufficient == 0:
''',
    "certificate refresh condition",
)

premat = replace_once(
    premat,
    '''                        block_summary = _same_stream_inactive_block_summary(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
                        largest = int(block_summary["largest_block_bytes"])
''',
    '''                        block_sizes = _same_stream_inactive_block_sizes(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
                        block_summary = _same_stream_inactive_block_summary(
                            snapshot,
                            device_index=int(device_index),
                            stream_id=stream_id,
                            request_bytes=envelope.materialisation_peak_bytes,
                        )
''',
    "snapshot block pool collection",
)

premat = replace_once(
    premat,
    '''                        self._allocator_certificate_generation[pool] = (
                            int(self._allocator_certificate_generation.get(pool, 0)) + 1
                        )
                        self._allocator_certificate_capacity_bytes[pool] = largest
                        self._allocator_certificate_live_charge_bytes[pool] = 0
                        self._allocator_certificate_pool_reserved_floor_bytes[pool] = (
                            pool_reserved_bytes
                        )
                        self._allocator_certificate_valid[pool] = True
''',
    '''                        self._allocator_certificate_generation[pool] = (
                            int(self._allocator_certificate_generation.get(pool, 0)) + 1
                        )
                        self._allocator_certificate_blocks[pool] = [
                            {
                                "block_id": index,
                                "capacity_bytes": int(size),
                                "live_charge_bytes": 0,
                            }
                            for index, size in enumerate(block_sizes, start=1)
                        ]
                        self._sync_allocator_certificate_totals(pool)
                        self._allocator_certificate_pool_reserved_floor_bytes[pool] = (
                            pool_reserved_bytes
                        )
                        self._allocator_certificate_valid[pool] = True
''',
    "snapshot certificate pool install",
)

premat = replace_once(
    premat,
    '''            detail["certificate_capacity_bytes"] = certificate_capacity
            detail["certificate_live_charge_bytes"] = int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            )
            detail["certificate_available_bytes"] = certificate_available
''',
    '''            detail["certificate_capacity_bytes"] = certificate_capacity
            detail["certificate_block_count"] = len(
                self._allocator_certificate_blocks.get(pool, ())
            )
            detail["certificate_sufficient_block_count"] = (
                self._allocator_certificate_sufficient_block_count(
                    pool, required_certificate_charge
                )
            )
            detail["certificate_largest_available_block_bytes"] = (
                self._allocator_certificate_largest_available_block_bytes(pool)
            )
            detail["certificate_live_charge_bytes"] = int(
                self._allocator_certificate_live_charge_bytes.get(pool, 0)
            )
            detail["certificate_available_bytes"] = certificate_available
''',
    "certificate detail refresh",
)

premat = replace_once(
    premat,
    '''            detail["largest_eligible_inactive_block_bytes"] = certificate_capacity
            detail["certified_premat_stream_cache_bytes"] = certificate_capacity
''',
    '''            detail["largest_eligible_inactive_block_bytes"] = (
                self._allocator_certificate_largest_available_block_bytes(pool)
            )
            detail["certified_premat_stream_cache_bytes"] = certificate_capacity
''',
    "certificate cache telemetry",
)

premat = replace_once(
    premat,
    '''        reuse_qualified = (
            bool(self._allocator_certificate_valid.get(pool, False))
            and certificate_available >= required_certificate_charge
        )
''',
    '''        reuse_qualified = (
            bool(self._allocator_certificate_valid.get(pool, False))
            and self._allocator_certificate_sufficient_block_count(
                pool, required_certificate_charge
            ) > 0
        )
''',
    "reuse qualification by individual block",
)

premat = replace_once(
    premat,
    '''                    candidate.allocator_certificate_pool = str(
                        certificate_consumption["pool"]
                    )
                    candidate.allocator_certificate_charge_bytes = int(
''',
    '''                    candidate.allocator_certificate_pool = str(
                        certificate_consumption["pool"]
                    )
                    candidate.allocator_certificate_block_id = int(
                        certificate_consumption["block_id"]
                    )
                    candidate.allocator_certificate_charge_bytes = int(
''',
    "candidate selected block",
)

premat = replace_once(
    premat,
    '''                    allocator_certificate_pool=candidate.allocator_certificate_pool,
                    allocator_certificate_charge_bytes=(
''',
    '''                    allocator_certificate_pool=candidate.allocator_certificate_pool,
                    allocator_certificate_block_id=(
                        candidate.allocator_certificate_block_id
                    ),
                    allocator_certificate_charge_bytes=(
''',
    "pending release selected block",
)

premat = replace_once(
    premat,
    '''            candidate.allocator_certificate_generation = None
            candidate.allocator_certificate_pool = None
            candidate.allocator_certificate_charge_bytes = 0
''',
    '''            candidate.allocator_certificate_generation = None
            candidate.allocator_certificate_pool = None
            candidate.allocator_certificate_block_id = None
            candidate.allocator_certificate_charge_bytes = 0
''',
    "clear candidate selected block after consumption",
)

premat = replace_once(
    premat,
    '''            self._release_allocator_certificate_charge(
                generation=release.allocator_certificate_generation,
                pool=release.allocator_certificate_pool,
                charge_bytes=release.allocator_certificate_charge_bytes,
            )
''',
    '''            self._release_allocator_certificate_charge(
                generation=release.allocator_certificate_generation,
                pool=release.allocator_certificate_pool,
                block_id=release.allocator_certificate_block_id,
                charge_bytes=release.allocator_certificate_charge_bytes,
            )
''',
    "release exact certified block",
)

premat_path.write_text(premat)


store_path = Path("sheet/local_chart_store.py")
store = store_path.read_text()

store = replace_once(
    store,
    '''def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


# vvv THOG runtime resource metadata accepts only finite nonnegative scalar values
''',
    '''def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _premat_snapshot_complete(snapshot: Mapping[str, Any]) -> bool:
    if snapshot.get("pass_complete") is True:
        return True
    events = snapshot.get("events")
    return bool(
        isinstance(events, list)
        and events
        and isinstance(events[-1], Mapping)
        and events[-1].get("event") == "pass_end"
    )


def _store_premat_snapshot_if_mutable(
    connection: sqlite3.Connection,
    optimizer_update: int,
    payload: Mapping[str, Any],
) -> bool:
    """Store until the first complete pass for an optimizer update, then freeze it."""
    connection.execute("BEGIN IMMEDIATE")
    existing_row = connection.execute(
        "SELECT payload FROM premat_snapshots WHERE optimizer_update = ?",
        (int(optimizer_update),),
    ).fetchone()
    if existing_row is not None:
        existing = _decode_payload(existing_row["payload"])
        if _premat_snapshot_complete(existing):
            return False
    connection.execute(
        "INSERT OR REPLACE INTO premat_snapshots(optimizer_update, payload) VALUES (?, ?)",
        (int(optimizer_update), _encode_payload(payload)),
    )
    return True


# vvv THOG runtime resource metadata accepts only finite nonnegative scalar values
''',
    "premat immutable history helpers",
)

store = replace_once(
    store,
    '''        self.connection.execute(
            "INSERT OR REPLACE INTO premat_snapshots(optimizer_update, payload) VALUES (?, ?)",
            (update, _encode_payload(payload)),
        )
        self.connection.execute(
''',
    '''        stored = _store_premat_snapshot_if_mutable(
            self.connection, update, payload
        )
        if not stored:
            self.connection.commit()
            return
        self.connection.execute(
''',
    "LocalChartStore immutable premat row",
)

store = replace_once(
    store,
    '''            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS premat_snapshots (
                    optimizer_update INTEGER PRIMARY KEY,
                    payload BLOB NOT NULL
                )
                """
            )
        update = int(optimizer_update)
''',
    '''            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS premat_snapshots (
                    optimizer_update INTEGER PRIMARY KEY,
                    payload BLOB NOT NULL
                )
                """
            )
            self.connection.commit()
        update = int(optimizer_update)
''',
    "live writer schema commit",
)

# There are now two apparent INSERT blocks in the file; replace the remaining
# one, which belongs to LocalPrematLiveWriter.
store = replace_once(
    store,
    '''        self.connection.execute(
            "INSERT OR REPLACE INTO premat_snapshots(optimizer_update, payload) VALUES (?, ?)",
            (update, _encode_payload(payload)),
        )
        self.connection.execute(
''',
    '''        stored = _store_premat_snapshot_if_mutable(
            self.connection, update, payload
        )
        if not stored:
            self.connection.commit()
            return
        self.connection.execute(
''',
    "LocalPrematLiveWriter immutable premat row",
)

store_path.write_text(store)


test_path = Path("tests/test_premat_allocator_aware_admission.py")
test = test_path.read_text()

test = replace_once(
    test,
    '''    _largest_same_stream_inactive_block_bytes,
    _same_stream_inactive_block_summary,
''',
    '''    _largest_same_stream_inactive_block_bytes,
    _same_stream_inactive_block_sizes,
    _same_stream_inactive_block_summary,
''',
    "test import block sizes helper",
)

test = test.replace(
    '''        generation=reserve["generation"], pool=reserve["pool"],
        charge_bytes=reserve["charge_bytes"],
''',
    '''        generation=reserve["generation"], pool=reserve["pool"],
        block_id=reserve["block_id"], charge_bytes=reserve["charge_bytes"],
''',
)

old_shrink_pattern = r"def test_pool_reserved_shrink_debits_certificate_without_false_growth\(monkeypatch\) -> None:.*?(?=\ndef test_no_overlapping_snapshot_while_certificate_charge_live)"
new_shrink_test = '''def test_pool_reserved_shrink_invalidates_pool_until_live_charge_drains(monkeypatch) -> None:
    observation, envelope = _blocked_case()
    original = _original_decision(observation, envelope)
    runtime = _runtime()
    state = {"pool_reserved": 480, "snapshot": 0}
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda, "memory_stats",
        lambda _device=None: _stats(pool_reserved=state["pool_reserved"]),
    )
    def snapshot():
        state["snapshot"] += 1
        return _snapshot()
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    admitted, _ = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert admitted.admitted
    reserve = runtime._reserve_allocator_certificate(
        request_bytes=envelope.materialisation_peak_bytes, required=True
    )
    state["pool_reserved"] = 380
    blocked, detail = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert blocked == original
    assert detail["certificate_pool_reserved_shrink_debit_bytes"] == 100 * MIB
    assert detail["certificate_refresh_deferred_live_charge"] is True
    assert runtime._allocator_certificate_valid["large"] is False
    assert runtime._allocator_certificate_available_bytes("large") == 0
    assert state["snapshot"] == 1
    runtime._release_allocator_certificate_charge(
        generation=reserve["generation"], pool=reserve["pool"],
        block_id=reserve["block_id"], charge_bytes=reserve["charge_bytes"],
    )
    refreshed, detail2 = runtime._cautious_allocator_aware_rescue(
        observation=observation, envelope=envelope, original_decision=original
    )
    assert refreshed.admitted
    assert detail2["snapshot_performed"] is True
    assert state["snapshot"] == 2

'''
test = replace_regex(test, old_shrink_pattern, new_shrink_test, "pool shrink test")

insert_anchor = '''def test_advance_rejection_does_not_head_of_line_block_later_matrices(monkeypatch) -> None:
'''
new_pool_tests = '''def test_snapshot_sizes_expose_full_certified_pool() -> None:
    snapshot = [
        {"device": 0, "stream": 11, "segment_type": "large", "blocks": [
            {"state": "inactive", "size": 64 * MIB},
            {"state": "inactive", "size": 32 * MIB},
            {"state": "inactive", "size": 16 * MIB},
            {"state": "inactive", "size": 4 * MIB},
        ]},
    ]
    assert _same_stream_inactive_block_sizes(
        snapshot, device_index=0, stream_id=11, request_bytes=8 * MIB
    ) == (64 * MIB, 32 * MIB, 16 * MIB, 4 * MIB)


def test_certificate_pool_uses_individual_best_fit_blocks() -> None:
    runtime = _runtime()
    runtime._allocator_certificate_valid["large"] = True
    runtime._allocator_certificate_generation["large"] = 3
    runtime._allocator_certificate_blocks["large"] = [
        {"block_id": 1, "capacity_bytes": 64 * MIB, "live_charge_bytes": 0},
        {"block_id": 2, "capacity_bytes": 32 * MIB, "live_charge_bytes": 0},
        {"block_id": 3, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
    ]
    runtime._sync_allocator_certificate_totals("large")
    first = runtime._reserve_allocator_certificate(request_bytes=20 * MIB, required=True)
    assert first["charge_bytes"] == 32 * MIB
    assert first["block_id"] == 2
    second = runtime._reserve_allocator_certificate(request_bytes=12 * MIB, required=True)
    assert second["charge_bytes"] == 16 * MIB
    assert second["block_id"] == 3
    assert runtime._allocator_certificate_available_bytes("large") == 64 * MIB
    assert runtime._allocator_certificate_largest_available_block_bytes("large") == 64 * MIB
    runtime._release_allocator_certificate_charge(
        generation=first["generation"], pool=first["pool"], block_id=first["block_id"],
        charge_bytes=first["charge_bytes"],
    )
    assert runtime._allocator_certificate_available_bytes("large") == 96 * MIB


def test_certificate_pool_never_treats_fragmented_total_as_one_block() -> None:
    runtime = _runtime()
    runtime._allocator_certificate_valid["large"] = True
    runtime._allocator_certificate_generation["large"] = 4
    runtime._allocator_certificate_blocks["large"] = [
        {"block_id": 1, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
        {"block_id": 2, "capacity_bytes": 16 * MIB, "live_charge_bytes": 0},
    ]
    runtime._sync_allocator_certificate_totals("large")
    assert runtime._allocator_certificate_available_bytes("large") == 32 * MIB
    assert runtime._allocator_certificate_sufficient_block_count("large", 32 * MIB) == 0
    with pytest.raises(RuntimeError, match="lost its certified"):
        runtime._reserve_allocator_certificate(request_bytes=20 * MIB, required=True)


'''
test = replace_once(test, insert_anchor, new_pool_tests + insert_anchor, "insert pool tests")

test_path.write_text(test)


history_test = Path("tests/test_premat_history_immutability.py")
history_test.write_text('''from __future__ import annotations

from sheet.local_chart_store import LocalChartReader, LocalChartStore, LocalPrematLiveWriter


def _snapshot(pass_sequence: int, marker: str, *, complete: bool) -> dict[str, object]:
    return {
        "pass_sequence": pass_sequence,
        "pass_complete": complete,
        "events": [{"event": "pass_end"}] if complete else [{"event": "consumed"}],
        "marker": marker,
        "aggregate": {},
    }


def test_first_complete_premat_pass_for_update_is_immutable(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(47, _snapshot(1, "first", complete=True))
    store.append_premat_snapshot(47, _snapshot(2, "later", complete=True))
    rows = LocalChartReader(path).premat_snapshots()
    assert len(rows) == 1
    assert rows[0]["marker"] == "first"
    assert rows[0]["pass_sequence"] == 1
    store.close()


def test_incomplete_row_can_progress_until_first_complete_snapshot(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(50, _snapshot(1, "partial", complete=False))
    store.append_premat_snapshot(50, _snapshot(1, "complete", complete=True))
    store.append_premat_snapshot(50, _snapshot(2, "must-not-replace", complete=True))
    row = LocalChartReader(path).latest_premat_snapshot()
    assert row is not None
    assert row["marker"] == "complete"
    assert row["pass_sequence"] == 1
    store.close()


def test_live_writer_cannot_replace_completed_history_row(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(51, _snapshot(1, "first", complete=True))
    writer = LocalPrematLiveWriter(path)
    writer.append(51, _snapshot(2, "later", complete=True))
    writer.close()
    row = LocalChartReader(path).latest_premat_snapshot()
    assert row is not None
    assert row["marker"] == "first"
    store.close()
''')

print("patched PREMAT certificate pool and immutable history storage")
