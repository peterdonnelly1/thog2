from pathlib import Path

p = Path("tests/test_premat.py")
text = p.read_text()
old = '''    assert candidate.owner == "premat"
    assert candidate.state == CandidateState.CONSUMING
    assert candidate.critical_path_miss
    report = runtime.report()
    assert report["aggregate"]["waited_hits"] == 1
'''
new = '''    assert candidate.owner == "premat"
    assert candidate.state == CandidateState.CONSUMING
    # Host acquire() only knows that PREMAT is still MATERIALISING. The hit
    # outcome remains provisional until the GPU reaches the dependency marker.
    assert candidate.critical_path_miss is False
    assert candidate.final_outcome == "PENDING"
    report = runtime.report()
    assert candidate.critical_path_miss is True
    assert candidate.final_outcome == "PARTIAL HIT"
    assert report["aggregate"]["waited_hits"] == 1
'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one provisional-outcome assertion block, found {text.count(old)}")
text = text.replace(old, new, 1)
old = '''    wait_timing.start_event.elapsed_time_override = (
        lambda other: -0.050 if other is wait_timing.dependency_event else 0.008
    )
    runtime.end()
'''
new = '''    wait_timing.start_event.elapsed_time_override = (
        lambda other: -0.050 if other is wait_timing.dependency_event else 0.008
    )
    # Match the real model sequence: the consuming GEMM is submitted and then
    # the model tells PREMAT that the matrix has been consumed.
    runtime.consumed("DOWN", 5)
    runtime.end()
'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one delayed-report acquire block, found {text.count(old)}")
text = text.replace(old, new, 1)
old = '''    wait_timing.end_event.complete = True
    runtime.begin((3, 5, 7), reference=_FakeTensor())
'''
new = '''    # The fake Premat stream has no autonomous progress. Mark every CUDA timing
    # from pass 1 complete, exactly as the real device eventually would; the
    # Main Stream wait timing keeps its explicit signed dependency override.
    for timing in runtime._pending_timings:
        timing.end_event.complete = True
    runtime.begin((3, 5, 7), reference=_FakeTensor())
'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one delayed-report completion block, found {text.count(old)}")
text = text.replace(old, new, 1)
p.write_text(text)
print("aligned PREMAT GPU wait regression harness with provisional classification")
