from pathlib import Path

source_path = Path("sheet/local_dashboard_live_loss.py")
source = source_path.read_text()
source = source.replace(
    '_TIME_AXIS_MODES = ("relative_wall", "relative_process", "wall_time")\n',
    '_TIME_AXIS_MODES = ("relative_wall", "relative_process", "wall_time")\n'
    '# vvv THOG keep restart/backlog reconstruction anchored to the newest train.log tail rather than stale chunks\n'
    '_READ_CHUNK_BYTES = 1024 * 1024\n'
    '_LIVE_TAIL_BYTES = 4 * 1024 * 1024\n'
    '# ^^^ THOG\n',
    1,
)
old = '''            identity = (stat.st_dev, stat.st_ino)\n            if identity != self.identity or stat.st_size < self.offset:\n                self.offset = 0\n                self.pending = b""\n                self.values = {"train": {}, "val": {}}\n                # vvv THOG a rotated/restarted train.log starts a fresh provisional timing origin\n                self.wall_times = {"train": {}, "val": {}}\n                self.first_wall_time = None\n                # ^^^ THOG\n                self.revision += 1\n            self.path, self.identity = path, identity\n            with path.open("rb") as handle:\n                handle.seek(self.offset)\n                content = handle.read(1024 * 1024)\n                self.offset = handle.tell()\n            lines = (self.pending + content).split(b"\\n")\n'''
new = '''            identity = (stat.st_dev, stat.st_ino)\n            reset_reader = identity != self.identity or stat.st_size < self.offset\n            if reset_reader:\n                self.offset = 0\n                self.pending = b""\n                self.values = {"train": {}, "val": {}}\n                # vvv THOG a rotated/restarted train.log starts a fresh provisional timing origin\n                self.wall_times = {"train": {}, "val": {}}\n                self.first_wall_time = None\n                # ^^^ THOG\n                self.revision += 1\n            self.path, self.identity = path, identity\n            # vvv THOG restart from the newest bounded tail and drain any modest backlog to EOF before assigning mtime-derived timestamps\n            backlog_bytes = max(0, int(stat.st_size) - int(self.offset))\n            drain_to_eof = reset_reader or backlog_bytes > _READ_CHUNK_BYTES\n            discard_partial_prefix = False\n            if reset_reader:\n                self.offset = max(0, int(stat.st_size) - _LIVE_TAIL_BYTES)\n                discard_partial_prefix = self.offset > 0\n            elif drain_to_eof:\n                tail_offset = max(int(self.offset), int(stat.st_size) - _LIVE_TAIL_BYTES)\n                if tail_offset > self.offset:\n                    self.offset = tail_offset\n                    self.pending = b""\n                    discard_partial_prefix = True\n            with path.open("rb") as handle:\n                handle.seek(self.offset)\n                content = handle.read() if drain_to_eof else handle.read(_READ_CHUNK_BYTES)\n                self.offset = handle.tell()\n            if discard_partial_prefix:\n                newline = content.find(b"\\n")\n                content = content[newline + 1:] if newline >= 0 else b""\n            # ^^^ THOG\n            lines = (self.pending + content).split(b"\\n")\n'''
if old not in source:
    raise SystemExit("source block not found")
source = source.replace(old, new, 1)
source = source.replace(
    '            wall_cursor = float(stat.st_mtime)\n',
    '            # vvv THOG restat after the read so the anchor corresponds to the newest bytes actually consumed\n'
    '            wall_cursor = float(path.stat().st_mtime)\n'
    '            # ^^^ THOG\n',
    1,
)
source_path.write_text(source)

test_path = Path("tests/test_instra_live_loss.py")
tests = test_path.read_text()
anchor = '''\ndef test_rotation_summary_and_bounded_memory(tmp_path):\n'''
addition = '''\n\ndef test_restart_large_log_drains_to_newest_progress_rows_in_one_refresh(tmp_path):\n    # vvv THOG a restarted Instra must not timestamp an old 1-MiB prefix with the current train.log mtime\n    noise = "not a progress row\\n" * 70000\n    reader, *_ = reader_for(\n        tmp_path,\n        noise\n        + "T 10 120926-1400 0010 Δstep=4.0s loss=7.1\\n"\n        + "T 20 120926-1400 0020 Δstep=6.0s loss=6.9\\n",\n    )\n    assert reader.values["train"] == {10: 7.1, 20: 6.9}\n    assert reader.wall_times["train"][10] == reader.wall_times["train"][20] - 6.0\n    # ^^^ THOG\n'''
if anchor not in tests:
    raise SystemExit("test anchor not found")
tests = tests.replace(anchor, addition + anchor, 1)
test_path.write_text(tests)
