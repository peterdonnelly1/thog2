# vvv THOG
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_thog2_stage6_pilot import archive_incomplete_run
from run_thog2_stage6_pilot import completed_result
from sheet.stage6_source import verify_source_identity


class Stage6RestartTests(unittest.TestCase):
    def test_s6_18_source_commit_mismatch_is_rejected(self) -> None:
        expected = {
            "commit": "a" * 40,
            "origin_url": "git@example/repo.git",
            "tracked_worktree_clean": True,
        }
        actual = {
            "commit": "b" * 40,
            "origin_url": "git@example/repo.git",
            "tracked_worktree_clean": True,
        }
        with self.assertRaisesRegex(ValueError, "source commit differs"):
            verify_source_identity(expected, actual)

    def test_s6_19_dirty_tracked_worktree_is_rejected(self) -> None:
        identity = {
            "commit": "a" * 40,
            "origin_url": "git@example/repo.git",
            "tracked_worktree_clean": False,
        }
        with self.assertRaisesRegex(ValueError, "worktree is not clean"):
            verify_source_identity(identity, identity)

    def test_s6_20_completed_result_is_reusable_only_for_same_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            result_path = run_dir / "result.json"
            result_path.write_text(
                json.dumps({
                    "status": "completed",
                    "run_id": "dense",
                    "protocol_sha256": "locked",
                }),
                encoding="utf-8",
            )
            run = {"run_id": "dense", "out_dir": str(run_dir)}
            self.assertIsNotNone(completed_result(run, "locked"))
            self.assertIsNone(completed_result(run, "different"))

    def test_s6_21_incomplete_run_is_archived_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            marker = run_dir / "partial.txt"
            marker.write_text("preserve me", encoding="utf-8")
            archived = archive_incomplete_run(run_dir)
            self.assertIsNotNone(archived)
            self.assertFalse(run_dir.exists())
            self.assertTrue((archived / "partial.txt").exists())
            self.assertEqual(
                (archived / "partial.txt").read_text(encoding="utf-8"),
                "preserve me",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
