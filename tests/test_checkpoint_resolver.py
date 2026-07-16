# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sheet.checkpoint_resolver import resolve_checkpoint
from sheet.checkpoints import save_payload
from tests.enhanced_resume_test_support import write_checkpoint_stub


class CheckpointResolverTests(unittest.TestCase):
    def test_checkpoint_file_directory_artifact_name_and_leading_timestamp_selectors_resolve_same_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, _ = write_checkpoint_stub(root)
            selectors = (str(checkpoint), str(checkpoint.parent), config.artifact_name, "260715-1200")
            resolved = [resolve_checkpoint(value, root / "checkpoints").checkpoint_path.resolve() for value in selectors]
            self.assertEqual(resolved, [checkpoint.resolve()] * len(selectors))

    def test_timestamp_selector_matches_only_leading_timestamp_not_fork_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, _, _ = write_checkpoint_stub(root, start_label="260715-1200")
            child_dir = root / "checkpoints" / "260715-1201_TEST__FORK_1_FROM_260715-1200"
            save_payload({"lifecycle": {"lifecycle_schema_version": 0}}, child_dir / "ckpt.pt")
            self.assertEqual(resolve_checkpoint("260715-1200", root / "checkpoints").checkpoint_path, original)

    def test_zero_timestamp_matches_fail_without_newest_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no checkpoint matches"):
                resolve_checkpoint("260715-9999", Path(directory))

    def test_multiple_timestamp_matches_fail_and_list_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _, lifecycle = write_checkpoint_stub(root, start_label="260715-1200")
            second_dir = first.parent.parent / (first.parent.name + "_COPY")
            copied = dict(lifecycle)
            copied["artifact_name"] = second_dir.name
            save_payload({"lifecycle": copied}, second_dir / "ckpt.pt")
            with self.assertRaisesRegex(ValueError, "multiple checkpoints"):
                resolve_checkpoint("260715-1200", root / "checkpoints")

    def test_checkpoint_without_enhanced_lifecycle_schema_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints" / "260715-1200_OLD" / "ckpt.pt"
            save_payload({"completed_updates": 3}, path)
            with self.assertRaisesRegex(ValueError, "enhanced lifecycle"):
                resolve_checkpoint(str(path), Path(directory) / "checkpoints")

    def test_checkpoint_directory_must_match_lifecycle_artifact_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, lifecycle = write_checkpoint_stub(root)
            bad = dict(lifecycle)
            bad["artifact_name"] = "different"
            save_payload({"lifecycle": bad}, checkpoint)
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_checkpoint(str(checkpoint), root / "checkpoints")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
