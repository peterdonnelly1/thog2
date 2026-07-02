# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sheet.checkpoints import load_payload
from sheet.trainer import SharedTrainer
from tests.stage5_test_support import (
    assert_nested_close,
    run_ddp_worker,
    stage5_config,
    token_splits,
)


class Stage5DdpTests(unittest.TestCase):
    def test_s5_08_ddp_construction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = run_ddp_worker("construction", Path(directory))
        self.assertEqual(evidence["world_size"], 2)
        self.assertEqual(evidence["local_batch_size"], 2)
        self.assertEqual(evidence["model_state_max_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_max_delta"], 0.0)
        self.assertEqual(
            sorted(report["rank"] for report in evidence["rank_reports"]),
            [0, 1],
        )
        self.assertTrue(all(report["active"] for report in evidence["rank_reports"]))
        self.assertTrue(all(report["backend"] == "gloo" for report in evidence["rank_reports"]))

    def test_s5_09_ddp_update_matches_single_process_global_batch(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            evidence = run_ddp_worker("update", output_dir)
            payload = load_payload(output_dir / "ddp_update.pt")

            single = SharedTrainer(
                stage5_config(),
                train_tokens,
                validation_tokens,
            )
            try:
                single_history = single.run()
                assert_nested_close(
                    self,
                    single.raw_model.state_dict(),
                    payload["model"],
                    atol=2.0e-6,
                    rtol=2.0e-5,
                )
                assert_nested_close(
                    self,
                    single.optimizer.state_dict(),
                    payload["optimizer"],
                    atol=2.0e-6,
                    rtol=2.0e-5,
                )
                self.assertEqual(
                    [list(item) for item in single.batch_source.training_trace()],
                    evidence["training_trace"],
                )
                self.assertEqual(len(single_history), len(evidence["history"]))
                for single_item, ddp_item in zip(single_history, evidence["history"]):
                    for key in single_item:
                        self.assertAlmostEqual(
                            single_item[key],
                            ddp_item[key],
                            delta=2.0e-6 + 2.0e-5 * abs(single_item[key]),
                        )
            finally:
                single.close()

        self.assertEqual(evidence["completed_updates"], 2)
        self.assertEqual(evidence["model_state_max_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_max_delta"], 0.0)
        self.assertEqual(payload["distributed_training"]["world_size"], 2)
        self.assertEqual(payload["distributed_training"]["rank"], 0)

    def test_s5_10_ddp_state_remains_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = run_ddp_worker("update", Path(directory))
        self.assertEqual(evidence["model_state_max_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_max_delta"], 0.0)

    def test_s5_11_ddp_resume_preserves_lifecycle_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            evidence = run_ddp_worker("resume", output_dir)
            payload = load_payload(output_dir / "ddp_resume_final.pt")
        self.assertEqual(evidence["completed_updates"], 2)
        self.assertEqual(payload["completed_updates"], 2)
        self.assertEqual(payload["trainer_state"]["completed_updates"], 2)
        self.assertEqual(payload["batch_source"]["world_size"], 2)
        self.assertEqual(evidence["model_state_max_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_max_delta"], 0.0)
        self.assertEqual(len(evidence["history"]), 2)

    def test_s5_12_uneven_checkpoint_segment_boundary_under_ddp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = run_ddp_worker("boundary", Path(directory))
        self.assertEqual(evidence["logical_layers"], 5)
        self.assertEqual(evidence["segment_size"], 2)
        self.assertEqual(evidence["checkpoint_segments"], 3)
        self.assertEqual(evidence["model_state_max_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_max_delta"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
