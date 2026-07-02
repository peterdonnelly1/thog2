# vvv THOG
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from sheet.checkpoints import load_payload
from sheet.trainer import SharedTrainer
from tests.stage5_difference import nested_tensor_difference
from tests.stage5_test_support import (
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
        communication = evidence["communication_profile"]
        self.assertEqual(communication["backend"], "gloo")
        self.assertEqual(communication["world_size"], 2)
        self.assertGreater(communication["maximum_elapsed_seconds"], 0.0)
        self.assertTrue(math.isfinite(communication["maximum_seconds_per_collective"]))

    def test_s5_09_ddp_update_matches_single_process_global_batch(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            evidence = run_ddp_worker("update", output_dir)
            payload = load_payload(output_dir / "ddp_update.pt")
            gradient_payload = torch.load(
                output_dir / "ddp_gradient_probe.pt",
                map_location="cpu",
                weights_only=True,
            )

            previous_thread_count = torch.get_num_threads()
            torch.set_num_threads(1)
            single = SharedTrainer(
                stage5_config(),
                train_tokens,
                validation_tokens,
            )
            try:
                batch_state = single.batch_source.state_dict()
                single.model.train()
                single.optimizer.zero_grad(set_to_none=True)
                batch = single.batch_source.get_batch("train", device=single.device)
                with single.autocast_context():
                    _, gradient_loss = single.model(batch.inputs, batch.targets)
                self.assertIsNotNone(gradient_loss)
                gradient_loss.backward()
                single_gradients = {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in single.raw_model.named_parameters()
                    if parameter.grad is not None
                }
                gradient_difference = nested_tensor_difference(
                    single_gradients,
                    gradient_payload["gradients"],
                )
                self.assertLessEqual(
                    gradient_difference["max_absolute_delta"],
                    2.0e-5,
                    msg=f"DDP averaged-gradient maximum error is too large: {gradient_difference}",
                )
                self.assertLessEqual(
                    gradient_difference["relative_l2_error"],
                    2.0e-5,
                    msg=f"DDP averaged-gradient relative error is too large: {gradient_difference}",
                )
                self.assertAlmostEqual(
                    float(gradient_loss),
                    float(gradient_payload["loss"]),
                    delta=2.0e-6,
                )
                self.assertEqual(
                    list(single.batch_source.training_trace()[-1]),
                    list(gradient_payload["global_starts"]),
                )
                single.optimizer.zero_grad(set_to_none=True)
                single.batch_source.load_state_dict(batch_state)

                single_history = single.run()
                model_difference = nested_tensor_difference(
                    single.raw_model.state_dict(),
                    payload["model"],
                )
                self.assertLessEqual(
                    model_difference["relative_l2_error"],
                    5.0e-4,
                    msg=(
                        "bounded post-Adam whole-state relative error is too large: "
                        f"{model_difference}"
                    ),
                )

                probe_inputs = torch.arange(16, dtype=torch.long).view(2, 8) % 32
                probe_targets = torch.roll(probe_inputs, shifts=-1, dims=1)
                single.raw_model.eval()
                with torch.no_grad():
                    single_logits, single_loss = single.raw_model(
                        probe_inputs,
                        probe_targets,
                    )
                single.raw_model.load_state_dict(payload["model"])
                with torch.no_grad():
                    ddp_logits, ddp_loss = single.raw_model(
                        probe_inputs,
                        probe_targets,
                    )
                functional_difference = nested_tensor_difference(
                    single_logits,
                    ddp_logits,
                )
                self.assertLessEqual(
                    functional_difference["max_absolute_delta"],
                    1.0e-3,
                    msg=f"fixed-probe logit maximum error is too large: {functional_difference}",
                )
                self.assertLessEqual(
                    functional_difference["relative_l2_error"],
                    2.0e-4,
                    msg=f"fixed-probe logit relative error is too large: {functional_difference}",
                )
                self.assertIsNotNone(single_loss)
                self.assertIsNotNone(ddp_loss)
                self.assertAlmostEqual(
                    float(single_loss),
                    float(ddp_loss),
                    delta=1.0e-4,
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
                torch.set_num_threads(previous_thread_count)

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
