# vvv THOG
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_thog2_owt
from run_thog2_owt import (
    _BASE_OWT_TRAINER,
    _CheckpointExitOwtTrainer,
)
from sheet.interactive_interrupt import (
    CheckpointExitController,
    CheckpointExitRequested,
)


class _RequestedController:
    def __init__(self, requested: bool) -> None:
        self._requested = requested
        self.closed = False

    def requested(self) -> bool:
        return self._requested

    def close(self) -> None:
        self.closed = True


class CheckpointExitControlTests(unittest.TestCase):
    def setUp(self) -> None:
        run_thog2_owt._CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = False

    def tearDown(self) -> None:
        run_thog2_owt._CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = False

    def test_shared_request_file_becomes_visible_after_maturation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request_path = Path(temporary) / "checkpoint_exit.request"
            with mock.patch.dict(
                os.environ,
                {"THOG2_CHECKPOINT_EXIT_FILE": str(request_path)},
                clear=False,
            ):
                controller = CheckpointExitController(is_primary=False)
                self.assertFalse(controller.requested())
                request_path.write_text("requested\n", encoding="utf-8")
                old = time.time() - 1.0
                os.utime(request_path, (old, old))
                self.assertTrue(controller.requested())

    def test_public_trainer_saves_at_safe_timed_boundary_then_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "ckpt.pt"
            trainer = object.__new__(_CheckpointExitOwtTrainer)
            trainer._checkpoint_exit_controller = _RequestedController(True)
            trainer.distributed = SimpleNamespace(is_primary=True)
            trainer.config = SimpleNamespace(out_dir=temporary)
            trainer.state = SimpleNamespace(completed_updates=37)
            trainer.optimizer = mock.Mock()
            trainer.save_checkpoint = mock.Mock(return_value=checkpoint_path)

            with mock.patch.object(
                _BASE_OWT_TRAINER,
                "_timed",
                return_value=({"training_loss": 1.0}, 0.5),
            ):
                with self.assertRaises(CheckpointExitRequested) as raised:
                    trainer._timed(lambda: None)

            self.assertEqual(raised.exception.completed_updates, 37)
            self.assertEqual(raised.exception.checkpoint_path, checkpoint_path)
            trainer.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
            trainer.save_checkpoint.assert_called_once_with(checkpoint_path)
            self.assertTrue(run_thog2_owt._CHECKPOINT_EXIT_CLEAN_FINISH_PENDING)

    def test_public_trainer_preserves_normal_timed_result_without_request(self) -> None:
        trainer = object.__new__(_CheckpointExitOwtTrainer)
        trainer._checkpoint_exit_controller = _RequestedController(False)

        with mock.patch.object(
            _BASE_OWT_TRAINER,
            "_timed",
            return_value=("result", 0.25),
        ):
            self.assertEqual(trainer._timed(lambda: None), ("result", 0.25))

    def test_ctrl_g_forces_explicit_clean_telemetry_finish(self) -> None:
        recorded_finishes = []
        run_thog2_owt._CHECKPOINT_EXIT_CLEAN_FINISH_PENDING = True

        def record_finish(telemetry, *, exit_code=None, final_state="finished") -> None:
            recorded_finishes.append((exit_code, final_state))

        with mock.patch.object(
            run_thog2_owt,
            "_ORIGINAL_WANDB_TELEMETRY_FINISH",
            side_effect=record_finish,
        ):
            run_thog2_owt._finish_telemetry_with_checkpoint_exit_policy(
                object(),
                exit_code=None,
            )

        self.assertEqual(recorded_finishes, [(0, "stopped")])
        self.assertFalse(run_thog2_owt._CHECKPOINT_EXIT_CLEAN_FINISH_PENDING)

    def test_public_main_maps_completed_checkpoint_exit_to_131(self) -> None:
        request = CheckpointExitRequested(Path("/tmp/ckpt.pt"), 19)
        with mock.patch.object(
            run_thog2_owt,
            "_main_without_interrupt_checkpoint",
            side_effect=request,
        ):
            self.assertEqual(run_thog2_owt.main([]), 131)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
