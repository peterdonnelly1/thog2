# vvv THOG
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch


class TrainerStepMixin:
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        self._record("evaluation_started")
        results: Dict[str, float] = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(self.config.eval_batches):
                batch = self.batch_source.get_batch(split, device=self.device)
                with self.autocast_context():
                    _, loss = self.model(batch.inputs, batch.targets)
                local_finite = loss is not None and bool(torch.isfinite(loss).item())
                self.distributed.require_all_true(
                    local_finite,
                    f"non-finite {split} evaluation loss on at least one rank",
                )
                if loss is None:
                    raise RuntimeError("model did not return an evaluation loss")
                losses.append(self.distributed.mean_float(loss.detach()))
            results[split] = sum(losses) / len(losses)
        self.state.latest_validation_loss = results["val"]
        self.state.best_validation_loss = min(
            self.state.best_validation_loss,
            results["val"],
        )
        self._record("evaluation_completed", losses=results)
        self.model.train(was_training)
        return results

    # vvv THOG
    # Build compact, JSON-safe diagnostics for non-finite losses and gradients before deciding to raise or skip.
    @staticmethod
    def _jsonable_float(value: Optional[float]) -> Optional[float | str]:
        if value is None:
            return None
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "nan"
        if value > 0.0:
            return "inf"
        return "-inf"

    def _local_gradients_are_finite(self) -> bool:
        for parameter in self.raw_model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                return False
        return True

    def _local_nonfinite_gradient_reports(self) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        for parameter_name, parameter in self.raw_model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            detached = gradient.detach()
            finite_mask = torch.isfinite(detached)
            if bool(finite_mask.all().item()):
                continue
            inf_mask = torch.isinf(detached)
            finite_count = int(finite_mask.sum().item())
            finite_abs_max: Optional[float] = None
            if finite_count > 0:
                finite_abs_max = float(detached[finite_mask].abs().max().item())
            reports.append(
                {
                    "parameter_name": parameter_name,
                    "shape": tuple(int(value) for value in detached.shape),
                    "dtype": str(detached.dtype),
                    "nan_count": int(torch.isnan(detached).sum().item()),
                    "posinf_count": int((inf_mask & (detached > 0)).sum().item()),
                    "neginf_count": int((inf_mask & (detached < 0)).sum().item()),
                    "finite_count": finite_count,
                    "finite_abs_max": self._jsonable_float(finite_abs_max),
                }
            )
        return reports

    def _nonfinite_update_payload(
        self,
        *,
        reason: str,
        learning_rate: float,
        training_loss: Optional[float],
        gradient_norm: Optional[float],
        micro_step: Optional[int],
        microbatch_starts: List[Tuple[int, ...]],
    ) -> Dict[str, Any]:
        local_report = {
            "rank": int(self.distributed.rank),
            "reason": reason,
            "loss": self._jsonable_float(training_loss),
            "gradient_norm": self._jsonable_float(gradient_norm),
            "micro_step": micro_step,
            "microbatch_starts": [
                [int(value) for value in starts]
                for starts in microbatch_starts
            ],
            "gradient_reports": self._local_nonfinite_gradient_reports(),
        }
        gathered = self.distributed.all_gather_object(local_report)
        return {
            "attempted_update": int(
                self.state.completed_updates
                + self.state.skipped_nonfinite_updates
                + 1
            ),
            "completed_updates": int(self.state.completed_updates),
            "skipped_nonfinite_updates": int(self.state.skipped_nonfinite_updates),
            "failed_update_attempts": int(self.state.failed_update_attempts),
            "learning_rate": float(learning_rate),
            "nonfinite_reason": reason,
            "rank_reports": gathered,
        }

    def _handle_nonfinite_update(
        self,
        *,
        reason: str,
        learning_rate: float,
        training_loss: Optional[float],
        gradient_norm: Optional[float],
        micro_step: Optional[int],
        microbatch_starts: List[Tuple[int, ...]],
    ) -> Dict[str, Any]:
        payload = self._nonfinite_update_payload(
            reason=reason,
            learning_rate=learning_rate,
            training_loss=training_loss,
            gradient_norm=gradient_norm,
            micro_step=micro_step,
            microbatch_starts=microbatch_starts,
        )
        self.state.failed_update_attempts += 1
        payload["failed_update_attempts"] = int(self.state.failed_update_attempts)
        if self.config.nonfinite_update_policy == "raise":
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "non-finite update detected: "
                + json.dumps(payload, sort_keys=True)
            )
        if self.config.nonfinite_update_policy != "skip":
            raise RuntimeError(
                f"unsupported nonfinite_update_policy: {self.config.nonfinite_update_policy!r}"
            )
        if self.state.skipped_nonfinite_updates >= self.config.max_nonfinite_update_skips:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "non-finite update skip limit exceeded: "
                + json.dumps(payload, sort_keys=True)
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.state.skipped_nonfinite_updates += 1
        payload["skipped_nonfinite_updates"] = int(self.state.skipped_nonfinite_updates)
        self._record("nonfinite_update_skipped", **payload)
        return {
            "completed_updates": float(self.state.completed_updates),
            "training_loss": (
                training_loss if training_loss is not None else float("nan")
            ),
            "learning_rate": learning_rate,
            "gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            "skipped_update": 1.0,
            "skipped_nonfinite_updates": float(self.state.skipped_nonfinite_updates),
            "failed_update_attempts": float(self.state.failed_update_attempts),
            "nonfinite_reason": reason,
            "nonfinite_diagnostics": payload,
        }
    # ^^^ THOG

    def train_one_update(self) -> Dict[str, Any]:
        if self.state.completed_updates >= self.config.max_updates:
            raise RuntimeError("maximum completed updates already reached")
        self.model.train()
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accumulation_steps = self.config.gradient_accumulation_steps
        # vvv THOG
        # Keep starts from the attempted update so a skipped batch is diagnosable.
        microbatch_starts: List[Tuple[int, ...]] = []
        # ^^^ THOG
        for micro_step in range(accumulation_steps):
            batch = self.batch_source.get_batch("train", device=self.device)
            # vvv THOG
            microbatch_starts.append(tuple(int(value) for value in batch.starts))
            # ^^^ THOG
            self._record(
                "microbatch",
                micro_step=micro_step,
                starts=batch.starts,
            )
            synchronize = micro_step == accumulation_steps - 1
            with self.distributed.no_sync_context(
                self.model,
                synchronize=synchronize,
            ):
                with self.autocast_context():
                    _, loss = self.model(batch.inputs, batch.targets)
                    local_finite = loss is not None and bool(torch.isfinite(loss).item())
                    # vvv THOG
                    if not self.distributed.all_true(local_finite):
                        loss_value = (
                            float(loss.detach().to(dtype=torch.float64).item())
                            if loss is not None
                            else None
                        )
                        return self._handle_nonfinite_update(
                            reason="loss",
                            learning_rate=learning_rate,
                            training_loss=loss_value,
                            gradient_norm=None,
                            micro_step=micro_step,
                            microbatch_starts=microbatch_starts,
                        )
                    # ^^^ THOG
                    if loss is None:
                        raise RuntimeError("model did not return a training loss")
                    scaled_loss = loss / accumulation_steps
                total_loss += self.distributed.mean_float(loss.detach())
                self.scaler.scale(scaled_loss).backward()

        self.scaler.unscale_(self.optimizer)
        # vvv THOG
        if not self.distributed.all_true(self._local_gradients_are_finite()):
            return self._handle_nonfinite_update(
                reason="gradient",
                learning_rate=learning_rate,
                training_loss=total_loss / accumulation_steps,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=microbatch_starts,
            )
        # ^^^ THOG

        diagnostics_hook = None
        if hasattr(self, "_before_optimizer_step"):
            diagnostics_hook = self._before_optimizer_step
        if diagnostics_hook is not None:
            diagnostics_hook()

        gradient_norm: Optional[float] = None
        if self.config.grad_clip > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(),
                self.config.grad_clip,
            )
            gradient_norm = self.distributed.mean_float(norm.detach())
            if not math.isfinite(gradient_norm):
                # vvv THOG
                return self._handle_nonfinite_update(
                    reason="gradient_norm",
                    learning_rate=learning_rate,
                    training_loss=total_loss / accumulation_steps,
                    gradient_norm=gradient_norm,
                    micro_step=None,
                    microbatch_starts=microbatch_starts,
                )
                # ^^^ THOG

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.completed_updates += 1
        self.state.latest_training_loss = total_loss / accumulation_steps
        metrics = {
            "completed_updates": float(self.state.completed_updates),
            "training_loss": self.state.latest_training_loss,
            "learning_rate": learning_rate,
            "gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            # vvv THOG
            "skipped_update": 0.0,
            "skipped_nonfinite_updates": float(self.state.skipped_nonfinite_updates),
            "failed_update_attempts": float(self.state.failed_update_attempts),
            # ^^^ THOG
        }
        self._record("optimizer_step_completed", **metrics)
        return metrics
# ^^^ THOG
