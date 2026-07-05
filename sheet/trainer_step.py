# vvv THOG
from __future__ import annotations

import math
from typing import Dict, Optional

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

    def _local_gradients_are_finite(self) -> bool:
        for parameter in self.raw_model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                return False
        return True

    def train_one_update(self) -> Dict[str, float]:
        if self.state.completed_updates >= self.config.max_updates:
            raise RuntimeError("maximum completed updates already reached")
        self.model.train()
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accumulation_steps = self.config.gradient_accumulation_steps
        for micro_step in range(accumulation_steps):
            batch = self.batch_source.get_batch("train", device=self.device)
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
                    self.distributed.require_all_true(
                        local_finite,
                        "non-finite training loss on at least one rank",
                    )
                    if loss is None:
                        raise RuntimeError("model did not return a training loss")
                    scaled_loss = loss / accumulation_steps
                total_loss += self.distributed.mean_float(loss.detach())
                self.scaler.scale(scaled_loss).backward()

        self.scaler.unscale_(self.optimizer)
        self.distributed.require_all_true(
            self._local_gradients_are_finite(),
            "non-finite gradient on at least one rank",
        )

        diagnostics_hook = getattr(self, "_before_optimizer_step", None)
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
                raise FloatingPointError("non-finite gradient norm")

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
        }
        self._record("optimizer_step_completed", **metrics)
        return metrics
# ^^^ THOG
