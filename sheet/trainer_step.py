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
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite {split} evaluation loss"
                    )
                losses.append(float(loss.detach()))
            results[split] = sum(losses) / len(losses)
        self.state.latest_validation_loss = results["val"]
        self.state.best_validation_loss = min(
            self.state.best_validation_loss,
            results["val"],
        )
        self._record("evaluation_completed", losses=results)
        self.model.train(was_training)
        return results

    def train_one_update(self) -> Dict[str, float]:
        if self.state.completed_updates >= self.config.max_updates:
            raise RuntimeError("maximum completed updates already reached")
        self.model.train()
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro_step in range(self.config.gradient_accumulation_steps):
            batch = self.batch_source.get_batch("train", device=self.device)
            self._record(
                "microbatch",
                micro_step=micro_step,
                starts=batch.starts,
            )
            with self.autocast_context():
                _, loss = self.model(batch.inputs, batch.targets)
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                scaled_loss = loss / self.config.gradient_accumulation_steps
            total_loss += float(loss.detach())
            self.scaler.scale(scaled_loss).backward()

        gradient_norm: Optional[float] = None
        if self.config.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
            gradient_norm = float(norm.detach())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.completed_updates += 1
        self.state.latest_training_loss = (
            total_loss / self.config.gradient_accumulation_steps
        )
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
