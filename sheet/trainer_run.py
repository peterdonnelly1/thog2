# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainerRunMixin:
    def run(
        self,
        *,
        target_updates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        target = (
            self.config.max_updates
            if target_updates is None
            else target_updates
        )
        if target < self.state.completed_updates or target > self.config.max_updates:
            raise ValueError(
                "target_updates is outside the valid completed-update range"
            )
        history: List[Dict[str, Any]] = []
        if self.state.completed_updates == 0 and self.config.eval_interval > 0:
            history.append({"evaluation_update": 0.0, **self.evaluate()})
        while self.state.completed_updates < target:
            metrics = self.train_one_update()
            history.append(metrics)
            # vvv THOG skipped attempts do not trigger completed-update evaluation or checkpoint cadence
            if metrics.get("skipped_update", 0.0):
                continue
            # ^^^ THOG
            if (
                self.config.eval_interval > 0
                and self.state.completed_updates % self.config.eval_interval == 0
            ):
                history.append(
                    {
                        "evaluation_update": float(
                            self.state.completed_updates
                        ),
                        **self.evaluate(),
                    }
                )
            if (
                self.config.checkpoint_interval > 0
                and self.state.completed_updates
                % self.config.checkpoint_interval
                == 0
            ):
                self.save_checkpoint(
                    Path(self.config.out_dir) / "ckpt.pt"
                )
        return history
# ^^^ THOG
