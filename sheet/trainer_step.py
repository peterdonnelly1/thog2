# vvv THOG
from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .batch_source import DeterministicBatchSource
from .layer_dropout import LayerDropoutConfig                                                                                                               # <<< THOG deterministic stratified layer selection
# vvv THOG PLASTIC DEPTH discrete count controller interprets fixed-example loss evidence outside AdamW
from .plastic_depth import (
    PlasticDepthCandidateMeasurement,
    choose_plastic_depth_candidate,
)
# ^^^ THOG


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

    # vvv THOG PLASTIC DEPTH count decisions use fixed examples, preserve AdamW tensors, and occur only at optimiser boundaries
    def _plastic_depth_lattice(self):
        if not self.config.plastic__enabled:
            return None
        trajectory = self.raw_model.trajectory
        lattice = trajectory.plastic_sampling
        if lattice is None:
            raise RuntimeError("enabled PLASTIC DEPTH model has no sampling lattice")
        return lattice

    def _plastic_depth_controller_batches(self):
        source = DeterministicBatchSource(
            self.batch_source.train_tokens,
            self.batch_source.validation_tokens,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            data_seed=self.config.data_seed + 1_000_003,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
            trace_limit=0,
        )
        batch_count = max(self.config.eval_batches, self.config.gradient_accumulation_steps)
        return tuple(
            source.get_batch("val", device=self.device)
            for _ in range(batch_count)
        )

    def _plastic_depth_candidate_loss(self, active_layers: int, batches) -> float:
        self.raw_model.set_plastic_depth_active_layer_count(active_layers)
        losses = []
        with torch.no_grad():
            for batch in batches[: self.config.eval_batches]:
                with self.autocast_context():
                    _, loss = self.model(batch.inputs, batch.targets)
                if loss is None or not bool(torch.isfinite(loss).item()):
                    return float("inf")
                losses.append(self.distributed.mean_float(loss.detach()))
        return sum(losses) / len(losses)

    def _plastic_depth_device_synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _plastic_depth_candidate_training_time(self, active_layers: int, batches) -> float:
        self.raw_model.set_plastic_depth_active_layer_count(active_layers)
        self.optimizer.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.train(True)
        retained_materializations_active = self._begin_optimizer_update_materializations()
        try:
            self._plastic_depth_device_synchronize()
            started = time.perf_counter()
            for micro_step, batch in enumerate(
                batches[: self.config.gradient_accumulation_steps]
            ):
                synchronize = micro_step == self.config.gradient_accumulation_steps - 1
                with self.distributed.no_sync_context(
                    self.model,
                    synchronize=synchronize,
                ):
                    with self.autocast_context():
                        _, loss = self.model(batch.inputs, batch.targets)
                        if loss is None:
                            raise RuntimeError("model did not return a PLASTIC DEPTH timing-probe loss")
                        scaled_loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
            if retained_materializations_active:
                self._finalize_optimizer_update_materializations()
                retained_materializations_active = False
            self._plastic_depth_device_synchronize()
            forward_backward_seconds = time.perf_counter() - started
        finally:
            if retained_materializations_active:
                self._end_optimizer_update_materializations()
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)
        local_time = torch.tensor(
            forward_backward_seconds,
            dtype=torch.float64,
            device=self.device,
        )
        worst_rank_forward_backward = self.distributed.max_float(local_time)
        lattice = self._plastic_depth_lattice()
        if lattice is None:
            raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent during timing probe")
        optimizer_step_seconds = float(lattice.optimizer_step_time_ema.item())
        if not math.isfinite(optimizer_step_seconds):
            optimizer_step_seconds = 0.0
        return worst_rank_forward_backward + optimizer_step_seconds

    def _plastic_depth_candidate_memory(self, active_layers: int, batch) -> Tuple[Optional[float], Optional[float]]:
        if self.device.type != "cuda":
            return None, None
        self.raw_model.set_plastic_depth_active_layer_count(active_layers)
        was_training = self.model.training
        self.model.train(True)

        def run_probe() -> None:
            self.optimizer.zero_grad(set_to_none=True)
            retained_materializations_active = self._begin_optimizer_update_materializations()
            try:
                with self.autocast_context():
                    _, loss = self.model(batch.inputs, batch.targets)
                if loss is None:
                    raise RuntimeError("model did not return a PLASTIC DEPTH memory-probe loss")
                self.scaler.scale(loss).backward()
                if retained_materializations_active:
                    self._finalize_optimizer_update_materializations()
                    retained_materializations_active = False
                self._plastic_depth_device_synchronize()
            finally:
                if retained_materializations_active:
                    self._end_optimizer_update_materializations()
                self.optimizer.zero_grad(set_to_none=True)

        try:
            run_probe()
            torch.cuda.reset_peak_memory_stats(self.device)
            run_probe()
            local_allocated = torch.tensor(
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3),
                dtype=torch.float64,
                device=self.device,
            )
            local_reserved = torch.tensor(
                torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 3),
                dtype=torch.float64,
                device=self.device,
            )
            allocated = self.distributed.max_float(local_allocated)
            reserved = self.distributed.max_float(local_reserved)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)
        return allocated, reserved

    def _prepare_plastic_depth_for_update(self) -> None:
        if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:
            return
        lattice = self._plastic_depth_lattice()
        if lattice is None:
            raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent")
        completed_updates = self.state.completed_updates
        hold_updates = self.config.plastic__layer_count_hold_updates
        if completed_updates == 0 or completed_updates % hold_updates != 0:
            return
        if int(lattice.last_count_decision_update.item()) == completed_updates:
            return
        current = int(lattice.active_layer_count.item())
        reference_time = float(lattice.reference_training_time.item())
        if (
            self.config.plastic__layer_count_objective == "relative_training_wall_time"
            and not math.isfinite(reference_time)
        ):
            return
        candidates = tuple(
            value
            for value in (current - 1, current, current + 1)
            if 1 <= value <= lattice.maximum_layers
        )
        batches = self._plastic_depth_controller_batches()
        was_training = self.model.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        self.model.eval()
        measurements = []
        selected_installed = False
        try:
            for candidate in candidates:
                validation_loss = self._plastic_depth_candidate_loss(candidate, batches)
                observed_time = float(lattice.training_time_ema[candidate].item())
                if (
                    self.config.plastic__layer_count_objective == "relative_training_wall_time"
                    and not math.isfinite(observed_time)
                ):
                    self._plastic_depth_candidate_training_time(candidate, batches)
                    observed_time = self._plastic_depth_candidate_training_time(candidate, batches)
                peak_allocated_gib = None
                peak_reserved_gib = None
                if self.config.plastic__layer_count_objective == "memory_budget":
                    peak_allocated_gib, peak_reserved_gib = self._plastic_depth_candidate_memory(
                        candidate, batches[0]
                    )
                    if peak_allocated_gib is not None and peak_reserved_gib is not None:
                        lattice.record_memory_probe(
                            candidate,
                            peak_allocated_gib=peak_allocated_gib,
                            peak_reserved_gib=peak_reserved_gib,
                        )
                measurements.append(
                    PlasticDepthCandidateMeasurement(
                        active_layers=candidate,
                        validation_loss=validation_loss,
                        training_time=observed_time if math.isfinite(observed_time) else None,
                        peak_allocated_gib=peak_allocated_gib,
                        peak_reserved_gib=peak_reserved_gib,
                    )
                )
            reference_time = float(lattice.reference_training_time.item())
            selected, score_report = choose_plastic_depth_candidate(
                measurements,
                objective=self.config.plastic__layer_count_objective,
                maximum_layers=lattice.maximum_layers,
                cost_weight=float(self.config.plastic__layer_count_cost_weight),
                reference_training_time=reference_time if math.isfinite(reference_time) else None,
                memory_budget_gib=self.config.plastic__layer_memory_budget_gib,
            )
            self.distributed.assert_identical_object(
                selected.active_layers,
                "PLASTIC DEPTH selected layer count",
            )
            self.raw_model.set_plastic_depth_active_layer_count(selected.active_layers)
            selected_installed = True
            lattice.last_count_decision_update.fill_(completed_updates)
            lattice.count_decision_number.add_(1)
            self._record(
                "plastic_depth_count_decision",
                previous_active_layers=current,
                selected_active_layers=selected.active_layers,
                candidates=score_report,
                objective=self.config.plastic__layer_count_objective,
                public_coordinates=lattice.interval_report()["active_public_coordinates"],
            )
        finally:
            if not selected_installed:
                self.raw_model.set_plastic_depth_active_layer_count(current)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)
    # ^^^ THOG

    # vvv THOG one selection per resample bucket; all-active runs return before sampler construction
    def _prepare_layer_dropout_for_update(self) -> None:
        if not self.config.layer_dropout_enabled:
            return
        bucket = self.state.completed_updates // self.config.layer_dropout_resample_steps
        if getattr(self, "_layer_dropout_resample_bucket", None) == bucket:
            return
        plan = LayerDropoutConfig(
            n_layer=self.config.n_layer,
            stratum_size=int(self.config.layer_dropout_stratum_size),
            active_per_stratum=int(self.config.layer_dropout_active_per_stratum),
            resample_steps=self.config.layer_dropout_resample_steps,
            seed=self.config.model_seed,
        )
        active_layer_indices = plan.active_layer_indices(self.state.completed_updates)
        if active_layer_indices is None:
            raise RuntimeError("enabled layer dropout unexpectedly resolved to the all-active path")
        self.raw_model.set_active_layer_indices(active_layer_indices)
        self._layer_dropout_resample_bucket = bucket
        self._record(
            "layer_dropout_resampled",
            resample_bucket=bucket,
            n_active_layers=len(active_layer_indices),
            stratum_size=plan.stratum_size,
            active_per_stratum=plan.active_per_stratum,
            active_layer_indices=active_layer_indices,
        )
    # ^^^ THOG

    # vvv THOG bounded non-finite update recovery with correct pre/post-unscale cleanup
    @staticmethod
    def _jsonable_float(value: Optional[float]) -> Optional[float | str]:
        if value is None:
            return None
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"

    def _local_gradients_are_finite(self) -> bool:
        for parameter in self.raw_model.parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all().item()
            ):
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
            finite_abs_max = (
                float(detached[finite_mask].abs().max().item())
                if finite_count > 0
                else None
            )
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

    def _cleanup_failed_update(self, *, scaler_unscaled: bool) -> None:
        if scaler_unscaled:
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

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
        return {
            "attempted_update": int(
                self.state.completed_updates
                + self.state.skipped_nonfinite_updates
                + 1
            ),
            "completed_updates": int(self.state.completed_updates),
            "skipped_nonfinite_updates": int(
                self.state.skipped_nonfinite_updates
            ),
            "failed_update_attempts": int(self.state.failed_update_attempts),
            "learning_rate": float(learning_rate),
            "nonfinite_reason": reason,
            "rank_reports": self.distributed.all_gather_object(local_report),
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
        scaler_unscaled: bool,
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
        payload["failed_update_attempts"] = int(
            self.state.failed_update_attempts
        )
        if self.config.nonfinite_update_policy == "raise":
            self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
            headline = {
                "loss": "non-finite training loss on at least one rank",
                "gradient": "non-finite gradient on at least one rank",
                "gradient_norm": "non-finite gradient norm",
            }.get(reason, "non-finite update detected")
            raise FloatingPointError(
                headline + ": " + json.dumps(payload, sort_keys=True)
            )
        if self.config.nonfinite_update_policy != "skip":
            self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
            raise RuntimeError(
                "unsupported nonfinite_update_policy: "
                f"{self.config.nonfinite_update_policy!r}"
            )
        if (
            self.state.skipped_nonfinite_updates
            >= self.config.max_nonfinite_update_skips
        ):
            self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
            raise FloatingPointError(
                "non-finite update skip limit exceeded: "
                + json.dumps(payload, sort_keys=True)
            )
        self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
        self.state.skipped_nonfinite_updates += 1
        payload["skipped_nonfinite_updates"] = int(
            self.state.skipped_nonfinite_updates
        )
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
            "skipped_nonfinite_updates": float(
                self.state.skipped_nonfinite_updates
            ),
            "failed_update_attempts": float(
                self.state.failed_update_attempts
            ),
            "nonfinite_reason": reason,
            "nonfinite_diagnostics": payload,
        }
    # ^^^ THOG

    # vvv THOG fast_discard=false retains detached operational tensors and projects their accumulated gradients once per update
    def _begin_optimizer_update_materializations(self) -> bool:
        begin = getattr(self.raw_model, "begin_optimizer_update", None)
        if not callable(begin):
            return False
        return bool(begin())

    def _finalize_optimizer_update_materializations(self) -> None:
        finalize = getattr(self.raw_model, "finalize_optimizer_update", None)
        if not callable(finalize):
            raise RuntimeError(
                "model began update-retained materialisations without a finalize method"
            )
        projected_parameters = tuple(finalize())
        self.distributed.mean_gradients_(projected_parameters)

    def _end_optimizer_update_materializations(self) -> None:
        end = getattr(self.raw_model, "end_optimizer_update", None)
        if not callable(end):
            raise RuntimeError(
                "model began update-retained materialisations without an end method"
            )
        end()
    # ^^^ THOG

    def train_one_update(self) -> Dict[str, Any]:
        if self.state.completed_updates >= self.config.max_updates:
            raise RuntimeError("maximum completed updates already reached")
        # vvv THOG PLASTIC DEPTH count decisions are excluded from normal training-only timing and precede the fixed microstep count
        self._prepare_plastic_depth_for_update()
        self.model.train()
        self._prepare_layer_dropout_for_update()                                                                                                            # <<< THOG one active nominal set for the complete optimizer update
        # ^^^ THOG
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accumulation_steps = self.config.gradient_accumulation_steps
        # vvv THOG PLASTIC DEPTH timing excludes data acquisition while preserving the exact disabled execution path
        prepared_batches = (
            tuple(
                self.batch_source.get_batch("train", device=self.device)
                for _ in range(accumulation_steps)
            )
            if self.config.plastic__enabled
            else None
        )
        plastic_training_started = time.perf_counter() if self.config.plastic__enabled else None
        # ^^^ THOG
        microbatch_starts: List[Tuple[int, ...]] = []
        retained_materializations_active = self._begin_optimizer_update_materializations()                                                                  # <<< THOG activate update-lifetime materialisations only for fast_discard=false sheet models
        try:
            for micro_step in range(accumulation_steps):
                # batch = self.batch_source.get_batch("train", device=self.device)
                batch = (
                    prepared_batches[micro_step]
                    if prepared_batches is not None
                    else self.batch_source.get_batch("train", device=self.device)
                )
                microbatch_starts.append(
                    tuple(int(value) for value in batch.starts)
                )
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
                        local_finite = loss is not None and bool(
                            torch.isfinite(loss).item()
                        )
                        if not self.distributed.all_true(local_finite):
                            loss_value = (
                                float(
                                    loss.detach()
                                    .to(dtype=torch.float64)
                                    .item()
                                )
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
                                scaler_unscaled=False,
                            )
                        if loss is None:
                            raise RuntimeError(
                                "model did not return a training loss"
                            )
                        scaled_loss = loss / accumulation_steps
                    total_loss += self.distributed.mean_float(loss.detach())
                    self.scaler.scale(scaled_loss).backward()
            if retained_materializations_active:
                self._finalize_optimizer_update_materializations()                                                                                           # <<< THOG project scaled operational gradients before the standard scaler unscale
                retained_materializations_active = False
        finally:
            if retained_materializations_active:
                self._end_optimizer_update_materializations()                                                                                                # <<< THOG discard retained tensors on any failed or interrupted update

        self.scaler.unscale_(self.optimizer)
        if not self.distributed.all_true(
            self._local_gradients_are_finite()
        ):
            return self._handle_nonfinite_update(
                reason="gradient",
                learning_rate=learning_rate,
                training_loss=total_loss / accumulation_steps,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=microbatch_starts,
                scaler_unscaled=True,
            )

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
                return self._handle_nonfinite_update(
                    reason="gradient_norm",
                    learning_rate=learning_rate,
                    training_loss=total_loss / accumulation_steps,
                    gradient_norm=gradient_norm,
                    micro_step=None,
                    microbatch_starts=microbatch_starts,
                    scaler_unscaled=True,
                )

        # vvv THOG capture PLASTIC DEPTH lattice gradient health before the optimiser clears gradients
        plastic_geometry_gradient_norm = 0.0
        if self.config.plastic__enabled:
            lattice = self._plastic_depth_lattice()
            if lattice is None:
                raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent before optimiser step")
            geometry_gradient = lattice.raw_intervals.grad
            if geometry_gradient is not None:
                plastic_geometry_gradient_norm = float(geometry_gradient.detach().norm().item())
        # ^^^ THOG
        # vvv THOG isolate the count-independent optimiser component for relative training-wall-time candidate probes
        if self.config.plastic__enabled:
            self._plastic_depth_device_synchronize()
        plastic_optimizer_started = time.perf_counter() if self.config.plastic__enabled else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        plastic_optimizer_step_seconds_local = None
        if self.config.plastic__enabled:
            self._plastic_depth_device_synchronize()
            if plastic_optimizer_started is None:
                raise RuntimeError("PLASTIC DEPTH optimiser timer was not started")
            plastic_optimizer_step_seconds_local = time.perf_counter() - plastic_optimizer_started
        # ^^^ THOG
        self.optimizer.zero_grad(set_to_none=True)
        self.state.completed_updates += 1
        self.state.latest_training_loss = total_loss / accumulation_steps
        # vvv THOG record normal update time and current PLASTIC DEPTH geometry health without including controller probes
        plastic_metrics: Dict[str, Any] = {}
        if self.config.plastic__enabled:
            lattice = self._plastic_depth_lattice()
            if lattice is None:
                raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent after update")
            if plastic_training_started is None:
                raise RuntimeError("PLASTIC DEPTH training timer was not started")
            self._plastic_depth_device_synchronize()
            local_training_only_seconds = time.perf_counter() - plastic_training_started
            training_only_seconds = self.distributed.max_float(
                torch.tensor(
                    local_training_only_seconds,
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            if plastic_optimizer_step_seconds_local is None:
                raise RuntimeError("PLASTIC DEPTH optimiser-step duration was not captured")
            optimizer_step_seconds = self.distributed.max_float(
                torch.tensor(
                    plastic_optimizer_step_seconds_local,
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            active_layers = int(lattice.active_layer_count.item())
            update_reference = (
                active_layers == self.config.plastic__initial_active_layers
                and self.state.completed_updates >= self.config.warmup_updates
            )
            lattice.record_training_time(
                active_layers,
                training_only_seconds,
                update_reference=update_reference,
            )
            lattice.record_optimizer_step_time(optimizer_step_seconds)
            interval_report = lattice.interval_report()
            plastic_metrics = {
                "plastic_active_layers": float(active_layers),
                "plastic_training_only_seconds": float(training_only_seconds),
                "plastic_optimizer_step_seconds": float(optimizer_step_seconds),
                "plastic_geometry_gradient_norm": plastic_geometry_gradient_norm,
                "plastic_public_coordinates": interval_report["active_public_coordinates"],
                "plastic_minimum_interval": interval_report["minimum_interval"],
                "plastic_maximum_interval": interval_report["maximum_interval"],
                "plastic_mean_absolute_movement": interval_report["mean_absolute_movement"],
            }
        # ^^^ THOG
        metrics = {
            "completed_updates": float(self.state.completed_updates),
            "training_loss": self.state.latest_training_loss,
            "learning_rate": learning_rate,
            "gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            "skipped_update": 0.0,
            "skipped_nonfinite_updates": float(
                self.state.skipped_nonfinite_updates
            ),
            "failed_update_attempts": float(
                self.state.failed_update_attempts
            ),
        }
        metrics.update(plastic_metrics)
        self._record("optimizer_step_completed", **metrics)
        return metrics
# ^^^ THOG
