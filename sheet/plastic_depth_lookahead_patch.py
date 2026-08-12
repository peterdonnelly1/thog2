# vvv THOG
"""Full-radius PLASTIC DEPTH FINE probing and bounded count movement.

Every valid integer count in the inclusive configured radius is measured on
one shared first-microstep chain.  The robust winner records the desired probe
count, while max_step independently limits the committed prefix transition.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from . import plastic_depth_controller as _controller
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from .plastic_depth import PlasticDepthCandidateMeasurement, choose_plastic_depth_candidate
from .plastic_depth_cuda import PlasticDepthCudaAllocatorReserve
from .plastic_depth_inline import PlasticDepthInlineProbeRequest
from .plastic_depth_optimizer import (
    commit_plastic_depth_adamw_transition,
    prepare_plastic_depth_adamw_transition,
)


_RADIUS_ENV = "THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS"
_MAX_STEP_ENV = "THOG2_PLASTIC_LAYER_COUNT_MAX_STEP"


def _positive_int(value: Any, *, name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return resolved


def _config_probe_radius(config: Any) -> int:
    value = getattr(config, "plastic__layer_count_probe_radius", os.environ.get(_RADIUS_ENV, 1))
    return _positive_int(value, name="plastic__layer_count_probe_radius")


def _config_max_step(config: Any) -> int:
    value = getattr(config, "plastic__layer_count__max_allowable_layer_change", os.environ.get(_MAX_STEP_ENV, 1))
    return _positive_int(value, name="plastic__layer_count__max_allowable_layer_change")


# vvv THOG expose exact-radius controls through the existing CLI without changing canonical dataclasses yet
_ORIGINAL_ARGPARSE_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args


def _strip_plastic_lookahead_args(args: Optional[Sequence[str]]) -> Tuple[list[str], Optional[int], Optional[int]]:
    source = list(sys.argv[1:] if args is None else args)
    stripped: list[str] = []
    probe_radius: Optional[int] = None
    max_step: Optional[int] = None
    index = 0
    while index < len(source):
        argument = source[index]
        if argument == "--plastic-layer-count-probe-radius":
            if index + 1 >= len(source):
                raise ValueError("--plastic-layer-count-probe-radius requires a value")
            probe_radius = _positive_int(source[index + 1], name="plastic__layer_count_probe_radius")
            index += 2
            continue
        if argument.startswith("--plastic-layer-count-probe-radius="):
            probe_radius = _positive_int(argument.split("=", 1)[1], name="plastic__layer_count_probe_radius")
            index += 1
            continue
        if argument == "--plastic-layer-count-max-step":
            if index + 1 >= len(source):
                raise ValueError("--plastic-layer-count-max-step requires a value")
            max_step = _positive_int(source[index + 1], name="plastic__layer_count__max_allowable_layer_change")
            index += 2
            continue
        if argument.startswith("--plastic-layer-count-max-step="):
            max_step = _positive_int(argument.split("=", 1)[1], name="plastic__layer_count__max_allowable_layer_change")
            index += 1
            continue
        stripped.append(argument)
        index += 1
    return stripped, probe_radius, max_step


def _parse_known_args_with_plastic_lookahead(self: argparse.ArgumentParser, args=None, namespace=None):
    stripped, probe_radius, max_step = _strip_plastic_lookahead_args(args)
    if probe_radius is not None:
        os.environ[_RADIUS_ENV] = str(probe_radius)
    if max_step is not None:
        os.environ[_MAX_STEP_ENV] = str(max_step)
    parsed, extras = _ORIGINAL_ARGPARSE_PARSE_KNOWN_ARGS(self, stripped, namespace)
    parsed_probe_radius = getattr(parsed, "plastic__layer_count_probe_radius", None)
    parsed_max_step = getattr(parsed, "plastic__layer_count__max_allowable_layer_change", None)
    resolved_probe_radius = probe_radius if probe_radius is not None else parsed_probe_radius
    resolved_max_step = max_step if max_step is not None else parsed_max_step
    if resolved_probe_radius is None:
        resolved_probe_radius = os.environ.get(_RADIUS_ENV, 1)
    if resolved_max_step is None:
        resolved_max_step = os.environ.get(_MAX_STEP_ENV, 1)
    setattr(parsed, "plastic__layer_count_probe_radius", _positive_int(resolved_probe_radius, name="plastic__layer_count_probe_radius"))
    setattr(parsed, "plastic__layer_count__max_allowable_layer_change", _positive_int(resolved_max_step, name="plastic__layer_count__max_allowable_layer_change"))
    return parsed, extras


argparse.ArgumentParser.parse_known_args = _parse_known_args_with_plastic_lookahead
# ^^^ THOG


def _history_key(current_count: int, offset: int) -> str:
    if offset == 0:
        raise ValueError("PLASTIC DEPTH history offset must be non-zero")
    return f"{current_count}:{offset:+d}"


def _finite_score_by_count(score_report: Sequence[Mapping[str, object]]) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for item in score_report:
        count = int(item["active_layers"])
        feasible = bool(item.get("feasible", False))
        score = float(item.get("score", float("inf")))
        if feasible and math.isfinite(score):
            result[count] = score
    return result


def _robust_scale(values: Sequence[float], current_difference: float) -> Tuple[float, float, float]:
    median = float(__import__("statistics").median(values))
    absolute_deviations = tuple(abs(value - median) for value in values)
    mad = float(__import__("statistics").median(absolute_deviations))
    scale_floor = _controller.PLASTIC_DEPTH_MAD_SIGMA_FLOOR * max(1.0, abs(median), abs(current_difference))
    sigma = max(_controller.PLASTIC_DEPTH_MAD_SCALE * mad, scale_floor)
    return median, mad, sigma


def choose_plastic_depth_count_with_exact_radius(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    minimum_observations: int,
    noise_lambda: float,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    max_step: int = 1,
) -> _controller.PlasticDepthRobustCountDecision:
    """Choose one bounded step from exact-offset score probes."""

    if noise_window < 1:
        raise ValueError("noise_window must be at least 1")
    if minimum_observations < 1 or minimum_observations > noise_window:
        raise ValueError("minimum_observations must lie in [1, noise_window]")
    if not math.isfinite(noise_lambda) or noise_lambda < 0.0:
        raise ValueError("noise_lambda must be finite and non-negative")
    if update_number < 1:
        raise ValueError("update_number must be positive")
    if update_brake < 0:
        raise ValueError("update_brake must be non-negative")
    max_step = _positive_int(max_step, name="plastic__layer_count__max_allowable_layer_change")

    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(current_count)
    updated_histories: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved_values = tuple(float(value) for value in values[-noise_window:])
        if not all(math.isfinite(value) for value in resolved_values):
            raise ValueError(f"PLASTIC DEPTH paired-score history {key!r} contains a non-finite value")
        updated_histories[str(key)] = resolved_values
    brake_active = update_brake > 0 and last_count_change_update >= 0 and update_number - last_count_change_update < update_brake
    evidence = []
    passing = []
    candidate_offsets = tuple(sorted(count - current_count for count in score_by_count if count != current_count))
    for offset in candidate_offsets:
        candidate_count = current_count + offset
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(current_count, offset)
        values = list(updated_histories.get(key, ()))
        if feasible:
            paired_difference = float(candidate_score - current_score)
            values.append(paired_difference)
            values = values[-noise_window:]
            updated_histories[key] = tuple(values)
            median, mad, sigma = _robust_scale(values, paired_difference)
            standardized = -median / sigma
            favourable_count = sum(value < 0.0 for value in values)
            significant = (
                len(values) >= minimum_observations
                and median < -noise_lambda * sigma
                and paired_difference < 0.0
                and favourable_count * 2 > len(values)
            )
            if significant and not brake_active:
                passing.append((standardized, offset, candidate_count))
        evidence.append(
            _controller.PlasticDepthPairedDirectionEvidence(
                candidate_count=candidate_count,
                direction=offset,
                paired_difference=paired_difference,
                observation_count=len(values),
                median=median,
                mad=mad,
                sigma=sigma,
                standardized_improvement=standardized,
                significant=significant,
                feasible=feasible,
            )
        )

    selected_count = current_count
    if passing:
        _, selected_offset, _ = max(passing, key=lambda item: (item[0], -item[2]))
        step = max(-max_step, min(max_step, selected_offset))
        selected_count = current_count + step
        updated_histories = {}

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=current_count,
        update_number=update_number,
        brake_active=brake_active,
        last_count_change_update=last_count_change_update,
        histories=updated_histories,
        evidence=tuple(evidence),
    )


_controller.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_exact_radius
_trainer_step.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_exact_radius


def _lookahead_counts(current: int, maximum: int, radius: int, max_step: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    del max_step
    lower = max(1, current - radius)
    upper = min(maximum, current + radius)
    decision_counts = tuple(range(lower, upper + 1))
    return decision_counts, decision_counts


_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _begin_plastic_depth_inline_update_with_lookahead(self: Any) -> Optional[Dict[str, Any]]:
    if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:
        return None
    if getattr(self.config, "plastic__runtime_phase", "fine") == "coarse":
        return None
    lattice = self._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent")
    current = lattice.current_active_layers
    radius = _config_probe_radius(self.config)
    max_step = _config_max_step(self.config)
    decision_candidates, execution_candidates = _lookahead_counts(current, lattice.maximum_layers, radius, max_step)
    if not execution_candidates:
        raise RuntimeError("PLASTIC DEPTH inline probe resolved no candidate counts")

    recoverable_upward_count: Optional[int] = None
    allocator_reserve: Optional[PlasticDepthCudaAllocatorReserve] = None
    upward_preflight_feasible: Optional[bool] = None
    proposed_upward_count = current + radius
    if radius == 1 and self.device.type == "cuda" and proposed_upward_count in execution_candidates:
        allocator_reserve = PlasticDepthCudaAllocatorReserve(
            device=self.device,
            reserve_gib=float(self.config.plastic__layer_count__cuda_allocator_reserve_gib),
        )
        local_preflight_feasible = allocator_reserve.acquire()
        upward_preflight_feasible = self.distributed.all_true(local_preflight_feasible)
        if upward_preflight_feasible:
            recoverable_upward_count = proposed_upward_count
        else:
            # vvv THOG flush only a rank whose reserve acquisition actually OOMed
            allocator_reserve.release(empty_cache=not local_preflight_feasible)
            # ^^^ THOG
            allocator_reserve = None
            execution_candidates = tuple(count for count in execution_candidates if count != proposed_upward_count)
            decision_candidates = tuple(count for count in decision_candidates if count != proposed_upward_count)

    setter = getattr(self.raw_model, "set_plastic_depth_update_layer_count", None)
    if not callable(setter):
        raise RuntimeError("PLASTIC DEPTH training model lacks update-prefix control")
    setter(max(execution_candidates))
    context: Dict[str, Any] = {
        "current_count": current,
        "candidate_counts": execution_candidates,
        "decision_candidate_counts": decision_candidates,
        "probe_radius": radius,
        "max_step": max_step,
        "selected_count": None,
        "score_report": None,
        "paired_evidence": None,
        "score_evidence": None,
        "decision": None,
        "sampled_token_count": None,
        "cuda_allocator_reserve": allocator_reserve,
        "recoverable_upward_count": recoverable_upward_count,
        "upward_preflight_feasible": upward_preflight_feasible,
        "upward_candidate_feasible": upward_preflight_feasible,
    }
    self._plastic_depth_inline_update_context = context
    return context


def _plastic_depth_inline_probe_request_with_lookahead(self: Any, targets: torch.Tensor, context: Dict[str, Any]) -> PlasticDepthInlineProbeRequest:
    lattice = self._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent")
    sampled_token_indices = self._plastic_depth_sampled_token_indices(targets)
    context["sampled_token_count"] = int(sampled_token_indices.numel())

    def select(candidates: Tuple[Tuple[int, torch.Tensor], ...]) -> int:
        actual_counts = tuple(count for count, _ in candidates)
        expected_counts = tuple(context["candidate_counts"])
        recoverable_upward_count = context.get("recoverable_upward_count")
        upward_was_rejected = (
            recoverable_upward_count is not None
            and context.get("upward_candidate_feasible") is False
            and actual_counts == tuple(count for count in expected_counts if count != recoverable_upward_count)
        )
        if actual_counts != expected_counts and not upward_was_rejected:
            raise RuntimeError("PLASTIC DEPTH inline model returned unexpected candidates")
        measurements = []
        for count, local_loss in candidates:
            paired_loss = self.distributed.mean_float(local_loss)
            observed_time = float(lattice.training_time_ema[count].item())
            peak_allocated = float(lattice.peak_allocated_gib[count].item())
            peak_reserved = float(lattice.peak_reserved_gib[count].item())
            measurements.append(
                PlasticDepthCandidateMeasurement(
                    active_layers=count,
                    validation_loss=paired_loss,
                    training_time=observed_time if math.isfinite(observed_time) else None,
                    peak_allocated_gib=peak_allocated if math.isfinite(peak_allocated) else None,
                    peak_reserved_gib=peak_reserved if math.isfinite(peak_reserved) else None,
                )
            )
        reference_time = float(lattice.reference_training_time.item())
        try:
            _selected, score_report_all = choose_plastic_depth_candidate(
                measurements,
                objective=self.config.plastic__layer_count_objective,
                maximum_layers=lattice.maximum_layers,
                cost_weight=float(self.config.plastic__layer_count_cost_weight),
                reference_training_time=reference_time if math.isfinite(reference_time) else None,
                memory_budget_gib=self.config.plastic__layer_count__memory_budget_gib,
            )
        except RuntimeError as error:
            current_count = int(context["current_count"])
            score_report_all = tuple(
                {
                    "active_layers": measurement.active_layers,
                    "validation_loss": measurement.validation_loss,
                    "training_time": measurement.training_time,
                    "peak_allocated_gib": measurement.peak_allocated_gib,
                    "peak_reserved_gib": measurement.peak_reserved_gib,
                    "feasible": measurement.active_layers == current_count,
                    "score": measurement.validation_loss if measurement.active_layers == current_count else float("inf"),
                    "fallback_reason": str(error),
                }
                for measurement in measurements
            )
        decision_counts = set(int(value) for value in context["decision_candidate_counts"])
        score_report = tuple(dict(item) for item in score_report_all if int(item["active_layers"]) in decision_counts)
        decision = choose_plastic_depth_count_with_exact_radius(
            current_count=int(context["current_count"]),
            score_report=score_report,
            histories=self.state.plastic_depth_probe_histories,
            noise_window=self.config.plastic__layer_count_probe__window_size_as_number_of_probes,
            extrapolation_weight=float(self.config.plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence),
            noise_lambda=float(self.config.plastic__layer_count_probe_noise_lambda),
            update_number=int(self.state.completed_updates) + 1,
            last_count_change_update=int(self.state.plastic_depth_last_count_change_update),
            update_brake=self.config.plastic__layer_count_update_brake,
            max_step=int(context["max_step"]),
        )
        selected_count = int(decision.selected_count)
        if selected_count not in actual_counts:
            raise RuntimeError(
                "PLASTIC DEPTH bounded selector returned a count without an execution checkpoint; "
                f"selected={selected_count}, candidates={actual_counts}"
            )
        self.distributed.assert_identical_object(selected_count, "PLASTIC DEPTH inline selected layer count")
        context["selected_count"] = selected_count
        context["score_report"] = score_report
        context["paired_evidence"] = ()
        context["score_evidence"] = decision.report()
        context["decision"] = decision
        return selected_count

    recoverable_upward_count = context.get("recoverable_upward_count")

    def prepare_recoverable_upward() -> None:
        reserve = context.get("cuda_allocator_reserve")
        reserve_release = getattr(reserve, "release", None)
        if callable(reserve_release):
            reserve_release()

    def synchronize_recoverable_upward(local_feasible: bool) -> bool:
        globally_feasible = self.distributed.all_true(bool(local_feasible))
        context["upward_candidate_feasible"] = globally_feasible
        if not local_feasible and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return globally_feasible

    return PlasticDepthInlineProbeRequest(
        candidate_counts=context["candidate_counts"],
        sampled_token_indices=sampled_token_indices,
        selector=select,
        recoverable_upward_count=recoverable_upward_count,
        prepare_recoverable_upward=prepare_recoverable_upward if recoverable_upward_count is not None else None,
        synchronize_recoverable_upward=synchronize_recoverable_upward if recoverable_upward_count is not None else None,
    )


def _commit_plastic_depth_inline_update_with_lookahead(self: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if context is None:
        return {}
    selected = context.get("selected_count")
    decision = context.get("decision")
    if selected is None or decision is None:
        raise RuntimeError("PLASTIC DEPTH inline update completed without a robust count decision")
    selected_count = int(selected)
    current_count = int(context["current_count"])
    if selected_count != int(decision.selected_count):
        raise RuntimeError("PLASTIC DEPTH inline context and robust decision disagree")
    decision_update = int(self.state.completed_updates) + 1
    if decision_update != int(decision.update_number):
        raise RuntimeError("PLASTIC DEPTH robust decision is stale at commit time")
    transition_report: Dict[str, Any] = {}
    if selected_count != current_count:
        model_transition = self.raw_model.prepare_plastic_depth_count_transition(selected_count)
        adamw_transition = prepare_plastic_depth_adamw_transition(self.raw_model, self.optimizer, model_transition)
        transition_report = commit_plastic_depth_adamw_transition(self.raw_model, self.optimizer, adamw_transition)
    lattice = self._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent")
    self.state.plastic_depth_probe_histories = {key: [float(value) for value in values] for key, values in decision.histories.items()}
    if selected_count != current_count:
        self.state.plastic_depth_last_count_change_update = decision_update
    lattice.last_count_decision_update.fill_(decision_update)
    lattice.count_decision_number.add_(1)
    self._record(
        "plastic_depth_count_decision",
        previous_active_layers=current_count,
        selected_active_layers=selected_count,
        candidates=context["score_report"],
        paired_evidence=context["paired_evidence"],
        score_evidence=context["score_evidence"],
        probe_radius=int(context["probe_radius"]),
        max_step=int(context["max_step"]),
        decision_candidate_counts=tuple(int(value) for value in context["decision_candidate_counts"]),
        execution_candidate_counts=tuple(int(value) for value in context["candidate_counts"]),
        brake_active=bool(decision.brake_active),
        last_count_change_update=int(self.state.plastic_depth_last_count_change_update),
        objective=self.config.plastic__layer_count_objective,
        sampled_token_count=context["sampled_token_count"],
        probe_sequence=int(context.get("plastic_probe_sequence", lattice.count_decision_number.item())),
        public_coordinates=lattice.interval_report()["active_public_coordinates"],
        transition=transition_report,
    )
    return transition_report


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_with_lookahead
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _plastic_depth_inline_probe_request_with_lookahead
_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = _commit_plastic_depth_inline_update_with_lookahead


# vvv THOG render exact decision probes and raw loss gain without the old fixed L±1 change_z field
_ORIGINAL_STAGE6_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_STAGE6_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _format_offset(offset: int) -> str:
    if offset == 0:
        return "L"
    return f"L{offset:+d}"


def _format_gain(value: Any) -> str:
    if value is None:
        return f"{'-':>8}"
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>8}"
    return f"{numeric:+8.3f}"


def _format_score_z(value: Any) -> str:
    if value is None:
        return f"{'-':>9}"
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>9}"
    magnitude = abs(numeric)
    if magnitude != 0.0 and (magnitude < 0.01 or magnitude >= 1000.0):
        return f"{numeric:+9.2e}"
    return f"{numeric:+9.2f}"


def _latest_probe_report(trainer: Any) -> Optional[Dict[str, Any]]:
    if not bool(getattr(getattr(trainer, "config", None), "plastic__do_learn_layer_count", False)):
        return None
    for event in reversed(getattr(trainer, "events", ())) :
        if event.name != "plastic_depth_count_decision":
            continue
        payload = event.payload
        if "previous_active_layers" not in payload:
            return None
        current_count = int(payload["previous_active_layers"])
        counts = tuple(int(value) for value in payload.get("decision_candidate_counts", ()))
        if not counts:
            counts = tuple(int(item["active_layers"]) for item in payload.get("candidates", ()))
        losses_by_count: Dict[int, Optional[float]] = {}
        for item in payload.get("candidates", ()):
            try:
                count = int(item["active_layers"])
            except (KeyError, TypeError, ValueError):
                continue
            loss = item.get("validation_loss")
            losses_by_count[count] = None if loss is None else float(loss)
        current_loss = losses_by_count.get(current_count)
        offsets = tuple(count - current_count for count in counts)
        losses = tuple(losses_by_count.get(count) for count in counts)
        edge_offsets = tuple(offset for offset in offsets if offset != 0)
        loss_gain = tuple(
            None if current_loss is None or losses_by_count.get(current_count + offset) is None else float(current_loss - losses_by_count[current_count + offset])
            for offset in edge_offsets
        )
        score_z_by_offset: Dict[int, Optional[float]] = {}
        for item in payload.get("score_evidence", ()):
            try:
                offset = int(item["direction"])
            except (KeyError, TypeError, ValueError):
                continue
            value = item.get("standardized_improvement")
            score_z_by_offset[offset] = None if value is None else float(value)
        objective = str(payload.get("objective", ""))
        score_z = None if objective == "lowest_loss" else tuple(score_z_by_offset.get(offset) for offset in edge_offsets)
        return {
            "offsets": offsets,
            "edge_offsets": edge_offsets,
            "losses": losses,
            "loss_gain": loss_gain,
            "score_z": score_z,
        }
    return None


def _prepare_console_progress_payload_with_lookahead(self: Any, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    values = _ORIGINAL_STAGE6_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event in {"optimizer_progress", "evaluation_completed"}:
        report = _latest_probe_report(self)
        if report is not None:
            values["plastic_probe_offsets"] = report["offsets"]
            values["plastic_probe_edge_offsets"] = report["edge_offsets"]
            values["plastic_probe_losses"] = report["losses"]
            values["plastic_loss_gain"] = report["loss_gain"]
            if report["score_z"] is not None:
                values["plastic_score_z"] = report["score_z"]
    return values


def _format_progress_line_with_lookahead(run_id: str, event: str, payload: Dict[str, Any]) -> str:
    if event not in {"optimizer_progress", "evaluation_completed"} or "plastic_probe_offsets" not in payload:
        return _ORIGINAL_STAGE6_FORMAT_PROGRESS_LINE(run_id, event, payload)
    local_payload = dict(payload)
    offsets = tuple(int(value) for value in local_payload.pop("plastic_probe_offsets"))
    edge_offsets = tuple(int(value) for value in local_payload.pop("plastic_probe_edge_offsets", tuple(value for value in offsets if value != 0)))
    probe_losses = tuple(local_payload.pop("plastic_probe_losses", ()))
    loss_gain = tuple(local_payload.pop("plastic_loss_gain", ()))
    score_z = local_payload.pop("plastic_score_z", None)
    line = _ORIGINAL_STAGE6_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    fields = []
    if probe_losses:
        probe_label = ", ".join(_format_offset(offset) for offset in offsets)
        formatted_losses = ", ".join(_stage6._format_plastic_probe_loss(value) for value in probe_losses)
        fields.append(f"probe_losses [{probe_label}] = [{formatted_losses}]")
    if loss_gain:
        gain_label = ", ".join(_format_offset(offset) for offset in edge_offsets)
        fields.append(f"loss_gain [{gain_label}] = [{', '.join(_format_gain(value) for value in loss_gain)}]")
    if score_z:
        score_label = ", ".join(_format_offset(offset) for offset in edge_offsets)
        fields.append(f"score_z [{score_label}] = [{', '.join(_format_score_z(value) for value in score_z)}]")
    if not fields:
        return line
    inserted = "  ".join(fields)
    marker = "  layer indices = "
    if marker in line:
        return line.replace(marker, f"\t{inserted}\tlayer indices = ", 1)
    return f"{line}\t{inserted}"


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_with_lookahead
_stage6.format_progress_line = _format_progress_line_with_lookahead
# ^^^ THOG

__all__ = ["choose_plastic_depth_count_with_exact_radius"]
# ^^^ THOG
