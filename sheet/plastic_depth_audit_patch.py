# vvv THOG
"""Replayable audit records for every PLASTIC FINE count decision."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple

from . import trainer_step as _trainer_step


_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _begin_plastic_depth_inline_update_with_audit(self: Any) -> Optional[Dict[str, Any]]:
    context = _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    if context is not None:
        context["audit_history_before"] = copy.deepcopy(
            self.state.plastic_depth_probe_histories
        )
    return context


def _plastic_depth_inline_probe_request_with_audit(
    self: Any,
    targets: Any,
    context: Dict[str, Any],
):
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    indices = request.sampled_token_indices
    if indices is None:
        positions = tuple(range(int(targets.numel())))
    else:
        positions = tuple(int(value) for value in indices.detach().cpu().tolist())
    context["audit_sampled_token_positions"] = positions
    return request


def _winning_probe_count(decision: Any, current_count: int) -> int:
    passing = []
    for evidence in decision.evidence:
        if not evidence.feasible or not evidence.significant:
            continue
        standardized = evidence.standardized_improvement
        if standardized is None:
            continue
        passing.append(
            (
                float(standardized),
                int(evidence.candidate_count),
            )
        )
    if not passing or bool(decision.brake_active):
        return int(current_count)
    return max(passing, key=lambda item: (item[0], -item[1]))[1]


def _decision_reason(
    *,
    current_count: int,
    winning_probe_count: int,
    committed_count: int,
    brake_active: bool,
) -> str:
    if brake_active:
        return "update_brake"
    if winning_probe_count == current_count:
        return "robust_gate_hold"
    if committed_count != winning_probe_count:
        return "max_step_limited"
    return "winning_probe_committed"


def _evidence_payload(decision: Any) -> Tuple[Dict[str, object], ...]:
    return tuple(
        {
            "candidate_count": int(item.candidate_count),
            "offset": int(item.direction),
            "paired_difference": item.paired_difference,
            "observation_count": int(item.observation_count),
            "median": item.median,
            "mad": item.mad,
            "sigma": item.sigma,
            "standardized_improvement": item.standardized_improvement,
            "significant": bool(item.significant),
            "feasible": bool(item.feasible),
        }
        for item in decision.evidence
    )


def replay_plastic_depth_count_audit(
    audit: Mapping[str, Any],
) -> Dict[str, object]:
    """Recompute winner, bounded commit and reason from one durable audit row."""

    current_count = int(audit["previous_count"])
    max_step = int(audit["max_step"])
    if max_step < 1:
        raise ValueError("audit max_step must be positive")
    brake_active = bool(audit["brake_active"])
    passing = []
    for item in audit["robust_evidence"]:
        if not bool(item["feasible"]) or not bool(item["significant"]):
            continue
        standardized = item["standardized_improvement"]
        if standardized is None:
            continue
        passing.append((float(standardized), int(item["candidate_count"])))
    if not passing or brake_active:
        winning_probe_count = current_count
    else:
        winning_probe_count = max(
            passing,
            key=lambda item: (item[0], -item[1]),
        )[1]
    committed_offset = max(
        -max_step,
        min(max_step, winning_probe_count - current_count),
    )
    committed_count = current_count + committed_offset
    reason = _decision_reason(
        current_count=current_count,
        winning_probe_count=winning_probe_count,
        committed_count=committed_count,
        brake_active=brake_active,
    )
    replay = {
        "winning_probe_count": winning_probe_count,
        "committed_count": committed_count,
        "decision_reason": reason,
    }
    expected = {
        name: audit[name]
        for name in (
            "winning_probe_count",
            "committed_count",
            "decision_reason",
        )
    }
    if replay != expected:
        raise ValueError(
            "PLASTIC FINE audit replay mismatch: "
            f"recorded={expected}, replayed={replay}"
        )
    return replay


def _commit_plastic_depth_inline_update_with_audit(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None:
        return transition
    decision = context.get("decision")
    if decision is None:
        raise RuntimeError("PLASTIC FINE audit lacks a completed robust decision")
    current_count = int(context["current_count"])
    committed_count = int(decision.selected_count)
    winning_probe_count = _winning_probe_count(decision, current_count)
    lattice = self._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC FINE audit lacks its sampling lattice")
    audit: Dict[str, Any] = {
        "phase": "fine",
        "update_number": int(decision.update_number),
        "decision_number": int(lattice.count_decision_number.item()),
        "previous_count": current_count,
        "winning_probe_count": winning_probe_count,
        "committed_count": committed_count,
        "decision_reason": _decision_reason(
            current_count=current_count,
            winning_probe_count=winning_probe_count,
            committed_count=committed_count,
            brake_active=bool(decision.brake_active),
        ),
        "objective": self.config.plastic__layer_count_objective,
        "objective_cost_weight": float(self.config.plastic__layer_count_cost_weight),
        "memory_budget_gib": self.config.plastic__layer_memory_budget_gib,
        "probe_interval": int(self.config.plastic__layer_count_probe_interval),
        "probe_radius": int(context["probe_radius"]),
        "max_step": int(context["max_step"]),
        "update_brake": int(self.config.plastic__layer_count_update_brake),
        "brake_active": bool(decision.brake_active),
        "last_count_change_update": int(
            self.state.plastic_depth_last_count_change_update
        ),
        "decision_candidate_counts": tuple(
            int(value) for value in context["decision_candidate_counts"]
        ),
        "execution_candidate_counts": tuple(
            int(value) for value in context["candidate_counts"]
        ),
        "sampled_token_count": int(context["sampled_token_count"]),
        "sampled_token_positions": tuple(
            int(value) for value in context["audit_sampled_token_positions"]
        ),
        "score_table": tuple(dict(item) for item in context["score_report"]),
        "robust_evidence": _evidence_payload(decision),
        "histories_before": copy.deepcopy(context["audit_history_before"]),
        "histories_after": copy.deepcopy(self.state.plastic_depth_probe_histories),
        "active_public_coordinates_after": tuple(
            float(value)
            for value in lattice.interval_report()["active_public_coordinates"]
        ),
        "transition": copy.deepcopy(transition),
    }
    replay_plastic_depth_count_audit(audit)
    self.distributed.assert_identical_object(
        audit,
        "PLASTIC FINE complete count-decision audit",
    )
    self.state.plastic_depth_count_audit.append(copy.deepcopy(audit))
    self._record("plastic_depth_count_audit", **audit)
    return transition


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = (
    _begin_plastic_depth_inline_update_with_audit
)
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_with_audit
)
_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_audit
)


__all__ = ["replay_plastic_depth_count_audit"]
# ^^^ THOG
