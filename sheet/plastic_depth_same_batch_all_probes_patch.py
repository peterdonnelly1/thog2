# vvv THOG
"""PLASTIC v0.53 fixed-batch, non-overlapping FINE probe windows.

The established inline path remains authoritative when the new mode is off.
When it is on, decision evidence is collected on one dedicated cached training-
split batch under no-grad, while the ordinary optimizer update continues on its
normal fresh training microbatches.  One cached batch belongs to exactly one
complete evidence window and is retired after both CHANGE and STAY decisions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from . import batch_source as _batch_source
from . import run_config as _run_config
from . import trainer_checkpoint_resume as _checkpoint_resume
from . import trainer_checkpoint_save as _checkpoint_save
from . import trainer_step as _trainer_step
from . import training_config as _training_config
from . import training_model as _training_model
from .plastic_depth_inline import PlasticDepthInlineProbeReport, PlasticDepthInlineProbeRequest


_CONFIG_KEY = "plastic__layer_count__same_batch_all_probes"
_PUBLIC_OPTION = "--plastic__layer_count__same_batch_all_probes"
_PUBLIC_NO_OPTION = "--no-plastic__layer_count__same_batch_all_probes"
_RUNTIME_ENV = "THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES"
_EXPLICIT_ENV = "THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES_EXPLICIT"
_CHECKPOINT_STATE_KEY = "plastic_depth_same_batch_all_probes_state"
_STATE_ATTRIBUTE = "_plastic_depth_same_batch_all_probes_state"
_BATCH_CACHE_ATTRIBUTE = "_plastic_depth_same_batch_all_probes_batch_cache"
_SAMPLE_CACHE_ATTRIBUTE = "_plastic_depth_same_batch_all_probes_sample_cache"
_TRAINING_ONLY_SELECTOR_MARKER = "_thog_same_batch_training_only"
_TRAINING_ONLY_SELECTED_COUNT = "_thog_same_batch_selected_count"
_WINDOW_STATE_VERSION = 1
_BATCH_SEED_OFFSET = 31_337_009
_BATCH_SEED_STRIDE = 1_000_003
_SAMPLE_SEED_OFFSET = 43_211_009
_SAMPLE_SEED_STRIDE = 1_000_033
_RANK_SAMPLE_SEED_STRIDE = 97_409


# vvv THOG accept the exact new public Boolean option without perturbing the established parser/config constructor surface
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _parse_boolean_text(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false; got {value!r}")


def _runtime_enabled() -> bool:
    raw = os.environ.get(_RUNTIME_ENV)
    if raw is None:
        return False
    return _parse_boolean_text(raw, name=_RUNTIME_ENV)


def _set_runtime_enabled(value: bool, *, explicit: bool = False) -> None:
    os.environ[_RUNTIME_ENV] = "true" if bool(value) else "false"
    if explicit:
        os.environ[_EXPLICIT_ENV] = "true" if bool(value) else "false"


def _explicit_requested_mode() -> Optional[bool]:
    raw = os.environ.get(_EXPLICIT_ENV)
    if raw is None:
        return None
    return _parse_boolean_text(raw, name=_EXPLICIT_ENV)


def _strip_public_option(arguments: Optional[Sequence[str]]) -> Tuple[list[str], Optional[bool]]:
    source = list(sys.argv[1:] if arguments is None else arguments)
    remaining: list[str] = []
    requested: Optional[bool] = None
    for argument in source:
        if argument == _PUBLIC_OPTION:
            if requested is False:
                raise SystemExit(f"conflicting {_PUBLIC_OPTION} and {_PUBLIC_NO_OPTION}")
            requested = True
            continue
        if argument == _PUBLIC_NO_OPTION:
            if requested is True:
                raise SystemExit(f"conflicting {_PUBLIC_OPTION} and {_PUBLIC_NO_OPTION}")
            requested = False
            continue
        if argument.startswith(_PUBLIC_OPTION + "=") or argument.startswith(_PUBLIC_NO_OPTION + "="):
            raise SystemExit(f"{_PUBLIC_OPTION} is a Boolean flag and takes no value")
        remaining.append(argument)
    return remaining, requested


def _attach_mode_to_namespace(namespace: Any, requested: Optional[bool]) -> Any:
    if requested is not None:
        _set_runtime_enabled(requested, explicit=True)
    setattr(namespace, _CONFIG_KEY, _runtime_enabled())
    return namespace


def _parse_args_with_same_batch(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_public_option(args)
    parsed = _ORIGINAL_PARSE_ARGS(self, remaining, namespace)
    return _attach_mode_to_namespace(parsed, requested)


def _parse_known_args_with_same_batch(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_public_option(args)
    parsed, extras = _ORIGINAL_PARSE_KNOWN_ARGS(self, remaining, namespace)
    return _attach_mode_to_namespace(parsed, requested), extras


def _format_help_with_same_batch(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    if _PUBLIC_OPTION in rendered:
        return rendered
    if not any(action.dest == "plastic__enabled" for action in self._actions):
        return rendered
    return (
        rendered.rstrip()
        + "\n  --plastic__layer_count__same_batch_all_probes\n"
        + "                        reuse one fixed evidence batch for one strict non-overlapping PLASTIC probe window\n"
        + "  --no-plastic__layer_count__same_batch_all_probes\n"
        + "                        use the established rolling/multi-batch probe path (default)\n"
    )


argparse.ArgumentParser.parse_args = _parse_args_with_same_batch
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_same_batch
argparse.ArgumentParser.format_help = _format_help_with_same_batch
# ^^^ THOG


# vvv THOG expose the canonical control as a read-only resolved property and persist it only when active, preserving old false-mode metadata

def _same_batch_config_property(_self: Any) -> bool:
    return _runtime_enabled()


if not hasattr(_training_config.TrainingConfig, _CONFIG_KEY):
    setattr(_training_config.TrainingConfig, _CONFIG_KEY, property(_same_batch_config_property))
if not hasattr(_run_config.OwtRunConfig, _CONFIG_KEY):
    setattr(_run_config.OwtRunConfig, _CONFIG_KEY, property(_same_batch_config_property))


_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict
_ORIGINAL_TRAINING_COMPACT_IDENTITY = _training_config.TrainingConfig.compact_identity_metadata
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_RUN_COMPACT_IDENTITY = _run_config.OwtRunConfig.compact_identity
_ORIGINAL_NORMALIZE_PLASTIC_CONFIG = _training_config.normalize_plastic_v0541_config_fields


def _persistent_with_same_batch(original, config: Any) -> Dict[str, Any]:
    values = original(config)
    if bool(config.plastic__enabled) and _runtime_enabled():
        values[_CONFIG_KEY] = True
    return values


def _training_persistent_dict_with_same_batch(self: Any) -> Dict[str, Any]:
    return _persistent_with_same_batch(_ORIGINAL_TRAINING_PERSISTENT_DICT, self)


def _run_persistent_dict_with_same_batch(self: Any) -> Dict[str, Any]:
    return _persistent_with_same_batch(_ORIGINAL_RUN_PERSISTENT_DICT, self)


def _identity_with_same_batch(identity: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    if not plastic_enabled or not _runtime_enabled():
        return identity
    plastic_identity = identity.get("plastic_depth")
    if isinstance(plastic_identity, Mapping):
        updated = dict(identity)
        updated["plastic_depth"] = {**dict(plastic_identity), _CONFIG_KEY: True}
        return updated
    return identity


def _training_compact_identity_with_same_batch(self: Any) -> Dict[str, Any]:
    return _identity_with_same_batch(
        _ORIGINAL_TRAINING_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _run_compact_identity_with_same_batch(self: Any) -> Dict[str, Any]:
    return _identity_with_same_batch(
        _ORIGINAL_RUN_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _normalize_plastic_config_with_same_batch(values: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(values)
    same_batch = source.pop(_CONFIG_KEY, None)
    if same_batch is not None:
        if not isinstance(same_batch, bool):
            raise ValueError(f"{_CONFIG_KEY} must be bool; got {same_batch!r}")
        explicit = _explicit_requested_mode()
        if explicit is not None and explicit != same_batch:
            raise ValueError(
                "resume material parameter mismatch: "
                f"{_CONFIG_KEY}: checkpoint={same_batch!r}, requested={explicit!r}"
            )
        _set_runtime_enabled(same_batch)
    return _ORIGINAL_NORMALIZE_PLASTIC_CONFIG(source)


_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_with_same_batch
_training_config.TrainingConfig.compact_identity_metadata = _training_compact_identity_with_same_batch
_training_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_same_batch
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_with_same_batch
_run_config.OwtRunConfig.compact_identity = _run_compact_identity_with_same_batch
_checkpoint_resume.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_same_batch
# ^^^ THOG


# vvv THOG serializable window state owns deterministic reconstruction; device tensors remain runtime-only

def _new_window_state() -> Dict[str, Any]:
    return {
        "version": _WINDOW_STATE_VERSION,
        "last_window_id": 0,
        "active": None,
    }


def _window_state(trainer: Any) -> Dict[str, Any]:
    state = getattr(trainer, _STATE_ATTRIBUTE, None)
    if state is None:
        state = _new_window_state()
        setattr(trainer, _STATE_ATTRIBUTE, state)
    return state


def _validate_window_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    version = int(state.get("version", 0))
    if version != _WINDOW_STATE_VERSION:
        raise ValueError(
            "unsupported PLASTIC same-batch window state version: "
            f"{version!r}"
        )
    result = {
        "version": version,
        "last_window_id": int(state.get("last_window_id", 0)),
        "active": None,
    }
    active = state.get("active")
    if active is None:
        return result
    if not isinstance(active, Mapping):
        raise ValueError("PLASTIC same-batch active window must be a mapping or None")
    probe_count = int(active.get("probe_count", 0))
    probe_sequences = [int(value) for value in active.get("probe_sequences", ())]
    global_starts = [int(value) for value in active.get("global_starts", ())]
    if probe_count < 0 or len(probe_sequences) != probe_count:
        raise ValueError("PLASTIC same-batch checkpoint has inconsistent probe provenance")
    result["active"] = {
        "window_id": int(active["window_id"]),
        "current_count": int(active["current_count"]),
        "probe_count": probe_count,
        "probe_sequences": probe_sequences,
        "global_starts": global_starts,
        "batch_digest": str(active["batch_digest"]),
        "sample_seed_base": int(active["sample_seed_base"]),
    }
    return result


def _clear_runtime_window_cache(trainer: Any) -> None:
    setattr(trainer, _BATCH_CACHE_ATTRIBUTE, None)
    setattr(trainer, _SAMPLE_CACHE_ATTRIBUTE, None)


def _batch_digest(*, global_starts: Sequence[int], block_size: int, batch_size: int) -> str:
    payload = {
        "split": "train",
        "starts": [int(value) for value in global_starts],
        "block_size": int(block_size),
        "batch_size": int(batch_size),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _global_probe_starts(trainer: Any, *, window_id: int) -> Tuple[int, ...]:
    source = trainer.batch_source
    upper = source._storage_length(source.train_tokens) - source.block_size
    if upper < 1:
        raise RuntimeError("PLASTIC same-batch training split is too short")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(trainer.config.data_seed)
        + _BATCH_SEED_OFFSET
        + _BATCH_SEED_STRIDE * int(window_id)
    )
    starts = torch.randint(upper, (source.batch_size,), generator=generator)
    resolved = tuple(int(value) for value in starts.tolist())
    trainer.distributed.assert_identical_object(
        resolved,
        "PLASTIC same-batch global probe starts",
    )
    return resolved


def _build_probe_batch(trainer: Any, global_starts: Sequence[int]) -> _batch_source.Batch:
    source = trainer.batch_source
    starts_all = tuple(int(value) for value in global_starts)
    if len(starts_all) != source.batch_size:
        raise ValueError(
            "PLASTIC same-batch global start count is incompatible with batch_size; "
            f"starts={len(starts_all)}, batch_size={source.batch_size}"
        )
    local_start = source.rank * source.local_batch_size
    local_end = local_start + source.local_batch_size
    starts = starts_all[local_start:local_end]
    storage_length = source._storage_length(source.train_tokens)
    for start in starts:
        if start < 0 or start + source.block_size >= storage_length:
            raise ValueError(f"PLASTIC same-batch start index out of range: {start}")
    inputs = torch.stack(
        [
            source._slice(source.train_tokens, start, start + source.block_size)
            for start in starts
        ]
    )
    targets = torch.stack(
        [
            source._slice(source.train_tokens, start + 1, start + source.block_size + 1)
            for start in starts
        ]
    )
    if trainer.device.type == "cuda":
        inputs = inputs.pin_memory().to(trainer.device, non_blocking=True)
        targets = targets.pin_memory().to(trainer.device, non_blocking=True)
    else:
        inputs = inputs.to(trainer.device)
        targets = targets.to(trainer.device)
    return _batch_source.Batch(inputs=inputs, targets=targets, starts=starts, split="train")


def _cached_probe_batch(trainer: Any, active: Mapping[str, Any]) -> _batch_source.Batch:
    cached = getattr(trainer, _BATCH_CACHE_ATTRIBUTE, None)
    if isinstance(cached, tuple) and len(cached) == 2 and int(cached[0]) == int(active["window_id"]):
        return cached[1]
    batch = _build_probe_batch(trainer, active["global_starts"])
    setattr(trainer, _BATCH_CACHE_ATTRIBUTE, (int(active["window_id"]), batch))
    return batch


def _invalidate_active_window(trainer: Any, *, reason: str) -> None:
    state = _window_state(trainer)
    active = state.get("active")
    if active is None:
        return
    state["last_window_id"] = max(int(state["last_window_id"]), int(active["window_id"]))
    state["active"] = None
    trainer.state.plastic_depth_probe_histories = {}
    _clear_runtime_window_cache(trainer)
    trainer._record(
        "plastic_depth_same_batch_window_invalidated",
        window_id=int(active["window_id"]),
        probe_count=int(active["probe_count"]),
        probe_sequences=tuple(int(value) for value in active["probe_sequences"]),
        batch_digest=str(active["batch_digest"]),
        reason=str(reason),
    )


def _ensure_active_window(trainer: Any, *, current_count: int) -> Dict[str, Any]:
    state = _window_state(trainer)
    active = state.get("active")
    if active is not None and int(active["current_count"]) != int(current_count):
        _invalidate_active_window(trainer, reason="active_layer_count_changed")
        active = None
    if active is not None:
        return active

    window_id = int(state["last_window_id"]) + 1
    global_starts = _global_probe_starts(trainer, window_id=window_id)
    sample_seed_base = (
        int(trainer.config.model_seed)
        + _SAMPLE_SEED_OFFSET
        + _SAMPLE_SEED_STRIDE * window_id
    )
    active = {
        "window_id": window_id,
        "current_count": int(current_count),
        "probe_count": 0,
        "probe_sequences": [],
        "global_starts": list(global_starts),
        "batch_digest": _batch_digest(
            global_starts=global_starts,
            block_size=trainer.batch_source.block_size,
            batch_size=trainer.batch_source.batch_size,
        ),
        "sample_seed_base": sample_seed_base,
    }
    state["active"] = active
    trainer.state.plastic_depth_probe_histories = {}
    _clear_runtime_window_cache(trainer)
    return active


def _retire_active_window(trainer: Any) -> None:
    state = _window_state(trainer)
    active = state.get("active")
    if active is None:
        raise RuntimeError("PLASTIC same-batch window retirement lacks an active window")
    state["last_window_id"] = max(int(state["last_window_id"]), int(active["window_id"]))
    state["active"] = None
    _clear_runtime_window_cache(trainer)
# ^^^ THOG


# vvv THOG keep one deterministic sampled-token set for the complete fixed-batch window
_ORIGINAL_SAMPLED_TOKEN_INDICES = _trainer_step.TrainerStepMixin._plastic_depth_sampled_token_indices


def _plastic_depth_sampled_token_indices_same_batch(self: Any, targets: torch.Tensor) -> torch.Tensor:
    if not _runtime_enabled():
        return _ORIGINAL_SAMPLED_TOKEN_INDICES(self, targets)
    state = _window_state(self)
    active = state.get("active")
    if active is None:
        raise RuntimeError("PLASTIC same-batch sampled-token selection has no active window")
    window_id = int(active["window_id"])
    cached = getattr(self, _SAMPLE_CACHE_ATTRIBUTE, None)
    if isinstance(cached, tuple) and len(cached) == 2 and int(cached[0]) == window_id:
        return cached[1]

    flattened = targets.reshape(-1)
    valid = torch.nonzero(flattened != -1, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise RuntimeError("PLASTIC DEPTH same-batch probe found no non-ignored target tokens")
    requested = int(self.config.plastic__layer_count_probe__number_of_sampled_valid_tokens)
    if requested > int(valid.numel()):
        raise RuntimeError(
            "plastic__layer_count_probe__number_of_sampled_valid_tokens exceeds the actual valid-token count "
            f"in this rank-local fixed probe batch: requested={requested}, valid={int(valid.numel())}"
        )
    if requested == 0 or requested == int(valid.numel()):
        selected = valid
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(active["sample_seed_base"])
            + _RANK_SAMPLE_SEED_STRIDE * int(self.distributed.rank)
        )
        positions = torch.randperm(int(valid.numel()), generator=generator)[:requested]
        selected = valid.index_select(0, positions.to(device=valid.device))
    setattr(self, _SAMPLE_CACHE_ATTRIBUTE, (window_id, selected))
    return selected


_trainer_step.TrainerStepMixin._plastic_depth_sampled_token_indices = (
    _plastic_depth_sampled_token_indices_same_batch
)
# ^^^ THOG


# vvv THOG evaluate every candidate on the cached batch through one no-grad shared transformer chain
@torch.no_grad()
def _same_batch_candidate_probe(
    self: Any,
    idx: torch.Tensor,
    targets: torch.Tensor,
    request: PlasticDepthInlineProbeRequest,
) -> int:
    if not self.plastic_depth_enabled:
        raise RuntimeError("PLASTIC same-batch probe requires PLASTIC DEPTH")
    if not self.training:
        raise RuntimeError("PLASTIC same-batch probe expects training-mode module semantics")
    if self._active_layer_indices is not None:
        raise RuntimeError("PLASTIC same-batch probing cannot be combined with layer dropout")
    if self._torch_compile_mode == "regional":
        raise RuntimeError("regional torch.compile does not support PLASTIC DEPTH same-batch probing")
    if self._update_retained_materializations.active:
        raise RuntimeError("PLASTIC same-batch evidence must run before optimizer-update materialisation retention")
    request.validate(maximum_count=self.config.n_layer)

    self.trajectory.prepare_plastic_depth_basis_cache()
    try:
        _, sequence_length = idx.shape
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        hidden = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(positions)
        )

        if request.recoverable_upward_counts:
            checkpoint_by_count, candidate_losses, self.last_execution_report = (
                self._plastic_depth_recoverable_probe_candidate_suffix(
                    hidden,
                    targets,
                    request,
                )
            )
        elif request.recoverable_upward_count is not None:
            checkpoint_by_count, candidate_losses, self.last_execution_report = (
                self._plastic_depth_recoverable_probe_candidates(
                    hidden,
                    targets,
                    request,
                )
            )
        else:
            maximum_count = request.candidate_counts[-1]
            checkpoints, self.last_execution_report = (
                _training_model.execute_logical_layer_checkpoints(
                    hidden,
                    n_layer=self.config.n_layer,
                    segment_size=self.checkpoint_segment_size,
                    logical_block=self._logical_block,
                    training=self.training,
                    layer_indices=tuple(range(maximum_count)),
                    checkpoint_counts=request.candidate_counts,
                )
            )
            checkpoint_by_count = dict(checkpoints)
            candidate_losses = tuple(
                (
                    count,
                    self._plastic_depth_candidate_head_loss(
                        checkpoint_by_count[count],
                        targets,
                        request.sampled_token_indices,
                    ).detach(),
                )
                for count in request.candidate_counts
            )

        selected_count = int(request.selector(tuple(candidate_losses)))
        if selected_count not in checkpoint_by_count:
            raise RuntimeError(
                "PLASTIC same-batch selector returned a non-candidate count; "
                f"selected={selected_count}, candidates={tuple(checkpoint_by_count)}"
            )
        sampled_count = (
            int(targets.numel())
            if request.sampled_token_indices is None
            else int(request.sampled_token_indices.numel())
        )
        self.last_plastic_depth_inline_probe_report = PlasticDepthInlineProbeReport(
            candidate_counts=tuple(int(count) for count, _ in candidate_losses),
            local_candidate_losses=tuple(float(loss.item()) for _, loss in candidate_losses),
            selected_count=selected_count,
            sampled_token_count=sampled_count,
        )
        return selected_count
    finally:
        self.trajectory.clear_plastic_depth_basis_cache()


_training_model.TrainingSheetGPT.plastic_depth_same_batch_candidate_probe = (
    _same_batch_candidate_probe
)
# ^^^ THOG


# vvv THOG decouple decision evidence from the training microbatch while reusing the complete v0.541 selector/audit stack
_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update
_ORIGINAL_TRAINING_SHEET_FORWARD = _training_model.TrainingSheetGPT.forward


def _capture_probe_rng(trainer: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if trainer.device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(trainer.device)
    return cpu_state, cuda_state


def _restore_probe_rng(trainer: Any, state: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, trainer.device)


def _force_framework_hold(context: Dict[str, Any], *, reason: str) -> None:
    current_count = int(context["current_count"])
    decision = context.get("decision")
    if decision is None:
        raise RuntimeError("PLASTIC same-batch framework hold lacks a completed decision object")
    evidence = tuple(replace(item, significant=False) for item in decision.evidence)
    held = replace(decision, selected_count=current_count, evidence=evidence)
    context["decision"] = held
    context["selected_count"] = current_count
    context["paired_evidence"] = held.report()
    context["score_evidence"] = held.report()
    report = context.get("plastic_directional_report")
    if isinstance(report, dict):
        report["selected_count"] = current_count
    context["plastic_same_batch_framework_hold_reason"] = str(reason)


def _begin_plastic_depth_inline_update_same_batch(self: Any) -> Optional[Dict[str, Any]]:
    context = _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    if context is None or not _runtime_enabled():
        return context

    current_count = int(context["current_count"])
    active = _ensure_active_window(self, current_count=current_count)
    window_size = int(
        self.config.plastic__layer_count_probe__window_size_as_number_of_probes
    )
    if window_size < 1:
        raise RuntimeError("PLASTIC same-batch window size must be positive")
    pending_ordinal = int(active["probe_count"]) + 1
    if pending_ordinal > window_size:
        raise RuntimeError(
            "PLASTIC same-batch active window exceeded its configured size; "
            f"window={active['window_id']}, ordinal={pending_ordinal}, size={window_size}"
        )

    context["plastic_same_batch_all_probes"] = True
    context["plastic_same_batch_window_id"] = int(active["window_id"])
    context["plastic_same_batch_window_size"] = window_size
    context["plastic_same_batch_window_ordinal"] = pending_ordinal
    context["plastic_same_batch_batch_digest"] = str(active["batch_digest"])
    context["plastic_same_batch_global_starts"] = tuple(
        int(value) for value in active["global_starts"]
    )

    batch = _cached_probe_batch(self, active)
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, batch.targets, context)
    rng_state = _capture_probe_rng(self)
    try:
        with self.autocast_context():
            selected_count = int(
                self.raw_model.plastic_depth_same_batch_candidate_probe(
                    batch.inputs,
                    batch.targets,
                    request,
                )
            )
    except BaseException:
        self.raw_model._plastic_depth_v0521_candidate_token_losses = None
        raise
    finally:
        _restore_probe_rng(self, rng_state)

    probe_sequence = int(context.get("plastic_probe_sequence", 0))
    if probe_sequence < 1:
        raise RuntimeError("PLASTIC same-batch probe lacks durable probe sequence provenance")
    prior_sequences = tuple(int(value) for value in active["probe_sequences"])
    window_provenance = (*prior_sequences, probe_sequence)
    context["plastic_probe_provenance"] = window_provenance
    context["plastic_same_batch_window_provenance"] = window_provenance
    report = context.get("plastic_directional_report")
    vote_total = 0
    if isinstance(report, dict):
        report["probe_provenance"] = window_provenance
        report["same_batch_all_probes"] = True
        report["probe_window_id"] = int(active["window_id"])
        report["probe_window_ordinal"] = pending_ordinal
        report["probe_window_size"] = window_size
        report["probe_batch_digest"] = str(active["batch_digest"])
        vote_total = int(report.get("vote_total", 0))
    history_aligned = vote_total == pending_ordinal
    context["plastic_same_batch_history_aligned"] = history_aligned
    context["plastic_same_batch_window_complete"] = pending_ordinal == window_size

    resolved_selected = int(context.get("selected_count", selected_count))
    if pending_ordinal < window_size:
        if resolved_selected != current_count:
            _force_framework_hold(context, reason="incomplete_same_batch_window")
        else:
            context["selected_count"] = current_count
    elif not history_aligned:
        _force_framework_hold(context, reason="decision_history_reset_inside_window")

    selected_for_training = int(context.get("selected_count", current_count))
    setter = self.raw_model.set_plastic_depth_update_layer_count
    setter(selected_for_training)
    context["plastic_same_batch_precomputed"] = True
    return context


def _training_only_probe_request(
    self: Any,
    targets: torch.Tensor,
    context: Dict[str, Any],
) -> PlasticDepthInlineProbeRequest:
    del targets
    selected_count = int(context["selected_count"])

    def select_training_only(_candidates: Any) -> int:
        return selected_count

    setattr(select_training_only, _TRAINING_ONLY_SELECTOR_MARKER, True)
    setattr(select_training_only, _TRAINING_ONLY_SELECTED_COUNT, selected_count)
    return PlasticDepthInlineProbeRequest(
        candidate_counts=(selected_count,),
        sampled_token_indices=None,
        selector=select_training_only,
    )


def _plastic_depth_inline_probe_request_same_batch(
    self: Any,
    targets: torch.Tensor,
    context: Dict[str, Any],
) -> PlasticDepthInlineProbeRequest:
    if not _runtime_enabled() or not bool(context.get("plastic_same_batch_precomputed", False)):
        return _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    return _training_only_probe_request(self, targets, context)


def _training_sheet_forward_same_batch(
    self: Any,
    idx: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    *,
    plastic_depth_probe_request: Optional[PlasticDepthInlineProbeRequest] = None,
    plastic_depth_active_layers_override: Optional[int] = None,
):
    request = plastic_depth_probe_request
    selector = None if request is None else request.selector
    if selector is None or not bool(getattr(selector, _TRAINING_ONLY_SELECTOR_MARKER, False)):
        return _ORIGINAL_TRAINING_SHEET_FORWARD(
            self,
            idx,
            targets,
            plastic_depth_probe_request=plastic_depth_probe_request,
            plastic_depth_active_layers_override=plastic_depth_active_layers_override,
        )
    if plastic_depth_active_layers_override is not None:
        raise ValueError("PLASTIC same-batch training-only request cannot also supply an active-layer override")
    selected_count = int(getattr(selector, _TRAINING_ONLY_SELECTED_COUNT))
    probe_report = self.last_plastic_depth_inline_probe_report
    result = _ORIGINAL_TRAINING_SHEET_FORWARD(
        self,
        idx,
        targets,
        plastic_depth_probe_request=None,
        plastic_depth_active_layers_override=selected_count,
    )
    self.last_plastic_depth_inline_probe_report = probe_report
    return result


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = (
    _begin_plastic_depth_inline_update_same_batch
)
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_same_batch
)
_training_model.TrainingSheetGPT.forward = _training_sheet_forward_same_batch
# ^^^ THOG


# vvv THOG commit window evidence only after the authoritative optimizer step succeeds; retire every complete window including STAY

def _augment_latest_count_audit(self: Any, context: Mapping[str, Any]) -> None:
    rows = getattr(self, "plastic_depth_count_audit", None)
    if not isinstance(rows, list) or not rows:
        return
    row = rows[-1]
    decision = context.get("decision")
    if decision is None or int(row.get("update_number", -1)) != int(decision.update_number):
        return
    row.update(
        {
            "same_batch_all_probes": True,
            "probe_window_id": int(context["plastic_same_batch_window_id"]),
            "probe_window_ordinal": int(context["plastic_same_batch_window_ordinal"]),
            "probe_window_size": int(context["plastic_same_batch_window_size"]),
            "probe_window_complete": bool(context["plastic_same_batch_window_complete"]),
            "probe_window_history_aligned": bool(context["plastic_same_batch_history_aligned"]),
            "probe_batch_digest": str(context["plastic_same_batch_batch_digest"]),
            "probe_batch_global_starts": tuple(
                int(value) for value in context["plastic_same_batch_global_starts"]
            ),
            "probe_window_provenance": tuple(
                int(value) for value in context["plastic_same_batch_window_provenance"]
            ),
            "framework_hold_reason": context.get("plastic_same_batch_framework_hold_reason"),
        }
    )


def _commit_plastic_depth_inline_update_same_batch(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None or not _runtime_enabled() or not bool(context.get("plastic_same_batch_precomputed", False)):
        return transition

    state = _window_state(self)
    active = state.get("active")
    if active is None:
        raise RuntimeError("PLASTIC same-batch commit lost its active window")
    window_id = int(context["plastic_same_batch_window_id"])
    if int(active["window_id"]) != window_id:
        raise RuntimeError("PLASTIC same-batch commit window identity changed in flight")
    ordinal = int(context["plastic_same_batch_window_ordinal"])
    if ordinal != int(active["probe_count"]) + 1:
        raise RuntimeError("PLASTIC same-batch commit probe ordinal is inconsistent")
    probe_sequence = int(context["plastic_probe_sequence"])
    active["probe_count"] = ordinal
    active["probe_sequences"] = [
        *[int(value) for value in active["probe_sequences"]],
        probe_sequence,
    ]
    provenance = tuple(int(value) for value in active["probe_sequences"])
    if provenance != tuple(context["plastic_same_batch_window_provenance"]):
        raise RuntimeError("PLASTIC same-batch committed provenance differs from in-flight provenance")

    if ordinal == 1:
        self._record(
            "plastic_depth_same_batch_window_started",
            window_id=window_id,
            current_count=int(context["current_count"]),
            window_size=int(context["plastic_same_batch_window_size"]),
            batch_digest=str(active["batch_digest"]),
        )
    self._record(
        "plastic_depth_same_batch_probe_committed",
        window_id=window_id,
        ordinal=ordinal,
        window_size=int(context["plastic_same_batch_window_size"]),
        probe_sequence=probe_sequence,
        probe_provenance=provenance,
        batch_digest=str(active["batch_digest"]),
        history_aligned=bool(context["plastic_same_batch_history_aligned"]),
    )
    _augment_latest_count_audit(self, context)

    window_size = int(context["plastic_same_batch_window_size"])
    if ordinal < window_size:
        decision = context.get("decision")
        if decision is not None and int(decision.selected_count) != int(context["current_count"]):
            raise RuntimeError("PLASTIC same-batch partial window committed a layer-count change")
        return transition

    if ordinal != window_size:
        raise RuntimeError("PLASTIC same-batch completed window has an invalid ordinal")
    decision = context.get("decision")
    if decision is None:
        raise RuntimeError("PLASTIC same-batch completed window lacks its decision")
    selected_count = int(decision.selected_count)
    current_count = int(context["current_count"])
    disposition = "change" if selected_count != current_count else "stay"

    self.state.plastic_depth_probe_histories = {}
    rows = getattr(self, "plastic_depth_count_audit", None)
    if isinstance(rows, list) and rows:
        rows[-1]["histories_after_window_retirement"] = {}
        rows[-1]["probe_window_disposition"] = disposition
    self._record(
        "plastic_depth_same_batch_window_decision",
        window_id=window_id,
        previous_count=current_count,
        selected_count=selected_count,
        disposition=disposition,
        probe_provenance=provenance,
        batch_digest=str(active["batch_digest"]),
        history_aligned=bool(context["plastic_same_batch_history_aligned"]),
        framework_hold_reason=context.get("plastic_same_batch_framework_hold_reason"),
    )
    _retire_active_window(self)
    return transition


_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_same_batch
)
# ^^^ THOG


# vvv THOG make the active window checkpoint-authoritative while leaving false-mode checkpoint payloads unchanged
_ORIGINAL_CHECKPOINT_PAYLOAD = _checkpoint_save.TrainerCheckpointSaveMixin.checkpoint_payload
_ORIGINAL_FROM_CHECKPOINT = _checkpoint_resume.TrainerCheckpointResumeMixin.from_checkpoint.__func__


def _checkpoint_payload_with_same_batch(self: Any) -> Dict[str, Any]:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(self)
    if _runtime_enabled() and bool(self.config.plastic__enabled):
        payload[_CHECKPOINT_STATE_KEY] = copy.deepcopy(_window_state(self))
    return payload


def _from_checkpoint_with_same_batch(
    cls,
    path,
    train_tokens,
    validation_tokens,
    *,
    overrides=None,
    expected_config=None,
):
    payload = _training_model.torch.load(path, map_location="cpu", weights_only=False)
    trainer_config = payload.get("trainer_config", {})
    checkpoint_enabled = bool(
        isinstance(trainer_config, Mapping) and trainer_config.get(_CONFIG_KEY, False)
    )
    explicit = _explicit_requested_mode()
    if explicit is not None and explicit != checkpoint_enabled:
        raise ValueError(
            "resume material parameter mismatch: "
            f"{_CONFIG_KEY}: checkpoint={checkpoint_enabled!r}, requested={explicit!r}"
        )
    _set_runtime_enabled(checkpoint_enabled)
    trainer = _ORIGINAL_FROM_CHECKPOINT(
        cls,
        path,
        train_tokens,
        validation_tokens,
        overrides=overrides,
        expected_config=expected_config,
    )
    if checkpoint_enabled:
        restored = payload.get(_CHECKPOINT_STATE_KEY, _new_window_state())
        setattr(trainer, _STATE_ATTRIBUTE, _validate_window_state(restored))
        _clear_runtime_window_cache(trainer)
    elif _CHECKPOINT_STATE_KEY in payload:
        raise ValueError("disabled PLASTIC same-batch checkpoint carries active window state")
    return trainer


_checkpoint_save.TrainerCheckpointSaveMixin.checkpoint_payload = _checkpoint_payload_with_same_batch
_checkpoint_resume.TrainerCheckpointResumeMixin.from_checkpoint = classmethod(
    _from_checkpoint_with_same_batch
)
# ^^^ THOG


__all__ = [
    "_CONFIG_KEY",
    "_CHECKPOINT_STATE_KEY",
    "_batch_digest",
    "_build_probe_batch",
    "_explicit_requested_mode",
    "_runtime_enabled",
    "_validate_window_state",
    "_window_state",
]
# ^^^ THOG
