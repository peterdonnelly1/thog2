from __future__ import annotations

import gc
import hashlib
import io
import pickle
import random
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class PlasticFreshTrainingState:
    trainer: Any
    phase: str
    active_layer_count: int
    instrumentation_namespace: str
    fingerprint: Mapping[str, str]


def reset_plastic_fresh_random_state(*, model_seed: int, device: str) -> None:
    random.seed(int(model_seed))
    np.random.seed(int(model_seed) % (2**32))
    torch.manual_seed(int(model_seed))
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA fresh-state construction requested without CUDA")
        torch.cuda.manual_seed_all(int(model_seed))


def _torch_digest(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _pickle_digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _tensor_digest(tensor: Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    payload = contiguous.numpy().tobytes(order="C")
    metadata = f"{tuple(contiguous.shape)}|{contiguous.dtype}".encode("utf-8")
    return hashlib.sha256(metadata + payload).hexdigest()


def _batch_digest(batch: Any) -> str:
    payload = (
        _tensor_digest(batch.inputs),
        _tensor_digest(batch.targets),
        tuple(int(value) for value in batch.starts),
        str(batch.split),
    )
    return _pickle_digest(payload)


def _first_batch_fingerprint(trainer: Any) -> Dict[str, str]:
    batch_source = trainer.batch_source
    state = batch_source.state_dict()
    try:
        train_batch = batch_source.get_batch("train", device=trainer.device)
        validation_batch = batch_source.get_batch("val", device=trainer.device)
    finally:
        batch_source.load_state_dict(state)
    return {
        "first_train_batch": _batch_digest(train_batch),
        "first_validation_batch": _batch_digest(validation_batch),
    }


def plastic_fresh_state_fingerprint(trainer: Any) -> Mapping[str, str]:
    model = getattr(trainer, "raw_model", getattr(trainer, "model", None))
    if model is None:
        raise TypeError("fresh-state fingerprint requires trainer.raw_model or trainer.model")
    optimizer = trainer.optimizer
    scaler = getattr(trainer, "scaler", None)
    batch_source = trainer.batch_source
    values: Dict[str, str] = {
        "model": _torch_digest(model.state_dict()),
        "optimizer": _torch_digest(optimizer.state_dict()),
        "batch_source": _torch_digest(batch_source.state_dict()),
        "python_rng": _pickle_digest(random.getstate()),
        "numpy_rng": _pickle_digest(np.random.get_state()),
        "torch_cpu_rng": _torch_digest(torch.random.get_rng_state()),
    }
    if scaler is not None:
        values["scaler"] = _torch_digest(scaler.state_dict())
    device = torch.device(trainer.device)
    if device.type == "cuda":
        values["torch_device_rng"] = _torch_digest(torch.cuda.get_rng_state_all())
    values.update(_first_batch_fingerprint(trainer))
    return values


def _resolved_fresh_config(
    config: Any,
    *,
    phase: str,
    active_layer_count: int,
) -> Any:
    if phase not in {"coarse", "fine"}:
        raise ValueError(f"invalid PLASTIC lifecycle phase: {phase!r}")
    if active_layer_count < 1:
        raise ValueError("active_layer_count must be positive")
    if not is_dataclass(config):
        raise TypeError("fresh-state construction requires a dataclass training config")
    # vvv THOG fixed-count COARSE states use layers_to_sample; dynamic FINE states use initial_layer_count
    if bool(getattr(config, "plastic__do_learn_layer_count", False)):
        changes: Dict[str, Any] = {
            "plastic__layers_to_sample": None,
            "plastic__initial_layer_count": int(active_layer_count),
        }
    else:
        changes = {
            "plastic__layers_to_sample": int(active_layer_count),
            "plastic__initial_layer_count": None,
            "plastic__max_permitted_layers": None,
        }
    # ^^^ THOG
    if hasattr(config, "plastic__coarse_phase"):
        changes["plastic__coarse_phase"] = "disabled"
    if hasattr(config, "plastic__runtime_phase"):
        changes["plastic__runtime_phase"] = phase
    return replace(config, **changes)


def build_fresh_training_state(
    *,
    trainer_factory: Callable[[Any, Tensor, Tensor], Any],
    resolved_config: Any,
    train_tokens: Tensor,
    validation_tokens: Tensor,
    phase: str,
    active_layer_count: int,
    instrumentation_namespace: str,
) -> PlasticFreshTrainingState:
    if not instrumentation_namespace:
        raise ValueError("instrumentation_namespace must not be empty")
    fresh_config = _resolved_fresh_config(
        resolved_config,
        phase=phase,
        active_layer_count=active_layer_count,
    )
    reset_plastic_fresh_random_state(
        model_seed=int(fresh_config.model_seed),
        device=str(fresh_config.device),
    )
    trainer = trainer_factory(fresh_config, train_tokens, validation_tokens)
    if phase == "coarse" and bool(getattr(fresh_config, "plastic__enabled", False)):
        trajectory = getattr(trainer.raw_model, "trajectory", None)
        lattice = getattr(trajectory, "plastic_sampling", None)
        if lattice is None:
            close = getattr(trainer, "close", None)
            if callable(close):
                close()
            raise RuntimeError("PLASTIC COARSE trainer has no sampling lattice")
        for parameter in lattice.parameters():
            parameter.requires_grad_(False)
        if int(lattice.current_active_layers) != int(active_layer_count):
            close = getattr(trainer, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "PLASTIC COARSE active count differs from its candidate: "
                f"candidate={active_layer_count}, active={lattice.current_active_layers}"
            )
    completed_updates = int(getattr(trainer.state, "completed_updates", 0))
    if completed_updates != 0:
        close = getattr(trainer, "close", None)
        if callable(close):
            close()
        raise RuntimeError(
            "fresh-state factory constructed stale trainer state: "
            f"completed_updates={completed_updates}"
        )
    trainer.plastic_lifecycle_phase = phase
    trainer.instrumentation_namespace = instrumentation_namespace
    fingerprint = plastic_fresh_state_fingerprint(trainer)
    return PlasticFreshTrainingState(
        trainer=trainer,
        phase=phase,
        active_layer_count=int(active_layer_count),
        instrumentation_namespace=instrumentation_namespace,
        fingerprint=fingerprint,
    )


def destroy_fresh_training_state(state: PlasticFreshTrainingState) -> None:
    trainer = state.trainer
    close = getattr(trainer, "close", None)
    if callable(close):
        close()
    state.trainer = None
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()