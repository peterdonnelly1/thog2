# vvv THOG
"""Explicit optimizer reset preserves model/data position and records a new history origin."""
from .checkpoints import validate_compatibility, strip_compiled_prefix, restore_rng_state
from .trainer_checkpoint_resume import _trainer_state_from_payload


def reset_optimizer_fork(trainer_class, payload, config, train_tokens, validation_tokens):
    validate_compatibility(payload,config)
    trainer = trainer_class(config,train_tokens,validation_tokens)
    trainer.raw_model.load_state_dict(strip_compiled_prefix(payload["model"]))
    trainer.state = _trainer_state_from_payload(payload)
    trainer.batch_source.load_state_dict(payload["batch_source"])
    if payload.get("thogopt_grad_scaler"): trainer.scaler.load_state_dict(payload["thogopt_grad_scaler"])
    if payload.get("rng_state"): restore_rng_state(payload["rng_state"])
    trainer.dense_snapshot_metadata = payload.get("dense_snapshot_baselining")
    trainer.optimizer_reset_origin = int(trainer.state.completed_updates)
    trainer._record("optimizer_reset_fork",history_origin=trainer.optimizer_reset_origin)
    return trainer
# ^^^ THOG
