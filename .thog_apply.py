from pathlib import Path

# vvv THOG preserve historical collective-failure headlines while retaining structured diagnostics
trainer_path = Path("sheet/trainer_step.py")
trainer_text = trainer_path.read_text(encoding="utf-8")
old = '''        if self.config.nonfinite_update_policy == "raise":
            self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
            raise FloatingPointError(
                "non-finite update detected: "
                + json.dumps(payload, sort_keys=True)
            )
'''
new = '''        if self.config.nonfinite_update_policy == "raise":
            self._cleanup_failed_update(scaler_unscaled=scaler_unscaled)
            headline = {
                "loss": "non-finite training loss on at least one rank",
                "gradient": "non-finite gradient on at least one rank",
                "gradient_norm": "non-finite gradient norm",
            }.get(reason, "non-finite update detected")
            raise FloatingPointError(
                headline + ": " + json.dumps(payload, sort_keys=True)
            )
'''
if old in trainer_text:
    trainer_path.write_text(trainer_text.replace(old, new, 1), encoding="utf-8")
elif new not in trainer_text:
    raise SystemExit("expected non-finite raise block not found")
# ^^^ THOG

# vvv THOG remove temporary one-shot workflows now superseded by the tested source state
for workflow_name in (
    ".github/workflows/apply_nonfinite_message_compat.yml",
    ".github/workflows/apply_stage6_nonfinite_completion.yml",
):
    workflow_path = Path(workflow_name)
    if workflow_path.exists():
        workflow_path.unlink()
# ^^^ THOG
