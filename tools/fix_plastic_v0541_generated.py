from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# vvv THOG keep v0.521 paired-SE as the public probe wrapper while v0.541 provenance wraps its inner request
replace_once(
    "sheet/plastic_depth_v0541_patch.py",
    "from . import plastic_depth_directional_coherence_patch as _directional\nfrom . import stage6_trainer as _stage6",
    "from . import plastic_depth_directional_coherence_patch as _directional\nfrom . import plastic_depth_probe_se_v0521_patch as _probe_se\nfrom . import stage6_trainer as _stage6",
)
replace_once(
    "sheet/plastic_depth_v0541_patch.py",
    "_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request",
    "_ORIGINAL_INLINE_PROBE_REQUEST = _probe_se._ORIGINAL_INLINE_PROBE_REQUEST",
)
replace_once(
    "sheet/plastic_depth_v0541_patch.py",
    "_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_v0541\n_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _plastic_depth_inline_probe_request_v0541",
    "_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_v0541\n_probe_se._ORIGINAL_INLINE_PROBE_REQUEST = _plastic_depth_inline_probe_request_v0541\n_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _probe_se._inline_probe_request_with_paired_token_se",
)
# ^^^ THOG


# vvv THOG startup rendering remains tolerant of lightweight legacy/test config objects while real configs always carry v0.541 fields
replace_once(
    "run_thog2_owt.py",
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_discount:", _startup_float(config.plastic__wall_time_equivalent_time_gain_discount))',
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_discount:", _startup_float(getattr(config, "plastic__wall_time_equivalent_time_gain_discount", 0.9)))',
)
replace_once(
    "run_thog2_owt.py",
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_loss_rate_window:", str(config.plastic__wall_time_equivalent_time_gain_loss_rate_window))',
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_loss_rate_window:", str(getattr(config, "plastic__wall_time_equivalent_time_gain_loss_rate_window", 64)))',
)
replace_once(
    "run_thog2_owt.py",
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_loss_rate_min_observations:", str(config.plastic__wall_time_equivalent_time_gain_loss_rate_min_observations))',
    '_print_plastic_option("plastic__wall_time_equivalent_time_gain_loss_rate_min_observations:", str(getattr(config, "plastic__wall_time_equivalent_time_gain_loss_rate_min_observations", 16)))',
)
# ^^^ THOG

print("PLASTIC v0.541 generated integration fixes applied")
