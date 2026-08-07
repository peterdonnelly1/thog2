#!/usr/bin/env python3
# vvv THOG
"""Apply PLASTIC v0.521 probe-token sampling and hybrid probe-delta console semantics."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _replace_once(path: str, old: str, new: str) -> None:
    content = _read(path)
    if new in content:
        return
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one replacement target, found {count}: {old!r}")
    _write(path, content.replace(old, new, 1))


def _insert_after(path: str, anchor: str, addition: str) -> None:
    content = _read(path)
    if addition.strip() in content:
        return
    count = content.count(anchor)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one insertion anchor, found {count}: {anchor!r}")
    _write(path, content.replace(anchor, anchor + addition, 1))


# TrainingConfig: persistent canonical field, fresh default, static-capacity validation.
_replace_once(
    "sheet/training_config.py",
    '    "plastic__layer_count_probe__probe_every_n_steps",\n    "plastic__layer_count_probe_radius",',
    '    "plastic__layer_count_probe__probe_every_n_steps",\n    "plastic__layer_count_probe__number_of_sampled_valid_tokens",\n    "plastic__layer_count_probe_radius",',
)
_replace_once(
    "sheet/training_config.py",
    "    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None\n    plastic__layer_count_probe_radius: int = 1",
    "    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None\n    plastic__layer_count_probe__number_of_sampled_valid_tokens: int = 1024\n    plastic__layer_count_probe_radius: int = 1",
)
_insert_after(
    "sheet/training_config.py",
    "        self.plastic__layer_count_probe__probe_every_n_steps = resolved_probe_interval\n",
    "        # vvv THOG v0.521 fresh probe-token sampling uses 1024 by default and zero as the explicit all-valid sentinel\n"
    "        if (\n"
    "            isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, bool)\n"
    "            or not isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, int)\n"
    "            or self.plastic__layer_count_probe__number_of_sampled_valid_tokens < 0\n"
    "        ):\n"
    "            raise ValueError(\n"
    "                \"plastic__layer_count_probe__number_of_sampled_valid_tokens must be a non-negative integer; \"\n"
    "                f\"got {self.plastic__layer_count_probe__number_of_sampled_valid_tokens!r}\"\n"
    "            )\n"
    "        # ^^^ THOG\n",
)
_insert_after(
    "sheet/training_config.py",
    "        for name in (\"block_size\", \"vocab_size\", \"n_layer\", \"n_head\", \"n_embd\", \"depth_order\", \"base_row_order\", \"mlp_hidden_group_size\", \"hyperblock_common_family_order\", \"hyperblock_attention_family_order\", \"hyperblock_mlp_family_order\", \"hyperblock_depth_order\", \"hyperblock_d_model_order\", \"hyperblock_mlp_hidden_order\", \"hyperblock_attention_head_order\", \"hyperblock_attention_head_channel_order\", \"hyperblock_mlp_hidden_multiplier\", \"hyperblock_loop_count\", \"batch_size\", \"gradient_accumulation_steps\", \"layer_dropout_resample_steps\", \"max_updates\", \"decay_updates\", \"eval_batches\", \"log_interval\"):\n            value = getattr(self, name)\n            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:\n                raise ValueError(f\"{name} must be a positive integer; got {value!r}\")\n",
    "        # vvv THOG v0.521 reject impossible configured samples rather than silently clipping to microbatch capacity\n"
    "        probe_token_capacity = self.batch_size * self.block_size\n"
    "        if self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity:\n"
    "            raise ValueError(\n"
    "                \"plastic__layer_count_probe__number_of_sampled_valid_tokens must not exceed \"\n"
    "                f\"batch_size * block_size ({probe_token_capacity}); \"\n"
    "                f\"got {self.plastic__layer_count_probe__number_of_sampled_valid_tokens}\"\n"
    "            )\n"
    "        # ^^^ THOG\n",
)

# OwtRunConfig: public run identity, validation, artifact identity, trainer transfer.
_replace_once(
    "sheet/run_config.py",
    '    "plastic__layer_count_probe__probe_every_n_steps",\n    "plastic__layer_count_probe_radius",',
    '    "plastic__layer_count_probe__probe_every_n_steps",\n    "plastic__layer_count_probe__number_of_sampled_valid_tokens",\n    "plastic__layer_count_probe_radius",',
)
_replace_once(
    "sheet/run_config.py",
    "    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None\n    plastic__layer_count_probe_radius: int = 1",
    "    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None\n    plastic__layer_count_probe__number_of_sampled_valid_tokens: int = 1024\n    plastic__layer_count_probe_radius: int = 1",
)
_insert_after(
    "sheet/run_config.py",
    '        object.__setattr__(self, "plastic__layer_count_probe__probe_every_n_steps", resolved_probe_interval)\n',
    "        # vvv THOG v0.521 expose a strict non-negative probe-token count; zero means every valid token in the probe microbatch\n"
    "        if (\n"
    "            isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, bool)\n"
    "            or not isinstance(self.plastic__layer_count_probe__number_of_sampled_valid_tokens, int)\n"
    "            or self.plastic__layer_count_probe__number_of_sampled_valid_tokens < 0\n"
    "        ):\n"
    "            raise ValueError(\n"
    "                \"plastic__layer_count_probe__number_of_sampled_valid_tokens must be a non-negative integer\"\n"
    "            )\n"
    "        # ^^^ THOG\n",
)
_insert_after(
    "sheet/run_config.py",
    "        for name in positive:\n            value = getattr(self, name)\n            if isinstance(value, bool) or not isinstance(value, int) or value < 1:\n                raise ValueError(f\"{name} must be a positive integer\")\n",
    "        # vvv THOG v0.521 reject a requested sample larger than the physical first-microbatch token capacity\n"
    "        probe_token_capacity = self.batch_size * self.block_size\n"
    "        if self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity:\n"
    "            raise ValueError(\n"
    "                \"plastic__layer_count_probe__number_of_sampled_valid_tokens must not exceed \"\n"
    "                f\"batch_size * block_size ({probe_token_capacity})\"\n"
    "            )\n"
    "        # ^^^ THOG\n",
)
_replace_once(
    "sheet/run_config.py",
    '                    f"LPI_{self.plastic__layer_count_probe__probe_every_n_steps}",\n                    f"LPR_{self.plastic__layer_count_probe_radius}",',
    '                    f"LPI_{self.plastic__layer_count_probe__probe_every_n_steps}",\n                    f"LPT_{self.plastic__layer_count_probe__number_of_sampled_valid_tokens}",\n                    f"LPR_{self.plastic__layer_count_probe_radius}",',
)
_replace_once(
    "sheet/run_config.py",
    "            plastic__layer_count_probe__probe_every_n_steps=self.plastic__layer_count_probe__probe_every_n_steps,\n            plastic__layer_count_probe_radius=self.plastic__layer_count_probe_radius,",
    "            plastic__layer_count_probe__probe_every_n_steps=self.plastic__layer_count_probe__probe_every_n_steps,\n            plastic__layer_count_probe__number_of_sampled_valid_tokens=self.plastic__layer_count_probe__number_of_sampled_valid_tokens,\n            plastic__layer_count_probe_radius=self.plastic__layer_count_probe_radius,",
)

# Canonical runner CLI, resolved config, startup report, and exact old-checkpoint migration.
_replace_once(
    "run_thog2_owt_core.py",
    '    parser.add_argument("--plastic__layer_count_probe__probe_every_n_steps", dest="plastic__layer_count_probe__probe_every_n_steps", type=int)\n    parser.add_argument("--plastic__layer_count_probe_radius",',
    '    parser.add_argument("--plastic__layer_count_probe__probe_every_n_steps", dest="plastic__layer_count_probe__probe_every_n_steps", type=int)\n    parser.add_argument("--plastic__layer_count_probe__number_of_sampled_valid_tokens", dest="plastic__layer_count_probe__number_of_sampled_valid_tokens", type=int, default=1024)\n    parser.add_argument("--plastic__layer_count_probe_radius",',
)
_replace_once(
    "run_thog2_owt_core.py",
    "        plastic__layer_count_probe__probe_every_n_steps=arguments.plastic__layer_count_probe__probe_every_n_steps,\n        plastic__layer_count_probe_radius=arguments.plastic__layer_count_probe_radius,",
    "        plastic__layer_count_probe__probe_every_n_steps=arguments.plastic__layer_count_probe__probe_every_n_steps,\n        plastic__layer_count_probe__number_of_sampled_valid_tokens=arguments.plastic__layer_count_probe__number_of_sampled_valid_tokens,\n        plastic__layer_count_probe_radius=arguments.plastic__layer_count_probe_radius,",
)
_replace_once(
    "run_thog2_owt_core.py",
    '            f"noise_window={config.plastic__layer_count_probe__window_size_as_number_of_probes}  "\n            f"lambda={float(config.plastic__layer_count_probe_noise_lambda):g}",',
    '            f"noise_window={config.plastic__layer_count_probe__window_size_as_number_of_probes}  "\n            f"probe_tokens={config.plastic__layer_count_probe__number_of_sampled_valid_tokens}  "\n            f"lambda={float(config.plastic__layer_count_probe_noise_lambda):g}",',
)
_replace_once(
    "run_thog2_owt_core.py",
    '    stored = TrainingConfig(**payload["trainer_config"])\n',
    '    # vvv THOG v0.521 checkpoints written before the public probe-token knob retain their historical hard-coded 256-token semantics\n    stored_values = dict(payload["trainer_config"])\n    if stored_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in stored_values:\n        stored_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\n    stored = TrainingConfig(**stored_values)\n    # ^^^ THOG\n',
)
_replace_once(
    "run_thog2_owt_core.py",
    '        "layer_dropout_stratum_size", "layer_dropout_active_per_stratum", "layer_dropout_resample_steps",\n',
    '        "layer_dropout_stratum_size", "layer_dropout_active_per_stratum", "layer_dropout_resample_steps",\n        "plastic__layer_count_probe__number_of_sampled_valid_tokens",\n',
)

# Shared checkpoint resume path gets the same legacy-256 migration before TrainingConfig construction.
_replace_once(
    "sheet/trainer_checkpoint_resume.py",
    '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n',
    '        # vvv THOG v0.521 preserve exact pre-knob PLASTIC resume semantics: missing probe-token field means the historical fixed 256-token sample\n        checkpoint_config_values = dict(payload["trainer_config"])\n        if checkpoint_config_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in checkpoint_config_values:\n            checkpoint_config_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\n        checkpoint_config = TrainingConfig(**checkpoint_config_values)\n        # ^^^ THOG\n',
)

# Wrapper: public default/usage, both long-option forms, validation and forwarding.
_insert_after(
    "train_OWT_core.sh",
    'PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS=""\n',
    'PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS=1024\n',
)
_insert_after(
    "train_OWT_core.sh",
    '  --plastic__layer_count_probe__probe_every_n_steps N=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update brake}\n',
    '  --plastic__layer_count_probe__number_of_sampled_valid_tokens N=${PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS}  0=all valid tokens in first probe microbatch\n',
)
_replace_once(
    "train_OWT_core.sh",
    "--plastic__layer_count_probe__probe_every_n_steps|--plastic__layer_count_probe_radius",
    "--plastic__layer_count_probe__probe_every_n_steps|--plastic__layer_count_probe__number_of_sampled_valid_tokens|--plastic__layer_count_probe_radius",
)
_insert_after(
    "train_OWT_core.sh",
    '        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$2" ;;\n',
    '        --plastic__layer_count_probe__number_of_sampled_valid_tokens) PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS="$2" ;;\n',
)
_replace_once(
    "train_OWT_core.sh",
    "--plastic__layer_count_probe__probe_every_n_steps=*|--plastic__layer_count_probe_radius=*",
    "--plastic__layer_count_probe__probe_every_n_steps=*|--plastic__layer_count_probe__number_of_sampled_valid_tokens=*|--plastic__layer_count_probe_radius=*",
)
_insert_after(
    "train_OWT_core.sh",
    '        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$plastic_value" ;;\n',
    '        --plastic__layer_count_probe__number_of_sampled_valid_tokens) PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS="$plastic_value" ;;\n',
)
_insert_after(
    "train_OWT_core.sh",
    '[[ -z "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] || validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" "PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS"\n',
    'validate_nonnegative_uint "$PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS" "PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS"\n',
)
_insert_after(
    "train_OWT_core.sh",
    '    [[ -n "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] && optional_args+=(--plastic__layer_count_probe__probe_every_n_steps "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS")\n',
    '    optional_args+=(--plastic__layer_count_probe__number_of_sampled_valid_tokens "$PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS")\n',
)

# Help registry: the abbreviation also names the artifact field.
_insert_after(
    "sheet/help_registry_descriptor_patch.py",
    '            ("LPI", "--plastic__layer_count_probe__probe_every_n_steps N", "updates between count probes"),\n',
    '            ("LPT", "--plastic__layer_count_probe__number_of_sampled_valid_tokens N", "valid-token sample per probe microbatch; 0 means all valid tokens"),\n',
)

# Final overlay installs after directional-coherence formatting so existing decision semantics remain untouched.
_insert_after(
    "sheet/__init__.py",
    'from . import plastic_depth_directional_coherence_patch as _plastic_depth_directional_coherence_patch\n_plastic_depth_console_minor_patch._ALIGNMENT_LABELS = ("sampled =", "probe_losses", "score_z", "change_z")\n# ^^^ THOG\n',
    '\n# vvv THOG v0.521 makes probe-token sampling configurable and renders arbitrary-radius probe losses as signed deltas around bold-white L\nfrom . import plastic_depth_probe_sampling_v0521_patch as _plastic_depth_probe_sampling_v0521_patch\n# ^^^ THOG\n',
)

# New final overlay.
_write(
    "sheet/plastic_depth_probe_sampling_v0521_patch.py",
    r'''# vvv THOG
"""PLASTIC v0.521 configurable probe-token sampling and hybrid probe-delta console rendering."""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

import torch

import constants as _constants

from . import plastic_depth_console_cleanup_patch as _cleanup
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_PROBE_VECTOR = re.compile(
    r"probe_losses \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]*)\]"
)
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _plastic_depth_sampled_token_indices_v0521(self: Any, targets: torch.Tensor) -> torch.Tensor:
    flattened = targets.reshape(-1)
    valid = torch.nonzero(flattened != -1, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise RuntimeError("PLASTIC DEPTH inline probe found no non-ignored target tokens")
    requested = int(self.config.plastic__layer_count_probe__number_of_sampled_valid_tokens)
    if requested == 0:
        return valid
    if requested > int(valid.numel()):
        raise RuntimeError(
            "plastic__layer_count_probe__number_of_sampled_valid_tokens exceeds the actual valid-token count "
            f"in this probe microbatch: requested={requested}, valid={int(valid.numel())}"
        )
    if requested == int(valid.numel()):
        return valid
    generator = torch.Generator(device="cpu")
    seed = (
        int(self.config.model_seed)
        + 1_000_003 * int(self.state.completed_updates)
        + 97_409 * int(self.distributed.rank)
    )
    generator.manual_seed(seed)
    positions = torch.randperm(int(valid.numel()), generator=generator)[:requested]
    return valid.index_select(0, positions.to(device=valid.device))


def _format_probe_delta(value: Optional[float]) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:+.3f}"


def _format_probe_absolute(value: Optional[float]) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.3f}"


def _render_probe_delta_values(
    offsets: Sequence[Any],
    losses: Sequence[Any],
) -> Optional[str]:
    resolved_offsets = tuple(int(value) for value in offsets)
    resolved_losses = tuple(None if value is None else float(value) for value in losses)
    if len(resolved_offsets) != len(resolved_losses) or 0 not in resolved_offsets:
        return None
    current_index = resolved_offsets.index(0)
    current_loss = resolved_losses[current_index]
    if current_loss is None or not math.isfinite(current_loss):
        return None
    rendered = []
    for offset, loss in zip(resolved_offsets, resolved_losses):
        if offset == 0:
            rendered.append(
                f"{_constants.BOLD_WHITE}{_format_probe_absolute(loss)}{_constants.R}"
            )
            continue
        delta = None if loss is None else float(loss) - float(current_loss)
        text = _format_probe_delta(delta)
        if delta is not None and math.isfinite(delta) and delta < 0.0:
            text = f"{_cleanup._GREEN}{text}{_cleanup._RESET}"
        rendered.append(text)
    return ", ".join(rendered)


def _format_progress_line_with_probe_deltas(
    run_id: str,
    event: str,
    payload: dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    offsets = payload.get("plastic_probe_offsets")
    losses = payload.get("plastic_probe_losses")
    if offsets is None or losses is None:
        return line
    rendered = _render_probe_delta_values(offsets, losses)
    if rendered is None:
        return line

    def replace(match: re.Match[str]) -> str:
        return f"probe_Δloss [{match.group('label')}] = [{rendered}]"

    return _PROBE_VECTOR.sub(replace, line, count=1)


_trainer_step.TrainerStepMixin._plastic_depth_sampled_token_indices = (
    _plastic_depth_sampled_token_indices_v0521
)
_stage6.format_progress_line = _format_progress_line_with_probe_deltas


__all__ = [
    "_format_probe_delta",
    "_plastic_depth_sampled_token_indices_v0521",
    "_render_probe_delta_values",
]
# ^^^ THOG
''',
)

# Focused regression coverage.
_write(
    "tests/test_plastic_depth_probe_sampling_v0521.py",
    r'''# vvv THOG
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

import constants
from run_thog2_owt_core import build_parser, config_from_arguments
from sheet import plastic_depth_console_cleanup_patch as cleanup
from sheet import plastic_depth_probe_sampling_v0521_patch as probe_v0521
from sheet.run_config import OwtRunConfig
from sheet.trainer_step import TrainerStepMixin
from sheet.training_config import TrainingConfig


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _fake_sampler(*, requested: int, completed_updates: int = 7, rank: int = 0):
    return SimpleNamespace(
        config=SimpleNamespace(
            plastic__layer_count_probe__number_of_sampled_valid_tokens=requested,
            model_seed=1337,
        ),
        state=SimpleNamespace(completed_updates=completed_updates),
        distributed=SimpleNamespace(rank=rank),
    )


def test_probe_token_default_and_runner_resolution_are_1024() -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-type", "sheet"])
    assert args.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024
    config = config_from_arguments(args)
    assert config.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024
    training = config.to_training_config(vocab_size=50304, world_size=1, out_dir=__import__("pathlib").Path("out"))
    assert training.plastic__layer_count_probe__number_of_sampled_valid_tokens == 1024


def test_probe_token_static_capacity_validation_accepts_zero_and_exact_capacity() -> None:
    OwtRunConfig(
        model_type="dense",
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=0,
    )
    OwtRunConfig(
        model_type="dense",
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
    )
    TrainingConfig(
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
    )


@pytest.mark.parametrize("value", [-1, 17])
def test_probe_token_static_capacity_validation_rejects_negative_or_too_large(value: int) -> None:
    with pytest.raises(ValueError, match="plastic__layer_count_probe__number_of_sampled_valid_tokens"):
        OwtRunConfig(
            model_type="dense",
            batch_size=2,
            block_size=8,
            plastic__layer_count_probe__number_of_sampled_valid_tokens=value,
        )


def test_zero_uses_every_valid_token_without_random_subsampling() -> None:
    targets = torch.tensor([[10, 11, -1, 12], [13, -1, 14, 15]], dtype=torch.long)
    sampled = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=0),
        targets,
    )
    assert sampled.tolist() == [0, 1, 3, 4, 6, 7]


def test_positive_probe_token_count_is_exact_deterministic_random_subset() -> None:
    targets = torch.arange(20, dtype=torch.long).reshape(2, 10)
    first = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=7),
        targets,
    )
    second = TrainerStepMixin._plastic_depth_sampled_token_indices(
        _fake_sampler(requested=7),
        targets,
    )
    assert first.numel() == 7
    assert len(set(first.tolist())) == 7
    assert first.tolist() == second.tolist()
    assert first.tolist() != list(range(7))


def test_runtime_rejects_request_above_actual_valid_token_count() -> None:
    targets = torch.tensor([[10, 11, -1, 12], [13, -1, 14, 15]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="requested=7, valid=6"):
        TrainerStepMixin._plastic_depth_sampled_token_indices(
            _fake_sampler(requested=7),
            targets,
        )


def test_hybrid_probe_vector_keeps_absolute_l_and_signed_candidate_deltas() -> None:
    rendered = probe_v0521._render_probe_delta_values(
        (-2, -1, 0, 1, 2),
        (4.100, 4.050, 4.077, 4.134, 4.060),
    )
    assert rendered is not None
    plain = _plain(rendered)
    assert plain == "+0.023, -0.027, 4.077, +0.057, -0.017"
    assert f"{constants.BOLD_WHITE}4.077{constants.R}" in rendered
    assert f"{cleanup._GREEN}-0.027{cleanup._RESET}" in rendered
    assert f"{cleanup._GREEN}-0.017{cleanup._RESET}" in rendered
    assert cleanup._GREEN not in rendered.split(", ")[0]
    assert cleanup._GREEN not in rendered.split(", ")[3]


def test_final_console_renames_probe_vector_to_probe_delta_loss() -> None:
    line = (
        "T 20 layers = 7\tprobe_losses [L-2 ... L+2] = [4.100, 4.050, 4.077, 4.134, 4.060]  "
        "L/R/A=[0/0/6]/6=>stet"
    )
    rendered_values = probe_v0521._render_probe_delta_values(
        (-2, -1, 0, 1, 2),
        (4.100, 4.050, 4.077, 4.134, 4.060),
    )
    assert rendered_values is not None
    replaced = probe_v0521._PROBE_VECTOR.sub(
        lambda match: f"probe_Δloss [{match.group('label')}] = [{rendered_values}]",
        line,
        count=1,
    )
    assert "probe_losses" not in _plain(replaced)
    assert "probe_Δloss [L-2 ... L+2]" in _plain(replaced)
    assert _plain(replaced).endswith("L/R/A=[0/0/6]/6=>stet")


def test_artifact_and_help_registry_name_the_new_probe_token_control() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        plastic__enabled=True,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=4,
        plastic__max_permitted_layers=8,
        n_layer=8,
        o_depth=4,
        batch_size=2,
        block_size=8,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=8,
    )
    assert "LPT_8" in config.parameter_artifact_fragment()
    help_text = build_parser().format_help()
    assert "--plastic__layer_count_probe__number_of_sampled_valid_tokens" in help_text
# ^^^ THOG
''',
)

# Spec v0.521: retain v0.52 history while adding the +0.001 normative delta.
spec_source = _read("docs/THOG2_PLASTIC_Requirements_Specification_v0.52.txt")
spec = spec_source
spec = spec.replace(
    "0.52 | 7 August 2026 | Proposed architecture-control refinement",
    "0.521 | 7 August 2026 | Probe-sampling and operator-console refinement",
    1,
)
spec = spec.replace(
    "VERSION 0.52 DELTA\n",
    "VERSION 0.521 DELTA\n"
    "Version 0.521 is a +0.001 refinement over Version 0.52. It preserves the Version 0.52 controller and decision semantics unchanged while making FINE probe-token sampling explicit and configurable, and replacing the arbitrary-radius raw probe-loss vector with a scan-oriented hybrid delta display. Fresh runs default plastic__layer_count_probe__number_of_sampled_valid_tokens to 1024; zero means all valid target tokens in the first probe microbatch. Positive values request exactly that many valid tokens and are never silently clipped.\n"
    "Console probe vectors retain the absolute current-L probe loss in the centre, rendered bold white. Every L-k/L+k entry is rendered as signed candidate_loss minus L_loss; negative deltas are bold bright green and positive deltas are uncoloured. The public label is probe_Δloss while the positional L-r ... L+r label remains unchanged.\n"
    "Checkpoint migration is semantic: a pre-v0.521 PLASTIC checkpoint that lacks the new field is interpreted as 256 sampled valid tokens, matching the previously hard-coded implementation.\n"
    "\nVERSION 0.52 DELTA (retained historical basis)\n",
    1,
)
spec = spec.replace(
    "PLASTIC_DEPTH remains the implementation base. Version 0.52 supersedes Version 0.51 as the governing requirements document.",
    "PLASTIC_DEPTH remains the implementation base. Version 0.521 supersedes Version 0.52 as the governing requirements document.",
    1,
)
spec = spec.replace(
    "5.6 Objective score and paired differences\n",
    "5.5.1 Probe-token sampling\n"
    "plastic__layer_count_probe__number_of_sampled_valid_tokens is the canonical FINE probe-token sample-size control. The fresh-run default is 1024. A value of 0 means every valid target token in the first probe microbatch; a positive value N means exactly N valid target tokens selected by the existing deterministic random sampler from that microbatch.\n"
    "A configured positive N must not exceed batch_size × block_size. The implementation must reject such impossible configurations before training. At probe time, N must also not exceed the actual number of non-ignored targets in the selected microbatch; if ignored targets reduce the available set below N, probing must fail visibly rather than silently reducing the requested sample. Zero bypasses random subsampling and uses the complete valid-token set.\n"
    "All candidate layer counts in one probe event must continue to use exactly the same selected token positions. The sampler remains confined to the first probe microbatch; representativeness across training data is provided temporally by the multi-probe evidence window rather than by probing every gradient-accumulation microbatch.\n"
    "\n5.6 Objective score and paired differences\n",
    1,
)
spec = spec.replace(
    "The existing raw probe losses remain displayed because they are valuable direct evidence even for objectives that later add time, memory or layer-cost terms.",
    "The console shall not require operators to compare a long vector of raw losses. The current-L entry remains its absolute probe loss and is bold white. Every other displayed probe entry is candidate_loss − L_loss with an explicit sign; negative deltas are bold bright green and positive deltas are uncoloured. The field label is probe_Δloss and the existing positional L−r ... L+r label remains visible. Objective-specific scoring and internal raw losses remain unchanged and auditable.",
    1,
)
acceptance_insert = (
    "V0.521 probe-token and console acceptance additions\n"
    "• Fresh configuration resolves plastic__layer_count_probe__number_of_sampled_valid_tokens=1024 unless explicitly overridden.\n"
    "• Value 0 uses all valid tokens in the first probe microbatch; positive N uses exactly N deterministic randomly sampled valid positions shared by all candidate counts in that probe event.\n"
    "• Negative N, N > batch_size × block_size, and runtime N greater than the actual valid-target count are rejected; no implicit min(requested, available) clipping is permitted.\n"
    "• Pre-v0.521 PLASTIC checkpoints lacking the field retain the historical 256-token meaning.\n"
    "• Arbitrary-radius console vectors render signed candidate-loss deltas around an absolute bold-white L; every negative delta is bold bright green.\n"
    "\n"
)
if acceptance_insert not in spec:
    marker = "12. Known Issues and Limitations\n"
    if marker not in spec:
        raise RuntimeError("spec section-12 marker not found")
    spec = spec.replace(marker, acceptance_insert + marker, 1)
_write("docs/THOG2_PLASTIC_Requirements_Specification_v0.521.txt", spec)

print("Applied PLASTIC v0.521 probe sampling, console rendering, tests and spec.")
# ^^^ THOG
