#!/usr/bin/env python3
# vvv THOG
"""Apply post-migration v0.52 test, compatibility, console-spacing and historical-spec cleanup."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def _replace_once(path: str, old: str, new: str) -> None:
    text = _read(path)
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one cleanup anchor, found {count}: {old!r}")
    _write(path, text.replace(old, new, 1))


def _normalize_console_spacing() -> None:
    path = "sheet/plastic_depth_directional_coherence_patch.py"
    _replace_once(
        path,
        '    line = line.replace("grad norm=  ", "grad norm= ")\n',
        '    line = re.sub(r"grad norm=\\s+", "grad norm= ", line)\n',
    )


def _fix_new_tests() -> None:
    path = "tests/test_plastic_depth_directional_coherence_v052.py"
    text = _read(path)
    text = text.replace(
        '        score_by_count={1: 10.0, 2: 9.0, 3: 11.0},\n',
        '        score_by_count={1: 10.0, 2: 9.0},\n',
    )
    text = text.replace(
        '    score_report = (_score(9, 9.0), _score(10, 10.0), _score(11, 9.0))\n',
        '    score_report = (_score(9, 11.0), _score(10, 10.0), _score(11, 9.0))\n',
        1,
    )
    text = text.replace(
        '            "10:-1": (-1.0, -1.0),\n            "10:+1": (-1.0, -1.0),\n',
        '            "10:-1": (1.0, 1.0),\n            "10:+1": (-1.0, -1.0),\n',
        1,
    )
    text = text.replace(
        '    line = line.replace("loss  =", "loss=").replace("grad norm=  ", "grad norm= ")\n    assert line == "loss=   4.7168  grad norm=  0.269"\n',
        '    line = line.replace("loss  =", "loss=")\n    line = re.sub(r"grad norm=\\s+", "grad norm= ", line)\n    assert line == "loss=   4.7168  grad norm= 0.269"\n',
    )
    _write(path, text)


def _migrate_retired_min_probe_tests() -> None:
    simple_remove = (
        "tests/plastic_depth_ddp_probe.py",
        "tests/plastic_depth_ddp_worker.py",
        "tests/test_plastic_depth_coarse_fine_gpu_smoke.py",
    )
    for path in simple_remove:
        text = _read(path)
        text = "".join(
            line
            for line in text.splitlines(keepends=True)
            if "plastic__layer_count_min_probes=" not in line
        )
        _write(path, text)

    path = "tests/test_plastic_depth_audit.py"
    text = _read(path)
    text = "".join(
        line
        for line in text.splitlines(keepends=True)
        if "plastic__layer_count_min_probes=" not in line
    )
    text = text.replace('            "probe_interval",\n', '            "probe_every_n_steps",\n', 1)
    _write(path, text)

    path = "tests/test_plastic_depth.py"
    text = _read(path)
    text = text.replace('            "plastic__layer_count_min_probes",\n', "")
    text = text.replace(
        '                    plastic__layer_count_probe__window_size_as_number_of_probes=8,\n                    plastic__layer_count_min_probes=1,\n',
        '                    plastic__layer_count_probe__window_size_as_number_of_probes=1,\n',
        1,
    )
    text = "".join(
        line
        for line in text.splitlines(keepends=True)
        if "plastic__layer_count_min_probes=" not in line
    )
    _write(path, text)

    path = "tests/test_plastic_depth_inline_probe.py"
    text = _read(path)
    text = "".join(
        line
        for line in text.splitlines(keepends=True)
        if "plastic__layer_count_min_probes=" not in line
    )
    text = text.replace(
        '    trainer = _learned_trainer(gradient_accumulation_steps=2)\n',
        '    trainer = _learned_trainer(\n        gradient_accumulation_steps=2,\n        plastic__layer_count_probe__window_size_as_number_of_probes=1,\n    )\n',
        1,
    )
    brake_anchor = '''    trainer = _learned_trainer(\n        gradient_accumulation_steps=1,\n        max_updates=6,\n        plastic__layer_count_update_brake=5,\n    )\n    try:\n'''
    brake_replacement = '''    trainer = _learned_trainer(\n        gradient_accumulation_steps=1,\n        max_updates=6,\n        plastic__layer_count_update_brake=5,\n        plastic__layer_count_probe__window_size_as_number_of_probes=5,\n    )\n    trainer.state.plastic_depth_probe_histories = {\n        "3:-1": [1.0, 1.0, 1.0, 1.0],\n        "3:+1": [-1.0, -1.0, -1.0, -1.0],\n        "3:@LRA": [1.0, 1.0, 1.0, 1.0],\n    }\n    try:\n'''
    if brake_replacement not in text:
        if brake_anchor not in text:
            raise RuntimeError(f"{path}: missing five-update-brake anchor")
        text = text.replace(brake_anchor, brake_replacement, 1)
    _write(path, text)

    path = "tests/test_plastic_depth_checkpoint_format.py"
    text = _read(path)
    text = "".join(
        line
        for line in text.splitlines(keepends=True)
        if "plastic__layer_count_min_probes=" not in line
    )
    text = text.replace(
        '        "probe_noise_min_observations": plastic["plastic__layer_count_min_probes"],\n',
        '        "probe_noise_min_observations": 3,\n',
        1,
    )
    _write(path, text)

    path = "tests/test_plastic_depth_startup_lookahead_ui.py"
    text = _read(path)
    text = text.replace(
        '            "plastic__layer_count_update_brake": 30,\n',
        '            "plastic__layer_count_update_brake": 30,\n            "plastic__layer_count_probe__probe_every_n_steps": 10,\n',
        1,
    )
    text = text.replace(
        '            "plastic__layer_count_min_probes": 4,\n',
        '            "plastic__layer_count_extrapolation_weight": 0.8,\n',
        1,
    )
    _write(path, text)

    path = "tests/test_run_thog2_owt_startup_report.py"
    text = _read(path)
    text = text.replace(
        '        plastic__layer_count_update_brake=20,\n',
        '        plastic__layer_count_update_brake=20,\n        plastic__layer_count_probe__probe_every_n_steps=10,\n',
        1,
    )
    text = text.replace(
        '        plastic__layer_count_min_probes=6,\n',
        '        plastic__layer_count_extrapolation_weight=0.8,\n',
        1,
    )
    text = text.replace(
        '    assert "plastic__layer_count_min_probes:" in output\n',
        '    assert "plastic__layer_count_extrapolation_weight:" in output\n',
        1,
    )
    old = '''    min_observation_row = next(\n        line\n        for line in rows\n        if "plastic__layer_count_min_probes:" in line\n    )\n'''
    new = '''    extrapolation_row = next(\n        line\n        for line in rows\n        if "plastic__layer_count_extrapolation_weight:" in line\n    )\n'''
    text = text.replace(old, new, 1)
    text = text.replace(
        '    assert min_observation_row.endswith("   6")\n',
        '    assert extrapolation_row.endswith("   0.8")\n',
        1,
    )
    text = text.replace('    assert "initial layer indices:" in output\n', '    assert "active sample_layer:" in output\n', 1)
    text = text.replace('    assert "capacity layer indices:" in output\n', '    assert "capacity sample_layer:" in output\n', 1)
    text = text.replace('        if "initial layer indices:" in line\n', '        if "active sample_layer:" in line\n', 1)
    text = text.replace('    capacity_row = next(line for line in rows if "capacity layer indices:" in line)\n', '    capacity_row = next(line for line in rows if "capacity sample_layer:" in line)\n', 1)
    _write(path, text)


def _migrate_legacy_checkpoint_semantics() -> None:
    path = "sheet/checkpoints.py"
    text = _read(path)
    old = '''def _semantic_plastic_depth_identity(value: Any, *, maximum_layers: Any) -> Any:\n    if not isinstance(value, Mapping):\n        return value\n    if "plastic__enabled" not in value:\n        return value\n    return {\n        "version": value.get("version"),\n        "maximum_layers": maximum_layers,\n        "initial_active_layers": value.get("plastic__initial_active_layers"),\n        "learn_layer_count": value.get("plastic__do_learn_layer_count"),\n        "sampling_initialisation": value.get("plastic__layer_sampling_initialisation"),\n        "count_objective": value.get("plastic__layer_count_objective"),\n        "count_update_brake": value.get("plastic__layer_count_update_brake"),\n        "extrapolation_weight": value.get("plastic__layer_count_extrapolation_weight"),\n        "probe_noise_window": value.get("plastic__layer_count_probe__window_size_as_number_of_probes"),\n        "probe_noise_lambda": value.get("plastic__layer_count_probe_noise_lambda"),\n        "count_cost_weight": value.get("plastic__layer_count_cost_weight"),\n        "memory_budget_gib": value.get("plastic__layer_memory_budget_gib"),\n        "cuda_allocator_reserve_gib": value.get("plastic__cuda_allocator_reserve_gib"),\n        "geometry_lr_multiplier": value.get("plastic__geometry_learning_rate_multiplier"),\n        "freeze_geometry_during_warmup": value.get("plastic__freeze_geometry_during_warmup"),\n    }\n'''
    new = '''def _semantic_plastic_depth_identity(value: Any, *, maximum_layers: Any) -> Any:\n    if not isinstance(value, Mapping):\n        return value\n    # vvv THOG v0.52 compatibility maps both canonical identities and the retired short v0.3 identity to one semantic form\n    if "plastic__enabled" not in value:\n        return {\n            "version": value.get("version"),\n            "maximum_layers": value.get("maximum_layers", maximum_layers),\n            "initial_active_layers": value.get("initial_active_layers"),\n            "learn_layer_count": value.get("learn_layer_count"),\n            "sampling_initialisation": value.get("sampling_initialisation"),\n            "count_objective": value.get("count_objective"),\n            "count_update_brake": value.get("count_update_brake"),\n            "extrapolation_weight": value.get("extrapolation_weight", 0.8),\n            "probe_noise_window": value.get("probe_noise_window"),\n            "probe_noise_lambda": value.get("probe_noise_lambda"),\n            "count_cost_weight": value.get("count_cost_weight"),\n            "memory_budget_gib": value.get("memory_budget_gib"),\n            "cuda_allocator_reserve_gib": value.get("cuda_allocator_reserve_gib"),\n            "geometry_lr_multiplier": value.get("geometry_lr_multiplier"),\n            "freeze_geometry_during_warmup": value.get("freeze_geometry_during_warmup"),\n        }\n    return {\n        "version": value.get("version"),\n        "maximum_layers": maximum_layers,\n        "initial_active_layers": value.get("plastic__initial_active_layers"),\n        "learn_layer_count": value.get("plastic__do_learn_layer_count"),\n        "sampling_initialisation": value.get("plastic__layer_sampling_initialisation"),\n        "count_objective": value.get("plastic__layer_count_objective"),\n        "count_update_brake": value.get("plastic__layer_count_update_brake"),\n        "extrapolation_weight": value.get("plastic__layer_count_extrapolation_weight", 0.8),\n        "probe_noise_window": value.get("plastic__layer_count_probe__window_size_as_number_of_probes"),\n        "probe_noise_lambda": value.get("plastic__layer_count_probe_noise_lambda"),\n        "count_cost_weight": value.get("plastic__layer_count_cost_weight"),\n        "memory_budget_gib": value.get("plastic__layer_memory_budget_gib"),\n        "cuda_allocator_reserve_gib": value.get("plastic__cuda_allocator_reserve_gib"),\n        "geometry_lr_multiplier": value.get("plastic__geometry_learning_rate_multiplier"),\n        "freeze_geometry_during_warmup": value.get("plastic__freeze_geometry_during_warmup"),\n    }\n    # ^^^ THOG\n'''
    if new not in text:
        if old not in text:
            raise RuntimeError(f"{path}: missing semantic PLASTIC identity anchor")
        text = text.replace(old, new, 1)
    _write(path, text)


def _restore_historical_v04_names() -> None:
    path = "docs/THOG2_PLASTIC_DEPTH_Requirements_Specification_v0.4.txt"
    text = _read(path)
    text = text.replace(
        "plastic__layer_count_probe__probe_every_n_steps",
        "plastic__layer_count_probe_window_size",
    )
    text = text.replace(
        "plastic__layer_count_probe__window_size_as_number_of_probes",
        "plastic__layer_count_probe_noise_window",
    )
    _write(path, text)


def _update_as_built_summary() -> None:
    path = "docs/PLASTIC_DEPTH_AS_BUILT_SPECIFICATION.md"
    text = _read(path)
    text = text.replace(
        "- `plastic__layer_count_min_probes`: minimum observations before the significance gate can trigger.\n",
        "- Full-window readiness: no separate minimum-probes knob; count movement is ineligible until the complete configured probe-history window is present.\n",
    )
    if "plastic__layer_count_extrapolation_weight" not in text:
        anchor = "- `plastic__layer_count_probe_noise_lambda`: robust MAD significance multiplier.\n"
        text = text.replace(
            anchor,
            "- `plastic__layer_count_extrapolation_weight`: right/up directional-credibility and distance discount; default 0.8.\n" + anchor,
            1,
        )
    _write(path, text)


def main() -> None:
    _normalize_console_spacing()
    _fix_new_tests()
    _migrate_retired_min_probe_tests()
    _migrate_legacy_checkpoint_semantics()
    _restore_historical_v04_names()
    _update_as_built_summary()


if __name__ == "__main__":
    main()
# ^^^ THOG
