from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_plastic_field_names(path: str, tuple_name: str) -> None:
    replace_once(
        path,
        f'{tuple_name} = (\n    "plastic__enabled",\n',
        f'{tuple_name} = (\n    "plastic__enabled",\n'
        '    "plastic__coarse_phase",\n'
        '    "plastic__phase_1_n_steps",\n'
        '    "plastic__phase_1_starting_layer_count",\n'
        '    "plastic__phase_1__number_of_trials",\n'
        '    "plastic__phase_1_evaluation_steps_count",\n',
    )
    replace_once(
        path,
        '    "plastic__layer_count_update_brake",\n'
        '    "plastic__layer_count_probe__window_size_as_number_of_probes",\n',
        '    "plastic__layer_count_update_brake",\n'
        '    "plastic__layer_count_probe__probe_every_n_steps",\n'
        '    "plastic__layer_count_probe_radius",\n'
        '    "plastic__layer_count_max_step",\n'
        '    "plastic__layer_count_probe__window_size_as_number_of_probes",\n',
    )


def insert_plastic_dataclass_fields(path: str, frozen: bool) -> None:
    replace_once(
        path,
        '    plastic__enabled: bool = False\n'
        '    plastic__layers_to_sample: Optional[int] = None\n',
        '    plastic__enabled: bool = False\n'
        '    plastic__coarse_phase: str = "disabled"\n'
        '    plastic__phase_1_n_steps: Optional[int] = None\n'
        '    plastic__phase_1_starting_layer_count: Optional[int] = None\n'
        '    plastic__phase_1__number_of_trials: Optional[int] = None\n'
        '    plastic__phase_1_evaluation_steps_count: Optional[int] = None\n'
        '    plastic__layers_to_sample: Optional[int] = None\n',
    )
    replace_once(
        path,
        '    plastic__layer_count_update_brake: int = 5\n'
        '    plastic__layer_count_probe__window_size_as_number_of_probes: int = 50\n',
        '    plastic__layer_count_update_brake: int = 5\n'
        '    plastic__layer_count_probe__probe_every_n_steps: Optional[int] = None\n'
        '    plastic__layer_count_probe_radius: int = 1\n'
        '    plastic__layer_count_max_step: int = 1\n'
        '    plastic__layer_count_probe__window_size_as_number_of_probes: int = 50\n',
    )


def insert_coarse_import(path: str) -> None:
    anchor = 'from .residual_init import '
    text = (ROOT / path).read_text(encoding="utf-8")
    index = text.find(anchor)
    if index < 0:
        raise RuntimeError(f"{path}: residual import anchor missing")
    block = (
        '# vvv THOG PLASTIC COARSE/FINE lifecycle configuration and candidate resolution\n'
        'from .plastic_depth_coarse import (\n'
        '    resolve_plastic_coarse_config,\n'
        '    resolve_plastic_probe_interval,\n'
        '    validate_plastic_fine_count_controls,\n'
        ')\n'
        '# ^^^ THOG\n'
    )
    text = text[:index] + block + text[index:]
    (ROOT / path).write_text(text, encoding="utf-8")


def insert_config_validation(path: str, frozen: bool) -> None:
    old = (
        '        resolved_plastic_counts = resolve_plastic_depth_counts(\n'
        '            n_layer=self.n_layer,\n'
        '            enabled=self.plastic__enabled,\n'
        '            layers_to_sample=self.plastic__layers_to_sample,\n'
        '            do_learn_layer_count=self.plastic__do_learn_layer_count,\n'
        '            initial_layer_count=self.plastic__initial_layer_count,\n'
        '            max_permitted_layers=self.plastic__max_permitted_layers,\n'
        '        )\n'
    )
    setter = (
        '        object.__setattr__(self, "plastic__layer_count_probe__probe_every_n_steps", resolved_probe_interval)\n'
        if frozen
        else '        self.plastic__layer_count_probe__probe_every_n_steps = resolved_probe_interval\n'
    )
    new = (
        '        # vvv THOG resolve one-shot COARSE scheduling and canonical FINE lookahead controls before active-count construction\n'
        '        resolved_coarse = resolve_plastic_coarse_config(\n'
        '            coarse_phase=self.plastic__coarse_phase,\n'
        '            plastic_enabled=self.plastic__enabled,\n'
        '            do_learn_layer_count=self.plastic__do_learn_layer_count,\n'
        '            n_steps=self.plastic__phase_1_n_steps,\n'
        '            starting_layer_count=self.plastic__phase_1_starting_layer_count,\n'
        '            number_of_trials=self.plastic__phase_1__number_of_trials,\n'
        '            evaluation_steps_count=self.plastic__phase_1_evaluation_steps_count,\n'
        '            max_permitted_layers=self.plastic__max_permitted_layers,\n'
        '        )\n'
        '        resolved_probe_interval = resolve_plastic_probe_interval(\n'
        '            probe_interval=self.plastic__layer_count_probe__probe_every_n_steps,\n'
        '            update_brake=self.plastic__layer_count_update_brake,\n'
        '            enabled=self.plastic__enabled,\n'
        '            do_learn_layer_count=self.plastic__do_learn_layer_count,\n'
        '        )\n'
        + setter +
        '        validate_plastic_fine_count_controls(\n'
        '            probe_radius=self.plastic__layer_count_probe_radius,\n'
        '            max_step=self.plastic__layer_count_max_step,\n'
        '        )\n'
        '        initial_layer_count_for_resolution = (\n'
        '            resolved_coarse.candidate_layers[0]\n'
        '            if resolved_coarse.enabled\n'
        '            else self.plastic__initial_layer_count\n'
        '        )\n'
        '        # ^^^ THOG\n'
        '        resolved_plastic_counts = resolve_plastic_depth_counts(\n'
        '            n_layer=self.n_layer,\n'
        '            enabled=self.plastic__enabled,\n'
        '            layers_to_sample=self.plastic__layers_to_sample,\n'
        '            do_learn_layer_count=self.plastic__do_learn_layer_count,\n'
        '            initial_layer_count=initial_layer_count_for_resolution,\n'
        '            max_permitted_layers=self.plastic__max_permitted_layers,\n'
        '        )\n'
    )
    replace_once(path, old, new)


def update_identity_function() -> None:
    path = "sheet/plastic_depth.py"
    replace_once(path, 'PLASTIC_DEPTH_VERSION = "plastic_depth_v0_3"', 'PLASTIC_DEPTH_VERSION = "plastic_depth_v0_4"')
    replace_once(
        path,
        '    *,\n'
        '    layers_to_sample: Optional[int],\n',
        '    *,\n'
        '    coarse_phase: str = "disabled",\n'
        '    phase_1_n_steps: Optional[int] = None,\n'
        '    phase_1_starting_layer_count: Optional[int] = None,\n'
        '    phase_1_number_of_trials: Optional[int] = None,\n'
        '    phase_1_evaluation_steps_count: Optional[int] = None,\n'
        '    layer_count_probe__probe_every_n_steps: Optional[int] = None,\n'
        '    layer_count_probe_radius: int = 1,\n'
        '    layer_count_max_step: int = 1,\n'
        '    layers_to_sample: Optional[int],\n',
    )
    replace_once(
        path,
        '        "plastic__enabled": True,\n'
        '        "plastic__layers_to_sample": layers_to_sample,\n',
        '        "plastic__enabled": True,\n'
        '        "plastic__coarse_phase": coarse_phase,\n'
        '        "plastic__phase_1_n_steps": phase_1_n_steps,\n'
        '        "plastic__phase_1_starting_layer_count": phase_1_starting_layer_count,\n'
        '        "plastic__phase_1__number_of_trials": phase_1_number_of_trials,\n'
        '        "plastic__phase_1_evaluation_steps_count": phase_1_evaluation_steps_count,\n'
        '        "plastic__layer_count_probe__probe_every_n_steps": layer_count_probe__probe_every_n_steps,\n'
        '        "plastic__layer_count_probe_radius": int(layer_count_probe_radius),\n'
        '        "plastic__layer_count_max_step": int(layer_count_max_step),\n'
        '        "plastic__layers_to_sample": layers_to_sample,\n',
    )


def add_identity_call_arguments(path: str) -> None:
    replace_once(
        path,
        '            identity["plastic_depth"] = plastic_depth_identity_metadata(\n'
        '                layers_to_sample=self.plastic__layers_to_sample,\n',
        '            identity["plastic_depth"] = plastic_depth_identity_metadata(\n'
        '                coarse_phase=self.plastic__coarse_phase,\n'
        '                phase_1_n_steps=self.plastic__phase_1_n_steps,\n'
        '                phase_1_starting_layer_count=self.plastic__phase_1_starting_layer_count,\n'
        '                phase_1_number_of_trials=self.plastic__phase_1__number_of_trials,\n'
        '                phase_1_evaluation_steps_count=self.plastic__phase_1_evaluation_steps_count,\n'
        '                layer_count_probe__probe_every_n_steps=self.plastic__layer_count_probe__probe_every_n_steps,\n'
        '                layer_count_probe_radius=self.plastic__layer_count_probe_radius,\n'
        '                layer_count_max_step=self.plastic__layer_count_max_step,\n'
        '                layers_to_sample=self.plastic__layers_to_sample,\n',
    )


def update_run_config_propagation() -> None:
    path = "sheet/run_config.py"
    replace_once(
        path,
        '            plastic__enabled=self.plastic__enabled,\n'
        '            plastic__layers_to_sample=self.plastic__layers_to_sample,\n',
        '            plastic__enabled=self.plastic__enabled,\n'
        '            plastic__coarse_phase=self.plastic__coarse_phase,\n'
        '            plastic__phase_1_n_steps=self.plastic__phase_1_n_steps,\n'
        '            plastic__phase_1_starting_layer_count=self.plastic__phase_1_starting_layer_count,\n'
        '            plastic__phase_1__number_of_trials=self.plastic__phase_1__number_of_trials,\n'
        '            plastic__phase_1_evaluation_steps_count=self.plastic__phase_1_evaluation_steps_count,\n'
        '            plastic__layers_to_sample=self.plastic__layers_to_sample,\n',
    )
    replace_once(
        path,
        '            plastic__layer_count_update_brake=self.plastic__layer_count_update_brake,\n'
        '            plastic__layer_count_probe__window_size_as_number_of_probes=self.plastic__layer_count_probe__window_size_as_number_of_probes,\n',
        '            plastic__layer_count_update_brake=self.plastic__layer_count_update_brake,\n'
        '            plastic__layer_count_probe__probe_every_n_steps=self.plastic__layer_count_probe__probe_every_n_steps,\n'
        '            plastic__layer_count_probe_radius=self.plastic__layer_count_probe_radius,\n'
        '            plastic__layer_count_max_step=self.plastic__layer_count_max_step,\n'
        '            plastic__layer_count_probe__window_size_as_number_of_probes=self.plastic__layer_count_probe__window_size_as_number_of_probes,\n',
    )
    replace_once(
        path,
        '            plastic_fields = [\n'
        '                f"PLN_{self.plastic__initial_active_layers}",\n',
        '            plastic_fields = [\n'
        '                f"PLN_{self.plastic__initial_active_layers}",\n',
    )
    replace_once(
        path,
        '            if self.plastic__do_learn_layer_count:\n'
        '                plastic_fields.extend([\n',
        '            if self.plastic__coarse_phase == "enabled":\n'
        '                plastic_fields.extend([\n'
        '                    f"PLC_{self.plastic__phase_1_starting_layer_count}",\n'
        '                    f"PLCS_{self.plastic__phase_1_n_steps}",\n'
        '                    f"PLCT_{self.plastic__phase_1__number_of_trials}",\n'
        '                    f"PLCE_{self.plastic__phase_1_evaluation_steps_count}",\n'
        '                ])\n'
        '            if self.plastic__do_learn_layer_count:\n'
        '                plastic_fields.extend([\n'
        '                    f"PLPI_{self.plastic__layer_count_probe__probe_every_n_steps}",\n'
        '                    f"PLPR_{self.plastic__layer_count_probe_radius}",\n'
        '                    f"PLMS_{self.plastic__layer_count_max_step}",\n',
    )


def update_core_parser_and_mapping() -> None:
    path = "run_thog2_owt_core.py"
    replace_once(
        path,
        '    parser.add_argument("--plastic-enabled", dest="plastic__enabled", action=argparse.BooleanOptionalAction, default=False)\n'
        '    parser.add_argument("--plastic-layers-to-sample", dest="plastic__layers_to_sample", type=int)\n',
        '    parser.add_argument("--plastic-enabled", dest="plastic__enabled", action=argparse.BooleanOptionalAction, default=False)\n'
        '    parser.add_argument("--plastic-coarse-phase", dest="plastic__coarse_phase", choices=("enabled", "disabled"), default="disabled")\n'
        '    parser.add_argument("--plastic-phase-1-n-steps", dest="plastic__phase_1_n_steps", type=int)\n'
        '    parser.add_argument("--plastic-phase-1-starting-layer-count", dest="plastic__phase_1_starting_layer_count", type=int)\n'
        '    parser.add_argument("--plastic-phase-1-number-of-trials", dest="plastic__phase_1__number_of_trials", type=int)\n'
        '    parser.add_argument("--plastic-phase-1-evaluation-steps-count", dest="plastic__phase_1_evaluation_steps_count", type=int)\n'
        '    parser.add_argument("--plastic-layers-to-sample", dest="plastic__layers_to_sample", type=int)\n',
    )
    replace_once(
        path,
        '    parser.add_argument("--plastic-layer-count-update-brake", dest="plastic__layer_count_update_brake", type=int, default=5)\n'
        '    parser.add_argument("--plastic-layer-count-probe-noise-window", dest="plastic__layer_count_probe__window_size_as_number_of_probes", type=int, default=50)\n',
        '    parser.add_argument("--plastic-layer-count-update-brake", dest="plastic__layer_count_update_brake", type=int, default=5)\n'
        '    parser.add_argument("--plastic-layer-count-probe-interval", dest="plastic__layer_count_probe__probe_every_n_steps", type=int)\n'
        '    parser.add_argument("--plastic-layer-count-probe-radius", dest="plastic__layer_count_probe_radius", type=int, default=1)\n'
        '    parser.add_argument("--plastic-layer-count-max-step", dest="plastic__layer_count_max_step", type=int, default=1)\n'
        '    parser.add_argument("--plastic-layer-count-probe-noise-window", dest="plastic__layer_count_probe__window_size_as_number_of_probes", type=int, default=50)\n',
    )
    replace_once(
        path,
        '        plastic__enabled=arguments.plastic__enabled,\n'
        '        plastic__layers_to_sample=arguments.plastic__layers_to_sample,\n',
        '        plastic__enabled=arguments.plastic__enabled,\n'
        '        plastic__coarse_phase=arguments.plastic__coarse_phase,\n'
        '        plastic__phase_1_n_steps=arguments.plastic__phase_1_n_steps,\n'
        '        plastic__phase_1_starting_layer_count=arguments.plastic__phase_1_starting_layer_count,\n'
        '        plastic__phase_1__number_of_trials=arguments.plastic__phase_1__number_of_trials,\n'
        '        plastic__phase_1_evaluation_steps_count=arguments.plastic__phase_1_evaluation_steps_count,\n'
        '        plastic__layers_to_sample=arguments.plastic__layers_to_sample,\n',
    )
    replace_once(
        path,
        '        plastic__layer_count_update_brake=arguments.plastic__layer_count_update_brake,\n'
        '        plastic__layer_count_probe__window_size_as_number_of_probes=arguments.plastic__layer_count_probe__window_size_as_number_of_probes,\n',
        '        plastic__layer_count_update_brake=arguments.plastic__layer_count_update_brake,\n'
        '        plastic__layer_count_probe__probe_every_n_steps=arguments.plastic__layer_count_probe__probe_every_n_steps,\n'
        '        plastic__layer_count_probe_radius=arguments.plastic__layer_count_probe_radius,\n'
        '        plastic__layer_count_max_step=arguments.plastic__layer_count_max_step,\n'
        '        plastic__layer_count_probe__window_size_as_number_of_probes=arguments.plastic__layer_count_probe__window_size_as_number_of_probes,\n',
    )


def update_startup_report() -> None:
    path = "run_thog2_owt.py"
    replace_once(
        path,
        '    "plastic__enabled:",\n'
        '    "resolved count mode:",\n',
        '    "plastic__enabled:",\n'
        '    "plastic__coarse_phase:",\n'
        '    "plastic__phase_1_n_steps:",\n'
        '    "plastic__phase_1_starting_layer_count:",\n'
        '    "plastic__phase_1__number_of_trials:",\n'
        '    "plastic__phase_1_evaluation_steps_count:",\n'
        '    "coarse candidate layers:",\n'
        '    "resolved count mode:",\n',
    )
    replace_once(
        path,
        '    _print_plastic_option("plastic__enabled:", _startup_bool(config.plastic__enabled))\n'
        '    _print_plastic_option("resolved count mode:", "learned" if config.plastic__do_learn_layer_count else "fixed")\n',
        '    _print_plastic_option("plastic__enabled:", _startup_bool(config.plastic__enabled))\n'
        '    _print_plastic_option("plastic__coarse_phase:", str(config.plastic__coarse_phase))\n'
        '    _print_plastic_option("plastic__phase_1_n_steps:", _startup_optional(config.plastic__phase_1_n_steps))\n'
        '    _print_plastic_option("plastic__phase_1_starting_layer_count:", _startup_optional(config.plastic__phase_1_starting_layer_count))\n'
        '    _print_plastic_option("plastic__phase_1__number_of_trials:", _startup_optional(config.plastic__phase_1__number_of_trials))\n'
        '    _print_plastic_option("plastic__phase_1_evaluation_steps_count:", _startup_optional(config.plastic__phase_1_evaluation_steps_count))\n'
        '    if config.plastic__coarse_phase == "enabled":\n'
        '        from sheet.plastic_depth_coarse import resolve_plastic_coarse_config\n'
        '        coarse = resolve_plastic_coarse_config(\n'
        '            coarse_phase=config.plastic__coarse_phase,\n'
        '            plastic_enabled=config.plastic__enabled,\n'
        '            do_learn_layer_count=config.plastic__do_learn_layer_count,\n'
        '            n_steps=config.plastic__phase_1_n_steps,\n'
        '            starting_layer_count=config.plastic__phase_1_starting_layer_count,\n'
        '            number_of_trials=config.plastic__phase_1__number_of_trials,\n'
        '            evaluation_steps_count=config.plastic__phase_1_evaluation_steps_count,\n'
        '            max_permitted_layers=config.plastic__max_permitted_layers,\n'
        '        )\n'
        '        _print_plastic_option("coarse candidate layers:", ", ".join(str(value) for value in coarse.candidate_layers))\n'
        '    _print_plastic_option("resolved count mode:", "learned" if config.plastic__do_learn_layer_count else "fixed")\n',
    )
    replace_once(
        path,
        '    probe_radius = int(getattr(config, "plastic__layer_count_probe_radius", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS", 1)))\n'
        '    max_step = int(getattr(config, "plastic__layer_count_max_step", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_MAX_STEP", 1)))\n',
        '    probe_interval = getattr(config, "plastic__layer_count_probe__probe_every_n_steps", None)\n'
        '    probe_radius = int(config.plastic__layer_count_probe_radius)\n'
        '    max_step = int(config.plastic__layer_count_max_step)\n',
    )
    replace_once(
        path,
        '    _print_plastic_option("plastic__layer_count_update_brake:", str(config.plastic__layer_count_update_brake))\n'
        '    _print_plastic_option("plastic__layer_count_probe_radius:", str(probe_radius))\n',
        '    _print_plastic_option("plastic__layer_count_update_brake:", str(config.plastic__layer_count_update_brake))\n'
        '    _print_plastic_option("plastic__layer_count_probe__probe_every_n_steps:", _startup_optional(probe_interval))\n'
        '    _print_plastic_option("plastic__layer_count_probe_radius:", str(probe_radius))\n',
    )
    replace_once(
        path,
        '    "plastic__layer_count_update_brake:",\n'
        '    "plastic__layer_count_probe_radius:",\n',
        '    "plastic__layer_count_update_brake:",\n'
        '    "plastic__layer_count_probe__probe_every_n_steps:",\n'
        '    "plastic__layer_count_probe_radius:",\n',
    )


def update_lifecycle_material_mapping() -> None:
    path = "run_thog2_lifecycle.py"
    replace_once(
        path,
        '    "data_seed": "data_seed",\n'
        '    "artifact_name_limit": "artifact_name_limit",\n',
        '    "data_seed": "data_seed",\n'
        '    "plastic__coarse_phase": "plastic__coarse_phase",\n'
        '    "plastic__phase_1_n_steps": "plastic__phase_1_n_steps",\n'
        '    "plastic__phase_1_starting_layer_count": "plastic__phase_1_starting_layer_count",\n'
        '    "plastic__phase_1__number_of_trials": "plastic__phase_1__number_of_trials",\n'
        '    "plastic__phase_1_evaluation_steps_count": "plastic__phase_1_evaluation_steps_count",\n'
        '    "plastic__layer_count_probe__probe_every_n_steps": "plastic__layer_count_probe__probe_every_n_steps",\n'
        '    "plastic__layer_count_probe_radius": "plastic__layer_count_probe_radius",\n'
        '    "plastic__layer_count_max_step": "plastic__layer_count_max_step",\n'
        '    "artifact_name_limit": "artifact_name_limit",\n',
    )


def main() -> None:
    insert_coarse_import("sheet/training_config.py")
    insert_coarse_import("sheet/run_config.py")
    insert_plastic_field_names("sheet/training_config.py", "PLASTIC_TRAINING_CONFIG_FIELDS")
    insert_plastic_field_names("sheet/run_config.py", "PLASTIC_RUN_CONFIG_FIELDS")
    insert_plastic_dataclass_fields("sheet/training_config.py", frozen=False)
    insert_plastic_dataclass_fields("sheet/run_config.py", frozen=True)
    insert_config_validation("sheet/training_config.py", frozen=False)
    insert_config_validation("sheet/run_config.py", frozen=True)
    update_identity_function()
    add_identity_call_arguments("sheet/training_config.py")
    add_identity_call_arguments("sheet/run_config.py")
    update_run_config_propagation()
    update_core_parser_and_mapping()
    update_startup_report()
    update_lifecycle_material_mapping()


if __name__ == "__main__":
    main()
