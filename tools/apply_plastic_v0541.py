from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_EXTRAPOLATION = "plastic__layer_count_extrapolation_weight"
NEW_EXTRAPOLATION = "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence"
OLD_MAX_STEP = "plastic__layer_count_max_step"
NEW_MAX_STEP = "plastic__layer_count__max_allowable_layer_change"

DISCOUNT = "plastic__wall_time_equivalent_time_gain_discount"
LOSS_WINDOW = "plastic__wall_time_equivalent_time_gain_loss_rate_window"
LOSS_MIN = "plastic__wall_time_equivalent_time_gain_loss_rate_min_observations"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match for {old!r}, found {count}")
    write(path, text.replace(old, new, 1))


def replace_all(path: str, old: str, new: str, *, minimum: int = 1) -> None:
    text = read(path)
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f"{path}: expected at least {minimum} matches for {old!r}, found {count}")
    write(path, text.replace(old, new))


def active_sources() -> list[Path]:
    paths: set[Path] = set()
    for pattern in ("*.py", "sheet/**/*.py", "tests/**/*.py"):
        paths.update(ROOT.glob(pattern))
    for name in (
        "train_OWT.sh",
        "train_OWT_core.sh",
        "plastic_depth_lookahead_wrapper_options.sh",
    ):
        path = ROOT / name
        if path.exists():
            paths.add(path)
    paths.discard(ROOT / "tools" / "apply_plastic_v0541.py")
    return sorted(paths)


# vvv THOG v0.541 canonical public layer-count control names replace the superseded v0.531 spellings across active code/tests
for path in active_sources():
    text = path.read_text(encoding="utf-8")
    text = text.replace(OLD_EXTRAPOLATION, NEW_EXTRAPOLATION)
    text = text.replace(OLD_MAX_STEP, NEW_MAX_STEP)
    path.write_text(text, encoding="utf-8")
# ^^^ THOG


# vvv THOG v0.541 persist and validate the three equivalent-time-gain algorithm controls in TrainingConfig
path = "sheet/training_config.py"
replace_once(path, "from typing import Any, Dict, Optional\n", "from typing import Any, Dict, Mapping, Optional\n")
replace_once(
    path,
    f'    "plastic__layer_count_probe_noise_lambda",\n    "plastic__layer_count_cost_weight",',
    f'    "plastic__layer_count_probe_noise_lambda",\n    "{DISCOUNT}",\n    "{LOSS_WINDOW}",\n    "{LOSS_MIN}",\n    "plastic__layer_count_cost_weight",',
)
replace_once(
    path,
    "    plastic__layer_count_probe_noise_lambda: float = 3.0\n    plastic__layer_count_cost_weight: float = 0.0",
    f"    plastic__layer_count_probe_noise_lambda: float = 3.0\n    {DISCOUNT}: float = 0.9\n    {LOSS_WINDOW}: int = 64\n    {LOSS_MIN}: int = 16\n    plastic__layer_count_cost_weight: float = 0.0",
)
replace_once(
    path,
    "        # ^^^ THOG\n        if (\n            isinstance(self.plastic__layer_count_cost_weight, bool)",
    f'''        # ^^^ THOG\n        # vvv THOG v0.541 public equivalent-time-gain controls are explicit, bounded and checkpoint-persistent\n        if (\n            isinstance(self.{DISCOUNT}, bool)\n            or not isinstance(self.{DISCOUNT}, (int, float))\n            or not math.isfinite(float(self.{DISCOUNT}))\n            or not (0.0 <= float(self.{DISCOUNT}) <= 1.0)\n        ):\n            raise ValueError(\n                "{DISCOUNT} must be finite and lie in [0, 1]; "\n                f"got {{self.{DISCOUNT}!r}}"\n            )\n        if (\n            isinstance(self.{LOSS_WINDOW}, bool)\n            or not isinstance(self.{LOSS_WINDOW}, int)\n            or self.{LOSS_WINDOW} < 2\n        ):\n            raise ValueError(\n                "{LOSS_WINDOW} must be an integer >= 2; "\n                f"got {{self.{LOSS_WINDOW}!r}}"\n            )\n        if (\n            isinstance(self.{LOSS_MIN}, bool)\n            or not isinstance(self.{LOSS_MIN}, int)\n            or self.{LOSS_MIN} < 2\n            or self.{LOSS_MIN} > self.{LOSS_WINDOW}\n        ):\n            raise ValueError(\n                "{LOSS_MIN} must be an integer in [2, {LOSS_WINDOW}]; "\n                f"got {{self.{LOSS_MIN}!r}}"\n            )\n        # ^^^ THOG\n        if (\n            isinstance(self.plastic__layer_count_cost_weight, bool)''',
)
replace_once(
    path,
    ")\n# ^^^ THOG\n\nMODEL_COMPATIBILITY_FIELDS = (",
    f''')\n# ^^^ THOG\n\n# vvv THOG v0.541 accept superseded PLASTIC config keys only when reconstructing existing checkpoints; new writes use canonical names\nPLASTIC_V0541_RENAMED_CONFIG_FIELDS = {{\n    "{OLD_EXTRAPOLATION}": "{NEW_EXTRAPOLATION}",\n    "{OLD_MAX_STEP}": "{NEW_MAX_STEP}",\n}}\n\ndef normalize_plastic_v0541_config_fields(values: Mapping[str, Any]) -> Dict[str, Any]:\n    normalized = dict(values)\n    for old_name, new_name in PLASTIC_V0541_RENAMED_CONFIG_FIELDS.items():\n        if old_name not in normalized:\n            continue\n        old_value = normalized.pop(old_name)\n        if new_name in normalized and normalized[new_name] != old_value:\n            raise ValueError(\n                f"conflicting PLASTIC checkpoint fields {{old_name}} and {{new_name}}"\n            )\n        normalized[new_name] = old_value\n    return normalized\n# ^^^ THOG\n\nMODEL_COMPATIBILITY_FIELDS = (''',
)
# ^^^ THOG


# vvv THOG v0.541 expose the same controls in OwtRunConfig and propagate them into TrainingConfig
path = "sheet/run_config.py"
replace_once(
    path,
    f'    "plastic__layer_count_probe_noise_lambda",\n    "plastic__layer_count_cost_weight",',
    f'    "plastic__layer_count_probe_noise_lambda",\n    "{DISCOUNT}",\n    "{LOSS_WINDOW}",\n    "{LOSS_MIN}",\n    "plastic__layer_count_cost_weight",',
)
replace_once(
    path,
    "    plastic__layer_count_probe_noise_lambda: float = 3.0\n    plastic__layer_count_cost_weight: float = 0.0",
    f"    plastic__layer_count_probe_noise_lambda: float = 3.0\n    {DISCOUNT}: float = 0.9\n    {LOSS_WINDOW}: int = 64\n    {LOSS_MIN}: int = 16\n    plastic__layer_count_cost_weight: float = 0.0",
)
replace_once(
    path,
    "        # ^^^ THOG\n        if (\n            isinstance(self.plastic__layer_count_cost_weight, bool)",
    f'''        # ^^^ THOG\n        # vvv THOG v0.541 public equivalent-time-gain controls mirror TrainingConfig validation\n        if (\n            isinstance(self.{DISCOUNT}, bool)\n            or not isinstance(self.{DISCOUNT}, (int, float))\n            or not math.isfinite(float(self.{DISCOUNT}))\n            or not (0.0 <= float(self.{DISCOUNT}) <= 1.0)\n        ):\n            raise ValueError("{DISCOUNT} must be finite and lie in [0, 1]")\n        if (\n            isinstance(self.{LOSS_WINDOW}, bool)\n            or not isinstance(self.{LOSS_WINDOW}, int)\n            or self.{LOSS_WINDOW} < 2\n        ):\n            raise ValueError("{LOSS_WINDOW} must be an integer >= 2")\n        if (\n            isinstance(self.{LOSS_MIN}, bool)\n            or not isinstance(self.{LOSS_MIN}, int)\n            or self.{LOSS_MIN} < 2\n            or self.{LOSS_MIN} > self.{LOSS_WINDOW}\n        ):\n            raise ValueError(\n                "{LOSS_MIN} must lie in [2, {LOSS_WINDOW}]"\n            )\n        # ^^^ THOG\n        if (\n            isinstance(self.plastic__layer_count_cost_weight, bool)''',
)
replace_once(
    path,
    "            plastic__layer_count_probe_noise_lambda=float(self.plastic__layer_count_probe_noise_lambda),\n            plastic__layer_count_cost_weight=float(self.plastic__layer_count_cost_weight),",
    f"            plastic__layer_count_probe_noise_lambda=float(self.plastic__layer_count_probe_noise_lambda),\n            {DISCOUNT}=float(self.{DISCOUNT}),\n            {LOSS_WINDOW}=self.{LOSS_WINDOW},\n            {LOSS_MIN}=self.{LOSS_MIN},\n            plastic__layer_count_cost_weight=float(self.plastic__layer_count_cost_weight),",
)
# ^^^ THOG


# vvv THOG v0.541 include the new wall-time controls in canonical PLASTIC identity while retaining neutral defaults for older identities
path = "sheet/plastic_depth.py"
replace_once(
    path,
    "    layer_count_extrapolation_weight: float = 0.8,\n    layers_to_sample: Optional[int],",
    "    layer_count_extrapolation_weight: float = 0.8,\n    wall_time_equivalent_time_gain_discount: float = 0.9,\n    wall_time_equivalent_time_gain_loss_rate_window: int = 64,\n    wall_time_equivalent_time_gain_loss_rate_min_observations: int = 16,\n    layers_to_sample: Optional[int],",
)
replace_once(
    path,
    f'        "{NEW_EXTRAPOLATION}": float(layer_count_extrapolation_weight),\n        "plastic__layers_to_sample": layers_to_sample,',
    f'        "{NEW_EXTRAPOLATION}": float(layer_count_extrapolation_weight),\n        "{DISCOUNT}": float(wall_time_equivalent_time_gain_discount),\n        "{LOSS_WINDOW}": int(wall_time_equivalent_time_gain_loss_rate_window),\n        "{LOSS_MIN}": int(wall_time_equivalent_time_gain_loss_rate_min_observations),\n        "plastic__layers_to_sample": layers_to_sample,',
)
for config_path in ("sheet/run_config.py", "sheet/training_config.py"):
    replace_once(
        config_path,
        f"                layer_count_extrapolation_weight=float(self.{NEW_EXTRAPOLATION}),\n                layers_to_sample=self.plastic__layers_to_sample,",
        f"                layer_count_extrapolation_weight=float(self.{NEW_EXTRAPOLATION}),\n                wall_time_equivalent_time_gain_discount=float(self.{DISCOUNT}),\n                wall_time_equivalent_time_gain_loss_rate_window=self.{LOSS_WINDOW},\n                wall_time_equivalent_time_gain_loss_rate_min_observations=self.{LOSS_MIN},\n                layers_to_sample=self.plastic__layers_to_sample,",
    )
# ^^^ THOG


# vvv THOG v0.541 argparse exposes only the renamed layer-count controls plus the three equivalent-time-gain controls
path = "run_thog2_owt_core.py"
replace_once(
    path,
    "from sheet.training_config import TrainingConfig\n",
    "from sheet.training_config import TrainingConfig, normalize_plastic_v0541_config_fields\n",
)
replace_once(
    path,
    "    parser.add_argument(\"--plastic__layer_count_probe_noise_lambda\", dest=\"plastic__layer_count_probe_noise_lambda\", type=float, default=3.0)\n    parser.add_argument(\"--plastic__layer_count_cost_weight\", dest=\"plastic__layer_count_cost_weight\", type=float, default=0.0)",
    f"    parser.add_argument(\"--plastic__layer_count_probe_noise_lambda\", dest=\"plastic__layer_count_probe_noise_lambda\", type=float, default=3.0)\n    parser.add_argument(\"--{DISCOUNT}\", dest=\"{DISCOUNT}\", type=float, default=0.9)\n    parser.add_argument(\"--{LOSS_WINDOW}\", dest=\"{LOSS_WINDOW}\", type=int, default=64)\n    parser.add_argument(\"--{LOSS_MIN}\", dest=\"{LOSS_MIN}\", type=int, default=16)\n    parser.add_argument(\"--plastic__layer_count_cost_weight\", dest=\"plastic__layer_count_cost_weight\", type=float, default=0.0)",
)
replace_once(
    path,
    "        plastic__layer_count_probe_noise_lambda=arguments.plastic__layer_count_probe_noise_lambda,\n        plastic__layer_count_cost_weight=arguments.plastic__layer_count_cost_weight,",
    f"        plastic__layer_count_probe_noise_lambda=arguments.plastic__layer_count_probe_noise_lambda,\n        {DISCOUNT}=arguments.{DISCOUNT},\n        {LOSS_WINDOW}=arguments.{LOSS_WINDOW},\n        {LOSS_MIN}=arguments.{LOSS_MIN},\n        plastic__layer_count_cost_weight=arguments.plastic__layer_count_cost_weight,",
)
replace_once(
    path,
    '    stored = TrainingConfig(**payload["trainer_config"])\n',
    '    stored = TrainingConfig(**normalize_plastic_v0541_config_fields(payload["trainer_config"]))\n',
)
# ^^^ THOG


# vvv THOG v0.541 startup report shows canonical names and the tunable equivalent-time-gain settings
path = "run_thog2_owt.py"
replace_once(
    path,
    '    "plastic__layer_count_probe_noise_lambda:",\n    "plastic__layer_count_cost_weight:",',
    f'    "plastic__layer_count_probe_noise_lambda:",\n    "{DISCOUNT}:",\n    "{LOSS_WINDOW}:",\n    "{LOSS_MIN}:",\n    "plastic__layer_count_cost_weight:",',
)
replace_once(
    path,
    '    _print_plastic_option("plastic__layer_count_probe_noise_lambda:", _startup_float(config.plastic__layer_count_probe_noise_lambda))\n    _print_plastic_option("plastic__layer_count_cost_weight:", _startup_float(config.plastic__layer_count_cost_weight))',
    f'    _print_plastic_option("plastic__layer_count_probe_noise_lambda:", _startup_float(config.plastic__layer_count_probe_noise_lambda))\n    _print_plastic_option("{DISCOUNT}:", _startup_float(config.{DISCOUNT}))\n    _print_plastic_option("{LOSS_WINDOW}:", str(config.{LOSS_WINDOW}))\n    _print_plastic_option("{LOSS_MIN}:", str(config.{LOSS_MIN}))\n    _print_plastic_option("plastic__layer_count_cost_weight:", _startup_float(config.plastic__layer_count_cost_weight))',
)
# ^^^ THOG


# vvv THOG v0.541 existing checkpoints normalize only the two renamed config keys before TrainingConfig construction
path = "sheet/trainer_checkpoint_resume.py"
replace_once(
    path,
    "    TrainingConfig,\n)",
    "    TrainingConfig,\n    normalize_plastic_v0541_config_fields,\n)",
)
replace_once(
    path,
    '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n',
    '        checkpoint_config = TrainingConfig(**normalize_plastic_v0541_config_fields(payload["trainer_config"]))\n',
)
# ^^^ THOG


# vvv THOG v0.541 compatibility compares old/new public spellings semantically and includes the new wall-time knobs with their defaults
path = "sheet/checkpoints.py"
replace_once(
    path,
    f'        "extrapolation_weight": value.get("{NEW_EXTRAPOLATION}", 0.8),',
    f'        "extrapolation_weight": value.get("{NEW_EXTRAPOLATION}", value.get("{OLD_EXTRAPOLATION}", 0.8)),',
)
replace_once(
    path,
    '        "probe_noise_window": value.get("plastic__layer_count_probe__window_size_as_number_of_probes"),',
    f'        "probe_noise_window": value.get("plastic__layer_count_probe__window_size_as_number_of_probes"),\n        "wall_time_discount": value.get("{DISCOUNT}", 0.9),\n        "wall_time_loss_rate_window": value.get("{LOSS_WINDOW}", 64),\n        "wall_time_loss_rate_min_observations": value.get("{LOSS_MIN}", 16),',
)
# add the same neutral defaults to the retired short-identity semantic branch
replace_once(
    path,
    '            "probe_noise_window": value.get("probe_noise_window"),',
    '            "probe_noise_window": value.get("probe_noise_window"),\n            "wall_time_discount": 0.9,\n            "wall_time_loss_rate_window": 64,\n            "wall_time_loss_rate_min_observations": 16,',
)
# ^^^ THOG


# vvv THOG v0.541 make all wall-time algorithm constants true public hyperparameters rather than fixed runtime policy
path = "sheet/plastic_depth_wall_time_equivalent_time_gain_patch.py"
replace_once(
    path,
    '        "loss_history": deque(maxlen=WALL_TIME_LOSS_RATE_WINDOW),',
    f'        "loss_history": deque(\n            maxlen=int(getattr(trainer.config, "{LOSS_WINDOW}", WALL_TIME_LOSS_RATE_WINDOW))\n        ),',
)
replace_once(
    path,
    "    if len(points) < WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS:\n        return None",
    f'    minimum_observations = int(\n        getattr(\n            trainer.config,\n            "{LOSS_MIN}",\n            WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS,\n        )\n    )\n    if len(points) < minimum_observations:\n        return None',
)
replace_once(
    path,
    "def _bootstrap_score_report(\n    measurements: Sequence[Any],\n) -> Tuple[Any, Tuple[Dict[str, object], ...]]:",
    "def _bootstrap_score_report(\n    measurements: Sequence[Any],\n    *,\n    discount: float,\n) -> Tuple[Any, Tuple[Dict[str, object], ...]]:",
)
replace_once(
    path,
    "    current_count: int,\n    reason: str,\n) -> Tuple[Any, Tuple[Dict[str, object], ...]]:",
    "    current_count: int,\n    reason: str,\n    discount: float,\n) -> Tuple[Any, Tuple[Dict[str, object], ...]]:",
)
replace_all(
    path,
    '"wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,',
    '"wall_time_discount": float(discount),',
    minimum=3,
)
replace_once(
    path,
    "    loss_fit = _loss_rate_fit(trainer)\n    ready = timing_fit is not None and loss_fit is not None",
    f'    loss_fit = _loss_rate_fit(trainer)\n    discount = float(\n        getattr(\n            trainer.config,\n            "{DISCOUNT}",\n            WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,\n        )\n    )\n    ready = timing_fit is not None and loss_fit is not None',
)
replace_once(path, "            return _bootstrap_score_report(measurements)", "            return _bootstrap_score_report(measurements, discount=discount)")
# all HOLD call sites gain the same configured discount
text = read(path)
text = text.replace('            reason="timing_model_unavailable",\n        )', '            reason="timing_model_unavailable",\n            discount=discount,\n        )')
text = text.replace('            reason="loss_rate_unavailable",\n        )', '            reason="loss_rate_unavailable",\n            discount=discount,\n        )')
text = text.replace('            reason="invalid_predicted_current_time",\n        )', '            reason="invalid_predicted_current_time",\n            discount=discount,\n        )')
write(path, text)
replace_once(
    path,
    "                horizon_updates=horizon_updates,\n            )",
    "                horizon_updates=horizon_updates,\n                discount=discount,\n            )",
)
replace_once(
    path,
    '        "plastic_wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,',
    f'        "plastic_wall_time_discount": float(\n            getattr(\n                trainer.config,\n                "{DISCOUNT}",\n                WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,\n            )\n        ),',
)
# ^^^ THOG


# vvv THOG v0.541 probe sequence/provenance is durable TrainerState, so checkpoint/resume preserves audit readability
path = "sheet/trainer_state.py"
replace_once(
    path,
    "    plastic_depth_probe_histories: Dict[str, List[float]] = field(default_factory=dict)\n    plastic_depth_last_count_change_update: int = -1",
    "    plastic_depth_probe_histories: Dict[str, List[float]] = field(default_factory=dict)\n    plastic_depth_probe_sequence: int = 0\n    plastic_depth_probe_provenance: List[int] = field(default_factory=list)\n    plastic_depth_last_count_change_update: int = -1",
)
# ^^^ THOG


# vvv THOG v0.541 surface probe provenance directly in the durable FINE count-decision audit
path = "sheet/plastic_depth_audit_patch.py"
replace_once(
    path,
    '        "update_number": int(decision.update_number),\n        "decision_number": int(lattice.count_decision_number.item()),',
    '        "update_number": int(decision.update_number),\n        "probe_sequence": int(context.get("plastic_probe_sequence", 0)),\n        "probe_provenance": tuple(int(value) for value in context.get("plastic_probe_provenance", ())),\n        "decision_number": int(lattice.count_decision_number.item()),',
)
# ^^^ THOG


# vvv THOG v0.541 final overlay numbers probes, records exact temporal provenance, uses two-dot probe labels and larger text arrows
v0541_patch = ROOT / "sheet" / "plastic_depth_v0541_patch.py"
v0541_patch.write_text('''# vvv THOG\n"""PLASTIC v0.541 public-control, probe-provenance and final console refinement."""\n\nfrom __future__ import annotations\n\nimport re\nfrom dataclasses import replace\nfrom typing import Any, Dict, Optional, Sequence\n\nfrom . import plastic_depth_console_minor_patch as _console_minor\nfrom . import plastic_depth_directional_coherence_patch as _directional\nfrom . import stage6_trainer as _stage6\nfrom . import trainer_step as _trainer_step\n\n\n_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update\n_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request\n_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload\n_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line\n_POSTFIX_START = re.compile(r"(?P<postfix>[ \\t]+(?:\\x1b\\[[0-9;]*m)*<<<)")\n\n\ndef _begin_plastic_depth_inline_update_v0541(self: Any) -> Optional[Dict[str, Any]]:\n    # An older checkpoint may have robust histories but no v0.541 provenance IDs.\n    # Discard that evidence rather than attach invented probe provenance to it.\n    if self.state.plastic_depth_probe_histories and not self.state.plastic_depth_probe_provenance:\n        self.state.plastic_depth_probe_histories = {}\n    return _ORIGINAL_BEGIN_INLINE_UPDATE(self)\n\n\ndef _advance_probe_provenance(\n    previous: Sequence[int],\n    *,\n    probe_sequence: int,\n    vote_total: int,\n) -> tuple[int, ...]:\n    needed_prior = max(0, int(vote_total) - 1)\n    prior = tuple(int(value) for value in previous)\n    if len(prior) != needed_prior:\n        prior = prior[-needed_prior:] if needed_prior else ()\n    return (*prior, int(probe_sequence))\n\n\ndef _plastic_depth_inline_probe_request_v0541(\n    self: Any,\n    targets: Any,\n    context: Dict[str, Any],\n):\n    self.state.plastic_depth_probe_sequence = int(self.state.plastic_depth_probe_sequence) + 1\n    probe_sequence = int(self.state.plastic_depth_probe_sequence)\n    context["plastic_probe_sequence"] = probe_sequence\n    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)\n    original_selector = request.selector\n\n    def selector(candidates: Any) -> int:\n        selected = int(original_selector(candidates))\n        report = context.get("plastic_directional_report")\n        decision = context.get("decision")\n        if report is None or decision is None:\n            self.state.plastic_depth_probe_provenance = [probe_sequence]\n            context["plastic_probe_provenance"] = (probe_sequence,)\n            return selected\n        provenance = _advance_probe_provenance(\n            self.state.plastic_depth_probe_provenance,\n            probe_sequence=probe_sequence,\n            vote_total=int(report.get("vote_total", 1)),\n        )\n        report["probe_sequence"] = probe_sequence\n        report["probe_provenance"] = provenance\n        context["plastic_probe_provenance"] = provenance\n        if int(decision.selected_count) != int(context["current_count"]):\n            self.state.plastic_depth_probe_provenance = []\n        else:\n            self.state.plastic_depth_probe_provenance = list(provenance)\n        return selected\n\n    return replace(request, selector=selector)\n\n\n_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_v0541\n_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _plastic_depth_inline_probe_request_v0541\n\n\ndef _prepare_console_progress_payload_v0541(\n    self: Any,\n    event: str,\n    payload: Dict[str, Any],\n) -> Dict[str, Any]:\n    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)\n    if event not in {"optimizer_progress", "evaluation_completed"}:\n        return values\n    report = _directional._latest_directional_report(self)\n    if report is None or "probe_sequence" not in report:\n        return values\n    try:\n        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))\n    except (TypeError, ValueError):\n        return values\n    if completed_updates != int(report.get("update_number", -1)):\n        return values\n    values["plastic_probe_sequence"] = int(report["probe_sequence"])\n    values["plastic_probe_provenance"] = tuple(int(value) for value in report.get("probe_provenance", ()))\n    return values\n\n\n_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_v0541\n\n\ndef _provenance_text(values: Sequence[Any]) -> str:\n    resolved = tuple(int(value) for value in values)\n    if not resolved:\n        return ""\n    return " (P" + ",".join(str(value) for value in resolved) + ")"\n\n\ndef _finalize_console_v0541(\n    line: str,\n    *,\n    probe_sequence: Optional[int],\n    probe_provenance: Sequence[Any],\n) -> str:\n    line = re.sub(\n        r"(probe_Δloss \\[[^\\]]*?) \\.\\.\\. ([^\\]]*?\\])",\n        r"\\1 .. \\2",\n        line,\n        count=1,\n    )\n    if probe_sequence is not None and "probe_Δloss" in line:\n        line = line.replace("probe_Δloss", f"P{int(probe_sequence):4d}  probe_Δloss", 1)\n\n    # ANSI has no portable font-size control. ⇩/⇧ are larger text glyphs than ↓/↑\n    # without introducing emoji-width instability. Existing ANSI colour spans are preserved.\n    line = line.replace("↓", "⇩").replace("↑", "⇧")\n\n    provenance = _provenance_text(probe_provenance)\n    if provenance and "⇩|⇧|? =" in line:\n        summary_start = line.find("⇩|⇧|? =")\n        postfix = _POSTFIX_START.search(line, summary_start)\n        if postfix is None:\n            line = f"{line}{provenance}"\n        else:\n            line = f"{line[:postfix.start()]}{provenance}{line[postfix.start():]}"\n    return line\n\n\ndef _format_progress_line_v0541(\n    run_id: str,\n    event: str,\n    payload: Dict[str, Any],\n) -> str:\n    local_payload = dict(payload)\n    probe_sequence = local_payload.pop("plastic_probe_sequence", None)\n    probe_provenance = tuple(local_payload.pop("plastic_probe_provenance", ()))\n    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)\n    line = _finalize_console_v0541(\n        line,\n        probe_sequence=None if probe_sequence is None else int(probe_sequence),\n        probe_provenance=probe_provenance,\n    )\n    if event == "optimizer_progress":\n        _console_minor._record_alignment(run_id, line)\n    elif event == "evaluation_completed":\n        line = _console_minor._align_validation_row(run_id, line)\n    return line\n\n\n_stage6.format_progress_line = _format_progress_line_v0541\n\n\n__all__ = ["_advance_probe_provenance", "_finalize_console_v0541", "_provenance_text"]\n# ^^^ THOG\n''', encoding="utf-8")
# ^^^ THOG


# vvv THOG v0.541 overlay must win after v0.531 integration
path = "sheet/plastic_depth_console_postfix_patch.py"
text = read(path)
marker = "from . import plastic_depth_v0541_patch as _plastic_depth_v0541_patch"
if marker not in text:
    text = text.rstrip() + "\n\n# vvv THOG install v0.541 canonical controls, durable probe provenance and final console glyph refinement\n" + marker + "\n# ^^^ THOG\n"
    write(path, text)
# ^^^ THOG


# vvv THOG v0.541 focused acceptance coverage for canonical CLI/config and console provenance
(ROOT / "tests" / "test_plastic_v0541_controls_and_provenance.py").write_text(f'''from __future__ import annotations\n\nimport pytest\n\nimport run_thog2_owt_core as runner\nfrom sheet.plastic_depth_v0541_patch import _advance_probe_provenance, _finalize_console_v0541\nfrom sheet.run_config import OwtRunConfig\nfrom sheet.training_config import TrainingConfig, normalize_plastic_v0541_config_fields\n\n\ndef test_v0541_public_defaults_and_parser_names():\n    parser = runner.build_parser()\n    args = parser.parse_args([\n        "--model-type", "sheet",\n        "--{NEW_EXTRAPOLATION}", "0.91",\n        "--{NEW_MAX_STEP}", "3",\n        "--{DISCOUNT}", "0.85",\n        "--{LOSS_WINDOW}", "80",\n        "--{LOSS_MIN}", "20",\n    ])\n    config = runner.config_from_arguments(args)\n    assert config.{NEW_EXTRAPOLATION} == pytest.approx(0.91)\n    assert config.{NEW_MAX_STEP} == 3\n    assert config.{DISCOUNT} == pytest.approx(0.85)\n    assert config.{LOSS_WINDOW} == 80\n    assert config.{LOSS_MIN} == 20\n\n    defaults = OwtRunConfig(model_type="sheet")\n    assert defaults.{DISCOUNT} == pytest.approx(0.9)\n    assert defaults.{LOSS_WINDOW} == 64\n    assert defaults.{LOSS_MIN} == 16\n\n\ndef test_v0541_old_cli_names_are_rejected():\n    parser = runner.build_parser()\n    with pytest.raises(SystemExit):\n        parser.parse_args(["--model-type", "sheet", "--{OLD_MAX_STEP}", "2"])\n    with pytest.raises(SystemExit):\n        parser.parse_args(["--model-type", "sheet", "--{OLD_EXTRAPOLATION}", "0.9"])\n\n\ndef test_v0541_training_config_validates_wall_time_controls():\n    config = TrainingConfig(\n        {DISCOUNT}=0.75,\n        {LOSS_WINDOW}=40,\n        {LOSS_MIN}=12,\n    )\n    assert config.{DISCOUNT} == pytest.approx(0.75)\n    with pytest.raises(ValueError):\n        TrainingConfig({DISCOUNT}=1.01)\n    with pytest.raises(ValueError):\n        TrainingConfig({LOSS_WINDOW}=8, {LOSS_MIN}=9)\n\n\ndef test_v0541_checkpoint_key_normalization_is_explicit():\n    normalized = normalize_plastic_v0541_config_fields({{\n        "{OLD_MAX_STEP}": 2,\n        "{OLD_EXTRAPOLATION}": 0.9,\n    }})\n    assert normalized["{NEW_MAX_STEP}"] == 2\n    assert normalized["{NEW_EXTRAPOLATION}"] == pytest.approx(0.9)\n    assert "{OLD_MAX_STEP}" not in normalized\n    assert "{OLD_EXTRAPOLATION}" not in normalized\n\n\ndef test_v0541_probe_provenance_tracks_exact_window():\n    assert _advance_probe_provenance((), probe_sequence=14, vote_total=1) == (14,)\n    assert _advance_probe_provenance((14,), probe_sequence=15, vote_total=2) == (14, 15)\n    assert _advance_probe_provenance((14, 15), probe_sequence=16, vote_total=3) == (14, 15, 16)\n\n\ndef test_v0541_console_probe_id_two_dot_label_larger_arrows_and_provenance():\n    source = "T probe_Δloss [L-5 ... L+5] = [-0.1]  ↓|↑|? =[1/0/2]/3=>\\x1b[1m\\x1b[93m↑\\x1b[0m  <<< update brake on"\n    rendered = _finalize_console_v0541(\n        source,\n        probe_sequence=1,\n        probe_provenance=(14, 15, 16),\n    )\n    assert "P   1  probe_Δloss [L-5 .. L+5]" in rendered\n    assert "⇩|⇧|? =[1/0/2]/3=>\\x1b[1m\\x1b[93m⇧\\x1b[0m (P14,15,16)" in rendered\n    assert rendered.endswith("<<< update brake on")\n''', encoding="utf-8")
# ^^^ THOG


# vvv THOG v0.541 specification delta records the public names and exact provenance/glyph semantics
(ROOT / "docs" / "THOG2_PLASTIC_Requirements_Specification_v0.541.txt").write_text(f'''PLASTIC REQUIREMENTS SPECIFICATION\nTHOG2\nPlastic\n\nVERSION | DATE | STATUS\n0.541 | 8 August 2026 | Public wall-time controls, canonical layer-count names and probe provenance\n\nGOVERNING BASIS\nVersion 0.541 is a +0.01 revision over Version 0.531. Version 0.531 remains normative except where this delta explicitly supersedes it.\n\n1. PUBLIC wall_time_equivalent_time_gain CONTROLS\n\nThe following are public persisted hyperparameters:\n\n    {DISCOUNT} = 0.9\n    {LOSS_WINDOW} = 64\n    {LOSS_MIN} = 16\n\nThe discount is finite in [0,1]. The loss-rate window is an integer >=2. The minimum observation count is an integer >=2 and may not exceed the window. The algorithm shall use these configured values everywhere that Version 0.531 previously used fixed constants, including scoring, bootstrap diagnostics, telemetry and the rolling loss-rate fit.\n\n2. CANONICAL LAYER-COUNT CONTROL RENAMES\n\nThe public control\n\n    {OLD_EXTRAPOLATION}\n\nis superseded by\n\n    {NEW_EXTRAPOLATION}\n\nThe public control\n\n    {OLD_MAX_STEP}\n\nis superseded by\n\n    {NEW_MAX_STEP}\n\nNew CLI/configuration and newly written persistent metadata shall use only the new names. Existing checkpoints containing the superseded names may be normalized on read when values are unambiguous. The superseded CLI spellings are not accepted.\n\n3. PROBE SEQUENCE AND DECISION PROVENANCE\n\nEvery actual FINE layer-count probe receives one monotonically increasing sequence number. The sequence counter and the active temporal evidence-window probe IDs are TrainerState and survive checkpoint/resume. Warmup-suppressed updates do not consume probe numbers.\n\nThe probe row prefix is fixed-width through probe 9999:\n\n    P   1\n    P  14\n    P 9999\n\nAt P10000 the field expands naturally rather than truncating the sequence.\n\nEvery directional decision summary displays the exact probe IDs contributing to its current temporal vote window. Example:\n\n    ⇩|⇧|? =[1/0/2]/3=>⇧ (P14,15,16)\n\nThe same sequence/provenance shall be retained in the durable FINE count-decision audit. No probe IDs may be invented for evidence restored from an older checkpoint that predates provenance; such old robust history is discarded and rebuilt.\n\n4. PROBE LABEL\n\nThe console label\n\n    probe_Δloss [L-5 ... L+5]\n\nis superseded by\n\n    probe_Δloss [L-5 .. L+5]\n\nThe same two-dot convention applies for any radius.\n\n5. LARGER DIRECTION ARROWS\n\nANSI provides no portable terminal font-size control. Version 0.541 therefore replaces the small text arrows ↓ and ↑ with the larger text glyphs ⇩ and ⇧ in both the directional labels and the outcome. This avoids emoji-width instability. A committed ⇩ or ⇧ outcome retains the existing bold bright-yellow constants from constants.py; label arrows and the neutral ● remain unhighlighted.\n\n6. RETAINED SEMANTICS\n\nExcept for the public configuration and presentation changes above, Version 0.531 wall_time_equivalent_time_gain mathematics, directional evidence, braking, probe sampling, score_z formatting and PLASTIC-disabled equivalence remain unchanged.\n''', encoding="utf-8")
# ^^^ THOG


# vvv THOG final source-level assertions keep the public rename exact and prevent accidental duplicate wall-time control spelling
for path in active_sources():
    text = path.read_text(encoding="utf-8")
    if OLD_EXTRAPOLATION in text or OLD_MAX_STEP in text:
        # Only explicit checkpoint-compatibility literals are allowed to retain the old spellings.
        allowed = path.name in {"training_config.py", "checkpoints.py"}
        if not allowed:
            raise RuntimeError(f"superseded PLASTIC name remains in active source: {path}")
# ^^^ THOG

print("PLASTIC v0.541 applicator completed")
