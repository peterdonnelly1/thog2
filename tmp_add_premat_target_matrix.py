# vvv THOG temporary exact-source transformer for --premat_target_matrix
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def thog_line(code: str, explanation: str) -> str:
    marker = "# <<< THOG " + explanation
    if len(code) >= 155:
        raise RuntimeError(f"THOG line too long for column-156 marker: {code}")
    return code + (" " * (155 - len(code))) + marker + "\n"


# Public CLI and OwtRunConfig construction.
replace_once(
    "run_thog2_owt_core.py",
    '    parser.add_argument("--premat_target_layer", type=int, choices=(0, 1, 2, 10), default=1, help="PREMAT relative target: 10 means ordered +1 then +0 sweep")\n    parser.add_argument("--premat_weight_matrix_target_order", choices=("l_to_r", "r_to_l"), default="r_to_l")\n',
    '    parser.add_argument("--premat_target_layer", type=int, choices=(0, 1, 2, 10), default=1, help="PREMAT relative target: 10 means ordered +1 then +0 sweep")\n'
    + thog_line('    parser.add_argument("--premat_target_matrix", type=int, choices=(1, 2, 3, 4), default=None, help="Optional fused-family PREMAT selector: 1=QKV, 2=O, 3=UP, 4=DOWN")', "add fixed fused-family PREMAT diagnostic selector")
    + '    parser.add_argument("--premat_weight_matrix_target_order", choices=("l_to_r", "r_to_l"), default="r_to_l")\n',
)
replace_once(
    "run_thog2_owt_core.py",
    '        premat_attention_mode=arguments.premat_attention_mode,\n        premat_target_layer=arguments.premat_target_layer,\n        premat_weight_matrix_target_order=arguments.premat_weight_matrix_target_order,\n',
    '        premat_attention_mode=arguments.premat_attention_mode,\n        premat_target_layer=arguments.premat_target_layer,\n'
    + thog_line('        premat_target_matrix=arguments.premat_target_matrix,', "carry selected PREMAT matrix into persistent run configuration")
    + '        premat_weight_matrix_target_order=arguments.premat_weight_matrix_target_order,\n',
)

# OwtRunConfig persistence, validation and TrainingConfig propagation.
replace_once(
    "sheet/run_config.py",
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n    premat_weight_matrix_target_order: str = "r_to_l"\n',
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n'
    + thog_line('    premat_target_matrix: Optional[int] = None', "persist optional fused-family PREMAT selector in run identity")
    + '    premat_weight_matrix_target_order: str = "r_to_l"\n',
)
replace_once(
    "sheet/run_config.py",
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n'
    + thog_line('            target_matrix=self.premat_target_matrix,', "validate optional fused-family PREMAT selector")
    + '            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
)
replace_once(
    "sheet/run_config.py",
    '            premat_attention_mode=self.premat_attention_mode,\n            premat_target_layer=self.premat_target_layer,\n            premat_weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
    '            premat_attention_mode=self.premat_attention_mode,\n            premat_target_layer=self.premat_target_layer,\n'
    + thog_line('            premat_target_matrix=self.premat_target_matrix,', "propagate selected PREMAT matrix into TrainingConfig")
    + '            premat_weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
)

# TrainingConfig persistence/resume identity, validation, model arguments and metadata.
replace_once(
    "sheet/training_config.py",
    '    "premat_attention_mode",\n    "premat_target_layer",\n    "premat_weight_matrix_target_order",\n',
    '    "premat_attention_mode",\n    "premat_target_layer",\n'
    + thog_line('    "premat_target_matrix",', "make target-matrix selection resume compatibility identity")
    + '    "premat_weight_matrix_target_order",\n',
)
replace_once(
    "sheet/training_config.py",
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n    premat_weight_matrix_target_order: str = "r_to_l"\n',
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n'
    + thog_line('    premat_target_matrix: Optional[int] = None', "persist optional fused-family PREMAT selector through training/checkpoint config")
    + '    premat_weight_matrix_target_order: str = "r_to_l"\n',
)
replace_once(
    "sheet/training_config.py",
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n'
    + thog_line('            target_matrix=self.premat_target_matrix,', "validate selected PREMAT matrix in TrainingConfig")
    + '            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
)
replace_once(
    "sheet/training_config.py",
    '                    "premat_attention_mode": self.premat_attention_mode,\n                    "premat_target_layer": self.premat_target_layer,\n                    "premat_weight_matrix_target_order": self.premat_weight_matrix_target_order,\n',
    '                    "premat_attention_mode": self.premat_attention_mode,\n                    "premat_target_layer": self.premat_target_layer,\n'
    + thog_line('                    "premat_target_matrix": self.premat_target_matrix,', "pass selected PREMAT matrix into SheetGPTConfig")
    + '                    "premat_weight_matrix_target_order": self.premat_weight_matrix_target_order,\n',
)
replace_once(
    "sheet/training_config.py",
    '                "attention_mode": self.premat_attention_mode,\n                "target_layer": self.premat_target_layer,\n                "weight_matrix_target_order": self.premat_weight_matrix_target_order,\n',
    '                "attention_mode": self.premat_attention_mode,\n                "target_layer": self.premat_target_layer,\n'
    + thog_line('                "target_matrix": self.premat_target_matrix,', "expose selected PREMAT matrix in compact training identity")
    + '                "weight_matrix_target_order": self.premat_weight_matrix_target_order,\n',
)

# SheetGPTConfig and runtime construction.
replace_once(
    "sheet/model.py",
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n    premat_weight_matrix_target_order: str = "r_to_l"\n',
    '    premat_attention_mode: str = "fused"\n    premat_target_layer: int = 1\n'
    + thog_line('    premat_target_matrix: Optional[int] = None', "carry optional fused-family PREMAT selector into model runtime")
    + '    premat_weight_matrix_target_order: str = "r_to_l"\n',
)
replace_once(
    "sheet/model.py",
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
    '            attention_mode=self.premat_attention_mode,\n            target_layer=self.premat_target_layer,\n'
    + thog_line('            target_matrix=self.premat_target_matrix,', "validate model-level PREMAT matrix selection")
    + '            weight_matrix_target_order=self.premat_weight_matrix_target_order,\n',
)
replace_once(
    "sheet/model.py",
    '                attention_mode=config.premat_attention_mode,\n                target_layer=config.premat_target_layer,\n                weight_matrix_target_order=config.premat_weight_matrix_target_order,\n',
    '                attention_mode=config.premat_attention_mode,\n                target_layer=config.premat_target_layer,\n'
    + thog_line('                target_matrix=config.premat_target_matrix,', "give PREMAT runtime the selected fused matrix family")
    + '                weight_matrix_target_order=config.premat_weight_matrix_target_order,\n',
)

# Scheduler validation and filtering. All four candidates remain represented so
# non-selected families retain the ordinary MAIN fallback path.
replace_once(
    "sheet/premat.py",
    'PREMAT_TARGET_LAYERS = (0, 1, 2, 10)\nPREMAT_WEIGHT_MATRIX_TARGET_ORDERS = ("l_to_r", "r_to_l")\n',
    'PREMAT_TARGET_LAYERS = (0, 1, 2, 10)\n'
    '# vvv THOG fixed matrix numbering is deliberately independent of l_to_r/r_to_l scheduling order\n'
    'PREMAT_TARGET_MATRICES = (1, 2, 3, 4)\n'
    'PREMAT_FUSED_TARGET_MATRIX_FAMILIES = {1: "QKV", 2: "O", 3: "UP", 4: "DOWN"}\n'
    '# ^^^ THOG\n'
    'PREMAT_WEIGHT_MATRIX_TARGET_ORDERS = ("l_to_r", "r_to_l")\n',
)
replace_once(
    "sheet/premat.py",
    '    logging: str,\n    instra: str,\n) -> None:\n',
    '    logging: str,\n    instra: str,\n'
    + thog_line('    target_matrix: Optional[int] = None,', "optional fused-family PREMAT selector; omitted preserves all-matrix scheduling")
    + ') -> None:\n',
)
replace_once(
    "sheet/premat.py",
    '    if isinstance(target_layer, bool) or target_layer not in PREMAT_TARGET_LAYERS:\n        raise ValueError(\n            f"premat_target_layer must be one of {PREMAT_TARGET_LAYERS}; "\n            f"got {target_layer!r}"\n        )\n    if weight_matrix_target_order not in PREMAT_WEIGHT_MATRIX_TARGET_ORDERS:\n',
    '    if isinstance(target_layer, bool) or target_layer not in PREMAT_TARGET_LAYERS:\n        raise ValueError(\n            f"premat_target_layer must be one of {PREMAT_TARGET_LAYERS}; "\n            f"got {target_layer!r}"\n        )\n'
    '    # vvv THOG matrix-specific targeting is intentionally fused-only until an unfused experiment is requested\n'
    '    if target_matrix is not None:\n'
    '        if isinstance(target_matrix, bool) or target_matrix not in PREMAT_TARGET_MATRICES:\n'
    '            raise ValueError(f"premat_target_matrix must be one of {PREMAT_TARGET_MATRICES} or None; got {target_matrix!r}")\n'
    '        if attention_mode != "fused":\n'
    '            raise ValueError("premat_target_matrix currently requires premat_attention_mode=fused")\n'
    '    # ^^^ THOG\n'
    '    if weight_matrix_target_order not in PREMAT_WEIGHT_MATRIX_TARGET_ORDERS:\n',
)
replace_once(
    "sheet/premat.py",
    '        shadow_mode: bool = False,\n        logging_enabled: bool,\n    ) -> None:\n',
    '        shadow_mode: bool = False,\n        logging_enabled: bool,\n'
    + thog_line('        target_matrix: Optional[int] = None,', "runtime filter for one fixed fused matrix family")
    + '    ) -> None:\n',
)
replace_once(
    "sheet/premat.py",
    '        self._allocator_aware_admission = allocator_aware_admission\n        self._target_layer = int(target_layer)\n        self._weight_matrix_target_order = weight_matrix_target_order\n',
    '        self._allocator_aware_admission = allocator_aware_admission\n        self._target_layer = int(target_layer)\n'
    + thog_line('        self._target_matrix = None if target_matrix is None else int(target_matrix)', "retain optional fixed matrix-family selector")
    + '        self._weight_matrix_target_order = weight_matrix_target_order\n',
)
replace_once(
    "sheet/premat.py",
    '            "target_layer": self._target_layer,\n            "matrix_order": self._weight_matrix_target_order,\n',
    '            "target_layer": self._target_layer,\n'
    + thog_line('            "target_matrix": self._target_matrix,', "report active matrix-family selector in PREMAT telemetry")
    + '            "matrix_order": self._weight_matrix_target_order,\n',
)
replace_once(
    "sheet/premat.py",
    '                    if item.state == CandidateState.UNAVAILABLE\n                    and item.layer_index == target_layer_index\n                    and item.sequence not in excluded\n',
    '                    if item.state == CandidateState.UNAVAILABLE\n                    and item.layer_index == target_layer_index\n'
    '                    # vvv THOG target_matrix filters PREMAT launch eligibility only; MAIN fallback candidates remain intact\n'
    '                    and (\n'
    '                        self._target_matrix is None\n'
    '                        or item.family == PREMAT_FUSED_TARGET_MATRIX_FAMILIES[self._target_matrix]\n'
    '                    )\n'
    '                    # ^^^ THOG\n'
    '                    and item.sequence not in excluded\n',
)
replace_once(
    "sheet/premat.py",
    '            "target_offset": (\n                self._target_layer\n                if candidate_target_offset is None\n                else candidate_target_offset\n            ),\n            "target_order": self._weight_matrix_target_order,\n',
    '            "target_offset": (\n                self._target_layer\n                if candidate_target_offset is None\n                else candidate_target_offset\n            ),\n'
    + thog_line('            "target_matrix": self._target_matrix,', "retain matrix selector on every detailed PREMAT event")
    + '            "target_order": self._weight_matrix_target_order,\n',
)

# Tests: public surface, fused-only validation, exact family filtering, omission compatibility.
replace_once(
    "tests/test_premat.py",
    '    target_layer: int = 1,\n    weight_matrix_target_order: str = "r_to_l",\n',
    '    target_layer: int = 1,\n'
    + thog_line('    target_matrix: int | None = None,', "test helper can select one fixed fused PREMAT family")
    + '    weight_matrix_target_order: str = "r_to_l",\n',
)
replace_once(
    "tests/test_premat.py",
    '        attention_mode=attention_mode,\n        target_layer=target_layer,\n        weight_matrix_target_order=weight_matrix_target_order,\n',
    '        attention_mode=attention_mode,\n        target_layer=target_layer,\n'
    + thog_line('        target_matrix=target_matrix,', "exercise matrix-specific PREMAT runtime filtering")
    + '        weight_matrix_target_order=weight_matrix_target_order,\n',
)
replace_once(
    "tests/test_premat.py",
    'def test_public_cli_exposes_exactly_the_fifteen_premat_options() -> None:\n',
    'def test_public_cli_exposes_exactly_the_sixteen_premat_options() -> None:\n',
)
replace_once(
    "tests/test_premat.py",
    '        "--premat_target_layer",\n        "--premat_weight_matrix_target_order",\n',
    '        "--premat_target_layer",\n'
    + thog_line('        "--premat_target_matrix",', "new fused-family selector is an intentional core PREMAT option")
    + '        "--premat_weight_matrix_target_order",\n',
)
replace_once(
    "tests/test_premat.py",
    'def test_wrapper_reclaims_unused_allocator_cache_for_premat_by_default() -> None:\n',
    '''# vvv THOG matrix-specific targeting must isolate PREMAT work without removing ordinary MAIN candidates\n@pytest.mark.parametrize(\n    ("target_matrix", "expected_family"),\n    ((1, "QKV"), (2, "O"), (3, "UP"), (4, "DOWN")),\n)\ndef test_target_matrix_launches_only_the_selected_fused_family(monkeypatch, target_matrix, expected_family) -> None:\n    runtime, _fake_cuda, calls = _runtime(\n        monkeypatch,\n        stay_below_current_peak=False,\n        target_layer=1,\n        target_matrix=target_matrix,\n        weight_matrix_target_order="r_to_l",\n    )\n    runtime.layer_start(3)\n    assert calls == [(expected_family, 5)]\n    report = runtime.report()\n    assert report["target_matrix"] == target_matrix\n    layer_candidates = [item for item in report["candidates"] if item["layer_index"] == 5]\n    assert {item["family"] for item in layer_candidates} == {"QKV", "O", "UP", "DOWN"}\n    assert [item["family"] for item in layer_candidates if item["owner"] == "premat"] == [expected_family]\n\n\ndef test_target_matrix_is_rejected_for_unfused_attention() -> None:\n    with pytest.raises(ValueError, match="requires premat_attention_mode=fused"):\n        validate_premat_configuration(\n            premat="enabled",\n            attention_mode="unfused",\n            target_layer=1,\n            weight_matrix_target_order="r_to_l",\n            stay_below_current_peak=True,\n            stay_within_global_buffer=False,\n            gpu_memory_buffer_gb=1.0,\n            cuda_stream_priority="normal",\n            diagnostic_layer_delay_ms=0.0,\n            logging="disabled",\n            instra="disabled",\n            target_matrix=1,\n        )\n\n\ndef test_target_matrix_cli_propagates_to_training_config(tmp_path) -> None:\n    parser = build_parser()\n    arguments = parser.parse_args([\n        "--model-type", "sheet",\n        "--premat", "enabled",\n        "--premat_attention_mode", "fused",\n        "--premat_target_layer", "0",\n        "--premat_target_matrix", "2",\n        "--device", "cuda",\n    ])\n    run_config = config_from_arguments(arguments)\n    assert run_config.premat_target_matrix == 2\n    training_config = run_config.to_training_config(vocab_size=32, world_size=1, out_dir=tmp_path)\n    assert training_config.premat_target_matrix == 2\n    assert training_config.model_arguments["premat_target_matrix"] == 2\n\n\n# ^^^ THOG\n\ndef test_wrapper_reclaims_unused_allocator_cache_for_premat_by_default() -> None:\n''',
)

# Record the diagnostic addition without changing the technical spec yet.
log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text(encoding="utf-8")
entry = "- Added optional fused-only `--premat_target_matrix 1|2|3|4` (1=QKV, 2=O, 3=UP, 4=DOWN). The selector filters only PREMAT launch eligibility at the configured `premat_target_layer`; all four candidates remain represented so non-selected matrices continue through the ordinary Main Stream materialisation path. Omission preserves the established all-matrix scheduler. This is intended for clean Nsight isolation experiments before any change to PREMAT execution mechanics."
if entry not in log:
    log_path.write_text(log.rstrip() + "\n" + entry + "\n", encoding="utf-8")

print("PREMAT target-matrix selector implementation applied")
# ^^^ THOG
