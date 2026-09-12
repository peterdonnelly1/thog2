# vvv THOG
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1))


# vvv THOG restore the original trainer body indentation; combine the diagnostic
# context with the existing autocast context instead of nesting/reindenting stock work.
trainer_path = ROOT / "sheet/trainer_step.py"
trainer = trainer_path.read_text()
start_marker = "                    # vvv THOG capture exactly one first accumulation forward; backward/checkpoint replay remains outside the Nsight range\n"
end_marker = "                    # ^^^ THOG\n"
start = trainer.index(start_marker)
end = trainer.index(end_marker, start) + len(end_marker)
block = trainer[start:end]
autocast_line = "                        with self.autocast_context():\n"
autocast_at = block.index(autocast_line)
body_start = autocast_at + len(autocast_line)
body = block[body_start:-len(end_marker)]
for line in body.splitlines():
    if line and not line.startswith("    "):
        raise RuntimeError(f"unexpected trainer capture indentation: {line!r}")
body = "\n".join(line[4:] if line else line for line in body.splitlines()) + "\n"
replacement = '''                    # vvv THOG one selected forward is profiled without changing the indentation or execution order of the existing autocast body
                    # with self.autocast_context():
                    with processing_capture_scope(
                        enabled=(str(getattr(self.config, "premat_processing_logging", "disabled")) == "enabled"),
                        completed_updates=self.state.completed_updates,
                        max_updates=self.config.max_updates,
                        log_interval=self.config.log_interval,
                        micro_step=micro_step,
                        device=self.device,
                    ), self.autocast_context():
                    # ^^^ THOG
''' + body
trainer_path.write_text(trainer[:start] + replacement + trainer[end:])
# ^^^ THOG

# vvv THOG retain the displaced ordinary materialisation statements as comments,
# while the active diagnostic wrappers add only NVTX semantics.
replace_once(
    "sheet/model.py",
    '            if self._premat_runtime is None:\n                with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n                    return self._premat_materialize_candidate(family, layer_index)\n',
    '            if self._premat_runtime is None:\n                # vvv THOG retain the ordinary materialisation statement while wrapping its CUDA launches in one semantic range\n                # return self._premat_materialize_candidate(family, layer_index)\n                with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n                    return self._premat_materialize_candidate(family, layer_index)\n                # ^^^ THOG\n',
)
replace_once(
    "sheet/premat.py",
    '                with processing_operation_range("MAIN", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                    candidate.tensor = self._materialize(candidate.family, candidate.layer_index)\n',
    '                # vvv THOG preserve the ordinary Main materialisation statement; the active copy is only wrapped for profiling attribution\n                # candidate.tensor = self._materialize(candidate.family, candidate.layer_index)\n                with processing_operation_range("MAIN", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                    candidate.tensor = self._materialize(candidate.family, candidate.layer_index)\n                # ^^^ THOG\n',
)
replace_once(
    "sheet/premat.py",
    '        with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n            return self._materialize(family, layer_index)\n',
    '        # vvv THOG preserve checkpoint/replay Main materialisation while adding profiling attribution\n        # return self._materialize(family, layer_index)\n        with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n            return self._materialize(family, layer_index)\n        # ^^^ THOG\n',
)
replace_once(
    "sheet/premat.py",
    '                            with torch.no_grad():\n                                with processing_operation_range("PREMAT", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                                    candidate.tensor = self._materialize(\n                                        candidate.family,\n                                        candidate.layer_index,\n                                    )\n',
    '                            with torch.no_grad():\n                                # vvv THOG preserve the original side-stream materialisation block while adding profiling attribution\n                                # candidate.tensor = self._materialize(\n                                #     candidate.family,\n                                #     candidate.layer_index,\n                                # )\n                                with processing_operation_range("PREMAT", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                                    candidate.tensor = self._materialize(\n                                        candidate.family,\n                                        candidate.layer_index,\n                                    )\n                                # ^^^ THOG\n',
)
# ^^^ THOG

# vvv THOG preserve the pre-processing execution-override declaration as source history.
old_override = 'EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic", "premat_processing_logging", "premat_processing_logging_capture_frequency_hz"}                         # <<< THOG processing capture is diagnostic execution state, never checkpoint model identity\n'
new_override = '# vvv THOG processing capture is diagnostic execution state, never checkpoint model identity\n# EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic"}\n' + old_override + '# ^^^ THOG\n'
replace_once("sheet/training_config.py", old_override, new_override)
# ^^^ THOG

# vvv THOG preserve the previous dashboard API route set as a commented line.
replace_once(
    "run_thog2_local_dashboard.py",
    '                if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing"}:\n',
    '                # vvv THOG extend the existing local API without deleting the prior route declaration\n                # if path in {"/api/status", "/api/figures", "/api/premat"}:\n                if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing"}:\n                # ^^^ THOG\n',
)
# ^^^ THOG

# vvv THOG expose the exact public wrapper controls in ordinary argparse help
# without adding them to the core PREMAT scheduler option set.
replace_once(
    "run_thog2_owt_core.py",
    '    parser = _ThogArgumentParser(description="Train or resume one canonical THOG2 OpenWebText run")\n',
    '    parser = _ThogArgumentParser(description="Train or resume one canonical THOG2 OpenWebText run")\n    # vvv THOG wrapper consumes these before core argparse, but --help must still advertise the exact public names\n    parser.epilog = "Processing diagnostics: --premat_processing_logging enabled|disabled; --premat_processing_logging_capture_frequency_hz HZ (default 10000)"\n    # ^^^ THOG\n',
)
# ^^^ THOG

# vvv THOG missing metric values are absent, not zero; Number(\"\") would otherwise
# draw a false zero line and bias family means when a GPU lacks a sampled metric.
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '    const rows = (payload.samples || []).filter(row => Number.isFinite(Number(row[key])));\n',
    '    const rows = (payload.samples || []).filter(row => row[key] !== "" && row[key] !== null && row[key] !== undefined && Number.isFinite(Number(row[key])));\n',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '  const values = rows.map(row => Number(row[key])).filter(Number.isFinite);\n',
    '  const values = rows.filter(row => row[key] !== "" && row[key] !== null && row[key] !== undefined).map(row => Number(row[key])).filter(Number.isFinite);\n',
)
# ^^^ THOG

print("PREMAT processing finalisation applied")
# ^^^ THOG
