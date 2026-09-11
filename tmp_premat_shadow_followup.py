from pathlib import Path


p = Path("sheet/run_config.py")
text = p.read_text(encoding="utf-8")
old = (
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n"
    "            logging=self.premat_logging,\n"
    "            instra=self.premat_instra,\n"
)
new = (
    "            enable_gpu_timing_diagnostic=self.premat_enable_gpu_timing_diagnostic,\n"
    "            shadow_mode=self.premat_enable_shadow_mode,\n"
    "            logging=self.premat_logging,\n"
    "            instra=self.premat_instra,\n"
)
if text.count(old) != 1:
    raise RuntimeError(
        "run-config shadow validation insertion expected once, "
        f"found {text.count(old)}"
    )
p.write_text(text.replace(old, new, 1), encoding="utf-8")

p = Path("tests/test_premat.py")
text = p.read_text(encoding="utf-8")
marker = "\ndef test_premat_cuda_stream_priority_is_opt_in(monkeypatch) -> None:\n"
test = (
    "\n\ndef test_run_config_rejects_shadow_mode_without_premat() -> None:\n"
    "    with pytest.raises(ValueError, match=\"premat_enable_shadow_mode requires premat enabled\"):\n"
    "        OwtRunConfig(\n"
    "            model_type=\"sheet\",\n"
    "            premat=\"disabled\",\n"
    "            premat_enable_shadow_mode=True,\n"
    "        )\n"
)
if "test_run_config_rejects_shadow_mode_without_premat" not in text:
    if text.count(marker) != 1:
        raise RuntimeError("run-config shadow test insertion marker not found exactly once")
    text = text.replace(marker, test + marker, 1)
p.write_text(text, encoding="utf-8")

print("PREMAT shadow follow-up applied")
