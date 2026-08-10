# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PlasticDepthWrapperOptionsTests(unittest.TestCase):
    def _run_bash(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.pop("THOG2_PLASTIC_LAYER_COUNT_PROBE__PROBE_EVERY_N_STEPS", None)
        environment.pop("THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS", None)
        environment.pop("THOG2_PLASTIC_LAYER_COUNT_MAX_STEP", None)
        return subprocess.run(
            ("bash", *arguments),
            cwd=_REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_canonical_wrapper_help_accepts_double_underscore_controls(self) -> None:
        result = self._run_bash(
            "./train_OWT.sh",
            "--plastic__coarse_phase",
            "disabled",
            "--plastic__layer_count_probe__probe_every_n_steps",
            "7",
            "--plastic__layer_count_probe_radius",
            "3",
            "--plastic__layer_count__max_allowable_layer_change",
            "1",
            "-h",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--plastic__coarse_phase enabled|disabled", result.stdout)
        self.assertIn("--plastic__phase_1__number_of_trials N", result.stdout)
        self.assertIn("--plastic__layer_count_probe_radius N=3", result.stdout)
        self.assertIn("--plastic__layer_count__max_allowable_layer_change N=1", result.stdout)
        self.assertIn("--plastic__log_interval_coarse", result.stdout)
        self.assertIn("--plastic__coarse_phase_roll_through", result.stdout)
        self.assertIn("theil_sen_kendall_LRA", result.stdout)
        self.assertIn("sen_kendall__tau__stratified", result.stdout)
        self.assertIn("objective selected separately", result.stdout)
        self.assertIn("--plastic__layer_count_decision_algorithm__growth_side_discount X", result.stdout)
        self.assertIn("beneficial growth-side objective evidence", result.stdout)
        self.assertNotIn("wall_time__theil_sen_kendall_LRA", result.stdout)
        self.assertNotIn("wall_time__sen_kendall__tau__stratified", result.stdout)
        self.assertNotIn("wall_time__gradient__theil_sen_kendall_slope_tau", result.stdout)
        self.assertNotIn("--plastic__layer_count_gradient__minimum_absolute_kendall_tau", result.stdout)
        self.assertNotIn("plastic--phase", result.stdout + result.stderr)
        self.assertNotIn("Unknown option", result.stdout + result.stderr)

    def test_canonical_controls_retain_arguments_and_legacy_environment(self) -> None:
        command = """
set -- --plastic__layer_count_probe_radius 4 --plastic__layer_count__max_allowable_layer_change=2 marker
source ./plastic_depth_lookahead_wrapper_options.sh
printf '%s|%s|%s\\n' "$THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS" "$THOG2_PLASTIC_LAYER_COUNT_MAX_STEP" "$*"
"""
        result = self._run_bash("-c", command)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "4|2|--plastic__layer_count_probe_radius 4 --plastic__layer_count__max_allowable_layer_change=2 marker",
        )

    def test_v056_decision_controls_are_routed_after_separator(self) -> None:
        command = """
set -- -g RUN --plastic__layer_count_decision_algorithm sen_kendall__tau__stratified --plastic__layer_count_decision_algorithm__growth_side_discount 0.6
source ./plastic_depth_lookahead_wrapper_options.sh
printf '%s\\n' "$@"
"""
        result = self._run_bash("-c", command)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "-g",
                "RUN",
                "--",
                "--plastic__layer_count_decision_algorithm",
                "sen_kendall__tau__stratified",
                "--plastic__layer_count_decision_algorithm__growth_side_discount",
                "0.6",
            ],
        )

    def test_retired_kendall_tau_control_is_rejected(self) -> None:
        command = """
set -- --plastic__layer_count_gradient__minimum_absolute_kendall_tau 0.7
source ./plastic_depth_lookahead_wrapper_options.sh
"""
        result = self._run_bash("-c", command)
        self.assertEqual(result.returncode, 2)
        self.assertIn("removed in PLASTIC v0.55", result.stderr)

    def test_hyphen_and_single_underscore_aliases_are_rejected(self) -> None:
        for alias in ("--plastic-enabled", "--plastic_enabled"):
            result = self._run_bash("./train_OWT.sh", alias, "-h")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Non-canonical PLASTIC option rejected", result.stderr)

    def test_non_positive_values_fail_before_core_getopts(self) -> None:
        result = self._run_bash(
            "./train_OWT.sh",
            "--plastic__layer_count_probe_radius",
            "0",
            "-h",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected a positive integer", result.stderr)
        self.assertNotIn("Unknown option", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
