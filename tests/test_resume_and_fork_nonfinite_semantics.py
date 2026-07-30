# vvv THOG
from __future__ import annotations

import unittest

import run_thog2_lifecycle
import run_thog2_owt  # noqa: F401  # <<< THOG public entry registers final lifecycle material classifications before assertions are tested
from sheet.trainer_checkpoint_resume import _validate_override_fields


class ResumeAndForkNonfiniteSemanticsTests(unittest.TestCase):
    def test_public_lifecycle_preflight_classifies_nonfinite_controls_as_material(self) -> None:
        self.assertNotIn(
            "nonfinite_update_policy",
            run_thog2_lifecycle._OPERATIONAL_CONFIG_DESTINATIONS,
        )
        self.assertNotIn(
            "max_nonfinite_update_skips",
            run_thog2_lifecycle._OPERATIONAL_CONFIG_DESTINATIONS,
        )
        self.assertEqual(
            run_thog2_lifecycle._ARGUMENT_TO_CONFIG["nonfinite_update_policy"],
            "nonfinite_update_policy",
        )
        self.assertEqual(
            run_thog2_lifecycle._ARGUMENT_TO_CONFIG["max_nonfinite_update_skips"],
            "max_nonfinite_update_skips",
        )

    def test_matching_nonfinite_policy_is_assertion_not_override(self) -> None:
        overrides = {
            "device": "cpu",
            "nonfinite_update_policy": "skip",
        }
        validated = _validate_override_fields(
            overrides,
            checkpoint_values={"nonfinite_update_policy": "skip"},
        )
        self.assertEqual(validated, {"device": "cpu"})

    def test_mismatching_nonfinite_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "resume material parameter mismatch: nonfinite_update_policy",
        ):
            _validate_override_fields(
                {"nonfinite_update_policy": "raise"},
                checkpoint_values={"nonfinite_update_policy": "skip"},
            )

    def test_matching_nonfinite_skip_limit_is_assertion_not_override(self) -> None:
        overrides = {
            "max_updates": 200,
            "max_nonfinite_update_skips": 10,
        }
        validated = _validate_override_fields(
            overrides,
            checkpoint_values={"max_nonfinite_update_skips": 10},
        )
        self.assertEqual(validated, {"max_updates": 200})

    def test_mismatching_nonfinite_skip_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "resume material parameter mismatch: max_nonfinite_update_skips",
        ):
            _validate_override_fields(
                {"max_nonfinite_update_skips": 20},
                checkpoint_values={"max_nonfinite_update_skips": 10},
            )

    def test_pre_recovery_checkpoint_without_nonfinite_fields_keeps_legacy_override_path(self) -> None:
        validated = _validate_override_fields(
            {"nonfinite_update_policy": "skip", "max_nonfinite_update_skips": 10},
            checkpoint_values={},
        )
        self.assertEqual(
            validated,
            {"nonfinite_update_policy": "skip", "max_nonfinite_update_skips": 10},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
