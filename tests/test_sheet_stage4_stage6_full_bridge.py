# vvv THOG
"""Temporary hosted-CI bridge for the complete Stage 6 and finishing suite."""

from __future__ import annotations

import unittest

from tests import test_sheet_stage6_canonical_naming
from tests import test_sheet_stage6_controls
from tests import test_sheet_stage6_dense_checkpointing
from tests import test_sheet_stage6_diagnostics
from tests import test_sheet_stage6_memmap
from tests import test_sheet_stage6_naming
from tests import test_sheet_stage6_protocol
from tests import test_sheet_stage6_restart
from tests import test_sheet_stage6_run_config
from tests import test_sheet_stage6_runner_scripts
from tests import test_sheet_stage6_traces
from tests import test_sheet_stage6_trainer
from tests import test_sheet_stage6_wandb


MODULES = (
    test_sheet_stage6_canonical_naming,
    test_sheet_stage6_controls,
    test_sheet_stage6_dense_checkpointing,
    test_sheet_stage6_diagnostics,
    test_sheet_stage6_memmap,
    test_sheet_stage6_naming,
    test_sheet_stage6_protocol,
    test_sheet_stage6_restart,
    test_sheet_stage6_run_config,
    test_sheet_stage6_runner_scripts,
    test_sheet_stage6_traces,
    test_sheet_stage6_trainer,
    test_sheet_stage6_wandb,
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for module in MODULES:
        suite.addTests(loader.loadTestsFromModule(module))
    return suite

# ^^^ THOG
