# vvv THOG
"""Public THOG2 OWT entry point with resume/fork lifecycle dispatch.

The complete pre-enhancement runner is preserved unchanged in
run_thog2_owt_core.py. Existing imports remain available here while command-line
execution is routed through the lifecycle-aware main function.
"""

import run_thog2_owt_core as _core                                                                                                                          # <<< THOG keep the preserved runner as the implementation substrate
from sheet.run_naming import artifact_paths as _artifact_paths                                                                                              # <<< THOG lifecycle orchestration reuses the current artifact-path contract

_core.artifact_paths = _artifact_paths                                                                                                                       # <<< THOG expose the current naming helper to the preserved runner module used by lifecycle orchestration

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                        # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation
from run_thog2_lifecycle import main                                                                                                                         # <<< THOG lifecycle-aware CLI owns fresh/resume/fork orchestration


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
