# vvv THOG
"""Public THOG2 OWT entry point with resume/fork lifecycle dispatch.

The complete pre-enhancement runner is preserved unchanged in
run_thog2_owt_core.py.  Existing imports remain available here while command-line
execution is routed through the lifecycle-aware main function.
"""

from run_thog2_owt_core import *  # noqa: F401,F403                                                                                                      # <<< THOG preserve the complete current-master Python runner API and fresh-run implementation
from run_thog2_lifecycle import main                                                                                                                       # <<< THOG lifecycle-aware CLI owns fresh/resume/fork orchestration


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
