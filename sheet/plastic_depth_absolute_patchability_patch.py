# vvv THOG
"""Keep PLASTIC absolute-ruler controller tests and downstream patches effective."""

from __future__ import annotations

from typing import Any

from . import plastic_depth_lookahead_patch as _lookahead
from . import trainer_step as _trainer_step


# vvv THOG resolve selector dynamically so tests and downstream experiments can still monkey-patch trainer_step.choose_plastic_depth_candidate
def _candidate_selector_proxy(*args: Any, **kwargs: Any) -> Any:
    return _trainer_step.choose_plastic_depth_candidate(*args, **kwargs)


_lookahead.choose_plastic_depth_candidate = _candidate_selector_proxy
# ^^^ THOG
