# vvv THOG
"""Keep the descriptor registry single-shot across layered argparse overlays."""

from __future__ import annotations

import argparse


_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help
_DESCRIPTOR_HEADING = "getopt / artifact descriptor registry"


def _format_help_with_single_descriptor_registry(
    parser: argparse.ArgumentParser,
) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(parser)
    first = rendered.find(_DESCRIPTOR_HEADING)
    if first < 0:
        return rendered
    second = rendered.find(_DESCRIPTOR_HEADING, first + len(_DESCRIPTOR_HEADING))
    if second < 0:
        return rendered
    return rendered[:second].rstrip() + "\n"


argparse.ArgumentParser.format_help = _format_help_with_single_descriptor_registry
# ^^^ THOG
