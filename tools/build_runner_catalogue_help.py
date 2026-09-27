#!/usr/bin/env python3
# vvv THOG keep Runner field descriptions linked to the canonical getopt descriptor registry
"""Refresh catalogue help from the public THOG descriptor registry without importing torch."""

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thog_grid_runner import SHORT_OPTIONS  # noqa: E402


def main():
    registry = ast.parse((ROOT / "sheet/help_registry_descriptor_patch.py").read_text())
    sections = next(ast.literal_eval(node.value) for node in registry.body
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == "_DESCRIPTOR_SECTIONS")
    catalogue_path = ROOT / "instra_runner_catalogue.json"
    catalogue = json.loads(catalogue_path.read_text())
    short_options = {**SHORT_OPTIONS, "-p": "--geometry-preset", "-q": "--run-mode",
                     "-g": "--run-name", "-c": "--learning-rate", "-f": "--min-lr",
                     "-x": "--dry-run", "-P": "--o-depth", "-T": "--dtype", "-K": "--attention-backend"}
    help_text = {}
    short_labels = {}
    for _section, rows in sections:
        for short, option, explanation in rows:
            names = re.findall(r"--[a-z][a-z0-9_-]*", option)
            if option.startswith("DEPTH.order="):
                names.append("DEPTH.order")
            if short in short_options:
                names.append(short_options[short])
            for name in names:
                if name in catalogue:
                    help_text[name] = explanation
                    if short.startswith("-") and short != "—" and len(short) == 2:
                        short_labels[name] = short
    for name, item in catalogue.items():
        if name.startswith("--no-") and name not in help_text:
            help_text[name] = help_text.get("--" + name[5:], "")
        if name in help_text:
            item["help"] = help_text[name]
        if name in short_labels:
            item["short"] = short_labels[name]
        if item.get("kind") not in {"manual", "automatic"} and not item.get("ui_hidden") and not item.get("help"):
            raise ValueError(f"No descriptor-registry help for visible Runner field {name}")
    catalogue_path.write_text(json.dumps(catalogue, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
# ^^^ THOG
