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


def enrich_options(catalogue):
    """Document open-ended version syntax without making it a restrictive enum."""
    for item in catalogue.values():
        if item.get("options_help") in {"a text value as described above", "one text value per line", "a filesystem path accessible on the execution host"}:
            item.pop("options_help", None)
    versions = {}
    for family in ("chebyshev", "dct", "haar", "lapped_cosine"):
        module = ast.parse((ROOT / f"sheet/bases/{family}.py").read_text())
        versions[family] = next(ast.literal_eval(node.value) for node in module.body
                                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                and target.id.endswith("BASIS_VERSION") for target in node.targets))
    equivalence = ("Chebyshev uses endpoint-clustered root nodes and direct normalization. "
                   "It is mathematically equivalent to DCT-II up to ordering and signs; "
                   "these are no longer distinct basis choices. Training quality requires experiments.")
    version_options = "auto; " + "; ".join(f"{family}: {version}" for family, version in versions.items())
    version_options += "; lapped_cosine_dc_preserving_orthonormal_v1_wN_o500 (even window N >= 2)"
    families = ", ".join(versions)
    for key in ("--basis-family", "DEPTH.compressor", "WIDTH.compressor", "--hyperblock-compressor", "--mlp-hidden-compressor"):
        item = catalogue[key]
        item["help"] = item["help"].split(" Chebyshev uses endpoint-clustered")[0] + " " + equivalence
        item["options_help"] = families + ("; also family@version using the corresponding registered version" if key in {"DEPTH.compressor", "WIDTH.compressor"} else "")
    for key in ("--basis-version", "DEPTH.compressor_version", "WIDTH.compressor_version", "--hyperblock-compressor-version"):
        catalogue[key]["options_help"] = version_options + "; the version must match the selected compressor"
        catalogue[key]["help"] = catalogue[key]["help"].split(" Chebyshev uses endpoint-clustered")[0] + " " + equivalence
    documented = {
        "--attention-geometry": "legacy_sheet_col, depth, head_aware_block, conventional",
        "--mlp-geometry": "legacy_sheet_col, depth, jpeg_like_v1, mlp_block, conventional",
        "--instrumentation__depth_weight_curves__destination": "wandb, local, none",
        "--device": "cpu, cuda, or a device such as cuda:0 (available hardware required)",
        "--dataset": "a prepared dataset name under the data directory (for example openwebtext)",
        "--thogopt__momentum_history_coefficients": "auto or an integer from 1 through L (--n-layer)",
        "--thogopt__scaling_history_coefficients": "auto or an integer from 1 through L (--n-layer)",
        "--lapped-cosine-window-length": "even integers >= 2",
        "--lapped-cosine-overlap-fraction": "0.5",
        "WIDTH.order": "integers from 2 through D (--n-embd), including 512 and 1024 when D=1024",
        "DEPTH.order": "integers from 1 through L (--n-layer)",
        "--instrumentation__width_activation_curves__probe_orders": "auto or distinct comma-separated integers k satisfying 1 <= k < r (WIDTH.order)",
        "--premat_processing_logging": "enabled, disabled",
        "--premat_processing_profiler": "nsys, ncu",
        "--resume": "a checkpoint path or the checkpoint selector described above",
        "--fork": "a checkpoint path or the checkpoint selector described above",
        "--instrumentation__width_activation_curves__end_step": "-1 (unbounded) or nonnegative whole numbers",
    }
    for key, options in documented.items():
        if key in catalogue:
            catalogue[key]["options_help"] = options
    catalogue["--reset-optimizer"]["type"] = "flag"
    catalogue["--print-geometry-registry"]["type"] = "flag"
    catalogue["--plastic-layer-count-hold-updates"]["type"] = "int"
    catalogue["--plastic-layer-count-hold-updates"]["ui_hidden"] = True  # Retired parser option; do not offer it in new Recipes.
    catalogue["--optimizer-momentum"]["type"] = "float"
    catalogue["--direct-factorised-hyperblock-mlp"]["type"] = "flag"
    catalogue["--direct-factorised-hyperblock-mlp"]["boolean_optional"] = True
    for key in ("--attention-geometry", "--mlp-geometry"):
        catalogue[key]["choices"] = documented[key].split(", ")
    for key in ("--mlp-hidden-compressor", "--hyperblock-compressor"):
        catalogue[key]["choices"] = list(versions)
    for key in ("--premat_processing_logging", "--premat_processing_profiler"):
        catalogue[key]["choices"] = documented[key].split(", ")
    if "conventional" not in catalogue["--geometry-preset"]["choices"]:
        catalogue["--geometry-preset"]["choices"].append("conventional")
    for key, item in catalogue.items():
        if item.get("choices") or item.get("options_help") or item.get("type") in {"int", "float", "flag", "bool_value"}:
            continue
        if any(term in key for term in ("-root", "-dir", "snapshot", "resume-from")):
            item["options_help"] = "a filesystem path accessible on the execution host"
        elif item.get("kind") == "list":
            item["options_help"] = "one text value per line"
        else:
            item["options_help"] = "a text value as described above"


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
    enrich_options(catalogue)
    catalogue_path.write_text(json.dumps(catalogue, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
# ^^^ THOG
