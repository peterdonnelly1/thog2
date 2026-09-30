# vvv THOG validate and expand saved Runner Recipes without entering model execution
"""Pure Grid resolution and deterministic standalone Bash export."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
import shlex
import uuid

CATALOGUE = json.loads(Path(__file__).with_name("instra_runner_catalogue.json").read_text())
COMMON = ("--geometry-preset", "--optimizer", "--n-layer", "DEPTH.order", "--warmup-iters",
          "--block-size", "--n-embd", "--n-head", "--gradient-accumulation-steps",
          "--checkpoint-segment-size", "--learning-rate", "--min-lr", "--max-iters", "--batch-size",
          "--log-interval", "--eval-iters", "--eval-interval")
FORBIDDEN = {"--device", "--host-label", "--dtype", "--attention-backend", "--o-depth", "--run-name",
             "--print-resolved-json"}
WRAPPER_ENV_OPTIONS = {"--depth-materialisation-matmul": "THOG2_DEPTH_MATERIALISATION_MATMUL",
                       "--materialisation-profiling": "THOG2_MATERIALISATION_PROFILING",
                       "--torch-compile": "THOG2_TORCH_COMPILE"}
MAX_RUNS = 4096
SHORT_OPTIONS = {"-n": "--max-iters", "-b": "--batch-size", "-y": "--optimizer", "-A": "--gradient-accumulation-steps",
                 "-u": "--eval-iters", "-e": "--eval-interval", "-l": "--log-interval", "-w": "--warmup-iters",
                 "-k": "--checkpoint-interval", "-B": "--basis-family", "-v": "--basis-version",
                 "-W": "--lapped-cosine-window-length", "-i": "--lapped-cosine-overlap-fraction",
                 "-a": "--attention-geometry", "-m": "--mlp-geometry", "-L": "--n-layer",
                 "-s": "--layer-dropout-stratum-size", "-M": "--layer-dropout-active-per-stratum", "-H": "--n-head",
                 "-D": "--n-embd", "-C": "--block-size", "-Q": "--o-attn-d-model",
                 "-J": "--o-attn-qkv-per-channel", "-O": "--o-attn-out-per-channel",
                 "-X": "--o-mlp-d-model", "-Y": "--o-mlp-hidden", "-S": "--checkpoint-segment-size",
                 "-r": "--residual-init-policy", "-z": "--residual-init-depth-source", "-Z": "--residual-init-depth-value",
                 "-d": "--dataset", "-t": "--data-dir", "-o": "--checkpoint-root", "-j": "--log-root", "-R": "--result-root"}
MANUAL_SHORT = {"-q", "-g", "-G", "-I", "-F", "-N", "-U", "-V", "-P", "-E", "-T", "-K"}


def _scalar(name, value):
    item = CATALOGUE[name]
    if isinstance(value, (dict, list)) or value is None:
        raise ValueError(f"{name} requires a scalar value")
    kind = item["type"]
    if kind == "flag":
        if type(value) is not bool:
            raise ValueError(f"{name} requires true or false")
        return value
    if kind == "bool_value":
        if value not in ("true", "false", True, False):
            raise ValueError(f"{name} requires true or false")
        return str(value).lower()
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"Invalid {name} value")
    raw = str(value)
    if not raw or "\0" in raw or "\n" in raw or len(raw) > 2048:
        raise ValueError(f"Invalid {name} value")
    try:
        if kind == "int":
            if not re.fullmatch(r"[+-]?\d+", raw):
                raise ValueError()
            result = int(raw)
            if result < 0 or (name == "DEPTH.order" and result < 1):
                raise ValueError()
        elif kind == "float":
            import math
            result = float(raw)
            if not math.isfinite(result):
                raise ValueError()
        else:
            result = raw
    except ValueError as error:
        raise ValueError(f"Invalid {name} value: {raw[:80]}") from error
    if item.get("choices") and result not in item["choices"]:
        raise ValueError(f"{name} requires one of {item['choices']}")
    return result


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) - {"label", "parameters", "gpu_pool", "host_ids",
                                                      "max_parallel", "profilers", "headroom_mib", "power_caps"}:
        raise ValueError("Invalid Grid Recipe fields")
    label = recipe.get("label", "")
    if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120 or any(ord(c) < 32 for c in label):
        raise ValueError("A Grid Recipe label of 1–120 printable characters is required")
    parameters = recipe.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    choices = {}
    for name, value in parameters.items():
        if name not in CATALOGUE or name in FORBIDDEN or CATALOGUE[name]["kind"] in {"automatic", "manual"}:
            raise ValueError(f"Unsupported or automatic Runner option: {name}")
        item = CATALOGUE[name]
        if item["kind"] == "dimension":
            values = value if isinstance(value, list) else [value]
            if not values or len(values) > 64:
                raise ValueError(f"{name} needs 1–64 choices")
            choices[name] = [_scalar(name, entry) for entry in values]
            if len({json.dumps(entry) for entry in choices[name]}) != len(choices[name]):
                raise ValueError(f"Duplicate {name} choices")
        elif item["kind"] == "list":
            if not isinstance(value, list) or len(value) > 64:
                raise ValueError(f"{name} must be a single list value")
            choices[name] = [_scalar(name, entry) if item["type"] != "list" else str(entry) for entry in value]
            if any("\0" in str(entry) or "\n" in str(entry) for entry in choices[name]):
                raise ValueError(f"Invalid {name} list")
        else:
            choices[name] = _scalar(name, value)
    for field in ("gpu_pool", "host_ids"):
        if field in recipe and (not isinstance(recipe[field], list) or
                                any(not isinstance(v, str) or not re.fullmatch(r"[\w.:-]{1,150}", v) for v in recipe[field])):
            raise ValueError(f"Invalid {field}")
    parallel = recipe.get("max_parallel", 1)
    if type(parallel) is not int or parallel < 1:
        raise ValueError("max_parallel must be a positive integer")
    modes = recipe.get("profilers", ["none"])
    if not isinstance(modes, list) or not modes or len(modes) > 2 or len(set(modes)) != len(modes) or set(modes) - {"none", "nsys", "ncu"}:
        raise ValueError("profilers must be none or the NSYS/NCU pair")
    if len(modes) == 2 and set(modes) != {"nsys", "ncu"}:
        raise ValueError("Only NSYS and NCU may be paired")
    if type(recipe.get("headroom_mib", 512)) is not int or not 0 <= recipe.get("headroom_mib", 512) <= 65536:
        raise ValueError("Invalid memory headroom")
    caps = recipe.get("power_caps", {})
    if not isinstance(caps, dict) or any(not isinstance(key, str) or not re.fullmatch(r"[\w.:-]{1,150}", key)
                                       or type(value) is not int or not 50 <= value <= 600 for key, value in caps.items()):
        raise ValueError("power_caps must map GPU identities to integer watts in 50–600")
    return choices


def expand(recipe, *, stable_preview=False):
    choices = validate_recipe(recipe)
    keys = [key for key in choices if CATALOGUE[key]["kind"] == "dimension"]
    count = len(recipe.get("profilers", ["none"]))
    for key in keys:
        count *= len(choices[key])
        if count > MAX_RUNS:
            raise ValueError(f"Grid exceeds {MAX_RUNS} runs")
    fixed = {key: value for key, value in choices.items() if key not in keys}
    trials = []
    seen = set()
    seed = json.dumps(recipe, sort_keys=True, separators=(",", ":")) if stable_preview else ""
    presets = choices.get("--geometry-preset", [])
    mixed_presets = "dense" in presets and any(preset != "dense" for preset in presets)
    max_layers = max(choices.get("--n-layer", [0]))
    for selected in itertools.product(*(choices[key] for key in keys)):
        values = {**fixed, **dict(zip(keys, selected))}
        dense = values.get("--geometry-preset") == "dense" or values.get("--model-type") == "dense"
        if mixed_presets and not dense and values.get("--n-layer", max_layers) != max_layers:
            continue
        if dense:
            # A depth sweep contributes one dense reference, regardless of how many
            # DEPTH.order choices accompany the other (compact) trials.
            values.pop("DEPTH.order", None)
        for name in ("--max-iters", "--batch-size", "--block-size", "--n-layer", "--n-head", "--n-embd",
                     "--gradient-accumulation-steps", "--checkpoint-segment-size"):
            if name in values and int(values[name]) < 1:
                raise ValueError(f"{name} must be positive")
        if int(values.get("--n-embd", 768)) % int(values.get("--n-head", 12)):
            raise ValueError("--n-embd must be divisible by --n-head")
        if int(values.get("--warmup-iters", 10)) >= int(values.get("--max-iters", 100)):
            raise ValueError("--warmup-iters must be less than --max-iters")
        if "DEPTH.order" in values and values["DEPTH.order"] > int(values.get("--n-layer", 72)):
            raise ValueError("DEPTH.order must not exceed --n-layer")
        if "DEPTH.order" in values and any(entry.startswith("DEPTH.order=") for entry in values.get("--option", [])):
            raise ValueError("DEPTH.order cannot also appear in --option")
        if values.get("--learning-rate") is not None and values.get("--min-lr") is not None and float(values["--min-lr"]) > float(values["--learning-rate"]):
            raise ValueError("--min-lr must not exceed --learning-rate")
        identity = json.dumps(values, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            continue
        seen.add(identity)
        pairing_id = (uuid.uuid5(uuid.NAMESPACE_URL, f"thog2-preview-pair:{seed}:{identity}").hex
                      if stable_preview else uuid.uuid4().hex) if len(recipe.get("profilers", ["none"])) == 2 else None
        for profiler in recipe.get("profilers", ["none"]):
            run_id = (uuid.uuid5(uuid.NAMESPACE_URL, f"thog2-preview-run:{seed}:{identity}:{profiler}").hex
                      if stable_preview else uuid.uuid4().hex)
            trials.append({"run_id": run_id, "pairing_id": pairing_id,
                           "profiler": profiler, "parameters": values.copy()})
    return trials


def command_for(run, gpu, *, python="python", entry="run_thog2_owt", host_label=None):
    """No shell input reaches subprocess; the script renderer quotes every argument."""
    values = run["parameters"]
    args = [python, "-m", entry]
    if host_label is not None:
        args += ["--host-label", host_label]
    if run.get("dtype"):
        args += ["--dtype", run["dtype"]]
    if run.get("attention_backend"):
        args += ["--attention-backend", run["attention_backend"]]
    preset = values.get("--geometry-preset", "depth")
    args += ["--model-type", values.get("--model-type", "dense" if preset == "dense" else "sheet")]
    if (any(name.startswith("--plastic__") and name != "--plastic__enabled" and
            (CATALOGUE[name]["type"] != "flag" or value is True) for name, value in values.items())
            and "--plastic__enabled" not in values and "--no-plastic__enabled" not in values):
        args.append("--plastic__enabled")
    for name, value in values.items():
        if name == "DEPTH.order":
            args += ["--select-depth", "--option", f"DEPTH.order={value}"]
        elif name in {"--model-type", "--experiment-prefix", "--cuda-expandable-segment", *WRAPPER_ENV_OPTIONS} or (name == "--geometry-preset" and value == "dense"):
            continue
        elif CATALOGUE[name]["kind"] == "list":
            for entry in value:
                args += [name, str(entry)]
        elif CATALOGUE[name]["type"] == "flag":
            if value:
                args.append(name)
            elif name.startswith("--no-"):
                args.append("--" + name[5:])
            elif CATALOGUE[name].get("boolean_optional"):
                args.append("--no-" + name[2:])
        else:
            args += [name, str(value)]
    run_label = f"{run['grid_tag']}_{run['run_id'][:12]}_{run['profiler'].upper()}"
    prefix = values.get("--experiment-prefix", "")
    args += ["--run-name", run_label, "--experiment-prefix", f"{prefix}_{run_label}" if prefix else run_label]
    if run["profiler"] != "none":
        args += ["--premat_processing_logging", "enabled", "--premat_processing_profiler", run["profiler"]]
        if run["profiler"] == "ncu" and "--ncu-probe-layer" not in values:
            args += ["--ncu-probe-layer", "0"]
    return args


def environment_for(run):
    values = run["parameters"]
    result = {env_name: str(values[option]) for option, env_name in WRAPPER_ENV_OPTIONS.items() if option in values}
    if values.get("--cuda-expandable-segment", "enabled") == "enabled":
        result["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return result


def script_for(runs):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\"",
             "# THOG Runner: deterministic resolved local commands; verify paths and GPU ordinals on replay."]
    power_helper = "/usr/local/libexec/instra-power-control"
    restore = "; ".join(f"sudo -n {power_helper} {shlex.quote(uuid)} default" for uuid in
                        dict.fromkeys(run["gpu"].get("uuid") for run in runs) if uuid)
    if restore:
        lines.append(f"trap {shlex.quote(restore)} EXIT")
    for run in runs:
        gpu = run["gpu"]
        args = command_for(run, gpu, host_label=run.get("host_label"))
        lines.append(f"# {run.get('host_label', 'local')} GPU {gpu['ordinal']} · {run['run_id']} · {run['profiler']}")
        metadata = {"grid_id": run.get("grid_id"), "grid_tag": run["grid_tag"], "run_id": run["run_id"],
                    "pairing_id": run["pairing_id"], "recipe_id": run.get("recipe_id"), "profiler": run["profiler"],
                    "gpu_uuid": gpu.get("uuid"), "thog_host_id": run.get("host_id"),
                    "execution_profile": run.get("execution_profile")}
        lines.append(f"sudo -n {power_helper} {shlex.quote(gpu['uuid'])} "
                     f"{int(run['requested_power_w']) if run.get('requested_power_w') is not None else 'default'}")
        environment = {**environment_for(run), "THOG2_RUNNER_METADATA": json.dumps(metadata, separators=(',', ':')),
                       "CUDA_VISIBLE_DEVICES": str(gpu["ordinal"])}
        lines.append(" ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + " " + shlex.join(args))
    return "\n".join(lines) + "\n"


def classic_script_for(runs):
    """Export resolved runs through the established train_OWT.sh front end."""
    inverse = {name: flag for flag, name in SHORT_OPTIONS.items()}
    inverse.update({"--dtype": "-T", "--attention-backend": "-K", "--geometry-preset": "-p"})
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "${BASH_SOURCE[0]}")/../.."',
             "# Equivalent legacy-wrapper invocations; check host paths and GPU ordinals before replay."]
    power_helper = "/usr/local/libexec/instra-power-control"
    restore = "; ".join(f"sudo -n {power_helper} {shlex.quote(uuid)} default" for uuid in
                        dict.fromkeys(run["gpu"].get("uuid") for run in runs) if uuid)
    if restore:
        lines.append(f"trap {shlex.quote(restore)} EXIT")
    for run in runs:
        gpu = run["gpu"]
        resolved = command_for(run, gpu, host_label=run.get("host_label"))[3:]
        label = f"{run['grid_tag']}_{run['run_id'][:12]}_{run['profiler'].upper()}"
        flags = ["-g", label]
        forwarded = []
        index = 0
        while index < len(resolved):
            name = resolved[index]
            value = resolved[index + 1] if index + 1 < len(resolved) else None
            if name == "--run-name":
                index += 2
                continue
            if name == "--model-type":
                if value == "dense": flags += ["-p", "dense"]
                index += 2
                continue
            if name == "--experiment-prefix":
                if value != label: forwarded += [name, value]
                index += 2
                continue
            if name in {"--learning-rate", "--min-lr"}:
                code = float(value) * 100000
                if abs(round(code) - code) < 1e-8:
                    flags += ["-c" if name == "--learning-rate" else "-f", str(round(code))]
                else:
                    forwarded += [name, value]
                index += 2
                continue
            if name in inverse:
                flags += [inverse[name], value]
                index += 2
                continue
            forwarded.append(name)
            if value is not None and not value.startswith("--"):
                forwarded.append(value)
                index += 2
            else:
                index += 1
        environment = {**environment_for(run), "CUDA_VISIBLE_DEVICES": str(gpu["ordinal"]),
                       "THOG2_HOST_LABEL": run.get("host_label", "local")}
        lines.append(f"# {run.get('host_label', 'local')} GPU {gpu['ordinal']} · {run['run_id']} · {run['profiler']}")
        lines.append(f"sudo -n {power_helper} {shlex.quote(gpu['uuid'])} "
                     f"{int(run['requested_power_w']) if run.get('requested_power_w') is not None else 'default'}")
        lines.append(" ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) +
                     " ./train_OWT.sh " + shlex.join([*flags, *forwarded]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse
    import csv
    import subprocess
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--getopt-options"]:
        print(",".join(name[2:] + ("" if item["type"] == "flag" else ":") for name, item in CATALOGUE.items()
                       if name.startswith("--") and item["kind"] not in {"manual", "automatic"}) + ",grid-label:,gpu-ordinals:,output:,preview")
        return 0
    parser = argparse.ArgumentParser(description="Generate a standalone resolved THOG Grid script")
    parser.add_argument("--grid-label", default="Local Grid")
    parser.add_argument("--gpu-ordinals", default="0")
    parser.add_argument("--output")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("options", nargs=argparse.REMAINDER)
    # Parse metadata options without expanding or executing any shell text.
    meta = []
    options = []
    index = 0
    while index < len(argv):
        name = argv[index]
        if name == "--":
            index += 1
            continue
        if name == "-h":
            parser.print_help(); return 0
        if name in MANUAL_SHORT:
            parser.error(f"{name} is reserved for train_OWT.sh manual runs; use Runner's resolved controls")
        if name in SHORT_OPTIONS or name in {"-c", "-f", "-p", "-x"}:
            if index + 1 >= len(argv):
                parser.error(f"{name} requires a value")
            value = argv[index+1]
            if name in {"-c", "-f"}:
                if not value.isdigit() or not 1 <= int(value) <= (1000 if name == "-c" else 100):
                    parser.error(f"{name} requires an integer learning-rate code")
                options.append(("--learning-rate" if name == "-c" else "--min-lr", f"{int(value)}e-5"))
            elif name == "-p":
                options.append(("--model-type", "dense")) if value == "dense" else options.append(("--geometry-preset", value))
            elif name == "-x":
                if value not in {"true", "false"}: parser.error("-x requires true or false")
                if value == "true": options.append(("--dry-run", True))
            else:
                options.append((SHORT_OPTIONS[name], value))
            index += 2
            continue
        if name in {"--grid-label", "--gpu-ordinals", "--output"}:
            if index + 1 >= len(argv):
                parser.error(f"{name} requires a value")
            meta += argv[index:index+2]
            index += 2
        elif name == "--preview":
            meta.append(name)
            index += 1
        else:
            if name not in CATALOGUE or CATALOGUE[name]["kind"] in {"automatic", "manual"}:
                parser.error(f"Unsupported option: {name}")
            if CATALOGUE[name]["type"] == "flag":
                options.append((name, True))
                index += 1
            else:
                if index + 1 >= len(argv):
                    parser.error(f"{name} requires a value")
                options.append((name, argv[index+1]))
                index += 2
    flags = parser.parse_args(meta)
    parameters = {}
    for name, raw in options:
        item = CATALOGUE[name]
        if item["kind"] == "dimension":
            values = next(csv.reader([raw], strict=True)) if isinstance(raw, str) else [raw]
            parameters.setdefault(name, []).extend(values)
        elif item["kind"] == "list":
            parameters.setdefault(name, []).append(raw)
        elif name in parameters:
            parser.error(f"{name} may appear only once")
        else:
            parameters[name] = raw
    if "--dataset" in parameters and "--data-dir" not in parameters:
        parameters["--data-dir"] = "data/" + str(parameters["--dataset"])
    recipe = {"label": flags.grid_label, "parameters": parameters}
    trials = expand(recipe)
    if not re.fullmatch(r"\d+(,\d+)*", flags.gpu_ordinals):
        parser.error("--gpu-ordinals requires comma-separated GPU ordinals")
    ordinals = [int(part) for part in flags.gpu_ordinals.split(",")]
    if len(set(ordinals)) != len(ordinals):
        parser.error("Duplicate GPU ordinal")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=8, check=True)
        caps = {int(parts[0].strip()): int(parts[1].strip().split(".")[0]) for line in result.stdout.splitlines()
                if len(parts := line.split(",")) == 2}
    except (OSError, ValueError, subprocess.SubprocessError):
        caps = {}
    for index, trial in enumerate(trials):
        ordinal = ordinals[index % len(ordinals)]
        trial.update(grid_tag="G-LOCAL", gpu={"ordinal": ordinal}, dtype="bfloat16" if caps.get(ordinal, 0) >= 8 else "float16",
                     attention_backend="sdpa")
    if flags.preview:
        print(json.dumps({"total_runs": len(trials), "runs": trials}, indent=2))
    else:
        destination = Path(flags.output or "grid-scripts/G-LOCAL.sh")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(script_for(trials))
        destination.chmod(0o700)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
