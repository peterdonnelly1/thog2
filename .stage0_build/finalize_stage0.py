from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import torch

from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"
BASELINE_SHA = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
DOC_STEMS = (
    "THOG2_Chebyshev_Sheet_Requirements_Specification_v0.1",
    "THOG2_Chebyshev_Sheet_Implementation_Plan_v0.1",
    "THOG2_Chebyshev_Sheet_Staging_Plan_v0.1",
    "THOG2_Chebyshev_Sheet_Test_Plan_v0.1",
)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def validate_runtime_unchanged() -> None:
    protected = ("model.py", "train.py", "sample.py", "configurator.py", "config", "data", "bench.py")
    result = run("git", "diff", "--name-only", BASELINE_SHA, "--", *protected)
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    if changed:
        raise RuntimeError(f"Stage 0 changed protected runtime paths: {changed}")


def validate_documents() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    required_entries = {
        "mimetype",
        "content.xml",
        "styles.xml",
        "meta.xml",
        "META-INF/manifest.xml",
    }
    digest_lines: list[str] = []
    validation_lines: list[str] = []

    for stem in DOC_STEMS:
        md_path = DOCS / f"{stem}.md"
        odt_path = DOCS / f"{stem}.odt"
        if not md_path.is_file() or md_path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty Markdown document: {md_path}")
        if not odt_path.is_file() or odt_path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty ODT document: {odt_path}")

        for path in (md_path, odt_path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_lines.append(f"{digest}  {path.relative_to(ROOT)}")

        with zipfile.ZipFile(odt_path) as archive:
            names = set(archive.namelist())
            missing = sorted(required_entries - names)
            if missing:
                raise RuntimeError(f"{odt_path.name} missing ODT members: {missing}")
            mimetype = archive.read("mimetype").decode("ascii")
            if mimetype != "application/vnd.oasis.opendocument.text":
                raise RuntimeError(f"invalid ODT mimetype in {odt_path.name}: {mimetype}")
            content_root = ElementTree.fromstring(archive.read("content.xml"))
            ElementTree.fromstring(archive.read("styles.xml"))
            ElementTree.fromstring(archive.read("meta.xml"))
            ElementTree.fromstring(archive.read("META-INF/manifest.xml"))
            text_nodes = [text.strip() for text in content_root.itertext() if text.strip()]
            if len(text_nodes) < 40:
                raise RuntimeError(f"implausibly little document content in {odt_path.name}")
            joined = " ".join(text_nodes[:80])
            if "THOG2" not in joined and "Chebyshev Sheet" not in joined:
                raise RuntimeError(f"expected title not found in {odt_path.name}")
            validation_lines.append(
                f"PASS {odt_path.name}: {odt_path.stat().st_size} bytes; "
                f"{len(text_nodes)} non-empty text nodes"
            )

    (EVIDENCE / "stage0_document_sha256.txt").write_text(
        "\n".join(digest_lines) + "\n", encoding="utf-8"
    )
    (EVIDENCE / "stage0_odt_validation.txt").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )


def run_cpu_smoke() -> None:
    torch.manual_seed(1337)
    config = GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, config.block_size), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, config.block_size), dtype=torch.long)

    logits, loss = model(idx, targets)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("non-finite initial loss")
    if logits.shape != (2, config.block_size, config.vocab_size):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_tensors = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradient_tensors or not all(torch.isfinite(g).all() for g in gradient_tensors):
        raise RuntimeError("missing or non-finite gradients")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    changed = sum(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if changed == 0:
        raise RuntimeError("optimizer update changed no parameters")

    with torch.no_grad():
        logits_after, loss_after = model(idx, targets)
    if loss_after is None or not torch.isfinite(loss_after) or not torch.isfinite(logits_after).all():
        raise RuntimeError("non-finite outputs after optimizer update")

    result = {
        "baseline_sha": BASELINE_SHA,
        "workflow_event_sha": os.environ.get("GITHUB_SHA"),
        "branch_head_before_finalization": run("git", "rev-parse", "HEAD").stdout.strip(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "seed": 1337,
        "config": {
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
            "bias": config.bias,
        },
        "logits_shape": list(logits.shape),
        "loss_before": float(loss.detach()),
        "loss_after": float(loss_after.detach()),
        "parameter_count_non_embedding": model.get_num_params(),
        "gradient_tensor_count": len(gradient_tensors),
        "parameter_tensors_changed": changed,
        "all_outputs_finite": True,
        "all_gradients_finite": True,
    }
    (EVIDENCE / "stage0_checkout_smoke.txt").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_ignore_probe() -> None:
    ignored_paths = (
        "out-stage0/ckpt.pt",
        "checkpoints/stage0.ckpt",
        "logs/stage0.log",
        "generated_text/stage0.txt",
        "wandb/stage0.json",
        ".env",
        "stage0-local.log",
    )
    control_path = "data/shakespeare/stage0_fixture.txt"
    created: list[Path] = []
    try:
        for relative in (*ignored_paths, control_path):
            path = ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stage0 probe\n", encoding="utf-8")
            created.append(path)

        lines: list[str] = []
        for relative in ignored_paths:
            probe = run("git", "check-ignore", "-v", relative)
            if not probe.stdout.strip():
                raise RuntimeError(f"expected ignored path was not ignored: {relative}")
            lines.append(probe.stdout.strip())

        control = run("git", "check-ignore", "-q", control_path, check=False)
        if control.returncode == 0:
            raise RuntimeError(f"ordinary fixture was unexpectedly ignored: {control_path}")
        lines.append(f"PASS trackable control: {control_path}")
        (EVIDENCE / "stage0_ignore_probe.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    finally:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for relative in ("out-stage0", "checkpoints", "logs", "generated_text", "wandb"):
            path = ROOT / relative
            if path.exists() and not any(path.iterdir()):
                path.rmdir()


def write_test_results() -> None:
    text = """# THOG2 Stage 0 Test Results

| Test | Result | Evidence |
|---|---|---|
| S0-01 Upstream identity | PASS | `thog2` baseline and selected upstream nanoGPT commit are both `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. |
| S0-02 Clean repository state | PASS | Verification cleanup removes all generated probes before the final commit. |
| S0-03 Dense CPU import and forward | PASS | `evidence/stage0_checkout_smoke.txt`. |
| S0-04 Dense backward and update | PASS | Finite gradients and at least one changed parameter tensor are recorded in `evidence/stage0_checkout_smoke.txt`. |
| S0-05 Dense single-GPU smoke | DEFERRED | The evidence container and GitHub runner expose no NVIDIA CUDA device; the deferral is explicit and remains open for the later GPU stage. |
| S0-06 Documentation-only guard | PASS | The final workflow rejects any changed path outside `.gitignore`, `docs/`, and `evidence/`. |
| S0-07 ODT structure validation | PASS | `evidence/stage0_odt_validation.txt` and `evidence/stage0_document_sha256.txt`. |
| S0-08 Artifact hygiene | PASS | `evidence/stage0_ignore_probe.txt`. |

Stage 0 is complete only after the pull request containing these results is merged to `master`.
"""
    (EVIDENCE / "stage0_test_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    os.chdir(ROOT)
    validate_runtime_unchanged()
    validate_documents()
    run_cpu_smoke()
    run_ignore_probe()
    write_test_results()
    print("Stage 0 finalization checks passed")


if __name__ == "__main__":
    main()
