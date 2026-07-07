# vvv THOG
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

import run_thog2_owt as base_runner
from sheet.basis import BASIS_VERSION
from sheet.compact_identity import BASIS_FAMILY_CHEBYSHEV, GEOMETRY_PRESET_CURVE, compact_identity_metadata
from sheet.residual_run_config import OwtRunConfig as ResidualOwtRunConfig
from sheet.run_naming import build_artifact_name, truncate_component
from sheet.training_config import ROW_ORDER_SCALING_RULE, TrainingConfig


_BASE_BUILD_PARSER = base_runner.build_parser
_BASE_CONFIG_FROM_ARGUMENTS = base_runner.config_from_arguments
BASIS_LABELS = {"chebyshev": "CHEBY", "dct": "DCT"}
DEFAULT_MLP_CHANNEL_ORDER = 256


@dataclass(frozen=True)
class Stage8OwtRunConfig(ResidualOwtRunConfig):
    geometry_preset: Optional[str] = GEOMETRY_PRESET_CURVE
    attention_geometry: Optional[str] = None
    mlp_geometry: Optional[str] = None
    basis_family: Optional[str] = BASIS_FAMILY_CHEBYSHEV
    basis_version: str = BASIS_VERSION
    attention_backend: str = "auto"
    mlp_channel_order: int = DEFAULT_MLP_CHANNEL_ORDER

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attention_backend not in ("auto", "flash2", "sdpa", "math"):
            raise ValueError("attention_backend must be auto, flash2, sdpa, or math")
        if isinstance(self.mlp_channel_order, bool) or not isinstance(self.mlp_channel_order, int) or self.mlp_channel_order <= 0:
            raise ValueError("mlp_channel_order must be a positive integer")
        if self.model_type == "sheet" and self.mlp_channel_order > 4 * self.n_embd:
            raise ValueError("mlp_channel_order must not exceed 4*n_embd")
        if self.model_type == "sheet":
            object.__setattr__(self, "basis_version", str(self.compact_identity()["basis_version"]))

    def compact_identity(self) -> Dict[str, Any]:
        return compact_identity_metadata(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            depth_order=self.depth_order,
            base_row_order=self.base_row_order,
            basis_version=self.basis_version,
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
            geometry_preset=self.geometry_preset,
            attention_geometry=self.attention_geometry,
            mlp_geometry=self.mlp_geometry,
            basis_family=self.basis_family,
        )

    def compact_artifact_fragment(self) -> Optional[str]:
        if self.model_type != "sheet":
            return None
        identity = self.compact_identity()
        basis_label = BASIS_LABELS.get(str(identity["basis_family"]), str(identity["basis_family"]).upper())
        return "__".join((f"{basis_label}_{str(identity['geometry_preset']).upper()}", f"A_{str(identity['attention_geometry']).upper()}", f"M_{str(identity['mlp_geometry']).upper()}", f"R_{self.mlp_channel_order}", f"V_{str(identity['basis_version']).upper()}"))

    def compact_suffix(self) -> Optional[str]:
        parts = []
        compact = self.compact_artifact_fragment()
        if compact:
            parts.append(compact)
        if self.artifact_suffix:
            parts.append(self.artifact_suffix.strip().upper())
        return "__".join(parts) if parts else None

    @property
    def artifact_name(self) -> str:
        base_name = build_artifact_name(
            model_type=self.internal_model_type,
            host_label=self.host_label,
            run_name=self.run_name,
            dataset_name=self.dataset,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_iters=self.max_iters,
            checkpoint_interval=self.checkpoint_interval,
            warmup_iters=self.warmup_iters,
            checkpoint_segment_size=self.checkpoint_segment_size,
            depth_order=self.depth_order if self.model_type == "sheet" else None,
            base_row_order=self.base_row_order if self.model_type == "sheet" else None,
            suffix=None,
            max_length=max(1000, self.artifact_name_limit),
        )
        residual_fragment = self.residual_init_artifact_fragment()
        segment_suffix = f"_S_{self.checkpoint_segment_size}"
        artifact_name = f"{base_name[:-len(segment_suffix)]}_{residual_fragment}{segment_suffix}" if base_name.endswith(segment_suffix) else f"{base_name}_{residual_fragment}"
        compact_suffix = self.compact_suffix()
        if compact_suffix:
            artifact_name = f"{artifact_name}__{compact_suffix}"
        return truncate_component(artifact_name, max_length=self.artifact_name_limit)

    def to_training_config(self, *, vocab_size: int, world_size: int, out_dir) -> TrainingConfig:
        config = super().to_training_config(vocab_size=vocab_size, world_size=world_size, out_dir=out_dir)
        if self.model_type != "sheet":
            return config
        values = asdict(config)
        values.update({"basis_version": self.basis_version, "geometry_preset": self.geometry_preset, "attention_geometry": self.attention_geometry, "mlp_geometry": self.mlp_geometry, "basis_family": self.basis_family, "mlp_channel_order": self.mlp_channel_order})
        return TrainingConfig(**values)

    def canonical_dict(self, *, world_size: int) -> Dict[str, Any]:
        values = super().canonical_dict(world_size=world_size)
        values["attention_backend"] = self.attention_backend
        if self.model_type == "sheet":
            values["geometry_preset"] = self.geometry_preset
            values["attention_geometry"] = self.attention_geometry
            values["mlp_geometry"] = self.mlp_geometry
            values["basis_family"] = self.basis_family
            values["basis_version"] = self.basis_version
            values["mlp_channel_order"] = self.mlp_channel_order
            values["compact_identity"] = self.compact_identity()
            values["compact_artifact_fragment"] = self.compact_artifact_fragment()
        return values


def build_parser() -> argparse.ArgumentParser:
    parser = _BASE_BUILD_PARSER()
    parser.add_argument("--geometry-preset", default=GEOMETRY_PRESET_CURVE)
    parser.add_argument("--attention-geometry")
    parser.add_argument("--mlp-geometry")
    parser.add_argument("--basis-family", default=BASIS_FAMILY_CHEBYSHEV)
    parser.add_argument("--basis-version", default="auto")
    parser.add_argument("--attention-backend", choices=("auto", "flash2", "sdpa", "math"), default="auto")
    parser.add_argument("--mlp-channel-order", type=int, default=DEFAULT_MLP_CHANNEL_ORDER)
    return parser


def configure_attention_backend(attention_backend: str) -> None:
    if attention_backend == "auto" or not torch.cuda.is_available():
        return
    if attention_backend == "flash2":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return
    if attention_backend == "sdpa":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return
    raise ValueError(f"unsupported attention backend: {attention_backend}")


def config_from_arguments(arguments: argparse.Namespace) -> Stage8OwtRunConfig:
    base_runner.OwtRunConfig = ResidualOwtRunConfig
    base_config = _BASE_CONFIG_FROM_ARGUMENTS(arguments)
    values = asdict(base_config)
    basis_version = BASIS_VERSION if arguments.basis_version == "auto" else arguments.basis_version
    values.update({"geometry_preset": arguments.geometry_preset, "attention_geometry": arguments.attention_geometry, "mlp_geometry": arguments.mlp_geometry, "basis_family": arguments.basis_family, "basis_version": basis_version, "attention_backend": arguments.attention_backend, "mlp_channel_order": arguments.mlp_channel_order})
    config = Stage8OwtRunConfig(**values)
    configure_attention_backend(config.attention_backend)
    return config


base_runner.build_parser = build_parser
base_runner.config_from_arguments = config_from_arguments
base_runner.OwtRunConfig = Stage8OwtRunConfig


if __name__ == "__main__":
    raise SystemExit(base_runner.main())
# ^^^ THOG
