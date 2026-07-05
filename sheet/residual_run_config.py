# vvv THOG
from __future__ import annotations

from .run_config import OwtRunConfig as _BaseOwtRunConfig
from .run_naming import build_artifact_name, truncate_component


class OwtRunConfig(_BaseOwtRunConfig):
    def residual_init_artifact_fragment(self) -> str:
        residual_init = self.residual_init_config()
        if self.residual_init_policy == "unscaled":
            return "r_unscaled"
        parts = [
            f"r_{self.residual_init_policy}",
            f"z_{residual_init.depth_source}",
        ]
        if residual_init.depth_source == "user_forced_depth":
            parts.append(f"Z_{self.residual_init_depth_value}")
        return "_".join(parts)

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
        if base_name.endswith(segment_suffix):
            artifact_name = f"{base_name[:-len(segment_suffix)]}_{residual_fragment}{segment_suffix}"
        else:
            artifact_name = f"{base_name}_{residual_fragment}"
        if self.artifact_suffix:
            artifact_name = f"{artifact_name}__{self.artifact_suffix.strip().upper()}"
        return truncate_component(artifact_name, max_length=self.artifact_name_limit)


__all__ = ["OwtRunConfig"]
# ^^^ THOG
