# vvv THOG
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import SheetGPT


class TrainingSheetGPT(SheetGPT):
    """SheetGPT wrapper with float32 generated LayerNorm affine state."""

    def _sheet_layer_norm(
        self,
        inputs: Tensor,
        weight_name: str,
        bias_name: str,
        layer_index: int,
    ) -> Tensor:
        with torch.autocast(
            device_type=inputs.device.type,
            enabled=False,
        ):
            weight = self.trajectory.materialize_vector(
                weight_name,
                layer_index,
            ).float()
            bias = self._optional_bias(bias_name, layer_index)
            if bias is not None:
                bias = bias.float()
            return F.layer_norm(
                inputs,
                (self.config.n_embd,),
                weight,
                bias,
                1.0e-5,
            )
# ^^^ THOG
