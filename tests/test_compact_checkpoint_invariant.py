# vvv THOG
from __future__ import annotations

import unittest

from model import GPT, GPTConfig
from sheet.checkpoints import compact_model_state
from sheet.compact_identity import (
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_JPEG_LIKE_V1,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
)
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import LEGACY_ATTENTION_INPUT_WEIGHT


_COMPACT_PRESETS = (
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_JPEG_LIKE_V1,
    GEOMETRY_PRESET_MLP_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_FULL_BLOCK,
)

_ALLOWED_SHEET_STATE_PREFIXES = (
    "trajectory.coefficients.",
    "trajectory.depth.coefficients.",
    "transformer.wte.",
    "transformer.wpe.",
    "transformer.ln_f.",
    "lm_head.",
)


def _sheet_config(preset: str) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=8,
        depth_order=2,
        base_row_order=4,
        mlp_channel_order=4,
        o_attn_d_model=4,
        o_attn_qkv_per_channel=2,
        o_attn_out_per_channel=2,
        o_mlp_d_model=4,
        o_mlp_hidden=4,
        mlp_hidden_group_size=8,
        geometry_preset=preset,
        basis_family="chebyshev",
    )


class CompactCheckpointInvariantTest(unittest.TestCase):
    def test_all_compact_presets_save_native_state_only(self):
        for preset in _COMPACT_PRESETS:
            with self.subTest(preset=preset):
                model = SheetGPT(_sheet_config(preset))
                state_before = compact_model_state(model, "thog2_sheet")

                self.assertTrue(
                    any(".coefficients." in key for key in state_before),
                    f"{preset} checkpoint contains no coefficient tensors",
                )
                self.assertFalse(
                    any("bases." in key for key in state_before),
                    f"{preset} checkpoint persisted reproducible fixed bases",
                )
                self.assertFalse(
                    any(key.startswith("transformer.h.") for key in state_before),
                    f"{preset} checkpoint persisted repeated dense transformer blocks",
                )
                unexpected = sorted(
                    key
                    for key in state_before
                    if not key.startswith(_ALLOWED_SHEET_STATE_PREFIXES)
                )
                self.assertEqual(unexpected, [])

                model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 0)
                state_after = compact_model_state(model, "thog2_sheet")
                self.assertEqual(tuple(state_after), tuple(state_before))

    def test_dense_checkpoint_keeps_dense_layer_weights(self):
        model = GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=True,
            )
        )
        state = compact_model_state(model, "dense")
        self.assertIn("transformer.h.0.attn.c_attn.weight", state)
        self.assertIn("transformer.h.0.mlp.c_fc.weight", state)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
