# vvv THOG
from sheet import processing_ncu_compatibility as compatibility
from sheet import processing_ncu_semantic_suffix_patch as repair


def test_ncu_kernel_rename_suffix_is_accepted() -> None:
    label = (
        "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=8/"
        "void at::native::elementwise_kernel<(int)128>"
    )
    assert repair.parse_processing_nvtx_label_with_ncu_suffix(label) == {
        "role": "PREMAT",
        "operation": "materialize",
        "family": "DOWN",
        "layer": 8,
    }


def test_ncu_nvtx_range_suffix_is_accepted() -> None:
    label = (
        "1614690  \"<default domain>:THOG2_PROCESSING|owner=MAIN|operation=consume|"
        "family=DOWN|layer=8:none:none:none:none:none:none\""
    )
    assert repair.parse_processing_nvtx_label_with_ncu_suffix(label) == {
        "role": "MAIN",
        "operation": "consume",
        "family": "DOWN",
        "layer": 8,
    }


def test_runtime_parser_is_repaired() -> None:
    label = "THOG2_PROCESSING|owner=MAIN|operation=consume|family=DOWN|layer=8/kernel"
    assert compatibility.parse_processing_nvtx_label(label) == {
        "role": "MAIN",
        "operation": "consume",
        "family": "DOWN",
        "layer": 8,
    }
# ^^^ THOG
