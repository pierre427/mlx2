"""Row-invariant small-M matmul for every weight format (see ``matmul``)."""

from .install import LAW_ID, install, installed, set_enabled, uninstall
from .matmul import (
    GROUP_SIZES,
    MAX_ROWS,
    QUANT_BITS,
    UNQUANTIZED_BITS,
    LaneUnsupported,
    LaneWeights,
    available,
    lane_matmul,
    prepare,
    split_k,
)

__all__ = [
    "GROUP_SIZES", "LAW_ID", "MAX_ROWS", "QUANT_BITS", "UNQUANTIZED_BITS",
    "LaneUnsupported", "LaneWeights", "available", "install", "installed",
    "lane_matmul", "prepare", "set_enabled", "split_k", "uninstall",
]
