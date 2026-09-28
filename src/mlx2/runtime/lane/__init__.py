"""Row-invariant small-M matmul for every weight format (see ``matmul``)."""

from .installer import LAW_ID, install, installed, law_id, set_enabled, uninstall
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
    "GROUP_SIZES",
    "LAW_ID",
    "MAX_ROWS",
    "QUANT_BITS",
    "UNQUANTIZED_BITS",
    "LaneUnsupported",
    "LaneWeights",
    "available",
    "install",
    "installed",
    "lane_matmul",
    "law_id",
    "prepare",
    "set_enabled",
    "split_k",
    "uninstall",
]
