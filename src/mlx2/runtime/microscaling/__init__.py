"""Opt-in integer-exact microscaling representation and execution primitives.

Importing the package does not initialize MLX, touch a GPU, or enable a route.
"""
from .format import (
    PackedSCWeight,
    SC_FORMAT,
    SCFormat,
    SCFormatError,
    decode_numpy,
    from_mxfp4_numpy,
    pack_numpy,
    metadata_from_mxfp4_scales_numpy,
    to_mxfp4_scales_numpy,
)

__all__ = [
    "PackedSCWeight", "SC_FORMAT", "SCFormat", "SCFormatError", "decode_numpy",
    "from_mxfp4_numpy", "pack_numpy", "to_mxfp4_scales_numpy",
    "metadata_from_mxfp4_scales_numpy",
]
