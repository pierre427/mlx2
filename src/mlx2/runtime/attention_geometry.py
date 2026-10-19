"""Adapter-declared KV attention geometry for cost planning.

A model declares which of its layers keep a KV cache and how wide, so the
AST cost prior never has to recognise attention modules by attribute names.
The result prices context work only: bytes of key and value read per query
token per context token. Recurrent and linear-attention layers contribute
nothing; a sliding layer contributes under its window.
"""

from __future__ import annotations


def declared_attention_geometry(*, full_layers, kv_heads, head_dim, element_bytes=2,
                                sliding_layers=0, sliding_window=None):
    """``{"kv_bytes_per_token", "windows"}`` from counted layers and head geometry."""
    for name, value in (("full_layers", full_layers), ("kv_heads", kv_heads), ("head_dim", head_dim),
                        ("element_bytes", element_bytes), ("sliding_layers", sliding_layers)):
        if type(value) is not int or value < 0:
            raise ValueError(f"attention geometry {name} must be a nonnegative integer")
    per_layer = 2 * kv_heads * head_dim * element_bytes
    windows = {}
    if sliding_layers:
        if type(sliding_window) is not int or sliding_window < 1:
            raise ValueError("sliding attention layers need a positive window")
        windows[sliding_window] = sliding_layers * per_layer
    return {"kv_bytes_per_token": full_layers * per_layer, "windows": windows,
            "full_attention_layers": full_layers, "sliding_attention_layers": sliding_layers,
            "kv_heads": kv_heads, "head_dim": head_dim, "element_bytes": element_bytes}


def geometry_from_layer_types(layer_types, *, kv_heads, head_dim, sliding_window=None,
                              element_bytes=2):
    """Count ``full_attention`` and ``sliding_attention`` entries of a layer-type list."""
    kinds = [str(kind) for kind in layer_types]
    sliding = sum(kind == "sliding_attention" for kind in kinds)
    full = sum("attention" in kind and "linear" not in kind and kind != "sliding_attention" for kind in kinds)
    return declared_attention_geometry(full_layers=full, kv_heads=kv_heads, head_dim=head_dim,
                                       element_bytes=element_bytes, sliding_layers=sliding,
                                       sliding_window=sliding_window if sliding else None)
