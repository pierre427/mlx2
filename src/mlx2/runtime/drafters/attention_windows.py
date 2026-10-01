"""Startup-only context windows for draft layers; target cache is untouched.

A window W retains the most recent W committed target-context positions in that
layer's draft KV cache. The entire trained proposal block still self-attends.
These are approximate *draft features*, verified against the ordinary target;
no approximate target or APC target state is produced.
"""


def validate_attention_windows(value, num_layers):
    """CPU-safe validation before any tensor/model allocation."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != num_layers:
        raise ValueError("draft_attention_windows must have one entry per draft layer")
    if any(
        window is not None and (type(window) is not int or window <= 0)
        for window in value
    ):
        raise ValueError(
            "draft_attention_windows entries must be null or positive integers"
        )
    return tuple(value)


def configure_attention_windows(model, windows):
    """Apply validated overrides during construction, before caches exist."""
    model.draft_attention_windows = windows
    if windows is None:
        return
    for layer, window in zip(model.layers, windows):
        layer.self_attn.is_sliding = window is not None
        # The DFlash attention implementation subtracts one for its context
        # bound. Its proposal keys are appended separately, so W means W context.
        layer.self_attn.sliding_window = None if window is None else window + 1


def make_window_caches(model):
    from ..models.cache import KVCache, RotatingKVCache

    windows = getattr(model, "draft_attention_windows", None)
    if windows is None:
        return None
    return [
        KVCache() if window is None else RotatingKVCache(max_size=window, keep=0)
        for window in windows
    ]


def compact_window_caches(cache):
    """Bound post-chunk KV planes while keeping absolute cache offsets.

    DFlash attention consumes an unordered complete context window, and RoPE
    coordinates are already attached to each key. Ring storage order therefore
    does not change attention semantics. Long appends are compacted in temporal
    order before another append or paired cache publication.
    """
    for layer_entries in cache:
        entries = (
            layer_entries.rows if hasattr(layer_entries, "rows") else [layer_entries]
        )
        for entry in entries:
            compact = getattr(entry, "compact_to_window", None)
            if callable(compact):
                compact()
