"""Adapter-supplied native state geometry for request lifecycle checks."""

def native_layer_count(candidate):
    count = getattr(candidate, "native_layer_count", None)
    if count is None:
        count = len(candidate.model.layers)
    if type(count) is not int or count < 1:
        raise ValueError("native candidate requires a positive physical layer count")
    return count


def supports_native_checkpoint_candidate(candidate):
    """Adapter capability for bounded recurrent checkpoints and native KV.

    Tensor math remains in the adapter. The lifecycle checks complete declared
    contracts; it does not infer a capability from a model identifier.
    """
    return bool(
        getattr(candidate, 'state_planes', None) == ('kv', 'gdn') and
        getattr(candidate, 'bootstrap_generation', None) == 0 and
        getattr(candidate, 'supports_singleton', False) is True and
        getattr(candidate, 'owns_physical_dispatch_proof', False) is True and
        type(getattr(candidate, 'native_layer_count', None)) is int and
        candidate.native_layer_count > 0 and
        type(getattr(candidate, 'logical_layer_count', None)) is int and
        candidate.logical_layer_count > candidate.native_layer_count and
        callable(getattr(candidate, 'packed_lane', None)) and
        callable(getattr(candidate, 'forward_staged', None)) and
        getattr(candidate, 'backend', None) is not None and
        getattr(candidate, 'model', None) is not None)
