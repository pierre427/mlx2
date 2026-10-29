# SPDX-License-Identifier: MIT
"""Dependency-complete CED suffixes, including all future local cache rows.

The first W-1 outputs of a shortened block are warmup and never consumed.
All source pools are produced at full depth for every input row. This is an
exact dependency transformation, not bounded approximate SWA replay. Different
GEMM row shapes can change rounding; the route has a separate numerical identity.
"""


def dependency_starts(config, length):
    if type(length) is not int or length < 1 or config.sliding_window < 1:
        raise ValueError("invalid CED prefill geometry")
    last = config.num_hidden_layers - 1
    # Candidate source output is query-specific and shared with later layers.
    barrier = max((*config.kv_source_layer_ids, config.candidate_source_layer_id), default=-1)
    window = config.sliding_window
    return tuple(0 if layer <= barrier else
                 max(0, length - (window + (last - layer) * (window - 1)))
                 for layer in range(config.num_hidden_layers))


def trim_shared(shared, delta):
    if not delta:
        return
    for name in ("indices", "selected_mask", "candidates"):
        value = shared.get(name)
        if value is not None:
            shared[name] = value[:, delta:]
    # These descriptors bind a particular query length/start or gather object.
    shared.pop("_all_retained_chronological", None)
    shared.pop("_decode_selected_kv", None)
