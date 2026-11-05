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
    barrier = dependency_barrier(config)
    window = config.sliding_window
    return tuple(0 if layer <= barrier else
                 max(0, length - (window + (last - layer) * (window - 1)))
                 for layer in range(config.num_hidden_layers))


def dependency_barrier(config):
    """Last layer that must run every prompt row (sources and candidates)."""
    return max((*config.kv_source_layer_ids, config.candidate_source_layer_id), default=-1)


def state_only_admitted(config, rows_after, step):
    """Whether a chunk may skip every layer above the barrier.

    Ported from Sushi's state-only chunks (dsv41_model.extendState), adapted
    to this dependency-complete profile. The final logits read the input of
    decoder layer L only on the last ``window + (last - L)(window - 1)`` prompt
    rows; the widest such cone is the first decoder layer's. When at least that
    many rows of the same call follow, no row of this chunk is inside any
    cone. A later chunk that starts a decoder layer at zero reads this chunk's
    (now empty) local keys only for warmup outputs that nothing consumes, and
    refills a full window only if it is at least ``window`` rows wide.
    """
    last = config.num_hidden_layers - 1
    barrier = dependency_barrier(config)
    window = config.sliding_window
    if barrier >= last or window < 1 or any(layer > barrier for layer in config.engram_layer_ids):
        return False
    cone = window + (last - (barrier + 1)) * (window - 1)
    return int(rows_after) >= cone and int(step) >= window


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
