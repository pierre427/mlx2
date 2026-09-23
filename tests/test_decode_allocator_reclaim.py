"""Every decode route reclaims the MLX buffer pool periodically.

Ollama v0.34.2: freed KV/intermediate buffers pile up in the MLX pool during
speculative decode unless something clears it. The prompt-lookup route had no
clear in prefill or decode.
"""

import mlx.core as mx

from mlx2.runtime import generate as G
from mlx2.runtime import pld
from mlx2.runtime.sample_utils import LaneRNG


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 16,
    )
    mx.random.seed(7)
    m = TextModel(args)
    m.eval()
    mx.eval(m.parameters())
    return m


def _count_clears(monkeypatch):
    calls = {"n": 0}
    real = mx.clear_cache

    def counted():
        calls["n"] += 1
        real()

    monkeypatch.setattr(mx, "clear_cache", counted)
    return calls


def _drain(gen, n):
    emitted = 0
    while emitted < n:
        _p, rs = gen.next()
        emitted += len(rs)
        if any(r.finish_reason for r in rs):
            break
    return emitted


def test_self_mtp_decode_invokes_periodic_reclaim(monkeypatch):
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 16)
    model = tiny_model()
    gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=64,
                           self_mtp={"num_draft": 2, "persistent": True})
    try:
        gen.insert([list(range(2, 100))], max_tokens=[80], lane_rngs=[LaneRNG(1)],
                   self_mtp_configs=[{"sampling_temp": 0.0}])
        _p, rs = gen.next()  # prefill (clears per chunk)
        calls = _count_clears(monkeypatch)
        _drain(gen, 79)
        steps = gen._steps_counter
    finally:
        gen.close()
    assert calls["n"] >= steps // 16 >= 1


def test_prompt_lookup_prefill_and_decode_reclaim(monkeypatch):
    monkeypatch.setattr(pld, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 16)
    model = tiny_model()
    gen = pld.PromptLookupBatchGenerator(
        model, completion_batch_size=1, prefill_step_size=64,
        prompt_lookup={}, stop_tokens=[],
    )
    try:
        gen.insert([[(i % 17) + 2 for i in range(200)]], max_tokens=[120])
        calls = _count_clears(monkeypatch)
        while gen.next()[0]:
            pass  # prefill rounds
        assert calls["n"] >= 3, "PLD prefill never reclaims per chunk"
        calls["n"] = 0
        _drain(gen, 119)
        steps = gen._steps_counter
    finally:
        gen.close()
    assert steps >= 16
    assert calls["n"] >= steps // 16


def test_self_mtp_reclaim_follows_emitted_tokens(monkeypatch):
    # Steps alone would never reach the interval here; tokens do.
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 1 << 40)
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL", 16)
    model = tiny_model()
    gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=64,
                           self_mtp={"num_draft": 2, "persistent": True})
    try:
        gen.insert([list(range(2, 100))], max_tokens=[80], lane_rngs=[LaneRNG(1)],
                   self_mtp_configs=[{"sampling_temp": 0.0}])
        gen.next()  # prefill
        calls = _count_clears(monkeypatch)
        emitted = _drain(gen, 79)
    finally:
        gen.close()
    assert emitted >= 32
    assert calls["n"] >= emitted // 16 - 1


def _lazy_nodes(array):
    """Unevaluated primitive nodes hanging off ``array`` (0 when evaluated)."""
    import io
    import re

    buffer = io.StringIO()
    mx.export_to_dot(buffer, array)
    return len(re.findall(r"shape=rectangle", buffer.getvalue()))


def _ragged_rounds(cache, rounds, width, dim):
    """Append ``width`` tokens, evaluate the returned K/V, then trim raggedly.

    This is what a verify round does on a full-attention layer whose mask the
    forward never builds: its K/V are consumed, its ``left_padding`` is not.
    """
    rows = cache.left_padding.shape[0]
    for step in range(rounds):
        keys = mx.full((rows, 1, width, dim), float(step))
        mx.eval(cache.update_and_fetch(keys, keys + 1))
        cache.trim_ragged([(row + step) % width for row in range(rows)])


def test_ragged_verify_rounds_do_not_chain_row_metadata():
    from mlx2.runtime.models.cache import BatchKVCache, BatchQuantizedKVCache

    rounds, width, rows = 200, 3, 3
    expected_padding = [0] * rows
    for step in range(rounds):
        drops = [(row + step) % width for row in range(rows)]
        uniform = min(drops)
        expected_padding = [
            pad + drop - uniform for (pad, drop) in zip(expected_padding, drops)
        ]
    for cache in (BatchKVCache([0] * rows), BatchQuantizedKVCache([0] * rows)):
        _ragged_rounds(cache, rounds, width, dim=64)
        # One pending rebinding from the last trim is expected; one node per
        # round is the leak (mlx-lm#1911: live buffers until malloc refuses).
        assert _lazy_nodes(cache.left_padding) <= 2, type(cache).__name__
        assert _lazy_nodes(cache.offset) <= 2, type(cache).__name__
        assert cache.left_padding.tolist() == expected_padding


def _self_mtp_left_padding_run(monkeypatch, rounds, tie):
    from mlx2.runtime.models import cache as C

    if not tie:
        monkeypatch.setattr(
            C.BatchKVCache, "_tie_row_metadata", lambda self: None, raising=False
        )
    model = tiny_model()
    lanes = 3
    gen = G.BatchGenerator(model, completion_batch_size=lanes,
                           prefill_batch_size=lanes, prefill_step_size=64,
                           self_mtp={"num_draft": 2, "persistent": True})
    tokens = [[] for _ in range(lanes)]
    try:
        gen.insert(
            [[(i * (k + 3)) % 97 + 2 for i in range(20 + 13 * k)] for k in range(lanes)],
            max_tokens=[rounds * 4 + 50] * lanes,
            lane_rngs=[LaneRNG(11 + k) for k in range(lanes)],
            # Sampling makes lanes accept different draft counts, so every
            # verify round takes the ragged trim.
            self_mtp_configs=[{"sampling_temp": 1.0}] * lanes,
        )
        for _ in range(rounds):
            _p, responses = gen.next()
            for response in responses:
                tokens[response.uid].append(response.token)
        caches = gen._generation_batch.state.caches.target
        chains = [
            _lazy_nodes(c.left_padding)
            for c in caches
            if isinstance(c, C.BatchKVCache)
        ]
    finally:
        gen.close()
    monkeypatch.undo()
    return (tokens, chains)


def test_batched_self_mtp_keeps_unmasked_layer_metadata_evaluated(monkeypatch):
    """Only one full-attention layer's mask reads ``left_padding``.

    The tiny model has two full-attention layers; the one the forward does not
    build a mask for used to gain a lazy node and a live buffer per verify
    round until a membership change happened to evaluate it.
    """
    (leaky_tokens, leaky_chains) = _self_mtp_left_padding_run(
        monkeypatch, 120, tie=False
    )
    (tokens, chains) = _self_mtp_left_padding_run(monkeypatch, 120, tie=True)
    assert len(leaky_chains) == 2
    # The falsifier: without the tie the unmasked layer does chain.
    assert max(leaky_chains) > 50
    assert max(chains) <= 2
    assert tokens == leaky_tokens
    assert sum(map(len, tokens)) > 120
