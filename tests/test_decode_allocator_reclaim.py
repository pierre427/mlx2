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
    # BatchKVCache also drops the padding every row shares after each trim.
    shared = min(expected_padding)
    reclaimed = [pad - shared for pad in expected_padding]
    for cache, padding in (
        (BatchKVCache([0] * rows), reclaimed),
        (BatchQuantizedKVCache([0] * rows), expected_padding),
    ):
        _ragged_rounds(cache, rounds, width, dim=64)
        # One pending rebinding from the last trim is expected; one node per
        # round is the leak (mlx-lm#1911: live buffers until malloc refuses).
        assert _lazy_nodes(cache.left_padding) <= 2, type(cache).__name__
        assert _lazy_nodes(cache.offset) <= 2, type(cache).__name__
        assert cache.left_padding.tolist() == padding


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


def _ragged_reference_run(cache, rounds, pattern, *, qsa):
    """Drive ragged verify rounds and check every row against a host oracle.

    Each round appends three tokens per row (every fourth round with right
    padding, as a ragged self-MTP verify of unequal draft depth does) and
    then rewinds the rows by different amounts. The oracle is the list of
    token ids each row must still hold; row ``i`` of the batch cache must
    extract exactly that sequence, keys, values and (for QSA) raw index keys.
    Returns the shared left padding (``min(left_padding)``) after each round.
    """
    rows = cache.left_padding.shape[0]
    initial = cache.left_padding.tolist()
    oracle = [list(range(1000 * row, 1000 * row + 6 - pad)) for row, pad in
              enumerate(initial)]
    shared = []

    def append(tokens, right):
        width = max(len(t) + r for t, r in zip(tokens, right))
        grid = [[0.0] * (width - len(t) - r) + [float(v) for v in t] + [-1.0] * r
                for t, r in zip(tokens, right)]
        keys = mx.broadcast_to(mx.array(grid)[:, None, :, None], (rows, 1, width, 2))
        if qsa:
            cache.update_index_keys(mx.array(grid)[:, :, None] + 0.25)
        cache.update_and_fetch(keys, keys + 0.5)

    append([row[:] for row in oracle], [0] * rows)
    counter = 5000
    for step in range(rounds):
        right = [(row + step) % 3 for row in range(rows)] if step % 4 == 3 else [0] * rows
        fresh = []
        for row in range(rows):
            fresh.append(list(range(counter, counter + 3 - right[row])))
            counter += 3
        if max(right):
            cache.prepare(lengths=[3 - r for r in right], right_padding=right)
        append(fresh, right)
        if max(right):
            cache.finalize()
        for row in range(rows):
            oracle[row].extend(fresh[row])
        drops = pattern(step, [len(t) for t in oracle])
        cache.trim_ragged(drops)
        for row in range(rows):
            del oracle[row][len(oracle[row]) - drops[row]:]
            got = cache.extract(row)
            want = mx.array([float(v) for v in oracle[row]])
            assert got.offset == len(oracle[row])
            assert mx.array_equal(got.keys[0, 0, :, 0], want).item(), (step, row)
            assert mx.array_equal(got.values[0, 0, :, 0], want + 0.5).item()
            if qsa:
                assert mx.array_equal(got.index_keys[0, :, 0], want + 0.25).item()
        shared.append(min(cache.left_padding.tolist()))
    return shared


def test_ragged_trim_reclaims_shared_padding_and_keeps_every_row():
    """X3-4: the physical self-MTP cohort never calls ``filter()``.

    A ragged trim rolled each row right and left the added padding in place,
    so under alternating rejections every row gained padding each cycle and
    the shared width, the attention span and every later roll kept growing
    with columns no row reads. The trim now drops the padding all rows share
    in the same gather; every row's content must still match the oracle.
    """
    import random

    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.qwen4_exp import BatchQSAKVCache

    rng = random.Random(5)
    patterns = {
        "alternating": lambda step, _lengths: [2, 0, 1] if step % 2 else [0, 2, 1],
        "random": lambda _step, lengths: [rng.randint(0, 2) for _ in lengths],
    }
    for qsa in (False, True):
        for name, pattern in patterns.items():
            cls = BatchQSAKVCache if qsa else BatchKVCache
            cache = cls([0, 3, 1])
            shared = _ragged_reference_run(cache, 60, pattern, qsa=qsa)
            # Without the reclaim the shared padding grows by about one
            # column per cycle (about 60 here). With it only the right
            # padding of a verify that a uniform rewind followed is still
            # shared, and the next ragged trim takes it.
            assert max(shared) <= 2, (cls.__name__, name, shared[-5:])
    # A restore to an unknown geometry forgets the host floor: the trim must
    # then reclaim nothing it cannot prove is padding, and stay exact.
    cache = BatchKVCache([0, 3, 1])
    cache._host_padding_floor = None
    _ragged_reference_run(cache, 12, patterns["alternating"], qsa=False)
