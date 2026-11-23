"""Hy4-flash host synchronization: counts, placement and the device Engram hasher (CPU).

``SyncCounter`` records every host materialization a Python caller can make
of an MLX array (``mx.eval``, ``np.array``/``np.asarray`` of an ``mx.array``
in the instrumented modules, ``.tolist()``, ``.item()``) as a *wait*, and
every ``mx.async_eval`` as a *dispatch*, with its call site.  It counts
call sites, not device time: a wait on an array that is already
materialized costs nothing, which is why the placement (what was dispatched
before the first wait) is asserted next to the counts.

The device hasher (``engram_hash: device``, opt-in) must give the host
hasher's rows bit for bit, and every route must give the host route's
tokens (and logits) exactly.
"""

import sys
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import (  # noqa: E402
    LONG,
    TOKENS,
    _drain,
    _greedy,
    build,
    checkpoint,
    tiny_config,
)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models.hy_v4_flash_device_hash import (  # noqa: E402
    MAX_MODULUS,
    DeviceEngramHasher,
)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


@contextmanager
def hash_mode(model, mode):
    saved = model.engram_hash
    model.engram_hash = mode
    try:
        yield
    finally:
        model.engram_hash = saved


# ------------------------------------------------------------ the harness --

class _NumpyProxy:
    def __init__(self, counter):
        self._counter = counter

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, value, *args, **kwargs):
        if isinstance(value, mx.array):
            self._counter._record("wait", "np.array")
        return np.array(value, *args, **kwargs)

    def asarray(self, value, *args, **kwargs):
        if isinstance(value, mx.array):
            self._counter._record("wait", "np.asarray")
        return np.asarray(value, *args, **kwargs)


MODULES = ("mlx2.runtime.models.hy_v4_flash", "mlx2.runtime.generate",
           "mlx2.runtime.external_speculative", "mlx2.runtime.drafters.hy4_dfly",
           "mlx2.adapters.hy_v4_flash_ast")


class SyncCounter:
    """Waits and dispatches in call order: ``events = [(kind, what, site)]``."""

    def __init__(self):
        self.events = []

    def _record(self, kind, what):
        frame = sys._getframe(2)
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        site = "?" if frame is None else f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"
        self.events.append((kind, what, site))

    @property
    def waits(self):
        return [e for e in self.events if e[0] == "wait"]

    @property
    def dispatches(self):
        return [e for e in self.events if e[0] == "dispatch"]

    def clear(self):
        self.events = []


@contextmanager
def count_syncs():
    counter = SyncCounter()
    real_eval, real_async = mx.eval, mx.async_eval
    real_tolist, real_item = mx.array.tolist, mx.array.item

    def eval_(*a, **k):
        counter._record("wait", "mx.eval")
        return real_eval(*a, **k)

    def async_(*a, **k):
        counter._record("dispatch", "mx.async_eval")
        return real_async(*a, **k)

    def tolist(self):
        counter._record("wait", ".tolist")
        return real_tolist(self)

    def item(self):
        counter._record("wait", ".item")
        return real_item(self)

    saved = {}
    for name in MODULES:
        module = sys.modules.get(name) or __import__(name, fromlist=["_"])
        if hasattr(module, "np"):
            saved[name] = module.np
            module.np = _NumpyProxy(counter)
    mx.eval, mx.async_eval = eval_, async_
    mx.array.tolist, mx.array.item = tolist, item
    try:
        yield counter
    finally:
        mx.eval, mx.async_eval = real_eval, real_async
        mx.array.tolist, mx.array.item = real_tolist, real_item
        for name, value in saved.items():
            sys.modules[name].np = value


def _lazy_decode_events(model, mode, prompt, steps=3):
    """Per decode forward: events when the input token is still lazy (the
    previous step's argmax, never materialized), as in GenerationBatch._step."""
    with hash_mode(model, mode):
        cache = model.make_cache()
        token = mx.argmax(model(mx.array([prompt]), cache)[:, -1:], axis=-1)
        per_step, outputs = [], []
        for _ in range(steps):
            with count_syncs() as counter:
                logits = model(token.astype(mx.uint32), cache)
            per_step.append(counter.events)
            token = mx.argmax(logits[:, -1:], axis=-1)
            outputs.append(np.array(logits))
        return per_step, outputs, cache


# ------------------------------------------------------ device hasher: bits --

def _host_hashers(vocab, layer_ids, moduli_range, heads, n, pad, rng, compressed=None):
    out = []
    for lid in layer_ids:
        if compressed is None:
            table = rng.integers(0, vocab, size=vocab)
        else:
            table = rng.integers(0, compressed, size=vocab)
        table[-1] = table.max()
        sizes = [[int(x) for x in rng.integers(*moduli_range, size=heads)] for _ in range(n - 1)]
        out.append(H.EngramHasher(table, sizes, layer_id=lid, seed=0, max_ngram=n,
                                  heads_per_ngram=heads, pad_id=pad))
    return out


def _assert_rows_equal(hashers, tokens):
    device = DeviceEngramHasher(hashers)
    got = np.array(device(mx.array(tokens.astype(np.int32))))
    for i, hasher in enumerate(hashers):
        np.testing.assert_array_equal(got[i], hasher.table_rows(tokens[None])[0])


@pytest.mark.parametrize("n", [2, 3, 4])
def test_device_hasher_matches_host_at_production_geometry(n):
    """Hy4-flash geometry: 120832-token map, 4 layers, 8 heads, ~10.3M primes."""
    rng = np.random.default_rng(n)
    vocab = 120832
    hashers = _host_hashers(vocab, (2, 12, 22, 32), (10_312_000, 10_313_202), 8, n, 120002, rng)
    tokens = rng.integers(0, vocab, size=(2048, n))
    edges = np.array([[vocab - 1] * n, [0] * n, [120002] * n, [vocab] * n, [vocab + 7] * n,
                      [2**31 - 1] * n, [vocab - 1, 0] + [vocab] * (n - 2)])
    _assert_rows_equal(hashers, np.concatenate([tokens, edges]))


def test_device_hasher_matches_host_at_the_modulus_and_product_bounds():
    rng = np.random.default_rng(7)
    # The largest admitted modulus, and a dense map so the multiplier bound
    # (compressed * m just below 2**63) is reached by real products.
    hashers = _host_hashers(4096, (1, 2), (MAX_MODULUS - 64, MAX_MODULUS + 1), 4, 3, 5, rng)
    for h in hashers:
        assert int(h.table.max()) * int(h.multipliers.max()) < 2**63
    tokens = rng.integers(0, 4100, size=(4096, 3))
    _assert_rows_equal(hashers, tokens)


def test_device_hasher_matches_tiny_fixture(tiny):
    _, _, model, _ = tiny
    hashers = [model.engram_hashers[i] for i in sorted(model.engram_hashers)]
    rng = np.random.default_rng(0)
    _assert_rows_equal(hashers, rng.integers(0, 120, size=(512, 3)))


def test_device_hasher_refuses_what_it_cannot_compute_exactly():
    rng = np.random.default_rng(1)
    big = _host_hashers(64, (1,), (MAX_MODULUS + 1, MAX_MODULUS + 2), 2, 3, 0, rng)
    with pytest.raises(ValueError, match="moduli"):
        DeviceEngramHasher(big)
    a = _host_hashers(64, (1,), (11, 12), 2, 3, 0, rng)
    b = _host_hashers(64, (2,), (11, 12), 3, 3, 0, rng)
    with pytest.raises(ValueError, match="geometry"):
        DeviceEngramHasher(a + b)


# ----------------------------------------------------- model: exact routes --

def test_device_ngrams_equal_host_ngrams_across_chunk_boundaries(tiny):
    """Prompt start (pad history), a tail shorter than the n-gram, chunk edges."""
    _, _, model, _ = tiny
    for cuts in ([1, 2, 3, 9], [5, 6, 12], [11, 12, 13, 24]):
        cache = model.make_cache()
        previous = 0
        for cut in cuts:
            chunk = np.array([TOKENS[previous:cut]], np.int64)
            first = cache[min(model.engram_hashers)]
            tail_rows = first.token_tail
            tail = np.array(tail_rows, np.int64) if tail_rows and len(tail_rows[0]) else \
                np.zeros((1, 0), np.int64)
            host = model._ngram_tokens(chunk, cache[0].offset, tail)
            device = model._device_ngrams(mx.array(chunk.astype(np.int32)), cache, cache[0].offset)
            np.testing.assert_array_equal(np.array(device), host.reshape(-1, host.shape[-1]))
            model(mx.array(chunk), cache)
            previous = cut


def test_device_route_logits_equal_host_route_bit_for_bit(tiny):
    _, _, model, _ = tiny
    runs = {}
    for mode in ("host", "device"):
        with hash_mode(model, mode):
            cache = model.make_cache()
            parts = [model(mx.array([TOKENS[:9]]), cache), model(mx.array([TOKENS[9:15]]), cache)]
            parts += [model(mx.array([[t]]), cache) for t in TOKENS[15:]]
            runs[mode] = (np.array(mx.concatenate(parts, axis=1)),
                          [c.token_tail for c in cache], [c.offset for c in cache])
    np.testing.assert_array_equal(runs["device"][0], runs["host"][0])
    assert runs["device"][1:] == runs["host"][1:]
    assert model.engram_hash_stats["device_forwards"] > 0


def test_device_route_long_decode_and_rollback_are_exact(tiny):
    _, _, model, _ = tiny
    prompt = LONG[:30]
    results = {}
    for mode in ("host", "device"):
        with hash_mode(model, mode):
            greedy = _greedy(model, prompt, 20)
            cache = model.make_cache()
            model(mx.array([prompt]), cache)
            model(mx.array([[40, 41, 42, 43, 44]]), cache)  # a rejected block
            for c in cache:
                c.trim(5)
            after = np.array(model(mx.array([LONG[30:33]]), cache))
            results[mode] = (greedy, after)
    assert results["device"][0] == results["host"][0]
    np.testing.assert_array_equal(results["device"][1], results["host"][1])


def _cache_state(cache):
    """Every non-array field, the buffers' live views (rows past a fill
    level are never read) and the recorded selections."""
    rows = []
    for c in cache:
        fields = {k: v for k, v in c.__dict__.items() if not isinstance(v, (mx.array, dict))}
        arrays = {p: np.array(v) for p, v in zip(c._buffers(), c.state) if v is not None}
        selections = {p: None if v is None else np.array(v) for p, v in c.selections.items()}
        rows.append((fields, arrays, selections, c.meta_state))
    return rows


def _assert_same_state(a, b):
    assert len(a) == len(b)
    for (fa, aa, sa, ma), (fb, ab, sb, mb) in zip(a, b):
        assert fa == fb and ma == mb and set(aa) == set(ab) and set(sa) == set(sb)
        for k in aa:
            np.testing.assert_array_equal(aa[k], ab[k])
        for k in sa:
            assert (sa[k] is None) == (sb[k] is None)
            if sa[k] is not None:
                np.testing.assert_array_equal(sa[k], sb[k])


@pytest.mark.parametrize("chunk", [1, 64])
def test_device_route_refused_token_restores_the_cache_exactly(tiny, chunk):
    """Index Share recording on, past the retained-selection window, a
    64-token chunk (the largest device forward): every field, buffer and
    recorded selection is as before, and decode continues exactly."""
    config, W, _, _ = tiny
    model, _ = build(tiny_config(image_token_id=96), W)  # 96 is not in LONG
    model.record_layers = frozenset({0, 4, 5})
    model.engram_hash = "device"
    prompt = (LONG * 3)[:90]
    cache = model.make_cache()
    model(mx.array([prompt[:60]]), cache)
    model(mx.array([prompt[60:90]]), cache)
    before = _cache_state(cache)
    refused = [44] * (chunk - 1) + [96]
    with pytest.raises(ValueError, match="pixel_values"):
        model(mx.array([refused]), cache)
    _assert_same_state(_cache_state(cache), before)
    after = np.array(model(mx.array([[44]]), cache))
    model.engram_hash = "host"
    ref = model.make_cache()
    model(mx.array([prompt[:60]]), ref)
    model(mx.array([prompt[60:90]]), ref)
    np.testing.assert_array_equal(after, np.array(model(mx.array([[44]]), ref)))


def test_device_route_reader_failure_restores_the_prefix_caches(tiny):
    _, _, model, _ = tiny

    class Broken:
        rows = None

        def prefetch(self, rows, **_kw):
            raise RuntimeError("reader closed")

    with hash_mode(model, "device"):
        cache = model.make_cache()
        model(mx.array([TOKENS[:12]]), cache)
        before = _cache_state(cache)
        saved = dict(model.engram_rows)
        model.engram_rows = {i: Broken() for i in saved}
        try:
            with pytest.raises(RuntimeError, match="reader closed"):
                model(mx.array([[44]]), cache)
        finally:
            model.engram_rows = saved
        _assert_same_state(_cache_state(cache), before)


def test_device_route_ragged_lanes_with_empty_and_offset_spans_equal_host(tiny):
    """A lane with no tokens in the chunk (count 0) and a left-padded lane
    (start > 0), as the AST joint waves and ragged prefill produce them."""
    _, _, model, _ = tiny
    layouts = [dict(lengths=[2, 0], right_padding=[0, 2]),
               dict(lengths=[1, 3], right_padding=[2, 0]),
               dict(lengths=[2, 1], left_padding=[1, 2])]
    out = {}
    for mode in ("host", "device"):
        with hash_mode(model, mode):
            a, b = model.make_cache(), model.make_cache()
            model(mx.array([TOKENS[:9]]), a)
            model(mx.array([TOKENS[2:14]]), b)
            merged = [type(x).merge([x, y]) for x, y in zip(a, b)]
            logits = []
            for step, prep in enumerate(layouts):
                for c in merged:
                    c.prepare(**prep)
                width = 3 if step else 2
                tokens = [[40 + step + j for j in range(width)], [50 + step + j for j in range(width)]]
                logits.append(np.array(model(mx.array(tokens), merged)))
                for c in merged:
                    c.finalize()
            out[mode] = (logits, [_cache_state(list(c.rows)) for c in merged])
    for x, y in zip(out["device"][0], out["host"][0]):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(out["device"][1], out["host"][1]):
        _assert_same_state(x, y)


def test_device_route_after_a_media_prefill_equals_host():
    """Text decode after image spans: n-gram document positions come from
    the spans (``slot_positions``), not the slots."""
    import test_hy_v4_flash_vision_cpu as VT

    config = VT.vision_config()
    model, _ = build(config, checkpoint(config))
    model.attach_vision(VT.V.load_vision_tower(config, {k: mx.array(v) for k, v in VT.vit_checkpoint().items()}))
    prompt = VT.image_prompt()
    pixels = mx.array(VT.pixels_for(VT.GRIDS))
    out = {}
    for mode in ("host", "device"):
        model.engram_hash = mode
        cache = model.make_cache()
        parts = [model(mx.array([prompt[:-1]]), cache, pixel_values=pixels, image_grid_thw=VT.GRIDS)]
        parts += [model(mx.array([[t]]), cache) for t in [prompt[-1], 30, 31, 33]]
        parts.append(model(mx.array([[35, 37, 39]]), cache))
        out[mode] = (np.array(mx.concatenate(parts, axis=1)), _cache_state(cache))
        assert cache[min(model.engram_hashers)].mrope_spans
    assert model.engram_hash_stats["device_forwards"] >= 5
    np.testing.assert_array_equal(out["device"][0], out["host"][0])
    _assert_same_state(out["device"][1], out["host"][1])


def test_device_hasher_geometry_is_checked_at_bind(tiny):
    config, W, _, _ = tiny
    model, _ = build(config, W)
    tables = {i: model.engram_rows[i] for i in model.engram_rows}
    model.engram_hash = "device"
    model.bind_engram(tables)
    assert model._device_hasher is not None  # built (and validated) at bind, not at decode


def test_device_route_ragged_batch_generator_equals_host(tiny):
    from mlx2.runtime.generate import BatchGenerator

    _, _, model, _ = tiny
    prompts = [TOKENS[:10], TOKENS[:17], TOKENS[3:7]]
    out = {}
    for mode in ("host", "device"):
        with hash_mode(model, mode):
            before = model.engram_hash_stats["device_forwards"]
            batch = BatchGenerator(model, max_tokens=8, completion_batch_size=3,
                                   prefill_batch_size=2, prefill_step_size=4)
            uids = batch.insert(prompts, max_tokens=[8] * 3)
            drained = _drain(batch, uids, 8)
            out[mode] = [drained[u][:8] for u in uids]
            if mode == "device":
                assert model.engram_hash_stats["device_forwards"] > before
    assert out["device"] == out["host"]
    for tokens, prompt in zip(out["device"], prompts):
        assert tokens == _greedy(model, prompt, 8)


@pytest.mark.parametrize("tree", [False, True])
def test_device_route_dfly2_engine_equals_host(tiny, tree):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    _, _, model, head = tiny
    model.record_layers = frozenset(head.index_layers)
    out = {}
    try:
        for mode in ("host", "device"):
            with hash_mode(model, mode):
                draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
                options = dict(dynamic_singleton_tree=True, dynamic_tree_max_width=1,
                               target_tree_executor_provider=Hy4ChainTreeExecutor.create(),
                               tree_node_budget_by_lanes={1: 7}) if tree else {}
                batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                                    num_draft=3, prefill_step_size=4, **options)
                uids = batch.insert([TOKENS[:13]], max_tokens=[16],
                                    sampling_configs=[{"sampling_temp": 0}])
                out[mode] = _drain(batch, uids, 16)[uids[0]][:16]
                assert draft.stats["block_drafts"] > 0
    finally:
        model.record_layers = frozenset()
    assert out["device"] == out["host"] == _greedy(model, TOKENS[:13], 16)


def test_device_route_ast_multilane_cohort_equals_host(tmp_path, tiny):
    from test_hy_v4_flash_ast_multilane_cpu import PROMPTS, _adapter

    out = {}
    for mode in ("host", "device"):
        (tmp_path / mode).mkdir()
        adapter = _adapter(tmp_path / mode, tiny, {"draft_model": "embedded_dfly", "num_draft": 3,
                                                   "engram_hash": mode})
        try:
            assert adapter.model.engram_hash == mode
            batch = adapter.create_external_batch(completion_batch_size=3, prefill_step_size=4)
            uids = batch.insert(PROMPTS[:3], max_tokens=[12] * 3,
                                sampling_configs=[{"sampling_temp": 0}] * 3)
            drained = _drain(batch, uids, 12)
            out[mode] = [drained[u][:12] for u in uids]
            assert batch.scheduler_stats["external_target_tree_packed_target_rounds"] > 0
            receipt = adapter.diagnostics()["engram_hash"]
            assert receipt["mode"] == mode and receipt["qualified"] is False
            assert receipt["observed_used"] is (mode == "device")
            for tokens, prompt in zip(out[mode], PROMPTS[:3]):
                assert tokens == _greedy(adapter.model, prompt, 12)
        finally:
            adapter.close()
    assert out["device"] == out["host"]


def test_engram_hash_policy_key_fails_closed():
    from mlx2.adapters import hy_v4_flash as A

    assert A._parse_policy({"engram_hash": "device"})[0] == {"engram_hash": "device"}
    with pytest.raises(ValueError, match="engram_hash"):
        A._parse_policy({"engram_hash": "gpu"})


# --------------------------------------------------- sync counts / placement --

def test_decode_forward_waits_once_and_device_route_dispatches_before_it(tiny):
    """One wait per decode forward on both routes; the device route moves it
    behind the hash and the pre-Engram layers instead of before all work."""
    _, _, model, _ = tiny
    host, host_out, host_cache = _lazy_decode_events(model, "host", TOKENS[:12])
    device, device_out, device_cache = _lazy_decode_events(model, "device", TOKENS[:12])
    for a, b in zip(host_out, device_out):
        np.testing.assert_array_equal(a, b)
    engram_layers = len(model.engram_hashers)
    for events in host:
        waits = [e for e in events if e[0] == "wait"]
        assert [(w[1], w[2].split(":")[0]) for w in waits] == [("np.array", "hy_v4_flash.py")]
        assert events[0][0] == "wait"  # nothing dispatched before the token wait
        # One dispatch after the first Engram layer, one before each later one.
        assert sum(e[0] == "dispatch" for e in events) == 1 + (engram_layers - 1)
    for events in device:
        waits = [e for e in events if e[0] == "wait"]
        assert [(w[1], w[2].split(":")[0]) for w in waits] == [("np.array", "hy_v4_flash.py")]
        first_wait = events.index(waits[0])
        # The hash and the layers before the first Engram layer are queued
        # before the host waits for the token.
        assert [e[0] for e in events[:first_wait]] == ["dispatch", "dispatch"]
        assert sum(e[0] == "dispatch" for e in events) == 2 + 1 + (engram_layers - 1)


def test_batch_generator_decode_step_sync_count(tiny):
    """GenerationBatch._step on Hy4: the model's token wait plus the generator's
    eval/tolist of the previous step's tokens, on both routes."""
    from mlx2.runtime.generate import BatchGenerator

    _, _, model, _ = tiny
    counts = {}
    for mode in ("host", "device"):
        with hash_mode(model, mode):
            batch = BatchGenerator(model, max_tokens=12, completion_batch_size=2,
                                   prefill_batch_size=2)
            uids = batch.insert([TOKENS[:10], TOKENS[3:9]], max_tokens=[12, 12])
            for _ in range(4):  # prefill and the first decode steps
                batch.next()
            with count_syncs() as counter:
                for _ in range(4):
                    batch.next()
            sites = {}
            for kind, what, site in counter.waits:
                sites[(what, site.split(":")[0])] = sites.get((what, site.split(":")[0]), 0) + 1
            counts[mode] = sites
            del uids
    for mode, sites in counts.items():
        assert sites.get(("np.array", "hy_v4_flash.py")) == 4, (mode, sites)
        model_waits = sum(v for (what, f), v in sites.items() if f.startswith("hy_v4_flash"))
        assert model_waits == 4, (mode, sites)
