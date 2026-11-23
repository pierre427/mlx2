"""Inkling 2 attention I/O changes that must stay bit-identical, CPU only.

- ``inkling2_sink_slot`` spare slot (``language.SINK_SLOT_SPARE``): an append
  that leaves a plain ``KVCache`` buffer exactly full grows it by one cache
  step instead of concatenating the zero sink row onto the whole K/V.
- split-K ring read (``splitk.RING_DIRECT``): sliding rings reach the split-K
  kernel as their physical buffers plus a start row, not as a reordered
  copy. On the CPU the kernel declines (the copy is made then, counted) or a
  stand-in that reads the ring through ``splitk.ring_gather`` checks the
  wiring.
- split-K direct-kernel vector loads (``splitk.VECTOR_LOADS``): only the
  template selection is checkable on the CPU; the kernels are Metal tests.

Every arm is compared bitwise with the reference route (the option or the
switch off) on the tiny Inkling 2 config (window 8, so the ring wraps): B1
and B2 prefill plus decode, DSpark verify rows, and an APCv2 store/restore.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache
from mlx2.runtime.models.inkling2 import language, splitk
from test_inkling2_cpu import _tiny_model


@pytest.fixture(autouse=True)
def _io_switches_on(monkeypatch):
    """The ring-direct and vector-load switches ship off until their Metal
    tests run; these tests exercise them (and set them off where needed)."""
    monkeypatch.setattr(splitk, "RING_DIRECT", True)
    monkeypatch.setattr(splitk, "VECTOR_LOADS", True)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _run(model, tokens, plan, cache=None):
    cache = model.make_cache() if cache is None else cache
    outs = [model(tokens[:, a:b], cache=cache) for a, b in plan]
    return mx.concatenate(outs, axis=1), cache


def _receipt_after(model, tokens, plan, **options):
    model.configure_kernels(**options)
    try:
        got, _ = _run(model, tokens, plan)
        return got, model.kernel_receipt()
    finally:
        model.configure_kernels()


# ----- sink slot: a spare slot instead of the full-buffer concatenation -----


@pytest.mark.parametrize("rows", [1, 2])
def test_sink_slot_spare_at_exact_fill_is_bitwise(rows):
    """Two 256-token chunks fill the global buffers exactly (the shape of a
    512-token serving chunk); the spare step replaces the concatenation, so
    only the rings (``cache_type``) fall back. With the switch off the old
    ``full`` fallback is back and the logits are the same bits."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (rows, 516))
    plan = [(0, 256), (256, 512)] + [(i, i + 1) for i in range(512, 516)]
    base, _ = _run(model, tokens, plan)
    got, receipt = _receipt_after(model, tokens, plan, sink_slot=True)
    assert mx.array_equal(got, base)
    assert receipt["sink_slot_calls"] > 0
    # A chunk of exactly one cache step fills the buffer each time (the
    # cache does not grow for the second chunk): one grow per global visit
    # (8 of the 48) per chunk, each replacing one concatenation.
    assert receipt["sink_slot_grows"] == 16
    assert set(receipt["sink_slot_fallback_reasons"]) == {"cache_type"}
    old = language.SINK_SLOT_SPARE
    language.SINK_SLOT_SPARE = False
    try:
        # Counters are cumulative since load: a fresh copy of the model.
        off, off_receipt = _receipt_after(_tiny_model(), tokens, plan, sink_slot=True)
    finally:
        language.SINK_SLOT_SPARE = old
    assert mx.array_equal(off, base)
    assert off_receipt["sink_slot_grows"] == 0
    assert off_receipt["sink_slot_fallback_reasons"]["full"] > 0


def test_sink_slot_spare_survives_cache_growth(monkeypatch):
    """Chunks of two cache steps (512-token chunks over the 256 step): only
    the first exact fill grows; the cache's own growth appends to the full
    buffer, so the spare step persists and later chunks find their slot."""
    monkeypatch.setattr(KVCache, "step", 4)
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 34))
    plan = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 33), (33, 34)]
    base, _ = _run(model, tokens, plan)
    got, receipt = _receipt_after(model, tokens, plan, sink_slot=True)
    assert mx.array_equal(got, base)
    assert receipt["sink_slot_grows"] == 8
    assert set(receipt["sink_slot_fallback_reasons"]) == {"cache_type"}


@pytest.mark.parametrize("rows", [1, 2])
def test_sink_slot_spare_small_step_chunks_decode_and_trim(rows, monkeypatch):
    """A 4-token cache step makes exact fills frequent: chunks of 4, 8, 1 and
    3 tokens, decode across several growths, and a speculative trim that
    leaves stale keys past offset. Bitwise against the reference with the
    same step, no ``full`` fallback."""
    monkeypatch.setattr(KVCache, "step", 4)
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (rows, 40))

    def run():
        plan = [(0, 4), (4, 12), (12, 13), (13, 16), (16, 24)]
        out, cache = _run(model, tokens, plan + [(i, i + 1) for i in range(24, 33)])
        for plane in cache:
            if type(plane) is KVCache:
                plane.trim(3)
        rest, _ = _run(model, tokens, [(30, 31), (31, 36), (36, 40)], cache=cache)
        return mx.concatenate([out, rest], axis=1)

    base = run()
    model.configure_kernels(sink_slot=True)
    try:
        got = run()
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["sink_slot_grows"] > 0
    assert set(receipt["sink_slot_fallback_reasons"]) == {"cache_type"}


def test_sink_slot_spare_dspark_verify_rows(monkeypatch):
    """DSpark verify rows (L = 4) and their rollbacks with an 8-token cache
    step, so the 16-token prefill chunks fill the buffers exactly: greedy
    output equals ordinary decode, and no visit falls back as ``full``."""
    import test_inkling2_dspark_cpu as d

    monkeypatch.setattr(KVCache, "step", 8)
    target, draft = _dspark_pair()
    prompt = [int(t) for t in mx.random.randint(1, 400, (48,)).tolist()]
    expect = d._ordinary_greedy(target, prompt)
    target.configure_kernels(sink_slot=True)
    try:
        got = d._dspark_greedy(target, draft, prompt)
        receipt = target.kernel_receipt()
    finally:
        target.configure_kernels()
    assert got == expect
    assert receipt["sink_slot_grows"] > 0
    assert "full" not in receipt["sink_slot_fallback_reasons"]


def _dspark_pair():
    import test_inkling2_dspark_cpu as d
    from mlx2.runtime.drafters.inkling2_dspark import (
        Inkling2DSparkConfig,
        Inkling2DSparkDraftModel,
    )
    from mlx2.runtime.models.inkling2 import Model, TextConfig

    target = Model(TextConfig.from_dict(d.TEXT))
    d._random(target, 0)
    draft = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf(d._config()))
    d._random(draft, 1)
    draft.bind(target)
    return target, draft


def _apc_restore_logits(model, tokens, stored, plan):
    """Prefill ``tokens[:stored]``, store it in APCv2, restore it by lookup
    and run ``plan`` (positions past ``stored``) on the restored cache."""
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    _, cache = _run(model, tokens, [(0, stored // 2), (stored // 2, stored)])
    mx.eval([c.state for c in cache if c.state is not None])
    apc = APCv2(max_size=4, layout_name=model.apc_v2_layout)
    ids = [int(t) for t in tokens[0].tolist()]
    apc.store(APCKey("inkling2"), ids[:stored], cache)
    hit = apc.lookup(APCKey("inkling2"), ids)
    assert hit.hit and hit.cached_tokens == stored, hit.miss_reason
    out, _ = _run(model, tokens, plan, cache=hit.cache)
    return out


def test_sink_slot_spare_apcv2_restore(monkeypatch):
    """An APCv2 hit restores the cache (its spare step included) and the
    suffix prefill and decode on it match the reference route bitwise."""
    monkeypatch.setattr(KVCache, "step", 8)
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 44))
    plan = [(32, 40)] + [(i, i + 1) for i in range(40, 44)]
    base = _apc_restore_logits(model, tokens, 32, plan)
    model.configure_kernels(sink_slot=True)
    try:
        got = _apc_restore_logits(model, tokens, 32, plan)
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["sink_slot_grows"] > 0
    assert set(receipt["sink_slot_fallback_reasons"]) == {"cache_type"}


# ----- split-K ring read: the physical ring plus a start row -----


def _chronological_copy(kv):
    """What ``Attention._chronological`` returns, without mutating ``kv``."""
    if type(kv) is RotatingKVCache:
        return kv._temporal_order(kv.keys)
    if kv.rotated:
        return mx.roll(kv.keys, -kv._idx, axis=2)
    return kv.keys[..., : kv._idx, :]


@pytest.mark.parametrize("kind", ["ring", "batch_ring"])
def test_ring_layout_reads_the_chronological_order(kind):
    """Through prefill chunks, one-token updates past the wrap, multi-row
    verify updates and window crops: ``ring_gather`` of the physical buffer
    at ``_ring_layout`` is the chronological copy, bitwise."""
    from mlx2.runtime.models.inkling2.language import Attention

    width, b = 8, (1 if kind == "ring" else 2)
    kv = RotatingKVCache(width) if kind == "ring" else BatchRotatingKVCache(width, [0, 2])
    mx.random.seed(3)
    seen = set()
    for n in (5, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1):
        k = mx.random.normal((b, 2, n, 4))
        kv.update_and_fetch(k, k + 1)
        rot, size = Attention._ring_layout(kv)
        seen.add(rot)
        want = _chronological_copy(kv)
        assert want.shape[2] == size
        assert mx.array_equal(splitk.ring_gather(kv.keys, (rot, size)), want)
        for start in range(1, size):
            ring = ((rot + start) % kv.keys.shape[2], size - start)
            assert mx.array_equal(splitk.ring_gather(kv.keys, ring), want[:, :, start:])
    assert len(seen) > 3, "the ring never wrapped at several offsets"


def test_ring_layout_declines_keep_rows():
    from mlx2.runtime.models.inkling2.language import Attention

    kv = RotatingKVCache(8, keep=2)
    kv.update_and_fetch(mx.zeros((1, 2, 3, 4)), mx.zeros((1, 2, 3, 4)))
    assert Attention._ring_layout(kv) is None


def _engage(monkeypatch):
    """Split-K engaged on the CPU with the composed law reading the ring the
    way the kernel does (``reference_attention(ring=...)`` gathers it)."""
    seen = []

    def fake(q, k, v, rel, sinks, **kw):
        seen.append((q.shape[2], kw["window"], kw.get("ring")))
        return splitk.reference_attention(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(splitk, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(splitk, "gqa_splitk_attention", fake)
    return seen


def _batched(model, sliding_ring=True):
    cache = model.make_cache()
    for i in range(0, len(cache), 2):
        cache[i] = (BatchKVCache([0, 3]) if type(cache[i]) is KVCache or not sliding_ring
                    else BatchRotatingKVCache(8, [0, 3]))
    return cache


@pytest.mark.parametrize("rows", ["b1", "b2", "b2_padded"])
def test_ring_direct_engaged_is_bitwise(monkeypatch, rows):
    """Prefill 9, verify 8, then decode well past the wrap (window 8): every
    sliding visit reaches the kernel stand-in as its physical ring with a
    start row, and the logits are the composed path's, bitwise. B2 runs a
    plain two-row ring; ``b2_padded`` BatchRotatingKVCache rows with left
    padding (its in-place ring is no longer rolled every step)."""
    model = _tiny_model()
    b = 1 if rows == "b1" else 2
    tokens = mx.random.randint(0, 500, (b, 34))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 34)]
    factory = (lambda: _batched(model)) if rows == "b2_padded" else model.make_cache
    base, _ = _run(model, tokens, plan, cache=factory())
    seen = _engage(monkeypatch)
    model.configure_kernels(verify_splitk=True)
    try:
        got, _ = _run(model, tokens, plan, cache=factory())
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_calls"] == 48 * len(plan)
    ring = receipt["verify_splitk_ring"]
    assert ring.get("calls", 0) == 40 * len(plan) and "copies" not in ring
    rings = [r for _, w, r in seen if w]
    assert all(r is not None for r in rings)
    assert len({r[0] for r in rings}) > 3, "no wrapped ring at several start rows"
    assert all(r is None for _, w, r in seen if not w)


def test_ring_direct_switch_off_is_the_copy(monkeypatch):
    """``RING_DIRECT = False``: the reordered copy reaches the kernel as
    before (no ring argument, no ring counters), same bits."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 24))
    plan = [(0, 9)] + [(i, i + 1) for i in range(9, 24)]
    base, _ = _run(model, tokens, plan)
    seen = _engage(monkeypatch)
    monkeypatch.setattr(splitk, "RING_DIRECT", False)
    got, receipt = _receipt_after(model, tokens, plan, verify_splitk=True)
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_ring"] == {}
    assert all(r is None for _, _, r in seen)


@pytest.mark.parametrize("rows", ["b1", "b2_padded"])
def test_ring_direct_declined_copies_and_is_bitwise(rows):
    """On the CPU the kernel declines (``cpu``): the copy is made after the
    decline (counted as ``copies``), bitwise the reference; for
    BatchRotatingKVCache that includes the deferred in-place roll."""
    model = _tiny_model()
    b = 1 if rows == "b1" else 2
    tokens = mx.random.randint(0, 500, (b, 30))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 30)]
    factory = (lambda: _batched(model)) if rows == "b2_padded" else model.make_cache
    base, _ = _run(model, tokens, plan, cache=factory())
    model.configure_kernels(verify_splitk=True)
    try:
        got, _ = _run(model, tokens, plan, cache=factory())
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_calls"] == 0
    assert receipt["verify_splitk_ring"] == {"copies": 40 * len(plan)}


def test_ring_direct_with_default_policy_options(monkeypatch):
    """The shipped default set (sink SDPA, sink slot, the four exact fusions,
    split-K) with the stand-in engaged: the ring read is what runs on the
    sliding decode visits, bitwise against the same set with the copy."""
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS

    options = {k: v for k, v in KERNEL_DEFAULTS.items()
               if isinstance(v, bool) and k not in ("moe_fused", "fuse_sconv_kernel")}
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 30))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 30)]
    _engage(monkeypatch)
    monkeypatch.setattr(splitk, "RING_DIRECT", False)
    base, _ = _receipt_after(model, tokens, plan, **options)
    monkeypatch.setattr(splitk, "RING_DIRECT", True)
    got, receipt = _receipt_after(model, tokens, plan, **options)
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_ring"].get("calls", 0) == 40 * len(plan)


def test_ring_direct_dspark_verify_rows(monkeypatch):
    """DSpark verify rows (L = 4) and ring rollbacks through the ring read:
    greedy output equals ordinary decode."""
    import test_inkling2_dspark_cpu as d

    target, draft = _dspark_pair()
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    expect = d._ordinary_greedy(target, prompt)
    seen = _engage(monkeypatch)
    target.configure_kernels(verify_splitk=True)
    try:
        got = d._dspark_greedy(target, draft, prompt)
        receipt = target.kernel_receipt()
    finally:
        target.configure_kernels()
    assert got == expect
    assert receipt["verify_splitk_ring"].get("calls", 0) > 0
    assert "copies" not in receipt["verify_splitk_ring"]
    assert any(length == 4 and r is not None for length, _, r in seen)


def test_ring_direct_apcv2_restore(monkeypatch):
    """A restored cache (wrapped rings) continues through the ring read with
    the reference route's logits, bitwise."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 44))
    plan = [(32, 40)] + [(i, i + 1) for i in range(40, 44)]
    base = _apc_restore_logits(model, tokens, 32, plan)
    _engage(monkeypatch)
    model.configure_kernels(verify_splitk=True)
    try:
        got = _apc_restore_logits(model, tokens, 32, plan)
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_ring"].get("calls", 0) > 0
    assert "copies" not in receipt["verify_splitk_ring"]


# ----- split-K direct kernel: 8-byte BF16 loads -----


def test_direct_vector_loads_template(monkeypatch):
    """BF16 with D / 32 a multiple of 4 (the Mini's D = 128) takes the
    8-byte loads; FP32, D = 32 (the tiny model) and the switch off do not.
    The value is fixed per dtype and geometry, never per step."""
    assert splitk.direct_vector_loads(mx.bfloat16, 128) == 1
    assert splitk.direct_vector_loads(mx.bfloat16, 256) == 1
    assert splitk.direct_vector_loads(mx.bfloat16, 32) == 0
    assert splitk.direct_vector_loads(mx.bfloat16, 64) == 0
    assert splitk.direct_vector_loads(mx.float32, 128) == 0
    monkeypatch.setattr(splitk, "VECTOR_LOADS", False)
    assert splitk.direct_vector_loads(mx.bfloat16, 128) == 0
