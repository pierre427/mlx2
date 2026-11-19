"""Hy4-flash APCv2 prefix reuse on the tiny random checkpoint (CPU only).

Hy4's cache has three bounded planes besides the append-only indexed KV:
the sliding layers' ring (``window - 1 + ROLLBACK`` rows), the Engram n-gram
token tail (``max_ngram - 1 + ROLLBACK`` ids) and, on DFly2 routes, the
Index Share selections (last ``ROLLBACK`` positions).  APCv2 restores an
entry by copy and lands a branch below it by trimming, so these tests show:

* a warm prefix hit continues bit for bit like the cold run (logprobs), with
  the sliding window already wrapped at the stored boundary;
* a branch landing below a published boundary, inside every plane's
  retention, is exact;
* a landing past retention is refused (``preflight_trim``) and the request
  runs cold, still exact;
* the Engram tail and DFly2 selections survive APCv2 store/lookup and the
  ``state``/``meta_state`` disk codec; the tail is long enough never to
  bound a landing tighter than the ring, a short one is refused, and DFly2
  recording never narrows the landings.

The engine runs the real ``HYV4FlashAdapter`` on a safetensors artifact, so
Engram tables are read from disk (PLE offload, the default).  Prompts are
chosen so warm chunks start on the cold run's prefill grid; only stored
state can then make the two differ.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving  # noqa: E402
from mlx2.runtime import os_memory  # noqa: E402
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models.cache import can_trim_prompt_cache_at  # noqa: E402
from test_hy_v4_flash_cpu import (  # noqa: E402
    _tiny_artifact,
    checkpoint,
    tiny,  # noqa: F401 - fixture
    tiny_config,
)

PREFILL_STEP = 16
SEQ = [int(t) for t in np.random.default_rng(9).integers(4, 90, size=220)]
SUFFIX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25]
RETAINED = 8 - 1 + H.ROLLBACK  # tiny sliding_window 8


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    config = tiny_config()
    return _tiny_artifact(tmp_path_factory.mktemp("hy4apc"), config, checkpoint(config))


def _engine(path, **kw):
    from mlx2.adapters import hy_v4_flash as A

    class Adapter(A.HYV4FlashAdapter):
        # Token-id requests: the tiny WordLevel tokenizer has no template.
        def prompt_tokens(self, request):
            if "tokens" in request:
                return list(request["tokens"])
            return super().prompt_tokens(request)

    engine = serving.ServingEngine(str(path), adapter_factory=Adapter, qualification_mode=True,
                                   mtp=False, max_lanes=1, prefill_step=PREFILL_STEP, **kw)
    assert engine.ready.wait(120), engine.error
    return engine


def _run(engine, tokens, n=4):
    job = engine.submit({"tokens": list(tokens), "max_tokens": n, "temperature": 0,
                         "logprobs": True, "top_logprobs": 5})
    rows = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "logprob" in event:
            row = event["logprob"]
            rows.append((row["id"], row["logprob"],
                         tuple((t["id"], t["logprob"]) for t in row["top_logprobs"])))
        if "finish_reason" in event:
            return rows, int(job.cached_tokens or 0)


def _cold(path, tokens, **kw):
    engine = _engine(path, **kw)
    try:
        rows, cached = _run(engine, tokens)
    finally:
        engine.close()
    assert cached == 0
    return rows


# ---------------------------------------------------------------- descriptor --

def test_descriptor_declares_apcv2_unqualified():
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.contracts import Capability
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities

    for dfly in (False, True):
        d = A.descriptor_for(dfly_selected=dfly)
        assert {Capability.PREFIX_REUSE, Capability.APC_V2} <= d.capabilities
        assert d.metadata["qualification"] == "pending"
        assert d.metadata["apc_v2"].startswith("implemented; unqualified")
    # DFly2 entries carry the Index Share plane: their own namespace.
    assert A.descriptor_for(dfly_selected=True).cache_layout == A.DFLY_CACHE_LAYOUT != A.CACHE_LAYOUT
    caps = inspect_apc_capabilities(H.Model(H.ModelArgs.from_hf_config(tiny_config())).make_cache())
    assert caps.exact_prefix and caps.topology == "checkpointed_hybrid"


# ------------------------------------------------- engine: cold versus warm --

def test_warm_prefix_hit_is_bitwise_the_cold_run(host, artifact):
    # The 97-token prompt publishes its len - 1 boundary (96): past the
    # sliding retention, so the restored ring has wrapped, and on the cold
    # run's chunk grid.  The second request diverges right there.
    prompt = SEQ[: 6 * PREFILL_STEP + 1]
    extended = prompt[:-1] + SUFFIX
    assert SUFFIX[0] != prompt[-1]
    assert len(prompt) - 1 > RETAINED
    engine = _engine(artifact)
    try:
        _run(engine, prompt)
        warm, cached = _run(engine, extended)
        assert engine.apc.apc_stats["hits"] >= 1
    finally:
        engine.close()
    assert cached == len(prompt) - 1
    assert warm == _cold(artifact, extended)


def test_branch_landing_below_a_published_boundary_is_exact(host, artifact):
    # The 200-token prompt publishes its len - 1 boundary (199).  A request
    # that diverges at 160 lands there by a 39-token trim: inside every
    # bounded plane's retention, and on the cold chunk grid.
    branch = SEQ[:160] + SUFFIX
    engine = _engine(artifact)
    try:
        _run(engine, SEQ[:200])
        warm, cached = _run(engine, branch)
    finally:
        engine.close()
    assert cached == 160
    assert warm == _cold(artifact, branch)


def test_branch_landing_past_retention_is_refused_and_runs_cold(host, artifact):
    # Diverging at 112 would trim 87 > ROLLBACK rows from the 199 boundary:
    # the sliding layers no longer hold the 7 keys before 112.
    branch = SEQ[:112] + SUFFIX
    engine = _engine(artifact)
    try:
        _run(engine, SEQ[:200])
        warm, cached = _run(engine, branch)
        stats = engine.apc.apc_stats
    finally:
        engine.close()
    assert cached == 0 and stats["hits"] == 0
    assert warm == _cold(artifact, branch)


def test_dfly2_route_hit_drafts_from_restored_selections(host, artifact, monkeypatch):
    # Route check: the DFly2 route's own namespace hits, the drafter keeps
    # drafting after the restore (a missing selection or feature would raise
    # ``_unavailable`` and decode the lane ordinarily) and the greedy tokens
    # equal a cold run.  Greedy verification keeps tokens exact whatever the
    # drafter sees, and the re-fed suffix records fresh selections, so the
    # survival of restored selections is shown by the store/lookup and codec
    # tests below, not here.
    from mlx2.runtime.drafters import hy4_dfly

    unavailable = []
    original = hy4_dfly._unavailable

    def record(reason):
        unavailable.append(reason)
        return original(reason)

    monkeypatch.setattr(hy4_dfly, "_unavailable", record)
    policy = {"execution_policy": {"draft_model": "embedded_dfly", "num_draft": 3}}
    prompt = SEQ[: 6 * PREFILL_STEP + 1]
    extended = prompt + SUFFIX
    engine = _engine(artifact, **policy)
    try:
        _run(engine, prompt)
        drafts = engine.adapter.draft_model.stats["block_drafts"]
        unavailable.clear()
        warm, cached = _run(engine, extended, n=8)
        assert "+dfly2-index-share" in engine.adapter.layout
        assert engine.adapter.draft_model.stats["block_drafts"] > drafts
    finally:
        engine.close()
    assert cached > 0 and unavailable == []
    cold_engine = _engine(artifact, **policy)
    try:
        cold, cold_cached = _run(cold_engine, extended, n=8)
    finally:
        cold_engine.close()
    assert cold_cached == 0 and [r[0] for r in warm] == [r[0] for r in cold]


# ------------------------------------------- APCv2 store/lookup and planes --

def _prefill(model, tokens, record=False):
    model.record_layers = frozenset({0, 4}) if record else frozenset()
    try:
        cache = model.make_cache()
        model(mx.array([tokens]), cache)
    finally:
        model.record_layers = frozenset()
    return cache


def _continue(model, cache, tokens):
    return np.array(model(mx.array([tokens]), cache))


def test_apcv2_lookup_keeps_engram_tail_and_dfly_selections(tiny):  # noqa: F811
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    _, _, model, _ = tiny
    tokens = SEQ[:100]
    cache = _prefill(model, tokens, record=True)
    apc = APCv2(max_size=4, layout_name="hy4-apcv2-test")
    key = APCKey("hy4-tiny")
    apc.store(key, tokens, cache)
    hit = apc.lookup(key, tokens + [5])
    assert hit.hit and hit.cached_tokens == 100
    for before, after in zip(cache, hit.cache):
        assert after.token_tail == before.token_tail and after.offset == before.offset
        assert sorted(after.selections) == sorted(before.selections)
        for pos, value in before.selections.items():
            if value is None:
                assert after.selections[pos] is None
            else:
                np.testing.assert_array_equal(np.array(after.selections[pos]), np.array(value))
    assert hit.cache[1].token_tail and hit.cache[4].selection_at(99) is not None
    np.testing.assert_array_equal(_continue(model, hit.cache, [5, 6]), _continue(model, cache, [5, 6]))
    # A landing past the ring is refused at lookup, not decoded.
    deep = apc.lookup(key, tokens[:20] + [5])
    assert deep.cached_tokens == 0 and deep.cache is None


def test_disk_codec_round_trip_keeps_every_plane(tiny, tmp_path):  # noqa: F811
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    _, _, model, _ = tiny
    cache = _prefill(model, SEQ[:90], record=True)
    path = str(tmp_path / "hy4.safetensors")
    save_prompt_cache(path, cache)
    loaded = load_prompt_cache(path)
    for a, b in zip(cache, loaded):
        assert (a.offset, a.start, a.token_tail, a.record) == (b.offset, b.start, b.token_tail, b.record)
        assert sorted(a.selections) == sorted(b.selections)
    np.testing.assert_array_equal(_continue(model, loaded, SUFFIX[:3]), _continue(model, cache, SUFFIX[:3]))


def test_engram_tail_never_binds_tighter_than_the_ring(tiny, monkeypatch):  # noqa: F811
    _, _, model, _ = tiny
    n = RETAINED - 1  # 70: the ring has not wrapped, so every trim is ring-safe
    cache = _prefill(model, SEQ[:n])
    assert all(c.start == 0 for c in cache)
    assert model.engram_tail_keep >= n + model.args.engram_max_ngram_size - 1
    # The tail keeps the whole short history: the deepest landing is exact.
    assert can_trim_prompt_cache_at(cache, n - 1)
    for c in cache:
        c.trim(n - 1)
    fresh = _prefill(model, SEQ[:1])
    np.testing.assert_allclose(_continue(model, cache, SUFFIX[:4]), _continue(model, fresh, SUFFIX[:4]),
                               rtol=1e-4, atol=1e-4)


def test_engram_tail_guard_refuses_a_short_history(tiny, monkeypatch):  # noqa: F811
    # A tail shorter than the next n-gram needs (the pre-APCv2 keep of
    # max_ngram - 1 + ROLLBACK ids, or an Engram cache with no tail at all)
    # is refused at the exact depth where decoding would go wrong.
    _, _, model, _ = tiny
    n, ngram = RETAINED - 1, model.args.engram_max_ngram_size - 1
    keep = ngram + H.ROLLBACK

    def short(tokens):
        cache = _prefill(model, tokens)
        for c in cache:
            if c.token_tail:
                c.token_tail = tuple(row[-keep:] for row in c.token_tail)
        return cache

    deepest = keep - ngram
    assert can_trim_prompt_cache_at(short(SEQ[:n]), deepest)
    assert not can_trim_prompt_cache_at(short(SEQ[:n]), deepest + 1)
    with pytest.raises(ValueError):
        short(SEQ[:n])[1].trim(deepest + 1)
    empty = _prefill(model, SEQ[:n])
    empty[1].token_tail = ()
    assert not can_trim_prompt_cache_at(empty, 1)
    # The refusal guards a real hazard: forced past it, the next n-gram
    # loses a history token and the logits move by whole units.
    forced = short(SEQ[:n])
    monkeypatch.setattr(H.HY4FlashLayerCache, "can_trim", lambda self, count: True)
    for c in forced:
        c.trim(deepest + 1)
    fresh = _prefill(model, SEQ[: n - deepest - 1])
    assert np.abs(_continue(model, forced, SUFFIX[:4]) - _continue(model, fresh, SUFFIX[:4])).max() > 0.1


def test_dfly_recording_does_not_narrow_landings(tiny):  # noqa: F811
    # A landing past the recorded selections is still exact: the drafter
    # reads only the last committed token's selection, recorded afresh when
    # the landing re-feeds its suffix.  Recording must not refuse more.
    _, _, model, _ = tiny
    n = RETAINED - 1
    recorded, plain = _prefill(model, SEQ[:n], record=True), _prefill(model, SEQ[:n])
    assert min(recorded[4].selections) == n - H.ROLLBACK
    for depth in (1, H.ROLLBACK, H.ROLLBACK + 1, n - 1):
        assert can_trim_prompt_cache_at(recorded, depth) == can_trim_prompt_cache_at(plain, depth) is True


def test_batch_cache_preflights_every_row(tiny):  # noqa: F811
    _, _, model, _ = tiny
    short, long = _prefill(model, SEQ[:20]), _prefill(model, SEQ[:150])
    merged = [type(a).merge([a, b]) for a, b in zip(short, long)]
    assert can_trim_prompt_cache_at(merged, H.ROLLBACK)
    # Row 1's ring has wrapped: 80 positions back is past its retention.
    assert not can_trim_prompt_cache_at(merged, 80)
