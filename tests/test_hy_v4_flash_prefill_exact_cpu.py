"""Hy4-flash exact prefill defaults on the tiny CPU checkpoint (Pierre 2026-11-21).

Three default-on options, each with an opt-out policy key, none of which may
change state or committed tokens:

* ``prefill_head_trim``: inside a prompt chunk's checkpoint-row request only
  each lane's checkpoint row runs through the fp32 head, padded to the
  calibrated head row floor.  The kept rows' logits are bitwise the full
  head's (dense fp32 and 8-bit quantized heads; B=1, B=2, ragged B=2).
* ``prefill_tail_skip``: the indexed last layer writes K/V, indexer token keys
  and pooled group keys over every row and computes the rest for the last
  row.  Every cache plane after prefill is bitwise the full forward's, and
  greedy tokens are identical; readers of all rows decline it.
* ``moe_sorted_weighted_sum``: on CPU it declines (Metal kernel); the
  ``keep_sorted`` restructure of ``SwitchGLU`` is checked bitwise here, the
  kernel itself in tests/test_hy_v4_flash_prefill_exact_metal.py.

Evidence for docs/research/model-feature-decisions.tsv (checkpoint: the tiny
random checkpoint of tests/test_hy_v4_flash_cpu.py, seed 0).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import LONG, TOKENS, _drain, _greedy, build, checkpoint, tiny_config  # noqa: E402

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_fusions as F  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402


@pytest.fixture(autouse=True)
def _tail_env(monkeypatch):
    monkeypatch.delenv(pts.ENV, raising=False)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    return config, checkpoint(config)


def _model(tiny, *, head="dense", fusions=None):
    config, W = tiny
    model, drafter = build(config, W)
    if head == "q8":
        nn.quantize(model, group_size=32, bits=8, class_predicate=lambda p, m: p == "lm_head")
        assert "scales" in model.lm_head
    if fusions:
        F.install(model, **fusions)
    return model, drafter


def _caches(model, lanes, lengths=None, width=None):
    if lanes == 1:
        return model.make_cache()
    per_lane = [model.make_cache() for _ in range(lanes)]
    caches = [H.HY4FlashBatchLayerCache(list(rows)) for rows in zip(*per_lane)]
    if lengths is not None and any(n != width for n in lengths):
        for c in caches:
            c.prepare(lengths=lengths, right_padding=[width - n for n in lengths])
    return caches


def _states(caches):
    out = []
    for c in caches:
        rows = getattr(c, "rows", None) or [c]
        for r in rows:
            out.append((tuple(np.array(a) for a in r.state), tuple(r.meta_state)))
    return out


def _assert_same_state(a, b):
    assert len(a) == len(b)
    for (sa, ma), (sb, mb) in zip(a, b):
        assert ma == mb
        assert len(sa) == len(sb)
        for x, y in zip(sa, sb):
            assert x.dtype == y.dtype and x.shape == y.shape
            np.testing.assert_array_equal(x, y)


def _batch_tokens(prompts, width):
    return mx.array([p + [0] * (width - len(p)) for p in prompts])


# ------------------------------------------------------------- head trim --

@pytest.mark.parametrize("head", ["dense", "q8"])
@pytest.mark.parametrize("lanes,lengths", [(1, [40]), (2, [40, 40]), (2, [40, 29])],
                         ids=["b1", "b2", "b2_ragged"])
def test_head_trim_checkpoint_rows_bitwise(tiny, head, lanes, lengths):
    model, _ = _model(tiny, head=head)
    width = max(lengths)
    prompts = [LONG[i * 7: i * 7 + n] for i, n in enumerate(lengths)]
    tokens = _batch_tokens(prompts, width)
    model.prefill_tail_skip = False  # head trim alone (the tail has its own test)
    full_caches = _caches(model, lanes, lengths, width)
    full = model(tokens, full_caches)
    caches = _caches(model, lanes, lengths, width)
    with prefill_checkpoint_rows(lengths, width=width) as request:
        out = model(tokens, caches)
    assert request.applied == 1 and out.shape == (lanes, 1, full.shape[-1])
    for b, n in enumerate(lengths):
        assert mx.array_equal(out[b, 0], full[b, n - 1]).item(), (head, lanes, b)
    _assert_same_state(_states(caches), _states(full_caches))
    status = model.prefill_exact_status()["head_trim"]
    assert status["observed_used"] and status["forwards"] == 1
    floor = model.head_row_floor()
    assert 1 <= floor <= 128
    assert status["rows_skipped"] == lanes * width - max(floor, lanes)


def test_head_trim_second_chunk_and_selection(tiny):
    """A chunk past candidate_topk (indexer selection active), after a first chunk."""
    model, _ = _model(tiny, head="q8")
    model.prefill_tail_skip = False
    ref = model.make_cache()
    model(mx.array([LONG[:30]]), ref)
    full = model(mx.array([LONG[30:70]]), ref)
    cache = model.make_cache()
    with prefill_checkpoint_rows([30], width=30):
        model(mx.array([LONG[:30]]), cache)
    with prefill_checkpoint_rows([40], width=40) as request:
        out = model(mx.array([LONG[30:70]]), cache)
    assert request.applied == 1
    assert mx.array_equal(out[0, 0], full[0, -1]).item()


def test_head_trim_leaves_other_forwards_full(tiny):
    model, _ = _model(tiny)
    tokens = mx.array([LONG[:40]])
    # No request (decode, verify, prompt scoring, taps): every row.
    assert model(tokens, model.make_cache()).shape == (1, 40, 97)
    # A chunk within the floor keeps every row (nothing to save); declined, counted.
    model.prefill_tail_skip = False
    floor = model.head_row_floor()
    short = mx.array([LONG[:max(2, floor)]])
    with prefill_checkpoint_rows([short.shape[1]], width=short.shape[1]) as request:
        out = model(short, model.make_cache())
    assert out.shape[1] == short.shape[1] and request.applied == 0
    assert model.prefill_exact_status()["head_trim"]["declined"]["chunk_within_floor"] == 1
    # DFly2 / AST taps and prefill bodies never read the request.
    with prefill_checkpoint_rows([40], width=40) as request:
        hidden = model.prefill_body(tokens, model.make_cache(), (5,))
    assert hidden.shape == (1, 40, 32) and request.applied == 0
    # Opt-out.
    model.prefill_head_trim = False
    model.prefill_tail_skip = False
    with prefill_checkpoint_rows([40], width=40) as request:
        out = model(tokens, model.make_cache())
    assert out.shape == (1, 40, 97) and request.applied == 0


# -------------------------------------------------------------- tail split --

def _chunked_prefill(model, caches, prompts, step):
    width = max(len(p) for p in prompts)
    tokens = _batch_tokens(prompts, width)
    lengths = [len(p) for p in prompts]
    outs = []
    for start in range(0, width, step):
        n = min(step, width - start)
        chunk = [max(0, min(n, length - start)) for length in lengths]
        with prefill_checkpoint_rows(chunk, width=n) as request:
            outs.append((model(tokens[:, start:start + n], caches), request.applied))
    return outs


@pytest.mark.parametrize("fusions", [None, dict(mla_inputs=True), dict(hc_seam=True, mla_inputs=True,
                                                                       engram_inputs=True)],
                         ids=["plain", "mla_fused", "fusions"])
@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("head", ["dense", "q8"])
def test_tail_state_bitwise_and_greedy_identical(tiny, fusions, lanes, head):
    model, _ = _model(tiny, head=head, fusions=fusions)
    prompts = [LONG[i * 5: i * 5 + 60] for i in range(lanes)]
    results = {}
    for tail in (True, False):
        model.prefill_tail_skip = tail
        before = model.prefill_exact_stats["tail_forwards"]
        caches = _caches(model, lanes)
        outs = _chunked_prefill(model, caches, prompts, step=20)
        engaged = model.prefill_exact_stats["tail_forwards"] - before
        state = _states(caches)
        # Greedy decode from the prefilled caches (one token per lane per step).
        nxt = mx.array([[p[-1]] for p in prompts])
        tokens = []
        for _ in range(6):
            logits = model(nxt, caches)
            nxt = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
            tokens.append(np.array(nxt).ravel().tolist())
        results[tail] = (state, tokens, engaged, outs)
    _assert_same_state(results[True][0], results[False][0])
    assert results[True][1] == results[False][1]
    assert results[True][2] == 3 and results[False][2] == 0
    for (out, applied), (ref, _) in zip(results[True][3], results[False][3]):
        # One row per lane, finite, and within rounding of the full forward's row.
        assert applied == 1 and out.shape == (lanes, 1, 97)
        assert bool(mx.isfinite(out).all().item())
        np.testing.assert_allclose(np.array(out[:, 0]), np.array(ref[:, 0]), rtol=1e-4, atol=1e-4)


def test_tail_engram_layer_and_batch_generator_greedy(tiny):
    """Engram on the last layer runs over all rows; the scheduler's greedy tokens match."""
    from mlx2.runtime.generate import BatchGenerator

    config = tiny_config(engram_layer_ids=[2, 6],
                         engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]],
                                                        "6": [[23, 29], [31, 37]]})
    W = checkpoint(config)
    model, _ = build(config, W)
    assert "engram_layer" in model.model.layers[-1]
    ref, _ = build(config, W)
    ref.prefill_tail_skip = False
    prompts = [LONG[:50], LONG[10:45], LONG[3:9]]
    batch = BatchGenerator(model, max_tokens=6, completion_batch_size=3, prefill_batch_size=2,
                           prefill_step_size=16)
    uids = batch.insert(prompts, max_tokens=[6] * 3)
    out = _drain(batch, uids, 6)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:6] == _greedy(ref, prompt, 6)
    assert model.prefill_exact_stats["tail_forwards"] > 0


def _decline_count(reason):
    return pts.STATS.get(f"declined_{reason}", 0)


def test_tail_declines_for_readers_of_every_row(tiny):
    model, drafter = _model(tiny)
    tokens = mx.array([LONG[:40]])

    def run(cache=None):
        cache = cache if cache is not None else model.make_cache()
        before = model.prefill_exact_stats["tail_forwards"]
        with prefill_checkpoint_rows([40], width=40) as request:
            out = model(tokens, cache)
        return model.prefill_exact_stats["tail_forwards"] - before, request.applied, out

    assert run()[0] == 1
    # DFly2 Index Share: the drafter's recorded layers include the last one.
    assert len(model.model.layers) - 1 in drafter.index_layers
    model.record_layers = frozenset(drafter.index_layers)
    n = _decline_count("hy_v4_flash_index_share_record")
    engaged, applied, out = run()
    assert engaged == 0 and _decline_count("hy_v4_flash_index_share_record") == n + 1
    assert applied == 1 and out.shape == (1, 1, 97)  # the head trim still applies
    model.record_layers = frozenset()
    # A layer-output reader (calculator tile, aMoEba capture) on the last layer.
    model.layer_output_readers[len(model.model.layers) - 1] = lambda *a: None
    n = _decline_count("hy_v4_flash_last_layer_reader")
    assert run()[0] == 0 and _decline_count("hy_v4_flash_last_layer_reader") == n + 1
    model.layer_output_readers.clear()
    # A reader on another layer still sees every row of its layer; the tail engages.
    seen = []
    model.layer_output_readers[2] = lambda layer, streams, ids, prefix: seen.append(streams.shape)
    assert run()[0] == 1 and seen[-1][1] == 40
    model.layer_output_readers.clear()
    # A wrapped linear in the last layer (a capture hook).
    attn = model.model.layers[-1].self_attn
    saved = attn.o_proj

    class Capture(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            return self.inner(x)

    attn.o_proj = Capture(saved)
    n = _decline_count("hy_v4_flash_wrapped_last_layer")
    assert run()[0] == 0 and _decline_count("hy_v4_flash_wrapped_last_layer") == n + 1
    attn.o_proj = saved
    # Weight overlays / router tiles bound on the model.
    object.__setattr__(model, "_mlx2_ablation_overlays", object())
    assert run()[0] == 0
    object.__delattr__(model, "_mlx2_ablation_overlays")
    # Ragged lanes: the checkpoint rows are not every lane's last, nothing is claimed.
    caches = _caches(model, 2, [40, 29], 40)
    before = model.prefill_exact_stats["tail_forwards"]
    with prefill_checkpoint_rows([40, 29], width=40) as request:
        model(_batch_tokens([LONG[:40], LONG[:29]], 40), caches)
    assert model.prefill_exact_stats["tail_forwards"] == before and request.applied == 1
    # Opt-outs: the environment switch and the policy attribute.
    import os

    os.environ[pts.ENV] = "0"
    try:
        assert run()[0] == 0
    finally:
        del os.environ[pts.ENV]
    model.prefill_tail_skip = False
    assert run()[0] == 0


def test_tail_declines_short_fused_chunk(tiny):
    model, _ = _model(tiny, fusions=dict(mla_inputs=True))
    # The load-time self-check bounds the CPU dense GEMM at 1 row; a device
    # whose bound covers the chunk takes its key rows from the fused GEMM.
    model.model.layers[-1].self_attn.fused_input_proj.set_max_rows(64)
    tokens = mx.array([LONG[:10]])  # within the fused GEMM's row bound
    n = _decline_count("hy_v4_flash_fused_input_rows")
    with prefill_checkpoint_rows([10], width=10):
        model(tokens, model.make_cache())
    assert _decline_count("hy_v4_flash_fused_input_rows") == n + 1


def test_family_default_and_status():
    assert pts.FAMILY_DEFAULTS["hy_v4_flash"] == "1"
    assert pts.tail_skip_mode(H.TAIL_SKIP_FAMILY) == "1"


# ------------------------------------------------------- sorted combine --

def test_keep_sorted_rows_unsort_to_the_default_output(tiny):
    from mlx2.runtime.models.switch_layers import _scatter_unsort

    model, _ = _model(tiny)
    moe = model.model.layers[1].mlp
    x = mx.array(np.random.default_rng(1).standard_normal((1, 30, 24)).astype(np.float32))
    indices, _ = moe.gate(mx.array(np.random.default_rng(2).standard_normal((1, 30, 32)).astype(np.float32)))
    plain = moe.experts(x, indices)
    rows, inv = moe.experts(x, indices, keep_sorted=True)
    assert mx.array_equal(_scatter_unsort(rows, inv, indices.shape).squeeze(-2), plain).item()
    # Below the sort threshold the plain output comes back.
    small = moe.experts(x[:, :2], indices[:, :2], keep_sorted=True)
    assert not isinstance(small, tuple)


def test_sorted_combine_declines_on_cpu_and_matches_eager(tiny):
    model, _ = _model(tiny)
    tokens = mx.array([TOKENS])
    on = model(tokens, model.make_cache())
    status = model.prefill_exact_status()["sorted_weighted_sum"]
    assert status["selected"] and not status["observed_used"]
    assert status["declined_layer_calls"].get("not_metal", 0) > 0
    model.set_moe_sorted_weighted_sum(False)
    off = model(tokens, model.make_cache())
    assert mx.array_equal(on, off).item()


# ------------------------------------------------------------- adapter --

def test_adapter_policy_defaults_opt_outs_and_receipt(tmp_path, tiny):
    from test_hy_v4_flash_cpu import _tiny_artifact

    from mlx2.adapters import hy_v4_flash as A

    config, W = tiny
    path = str(_tiny_artifact(tmp_path, config, W))
    for key in A.PREFILL_EXACT_DEFAULTS:
        with pytest.raises(ValueError, match=key):
            A._parse_policy({key: "yes"})
    adapter = A.HYV4FlashAdapter(path)
    try:
        assert adapter.model.prefill_head_trim and adapter.model.prefill_tail_skip
        assert all(m.sorted_weighted_sum for m in (l.mlp for l in adapter.model.model.layers)
                   if isinstance(m, H.MoE))
        receipt = adapter.diagnostics()["prefill_exact"]
        for key in A.PREFILL_EXACT_DEFAULTS:
            assert receipt[key]["selected"] and receipt[key]["default"] is True
            assert receipt[key]["qualified"] is False and receipt[key]["opt_out"] == f'{{"{key}": false}}'
        assert receipt["prefill_head_trim"]["row_floor"]  # calibrated at load
        layout = adapter.layout
        budget = adapter.cache_budget(mtp=False)
        assert budget.prefill_head_rows == adapter.model.head_row_floor()
    finally:
        adapter.close()
    off = A.HYV4FlashAdapter(path, execution_policy={k: False for k in A.PREFILL_EXACT_DEFAULTS})
    try:
        assert not off.model.prefill_head_trim and not off.model.prefill_tail_skip
        receipt = off.diagnostics()["prefill_exact"]
        assert not any(receipt[k]["selected"] for k in A.PREFILL_EXACT_DEFAULTS)
        # Exact options: no APCv2 namespace change.
        assert off.layout == layout
        assert off.cache_budget(mtp=False).prefill_head_rows is None
    finally:
        off.close()


# ------------------------------------------------- review follow-ups --

def test_receipt_does_not_claim_consumed_logits(tmp_path, tiny):
    from test_hy_v4_flash_cpu import _tiny_artifact

    from mlx2.adapters import hy_v4_flash as A

    config, W = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)))
    try:
        receipt = adapter.diagnostics()["prefill_exact"]
        assert "not consumed" in receipt["prefill_head_trim"]["relation"]
        assert "not consumed" in receipt["prefill_tail_skip"]["relation"]
    finally:
        adapter.close()


def test_env_off_also_disables_head_trim(tiny, monkeypatch):
    model, _ = _model(tiny)
    monkeypatch.setenv(pts.ENV, "0")
    with prefill_checkpoint_rows([40], width=40) as request:
        out = model(mx.array([LONG[:40]]), model.make_cache())
    assert out.shape == (1, 40, 97) and request.applied == 0
    # "head" keeps the trim and leaves the tail off.
    monkeypatch.setenv(pts.ENV, "head")
    before = model.prefill_exact_stats["tail_forwards"]
    with prefill_checkpoint_rows([40], width=40) as request:
        out = model(mx.array([LONG[:40]]), model.make_cache())
    assert out.shape == (1, 1, 97) and request.applied == 1
    assert model.prefill_exact_stats["tail_forwards"] == before


def test_tail_declines_left_padded_lanes(tiny):
    model, _ = _model(tiny)
    caches = _caches(model, 2)
    for c in caches:
        c.prepare(left_padding=[0, 3])
    tokens = mx.array([LONG[:40], [0, 0, 0] + LONG[:37]])
    n = _decline_count("hy_v4_flash_lane_layout")
    before = model.prefill_exact_stats["tail_forwards"]
    with prefill_checkpoint_rows([40, 40], width=40):
        model(tokens, caches)  # must not raise after a claim
    assert _decline_count("hy_v4_flash_lane_layout") == n + 1
    assert model.prefill_exact_stats["tail_forwards"] == before


def test_no_refusal_walk_without_a_request(tiny, monkeypatch):
    model, _ = _model(tiny)
    calls = []
    monkeypatch.setattr(type(model), "_tail_refusal", lambda self, *a: calls.append(1))
    model(mx.array([LONG[:40]]), model.make_cache())  # verify-shaped: no request
    assert calls == []


def test_sorted_combine_no_counter_on_decode_rows(tiny):
    model, _ = _model(tiny)
    cache = model.make_cache()
    model(mx.array([LONG[:2]]), cache)  # 2 rows x top-3 < 64 routed rows
    model(mx.array([[5]]), cache)
    assert model.prefill_exact_status()["sorted_weighted_sum"]["declined_layer_calls"] == {}
    model(mx.array([LONG[:40]]), model.make_cache())
    keys = set(model.prefill_exact_status()["sorted_weighted_sum"]["declined_layer_calls"])
    assert keys == {"not_metal"}


def test_sorted_combine_decline_keys_are_stable(tiny):
    config = tiny_config(n_routed_experts=32, num_experts_per_tok=16)
    model, _ = build(config, checkpoint(config))
    moe = model.model.layers[1].mlp
    import mlx2.runtime.models.qwen4_moe_weighted_sum as WS

    WS_runtime = WS.runtime_supported
    try:
        WS.runtime_supported = lambda: True
        # training (the test model is not .eval()'d): admission declines with a stable key.
        moe._sorted_combine(mx.zeros((1, 8, 24)), mx.zeros((1, 8, 16), dtype=mx.int32),
                            mx.ones((1, 8, 16)))
    finally:
        WS.runtime_supported = WS_runtime
    assert set(moe.sorted_sum_declines) == {"admission_training"}
    small = tiny_config(n_routed_experts=32, num_experts_per_tok=8)
    m2, _ = build(small, checkpoint(small))
    moe2 = m2.model.layers[1].mlp
    try:
        WS.runtime_supported = lambda: True
        moe2._sorted_combine(mx.zeros((1, 8, 24)), mx.zeros((1, 8, 8), dtype=mx.int32), mx.ones((1, 8, 8)))
    finally:
        WS.runtime_supported = WS_runtime
    assert set(moe2.sorted_sum_declines) == {"top_k_unsupported"}


def test_budget_charges_lanes_above_the_floor():
    from mlx2.adapters.hy_v4_flash_memory import HY4FlashCacheBudget

    a = H.ModelArgs.from_hf_config(tiny_config())
    b = HY4FlashCacheBudget.from_args(a, item_bytes=4, record_layers=0, prefill_head_rows=16)
    per_row = (a.vocab_size + a.hidden_size) * 4
    assert b.prefill_transient_parts(64, 40)["head_logits"] == 16 * per_row
    assert b.prefill_transient_parts(64, 40, lanes=4)["head_logits"] == 16 * per_row
    assert b.prefill_transient_parts(64, 40, lanes=24)["head_logits"] == 24 * per_row
    assert b.prefill_transient_parts(64, 3, lanes=2)["head_logits"] == 6 * per_row  # whole chunk
    assert b.prefill_transient_bytes(64, 40, lanes=24) > b.prefill_transient_bytes(64, 40)
    off = HY4FlashCacheBudget.from_args(a, item_bytes=4, record_layers=0)
    assert off.prefill_transient_parts(64, 40, lanes=2)["head_logits"] == 80 * per_row


def _greedy_from(model, cache, last, n=5):
    nxt, out = mx.array([[last]]), []
    for _ in range(n):
        nxt = mx.argmax(model(nxt, cache)[:, -1:], axis=-1).astype(mx.int32)
        out.append(int(nxt.item()))
    return out


def test_tail_after_apcv2_restore_is_bitwise(tiny):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    model, _ = _model(tiny)
    prefix, suffix = LONG[:48], LONG[48:88]
    model.prefill_tail_skip = False
    cache = model.make_cache()
    with prefill_checkpoint_rows([48], width=48):
        model(mx.array([prefix]), cache)
    apc = APCv2(max_size=4, layout_name="hy4-prefill-exact-test")
    key = APCKey("hy4-tiny")
    apc.store(key, prefix, cache)
    results = {}
    for tail in (True, False):
        model.prefill_tail_skip = tail
        hit = apc.lookup(key, prefix + suffix)
        assert hit.hit and hit.cached_tokens == 48
        before = model.prefill_exact_stats["tail_forwards"]
        with prefill_checkpoint_rows([40], width=40):
            model(mx.array([suffix]), hit.cache)
        results[tail] = (_states(hit.cache), model.prefill_exact_stats["tail_forwards"] - before,
                         _greedy_from(model, hit.cache, suffix[-1]))
    assert results[True][1] == 1 and results[False][1] == 0
    _assert_same_state(results[True][0], results[False][0])
    assert results[True][2] == results[False][2]


def test_tail_with_a_chunk_past_the_sparse_prefill_threshold(tiny, monkeypatch):
    # Past SPARSE_PREFILL_MIN_KEYS (monkeypatched from 4096) and GATHER_MAX_QUERY
    # rows, the full forward takes the sparse-prefill gate (on CPU a counted
    # dense fallback); the tail must leave the same state.
    monkeypatch.setattr(H, "SPARSE_PREFILL_MIN_KEYS", 64)
    model, _ = _model(tiny)
    model.set_sparse_prefill_attention(True)
    prompt = (LONG * 2)[:150]
    results = {}
    for tail in (True, False):
        model.prefill_tail_skip = tail
        cache = model.make_cache()
        for start, n in ((0, 70), (70, 80)):
            with prefill_checkpoint_rows([n], width=n):
                model(mx.array([prompt[start:start + n]]), cache)
        results[tail] = (_states(cache), _greedy_from(model, cache, prompt[-1]))
    declines = {}
    for layer in model.model.layers:
        for k, v in getattr(layer.self_attn, "sparse_prefill_declines", {}).items():
            declines[k] = declines.get(k, 0) + v
    assert declines.get("metal_unavailable", 0) > 0  # the gate was reached past the threshold
    _assert_same_state(results[True][0], results[False][0])
    assert results[True][1] == results[False][1]
    assert model.prefill_exact_stats["tail_forwards"] == 2
