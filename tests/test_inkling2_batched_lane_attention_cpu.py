"""``inkling2_batched_lane_attention`` (CPU, tiny Inkling 2).

One attention launch per visit across the lanes of a batched verify round.
Committed state must equal the per-lane path's exactly (the lanes' own
caches take the same appends); logits match the per-lane path to float
tolerance; padded rows are zero; the option defaults off.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

TOL = 1e-4


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _tiny(**kernels):
    import test_inkling2_cpu as t

    model = t._tiny_model()
    if kernels:
        model.configure_kernels(**kernels)
    return model


def _tokens(lane, n):
    return [(37 * lane + 11 * i + (i * i) % 13) % 500 + 3 for i in range(n)]


def _rows(model, prompts):
    rows = []
    for lane, p in enumerate(prompts):
        cache = model.make_cache()
        model(mx.array([_tokens(lane, 80)[:p]]), cache=cache)
        rows.append(cache)
    return rows


def _round(model, prompts, lengths, accepted, *, deferred=False):
    rows = _rows(model, prompts)
    txn = HybridVerifyRows(rows, deferred_commit=deferred, batched_rings=True).begin(list(lengths))
    width = max(lengths)
    blocks = [_tokens(i, 80)[p : p + n] + [0] * (width - n)
              for i, (p, n) in enumerate(zip(prompts, lengths))]
    logits = model(mx.array(blocks), cache=txn.caches)
    txn.commit(list(accepted))
    nxt = [model(mx.array([[_tokens(i, 80)[p + a]]]), cache=row)
           for i, (row, p, a) in enumerate(zip(rows, prompts, accepted))]
    return logits, rows, nxt


def test_defaults_off():
    from mlx2.adapters.inkling2 import validate_policy

    assert validate_policy(None)[0]["batched_lane_attention"] is False
    assert validate_policy({"inkling2_batched_lane_attention": True})[0]["batched_lane_attention"]


@pytest.mark.parametrize("prompts,lengths,accepted", [
    ((30, 5), (4, 4), (2, 4)),          # long (concatenated) ring + short lane
    ((30, 41, 12), (5, 1, 3), (3, 1, 2)),  # ragged widths, a one-token lane
    ((9, 30), (8, 8), (8, 1)),
])
@pytest.mark.parametrize("deferred", [False, True])
def test_batched_lane_attention_matches_per_lane_path(prompts, lengths, accepted, deferred):
    extra = {"verify_deferred_commit": True} if deferred else {}
    ref_model = _tiny(**extra)
    want_logits, want_rows, want_next = _round(ref_model, prompts, lengths, accepted,
                                               deferred=deferred)
    model = _tiny(batched_lane_attention=True, **extra)
    got_logits, got_rows, got_next = _round(model, prompts, lengths, accepted, deferred=deferred)
    for lane, n in enumerate(lengths):
        diff = mx.abs(got_logits[lane, :n] - want_logits[lane, :n]).max().item()
        assert diff < TOL, (lane, diff)
    for g, w in zip(got_next, want_next):
        assert mx.abs(g - w).max().item() < TOL
    for grow, wrow in zip(got_rows, want_rows):  # committed state: the same appends
        for g, w in zip(grow, wrow):
            assert getattr(g, "offset", None) == getattr(w, "offset", None)
    visits = sum(layer.self_attn.__dict__.get("segmented_lane_counts", {}).get("batched_visits", 0)
                 for layer in model.model.layers)
    assert visits == 40  # every sliding visit of the round took the batched launch


def _decoded_rows(model, lanes, prompt, extra):
    """Rows prefilled with ``prompt`` tokens then decoded one token at a time
    for ``extra`` tokens: the rings wrap and are no longer concatenated."""
    rows = []
    for lane in range(lanes):
        toks = _tokens(lane, prompt + extra + 40)
        cache = model.make_cache()
        model(mx.array([toks[:prompt]]), cache=cache)
        for t in toks[prompt : prompt + extra]:
            model(mx.array([[t]]), cache=cache)
        rows.append(cache)
    return rows


def _ring_contents(row):
    out = []
    for c in row:
        if getattr(c, "max_size", None) is not None and c.keys is not None:
            out.append(c._temporal_order(c.keys)[..., -min(c.offset, c.max_size):, :])
    return out


@pytest.mark.parametrize("deferred", [False, True])
def test_wrapped_rings_over_several_rounds_match_per_lane_contents(deferred):
    """Rings wrapped by one-token decode (physical order != time order), three
    back-to-back rounds with partial accepts: logits and the committed ring
    contents match the per-lane path to tolerance."""
    extra = {"verify_deferred_commit": True} if deferred else {}
    plans = [((4, 3, 5), (2, 3, 1)), ((5, 2, 4), (5, 1, 2)), ((3, 4, 2), (1, 4, 2))]
    results = []
    for batched in (False, True):
        model = _tiny(batched_lane_attention=batched, **extra) if batched or extra else _tiny()
        rows = _decoded_rows(model, 3, 6, 11)  # window 8: wrapped after 17 tokens
        owner = HybridVerifyRows(rows, deferred_commit=deferred, batched_rings=True)
        pos = [17, 17, 17]
        logits = []
        for lengths, accepted in plans:
            txn = owner.begin(list(lengths))
            width = max(lengths)
            blocks = [_tokens(i, 80)[p : p + n] + [0] * (width - n)
                      for i, (p, n) in enumerate(zip(pos, lengths))]
            logits.append(model(mx.array(blocks), cache=txn.caches))
            txn.commit(list(accepted))
            pos = [p + a for p, a in zip(pos, accepted)]
        results.append((logits, [_ring_contents(r) for r in rows], model))
    (want, want_rings, _), (got, got_rings, model) = results
    for (lengths, _), w, g in zip(plans, want, got):
        for lane, n in enumerate(lengths):
            assert mx.abs(g[lane, :n] - w[lane, :n]).max().item() < TOL
    for wr, gr in zip(want_rings, got_rings):
        assert len(wr) == len(gr) and wr
        for w, g in zip(wr, gr):
            assert w.shape == g.shape
            assert mx.abs(g.astype(mx.float32) - w.astype(mx.float32)).max().item() < TOL
    visits = sum(layer.self_attn.__dict__.get("segmented_lane_counts", {}).get("batched_visits", 0)
                 for layer in model.model.layers)
    assert visits == 40 * len(plans)
