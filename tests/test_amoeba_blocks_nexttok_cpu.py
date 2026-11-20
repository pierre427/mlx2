"""aMoEba next-token gating (opt-in): state, hysteresis, fail-open, prefill/speculative rules, verify +
recompute exactness against same-step gating, request fail-closed and receipts (CPU only)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.amoeba_blocks import layout as L
from mlx2.runtime.amoeba_blocks import nexttok as NT
from mlx2.runtime.amoeba_blocks import ops as O
from mlx2.runtime.amoeba_blocks import reference as R
from test_amoeba_blocks_cpu import _case, dense_wt, graph_small, packed, cpu_only  # noqa: F401


def _tables(rng, d=32):
    g = graph_small()
    pw = rng.standard_normal((len(g.boxes), d)).astype(np.float32)
    pb = rng.standard_normal(len(g.boxes)).astype(np.float32)
    return g, O.GateTables.from_graph(g, pw, pb)


def _gate(tables, seqs=1, **kw):
    cfg = NT.NextTokenConfig(**{k: kw.pop(k) for k in ("hysteresis", "fail_open_margin", "source") if k in kw})
    return NT.NextTokenGate(tables, seqs, cfg, gate_fn=NT.reference_tile_gates, **kw)


def test_reference_gate_law_matches_the_graph():
    rng = np.random.default_rng(0)
    g, t = _tables(rng)
    h = rng.standard_normal((9, 32)).astype(np.float32)
    raw = (h.astype(np.float64) @ np.array(t.probe_w, np.float64).T + np.array(t.probe_b)) > 0
    topen, un = NT.reference_tile_gates(mx.array(h), t, union=True)
    assert np.array_equal(np.array(topen), g.tile_open(raw))
    assert np.array_equal(np.array(un), g.tile_open(raw).max(0))
    bo = (rng.random((5, 4)) < 0.4).astype(np.uint8)
    assert np.array_equal(np.array(NT.reference_tile_gates(None, t, box_open=mx.array(bo))[0]), g.tile_open(bo))


def test_lagged_equals_same_step_when_the_topic_is_constant():
    """A constant topic: the same probe on step t's state predicts step t+1's same-step set exactly."""
    rng = np.random.default_rng(1)
    g, t = _tables(rng)
    topic = rng.standard_normal(32).astype(np.float32) * 4
    hs = topic[None] + 0.01 * rng.standard_normal((40, 32)).astype(np.float32)
    same = np.array(NT.reference_tile_gates(mx.array(hs), t)[0])
    assert (same == same[0]).all() and 0 < same[0].sum() < len(g.tiles)     # constant, non-trivial set
    gate = _gate(t)
    gate.prefill(0, mx.array(hs[:8]))
    for s in range(8, 40):
        assert np.array_equal(np.array(gate.current())[0], same[s])
        gate.advance(h_last=mx.array(hs[s:s + 1]))


def test_lag_is_exactly_one_step_at_a_topic_switch():
    """A topic switch: the prediction for step t+1 reads step t's state only, so the set switches exactly one
    step after the same-step set does (never earlier: no peeking at step t+1's own state)."""
    rng = np.random.default_rng(11)
    g, t = _tables(rng)
    a, b = None, None
    for _ in range(200):                                   # two topics with different non-trivial sets
        a, b = rng.standard_normal(32).astype(np.float32) * 4, rng.standard_normal(32).astype(np.float32) * 4
        sa = np.array(NT.reference_tile_gates(mx.array(a[None]), t)[0])[0]
        sb = np.array(NT.reference_tile_gates(mx.array(b[None]), t)[0])[0]
        if not np.array_equal(sa, sb):
            break
    hs = np.concatenate([np.repeat(a[None], 20, 0), np.repeat(b[None], 20, 0)])
    same = np.array(NT.reference_tile_gates(mx.array(hs), t)[0])
    gate = _gate(t)
    gate.prefill(0, mx.array(hs[:4]))
    gate.advance(h_last=mx.array(hs[3:4]))                 # the prefill's last op: predict step 4
    got = []
    for s in range(4, 40):
        got.append(np.array(gate.current())[0])
        gate.advance(h_last=mx.array(hs[s:s + 1]))
    got = np.array(got)
    assert np.array_equal(got[1:], same[4:39])               # step s uses step s-1's same-step set
    assert np.array_equal(got[16], same[19]) and np.array_equal(got[17], same[20])   # switch at 20 lands at 21
    assert not np.array_equal(got[16], same[20])


def test_hysteresis_opens_at_once_and_closes_after_n_steps_of_disuse():
    g = graph_small()
    t = O.GateTables.from_graph(g)
    probs = np.zeros((6, 4), np.float32)
    probs[1, 2] = 1.0                                   # token 1 -> box lang
    prior = NT.TokenPrior.from_probs(probs)
    lang = g.tiles.index("box:lang")
    for N in (0, 2, 5):
        gate = _gate(t, hysteresis=N, source="token", prior=prior)
        gate.prefill(0, box_open=mx.zeros((3, 4), mx.uint8))
        assert np.array(gate.current())[0].tolist() == [1, 0, 0, 0, 0, 0]
        seq = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        opened = []
        for tok in seq:
            opened.append(int(np.array(gate.advance(tokens=mx.array([tok])))[0, lang]))
        want = []
        last = None
        for i, tok in enumerate(seq):
            last = i if tok == 1 else last
            want.append(int(last is not None and i - last <= N))
        assert opened == want, (N, opened, want)
    gate = _gate(t, hysteresis=3, source="token", prior=prior)
    for _ in range(5):
        gate.advance(tokens=mx.array([0]), steps=1 << 19)      # ages saturate, never wrap
    assert int(np.array(gate.age)[:, 1:].min()) == 1 << 20 and np.array(gate.current())[0].tolist() == [1, 0, 0, 0, 0, 0]


def test_fail_open_opens_unsure_boxes_only():
    g = graph_small()
    d = 4
    pw = np.eye(4, d, dtype=np.float32)
    pb = np.zeros(4, np.float32)
    t = O.GateTables.from_graph(g, pw, pb)
    h = mx.array([[-0.1, -1.0, -0.1, -5.0]], mx.float32)       # sci and lang unsure, sci.phys and code closed
    plain = np.array(_gate(t).predict(h_last=h))[0]
    fo = np.array(_gate(t, fail_open_margin=0.2).predict(h_last=h))[0]
    assert plain.tolist() == [1, 0, 0, 0, 0, 0]
    want = g.tile_open(np.array([[1, 0, 1, 0]], bool))[0]
    assert fo.tolist() == want.tolist()
    with pytest.raises(ValueError):
        NT.NextTokenConfig(fail_open_margin=-0.1)
    with pytest.raises(O.BlockSparseUnsupported):
        NT.fail_open_tables(O.GateTables.from_graph(g), 0.2)


def test_prefill_union_speculative_union_rows_and_prefetch_hook():
    g = graph_small()
    t = O.GateTables.from_graph(g)
    probs = np.zeros((6, 4), np.float32)
    probs[1, 0] = probs[2, 2] = probs[3, 3] = 1.0
    prior = NT.TokenPrior.from_probs(probs)
    hook = NT.RecordingPrefetch()
    gate = _gate(t, seqs=2, hysteresis=1, source="token", prior=prior, prefetch=hook)
    bo = np.zeros((4, 4), np.uint8)
    bo[0, 0] = bo[3, 2] = 1                                     # the prompt opens sci then lang
    u = gate.prefill(0, box_open=mx.array(bo))
    want = g.tile_open(bo).max(0)
    assert np.array(u)[0].tolist() == want.tolist()
    assert np.array(gate.current())[0].tolist() == want.tolist()
    assert np.array(gate.current())[1].tolist() == [1, 0, 0, 0, 0, 0]   # the other row is untouched
    nxt = gate.advance(tokens=mx.array([3, 0]))
    assert hook.calls and hook.calls[-1][1] is nxt and hook.calls[-1][0] == [0, 1]   # the array itself
    code = g.tiles.index("pair:lang__code")
    assert int(np.array(nxt)[0, code]) == 1
    # one row only: the other row's age does not move
    before = np.array(gate.age)[1].copy()
    gate.advance(tokens=mx.array([2]), rows=[0])
    assert np.array_equal(np.array(gate.age)[1], before) and hook.calls[-1][0] == [0]
    # speculative: the verify pass uses the union of the current set and the drafts' predicted sets
    drafts = np.array(NT.reference_tile_gates(None, t, box_open=prior.box_open(mx.array([1, 3])))[0])
    su = np.array(gate.speculative_union(1, mx.array(drafts)))
    assert su.shape == (1, len(g.tiles))
    assert su[0].tolist() == np.maximum(np.array(gate.current())[1], drafts.max(0)).tolist()
    gate.reset([0])
    assert np.array(gate.current())[0].tolist() == [1, 0, 0, 0, 0, 0]
    with pytest.raises(O.BlockSparseUnsupported):
        gate.advance(tokens=mx.array([1]))                      # one prediction for two rows


def _mlp_dense(g, u, dn, lb, rows, per):
    gp, up = packed(g, lb), packed(u, lb)
    wt = dense_wt(rows, per)

    def mlp(x, topen, reads=None):
        return mx.array(R.emulate(np.array(x), gp, up, dn, lb, wt, np.array(topen), per_row=True, reads=reads)[1])
    return mlp, wt


@pytest.mark.parametrize("exact", [True, False])
def test_verify_recompute_is_bit_exact_against_same_step_dense(exact):
    rng = np.random.default_rng(5)
    g, u, dn, own, lb = _case(rng, E=1)
    rows = 12
    mlp, wt = _mlp_dense(g, u, dn, lb, rows, 8)
    x = mx.array(rng.standard_normal((rows, 128)).astype(np.float32))
    S = (rng.random((rows, lb.tiles)) < 0.4).astype(np.uint8)
    S[:, 0] = 1
    P = S.copy()
    P[1, 2] ^= 1                                     # flip: a missed open or an extra one
    P[4] = 1                                         # extra opens only
    S[6, 3], P[6, 3] = 1, 0                          # a missed open
    P[9, 1:] = 0
    S[9, 1:] = 1                                     # many missed opens
    y, redo = NT.verified(mlp, x, mx.array(P), mx.array(S), exact=exact)
    ref = mlp(x, mx.array(S))
    redo = np.array(redo)
    missed = (S.astype(bool) & ~P.astype(bool)).any(1)
    diff = (S != P).any(1)
    assert redo.tolist() == (diff if exact else missed).tolist()
    yn, rn = np.array(y), np.array(ref)
    if exact:
        assert np.array_equal(yn, rn)                # bit-exact against same-step gating, every row
    else:
        keep = ~missed
        assert np.array_equal(yn[missed], rn[missed])
        assert np.array_equal(yn[keep], np.array(mlp(x, mx.array(P)))[keep])
    # the recompute pass reads only for redone rows: an all-zero row adds no read
    _, t_redo = NT.recompute_rows(mx.array(P), mx.array(S), exact=exact)
    reads = R.Reads()
    mlp(x, t_redo, reads)
    pred = L.predicted_reads(lb, wt, np.array(t_redo), rmap=None, per_row=True, out_cols=32)
    assert pred["chunks"] == reads.chunk_reads
    only = np.zeros_like(S)
    only[np.array(redo)] = S[np.array(redo)]
    assert L.predicted_reads(lb, wt, only, rmap=None, per_row=True, out_cols=32) == pred


def test_verify_recompute_is_bit_exact_on_sorted_moe_routes():
    rng = np.random.default_rng(6)
    g, u, dn, own, lb = _case(rng)
    gp, up = packed(g, lb), packed(u, lb)
    tokens = 6
    experts = np.array([0, 1, 1, 0, 1, 0])
    order = np.argsort(experts, kind="stable")
    wt = np.array([[0, 3, 0, 0], [3, 3, 1, 0]])

    def mlp(x, topen):
        return mx.array(R.emulate(np.array(x), gp, up, dn, lb, wt, np.array(topen), rmap=order, omap=order,
                                  rows_out=tokens, per_row=True)[1])
    x = mx.array(rng.standard_normal((tokens, 128)).astype(np.float32))
    S = (rng.random((tokens, lb.tiles)) < 0.5).astype(np.uint8)
    S[:, 0] = 1
    P = S.copy()
    P[2, 1:] = 0
    P[5, 1:] = 1
    y, redo = NT.verified(mlp, x, mx.array(P), mx.array(S))
    assert np.array_equal(np.array(y), np.array(mlp(x, mx.array(S))))
    assert np.array(redo).tolist() == (S != P).any(1).tolist()


def test_verified_aligns_moe_shaped_outputs():
    x = mx.zeros((2, 3, 4))
    P = mx.array(np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]], np.uint8))
    S = mx.array(np.array([[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]], np.uint8))

    def mlp(x, t):                                   # [B, T, k, d] (the SwitchGLU contract), rows = B * T
        return mx.broadcast_to(t[:, 1].astype(mx.float32).reshape(2, 3, 1, 1), (2, 3, 2, 4))
    y, redo = NT.verified(mlp, x, P, S)
    assert np.array_equal(np.array(y), np.array(mlp(x, S)))
    with pytest.raises(O.BlockSparseUnsupported):
        NT.verified(mlp, x, P, S[:4])


def test_request_gate_mode_fails_closed_without_amoeba_blocks_and_receipts():
    assert NT.check_request_gate_mode(None, None) is None
    for m in NT.GATE_MODES:
        with pytest.raises(NT.GateModeUnavailable) as e:
            NT.check_request_gate_mode(None, m)
        assert e.value.http_status == 409
    with pytest.raises(ValueError):
        NT.check_request_gate_mode(None, "sometimes")

    class Route:
        gate_modes = ("same_step", "next_token")
    assert NT.check_request_gate_mode(Route(), "next_token") == "next_token"
    with pytest.raises(NT.GateModeUnavailable):
        NT.check_request_gate_mode(Route(), "next_token_verify")
    r = NT.receipt_fields(None)
    assert r["amoeba_gate_mode"] is None and r["amoeba_gate_observed"] is False
    assert r["amoeba_gate_mode_requested"] is None and r["amoeba_gate_qualified"] is False
    r = NT.receipt_fields("next_token")                      # asked, nothing observed: never echoed as run
    assert r["amoeba_gate_mode_requested"] == "next_token" and r["amoeba_gate_mode"] is None
    r = NT.receipt_fields("next_token_verify", {"executed_mode": "next_token_verify", "rows_recomputed": 3})
    assert r["amoeba_gate_mode"] == "next_token_verify" and r["amoeba_gate_observed"] is True
    assert r["amoeba_gate_stats"] == {"rows_recomputed": 3}
    with pytest.raises(ValueError):
        NT.receipt_fields(None, {"executed_mode": "bogus"})
    assert NT.DEFAULT_GATE_MODE == "same_step"              # opt-in: nothing selects next-token gating


def test_config_and_prior_guards():
    g = graph_small()
    t = O.GateTables.from_graph(g)
    with pytest.raises(O.BlockSparseUnsupported):
        NT.NextTokenGate(t, 1, NT.NextTokenConfig(source="token"))
    with pytest.raises(O.BlockSparseUnsupported):
        NT.NextTokenGate(t, 1, NT.NextTokenConfig(source="token"),
                         prior=NT.TokenPrior.from_probs(np.zeros((3, 2), np.float32)))
    with pytest.raises(ValueError):
        NT.NextTokenConfig(hysteresis=-1)
    with pytest.raises(ValueError):
        NT.NextTokenConfig(source="token", fail_open_margin=0.1)
    with pytest.raises(O.BlockSparseUnsupported):
        _gate(t).predict(h_last=None)

