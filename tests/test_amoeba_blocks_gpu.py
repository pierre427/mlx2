"""aMoEba block-sparse kernels on Metal (MLX2_RUN_GPU_TESTS=1; never part of the CPU suite).

Per backend this host has (mpp needs M5 tensor units; simd runs everywhere, M5 included):
* all open: the packed layout gives the unpacked (identity-table) layout's bits;
* gates closed: skipped == masked (SKIP=0, every block computed with zeros) bit for bit, and both
  equal the float64 masked reference within dtype tolerance;
* closed blocks are not read: device counters equal ``layout.predicted_reads`` (gate/up and
  down), and NaN poisoned into closed down groups' scales never reaches the output (a gate/up
  closed row is written as 0 whether or not it was computed, so poison cannot tell for gate/up:
  the counters are that proof);
* short chunks (1, 7, 31 rows) on the MPP extent match the reference;
* a row's output does not depend on the rows beside it (decode row == the same row in a batch);
* MoE (fewer than ops.PREFILL_MIN_ROWS tokens, one law): sorted and unsorted routes agree bit for
  bit; an expert with no open block yields zeros;
* tile gates: the kernel's open bits equal the numpy closure reference (and the union);
  the split kernels give the fused kernel's masks bit for bit; the host gate's router/token law too;
* prefill law (rows >= ops.PREFILL_MIN_ROWS): the same invariants (packed == unpacked, skip ==
  masked, counters == predicted reads, reference tolerance, row independence from neighbours),
  dense and sorted MoE, and its distance from the lane law.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
pytestmark = pytest.mark.skipif(not GPU or not mx.metal.is_available(),
                                reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")

from mlx2.runtime.amoeba_blocks import layout as L  # noqa: E402
from mlx2.runtime.amoeba_blocks import ops as O  # noqa: E402
from mlx2.runtime.amoeba_blocks import reference as R  # noqa: E402


def _backends():
    from mlx2.runtime.lane import matmul as lane
    return ["mpp", "simd"] if lane._tensor_units() else ["simd"]


@pytest.fixture(autouse=True)
def gpu():
    old = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(old)


def quantized(rng, E, N, Kd, bits):
    w = mx.array((rng.standard_normal((E, N, Kd)) * 0.05).astype(np.float32))
    q = mx.quantize(w, group_size=64, bits=bits)
    return O.QW(q[0], q[1].astype(mx.bfloat16), q[2].astype(mx.bfloat16), bits, 64)


def to_np(q: O.QW) -> R.QArrays:
    return R.QArrays(np.array(q.weight), np.array(q.scales.astype(mx.float32)),
                     np.array(q.biases.astype(mx.float32)), q.bits, q.group_size)


def case(rng, E, F, d, T, bits, core=0.6):
    own = rng.integers(1, T, size=(E, F)).astype(np.int32)
    own[rng.random((E, F)) < core] = 0
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    g, u, dn = quantized(rng, E, F, d, bits), quantized(rng, E, F, d, bits), quantized(rng, E, d, F, bits)
    return own, lb, g, u, dn


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 4, 37])
def test_dense_exactness_and_reads(backend, bits, rows):
    rng = np.random.default_rng(rows * 10 + bits)
    F, d, T = 512, 256, 6
    own, lb, g, u, dn = case(rng, 1, F, d, T, bits)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    ident = O.DeviceBlocks.from_layout(L.identity_layer(1, F, group_size=64, chunk=32, tiles=T))
    x = mx.array(rng.standard_normal((rows, d)).astype(np.float32)).astype(mx.bfloat16)
    ones = mx.ones((rows, T), mx.uint8)
    y_id, _ = O.block_mlp(x, g, u, dn, ident, ones, backend_name=backend)
    y_pk, _ = O.block_mlp(x, gp, up, dn, blocks, ones, backend_name=backend)
    assert mx.array_equal(y_id, y_pk).item(), "packed all-open != unpacked"
    topen_np = (rng.random((rows, T)) < 0.35).astype(np.uint8)
    topen_np[:, 0] = 1
    topen = mx.array(topen_np)
    y_sk, c_sk = O.block_mlp(x, gp, up, dn, blocks, topen, count=True, backend_name=backend)
    y_ms, _ = O.block_mlp(x, gp, up, dn, blocks, topen, skip=False, backend_name=backend)
    assert mx.array_equal(y_sk, y_ms).item(), "skipped != masked"
    ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
    got = np.array(y_sk.astype(mx.float32))
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    per = O.tile_rows(backend, rows, "dense")
    pred = L.predicted_reads(lb, np.array(O.dense_worktiles(rows, per)).reshape(-1, 4), topen_np, rmap=None,
                             per_row=True, out_cols=d, down_cols=O.down_cols(backend))
    assert O.counter_reads(c_sk) == pred
    # row invariance: each row alone gives its bits in the batch
    for r in (0, rows - 1):
        y1, _ = O.block_mlp(x[r:r + 1], gp, up, dn, blocks, topen[r:r + 1], backend_name=backend)
        assert mx.array_equal(y1[0], y_sk[r]).item()


@pytest.mark.parametrize("backend", _backends())
def test_closed_blocks_are_not_read_poison(backend):
    rng = np.random.default_rng(11)
    F, d, T = 512, 256, 4
    own, lb, g, u, dn = case(rng, 1, F, d, T, 4, core=0.5)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    topen_np = np.array([[1, 0, 1, 0]], np.uint8)
    topen = mx.array(topen_np)
    x = mx.array(rng.standard_normal((1, d)).astype(np.float32)).astype(mx.bfloat16)
    clean, _ = O.block_mlp(x, gp, up, dn, blocks, topen, backend_name=backend)
    # poison every closed chunk's gate/up scales and every closed down group's scales
    closed_rows = np.zeros(F, bool)
    for off, ln, t in lb.chunks:
        if not topen_np[0, t]:
            closed_rows[off:off + ln] = True
    nan = mx.array(np.float32(np.nan)).astype(mx.bfloat16)
    rows_mask = mx.array(closed_rows)[None, :, None]
    gs = O.QW(gp.weight, mx.where(rows_mask, nan, gp.scales), gp.biases, 4, 64)
    us = O.QW(up.weight, mx.where(rows_mask, nan, up.scales), up.biases, 4, 64)
    G = lb.groups
    closed_groups = np.array([not any(topen_np[0, t] for t in lb.gsig[lb.gsig_ptr[i]:lb.gsig_ptr[i + 1]])
                              for i in range(G)])
    assert closed_rows.any() and closed_groups.any(), "the case must have closed blocks"
    gm = mx.array(closed_groups)[None, None, :]
    ds = O.QW(dn.weight, mx.where(gm, nan, dn.scales), dn.biases, 4, 64)
    poisoned, _ = O.block_mlp(x, gs, us, ds, blocks, topen, backend_name=backend)
    assert mx.array_equal(clean, poisoned).item()
    masked, _ = O.block_mlp(x, gs, us, ds, blocks, topen, skip=False, backend_name=backend)
    assert np.isnan(np.array(masked.astype(mx.float32))).any(), "the masked arm reads them (control)"


@pytest.mark.parametrize("backend", _backends())
def test_short_chunks_match_reference(backend):
    rng = np.random.default_rng(14)
    F, d = 128, 256
    own = np.zeros(F, np.int32)
    own[:1] = 1                     # tile 1: 1 neuron, tile 2: 7, tile 3: 31, the core the rest
    own[1:8] = 2
    own[8:39] = 3
    rng.shuffle(own)
    lb = L.plan_layer(own, tiles=4, group_size=64, chunk=32)
    assert sorted(int(n) for _, n, t in lb.chunks if t) == [1, 7, 31]
    g, u, dn = quantized(rng, 1, F, d, 4), quantized(rng, 1, F, d, 4), quantized(rng, 1, d, F, 4)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.array(rng.standard_normal((3, d)).astype(np.float32)).astype(mx.bfloat16)
    for topen_np in (np.ones((3, 4), np.uint8), np.array([[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0]], np.uint8)):
        y, c = O.block_mlp(x, gp, up, dn, blocks, mx.array(topen_np), count=True, backend_name=backend)
        ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
        got = np.array(y.astype(mx.float32))
        assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
        pred = L.predicted_reads(lb, np.array(O.dense_worktiles(3, O.tile_rows(backend, 3, "dense"))).reshape(-1, 4),
                                 topen_np, rmap=None, per_row=True, out_cols=d, down_cols=O.down_cols(backend))
        assert O.counter_reads(c) == pred


@pytest.mark.parametrize("backend", _backends())
def test_moe_sorted_unsorted_and_empty_expert(backend):
    rng = np.random.default_rng(12)
    E, F, d, T, k = 6, 640, 256, 5, 3
    own, lb, g, u, dn = case(rng, E, F, d, T, 4)
    own[2] = rng.integers(1, T, size=F)                 # expert 2: no core
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    tokens = 30
    x = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    inds_np = np.stack([rng.choice(E, k, replace=False) for _ in range(tokens)]).astype(np.int32)
    inds_np[0] = [2, 0, 1]
    inds = mx.array(inds_np)
    topen_np = (rng.random((tokens, T)) < 0.4).astype(np.uint8)
    topen_np[:, 0] = 1
    topen_np[0, 1:] = 0                                 # token 0: core only, so expert 2 is empty
    topen = mx.array(topen_np)
    ys, cs = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=True, count=True, backend_name=backend)
    yu, cu = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=False, count=True, backend_name=backend)
    assert mx.array_equal(ys, yu).item(), "sorted != unsorted routes"
    assert not np.array(ys[0, 0].astype(mx.float32)).any()
    ym, _ = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, sort=True, skip=False, backend_name=backend)
    assert mx.array_equal(ys, ym).item()
    flat = inds_np.reshape(-1)
    ref = R.masked_reference(np.repeat(np.array(x.astype(mx.float32)), k, 0), to_np(g), to_np(u), to_np(dn),
                             own, topen_np, experts=flat, tokens=np.arange(tokens * k) // k)
    got = np.array(ys.astype(mx.float32)).reshape(tokens * k, d)
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    pred = L.predicted_reads(lb, np.array(O.pair_worktiles(inds)[0]).reshape(-1, 4), topen_np,
                             rmap=np.arange(tokens * k) // k, per_row=True, out_cols=d, down_cols=O.down_cols(backend))
    assert O.counter_reads(cu) == pred


def test_tile_gates_kernel():
    g = L.TileGraph(["sci", "sci.phys", "lang", "code"],
                    ["core", "box:sci", "box:sci.phys", "box:lang", "pair:lang__code", "joint"],
                    [(), (0,), (1,), (2,), (2, 3), (1, 2, 3)], parent=[-1, 0, -1, -1])
    rng = np.random.default_rng(13)
    d = 256
    pw = rng.standard_normal((4, d)).astype(np.float32)
    pb = rng.standard_normal(4).astype(np.float32)
    tables = O.GateTables.from_graph(g, pw, pb)
    h = mx.array(rng.standard_normal((9, d)).astype(np.float32)).astype(mx.bfloat16)
    topen, un = O.tile_gates(h, tables, union=True)
    logit = np.array(h.astype(mx.float32)).astype(np.float64) @ pw.T.astype(np.float64) + pb
    clear = np.abs(logit) > 1e-3                         # away from the sign boundary
    raw = logit > 0
    ref = g.tile_open(raw)
    got = np.array(topen)
    rows = clear.all(1)
    assert np.array_equal(got[rows], ref[rows])
    assert np.array_equal(np.array(un).astype(bool), got.any(0))
    box = mx.array(raw.astype(np.uint8))
    t2, _ = O.tile_gates(h, tables, box_open=box)
    assert np.array_equal(np.array(t2), ref)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("moe", [False, True])
def test_nexttok_verify_recompute_bitwise_equals_same_step(backend, moe):
    """Next-token gating with verify: the predicted pass, the recompute of disagreeing rows and the merge
    give the same-step output bit for bit with the same kernels (row invariance on Metal)."""
    from mlx2.runtime.amoeba_blocks import nexttok as NT
    rng = np.random.default_rng(21 + moe)
    E, F, d, T = (6, 640, 256, 5) if moe else (1, 512, 256, 5)
    own, lb, g, u, dn = case(rng, E, F, d, T, 4)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    tokens = 12
    x = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    inds = mx.array(np.stack([rng.choice(E, 3, replace=False) for _ in range(tokens)]).astype(np.int32))
    S = (rng.random((tokens, T)) < 0.4).astype(np.uint8)
    S[:, 0] = 1
    P = S.copy()
    P[[1, 5, 9]] = 1 - P[[1, 5, 9]]
    P[:, 0] = 1

    def mlp(xx, t):
        if moe:
            return O.block_switch_glu(xx, inds, gp, up, dn, blocks, t, backend_name=backend)[0]
        return O.block_mlp(xx, gp, up, dn, blocks, t, backend_name=backend)[0]
    for exact in (True, False):
        y, redo = NT.verified(mlp, x, mx.array(P), mx.array(S), exact=exact)
        ref = mlp(x, mx.array(S))
        r = np.array(redo)
        if exact:
            assert mx.array_equal(y, ref).item()
        else:
            assert mx.array_equal(y[mx.array(np.flatnonzero(r))], ref[mx.array(np.flatnonzero(r))]).item()



# ---------------------------------------------------------------------------
# Prefill law and split gates (2026-11-19)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [65, 130])
def test_dense_prefill_law_exactness_and_reads(backend, bits, rows):
    rng = np.random.default_rng(100 + rows + bits)
    F, d, T = 512, 256, 6
    own, lb, g, u, dn = case(rng, 1, F, d, T, bits)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    ident = O.DeviceBlocks.from_layout(L.identity_layer(1, F, group_size=64, chunk=32, tiles=T))
    x = mx.array(rng.standard_normal((rows, d)).astype(np.float32)).astype(mx.bfloat16)
    assert O.select_law(backend, rows, d) == "prefill"
    ones = mx.ones((rows, T), mx.uint8)
    y_id, _ = O.block_mlp(x, g, u, dn, ident, ones, backend_name=backend)
    y_pk, _ = O.block_mlp(x, gp, up, dn, blocks, ones, backend_name=backend)
    assert mx.array_equal(y_id, y_pk).item(), "packed all-open != unpacked (prefill law)"
    topen_np = (rng.random((rows, T)) < 0.35).astype(np.uint8)
    topen_np[:, 0] = 1
    topen_np[:64, 1:] = 0                               # one work tile with only the core: groups skip
    topen = mx.array(topen_np)
    y_sk, c_sk = O.block_mlp(x, gp, up, dn, blocks, topen, count=True, backend_name=backend)
    y_ms, _ = O.block_mlp(x, gp, up, dn, blocks, topen, skip=False, backend_name=backend)
    assert mx.array_equal(y_sk, y_ms).item(), "skipped != masked (prefill law)"
    ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
    got = np.array(y_sk.astype(mx.float32))
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    pred = L.predicted_reads(lb, np.array(O.dense_worktiles(rows, 64)).reshape(-1, 4), topen_np, rmap=None,
                             per_row=True, out_cols=d, down_cols=O.down_cols(backend, "prefill"))
    assert O.counter_reads(c_sk) == pred
    # a row's bits do not depend on its neighbours (same law, other batch)
    perm = np.r_[rows - 1, np.arange(rows - 1)]
    y_p, _ = O.block_mlp(x[mx.array(perm)], gp, up, dn, blocks, topen[mx.array(perm)], backend_name=backend)
    assert mx.array_equal(y_p[0], y_sk[rows - 1]).item()
    # the lane law on the same rows: different bits, same function
    y_ln, _ = O.block_mlp(x, gp, up, dn, blocks, topen, backend_name=backend, law="lane")
    lane = np.array(y_ln.astype(mx.float32))
    assert np.abs(got - lane).max() <= 0.02 * np.abs(lane).max() + 1e-3


@pytest.mark.parametrize("backend", _backends())
def test_moe_prefill_law_sorted(backend):
    rng = np.random.default_rng(120)
    E, F, d, T, k = 6, 640, 256, 5, 3
    own, lb, g, u, dn = case(rng, E, F, d, T, 4)
    own[2] = rng.integers(1, T, size=F)
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    ident = O.DeviceBlocks.from_layout(L.identity_layer(E, F, group_size=64, chunk=32, tiles=T))
    tokens = 90
    x = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    inds_np = np.stack([rng.choice(E, k, replace=False) for _ in range(tokens)]).astype(np.int32)
    inds = mx.array(inds_np)
    ones = mx.ones((tokens, T), mx.uint8)
    yi, _ = O.block_switch_glu(x, inds, g, u, dn, ident, ones, backend_name=backend)
    yp, _ = O.block_switch_glu(x, inds, gp, up, dn, blocks, ones, backend_name=backend)
    assert mx.array_equal(yi, yp).item()
    topen_np = (rng.random((tokens, T)) < 0.4).astype(np.uint8)
    topen_np[:, 0] = 1
    topen = mx.array(topen_np)
    ys, cs = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, count=True, backend_name=backend)
    ym, _ = O.block_switch_glu(x, inds, gp, up, dn, blocks, topen, skip=False, backend_name=backend)
    assert mx.array_equal(ys, ym).item()
    flat = inds_np.reshape(-1)
    ref = R.masked_reference(np.repeat(np.array(x.astype(mx.float32)), k, 0), to_np(g), to_np(u), to_np(dn),
                             own, topen_np, experts=flat, tokens=np.arange(tokens * k) // k)
    got = np.array(ys.astype(mx.float32)).reshape(tokens * k, d)
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    wt, rmap, _, _, _ = O.sorted_worktiles(inds, E, 64)
    pred = L.predicted_reads(lb, np.array(wt).reshape(-1, 4), topen_np, rmap=np.array(rmap), per_row=True,
                             out_cols=d, down_cols=O.down_cols(backend, "prefill"))
    assert O.counter_reads(cs) == pred


def _gate_graph(rng, boxes=79, tiles=60):
    names = [f"b{i}" for i in range(boxes)]
    members = [()] + [tuple(sorted(rng.choice(boxes, rng.integers(1, 4), replace=False).tolist()))
                      for _ in range(tiles - 1)]
    return L.TileGraph(names, ["core"] + [f"t{i}" for i in range(1, tiles)], members, parent=[-1] * boxes)


@pytest.mark.parametrize("tokens", [1, 8, 9, 37])
def test_split_gates_equal_fused_bitwise(tokens):
    rng = np.random.default_rng(130 + tokens)
    g = _gate_graph(rng)
    d = 2560
    nb = len(g.boxes)
    tables = O.GateTables.from_graph(g, rng.standard_normal((nb, d)).astype(np.float32) / np.sqrt(d),
                                     (rng.standard_normal(nb) * 0.01).astype(np.float32))
    h = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    f_t, f_u = O.tile_gates(h, tables, union=True, impl="fused")
    s_t, s_u = O.tile_gates(h, tables, union=True, impl="split")
    assert mx.array_equal(f_t, s_t).item() and mx.array_equal(f_u, s_u).item()
    box = mx.array((rng.random((tokens, nb)) < 0.05).astype(np.uint8))
    f2, _ = O.tile_gates(h, tables, box_open=box, impl="fused")
    s2, _ = O.tile_gates(h, tables, box_open=box, impl="split")
    assert mx.array_equal(f2, s2).item()
    host = O.HostGateTables.from_tables(tables)
    h2, _ = O.host_tile_gates(host, box_open=np.array(box))
    assert np.array_equal(h2, np.array(f2))



@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("moe", [False, True])
def test_prefill_law_closed_blocks_are_not_read_poison(backend, moe):
    """NaN in every closed down group's scales (and closed gate/up chunks' scales) never reaches the
    output on the prefill law; the masked arm (which reads them) is the control."""
    rng = np.random.default_rng(140 + moe)
    E, F, d, T = (4, 512, 256, 4) if moe else (1, 512, 256, 4)
    own, lb, g, u, dn = case(rng, E, F, d, T, 4, core=0.5)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    tokens = 100 if moe else 70
    topen_np = np.array([[1, 0, 1, 0]], np.uint8)
    topen = mx.array(topen_np)
    x = mx.array(rng.standard_normal((tokens, d)).astype(np.float32)).astype(mx.bfloat16)
    inds = mx.array(np.stack([rng.choice(E, 2, replace=False) for _ in range(tokens)]).astype(np.int32))

    def run(gq, uq, dq, skip=True):
        if moe:
            return O.block_switch_glu(x, inds, gq, uq, dq, blocks, topen, per_row=False, skip=skip, backend_name=backend,
                                      law="prefill")[0]
        return O.block_mlp(x, gq, uq, dq, blocks, topen, per_row=False, skip=skip, backend_name=backend)[0]
    if not moe:
        assert O.select_law(backend, tokens, d) == "prefill"
    clean = run(gp, up, dn)
    nan = mx.array(np.float32(np.nan)).astype(mx.bfloat16)
    rows = np.zeros((E, F), bool)
    for e in range(E):
        for off, ln, t in lb.chunks[lb.chunk_ptr[e]:lb.chunk_ptr[e + 1]]:
            if not topen_np[0, t]:
                rows[e, off:off + ln] = True
    rm = mx.array(rows)[:, :, None]
    G = lb.groups
    closed = np.array([[not any(topen_np[0, t] for t in lb.gsig[lb.gsig_ptr[e * G + i]:lb.gsig_ptr[e * G + i + 1]])
                        for i in range(G)] for e in range(E)])
    assert rows.any() and closed.any(), "the case must have closed blocks"
    gm = mx.array(closed)[:, None, :]
    gs_ = O.QW(gp.weight, mx.where(rm, nan, gp.scales), gp.biases, 4, 64)
    us_ = O.QW(up.weight, mx.where(rm, nan, up.scales), up.biases, 4, 64)
    ds_ = O.QW(dn.weight, mx.where(gm, nan, dn.scales), dn.biases, 4, 64)
    assert mx.array_equal(clean, run(gs_, us_, ds_)).item()
    assert np.isnan(np.array(run(gs_, us_, ds_, skip=False).astype(mx.float32))).any(), "control reads them"


def test_simd_prefill_law_output_width_not_a_multiple_of_64():
    """simd prefill (K steps of 32): d = 288, so down's last 64-column tile is partly past N."""
    rng = np.random.default_rng(150)
    F, d, T, rows = 256, 288, 4, 80
    own, lb, g, u, dn = case(rng, 1, F, d, T, 4)
    gp, up = O.pack_qweight(g, lb), O.pack_qweight(u, lb)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.array(rng.standard_normal((rows, d)).astype(np.float32)).astype(mx.bfloat16)
    topen_np = (rng.random((rows, T)) < 0.4).astype(np.uint8)
    topen_np[:, 0] = 1
    y, c = O.block_mlp(x, gp, up, dn, blocks, mx.array(topen_np), count=True, backend_name="simd")
    ym, _ = O.block_mlp(x, gp, up, dn, blocks, mx.array(topen_np), skip=False, backend_name="simd")
    assert mx.array_equal(y, ym).item()
    ref = R.masked_reference(np.array(x.astype(mx.float32)), to_np(g), to_np(u), to_np(dn), own, topen_np)
    got = np.array(y.astype(mx.float32))
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max() + 1e-3
    pred = L.predicted_reads(lb, np.array(O.dense_worktiles(rows, 64)).reshape(-1, 4), topen_np, rmap=None,
                             per_row=True, out_cols=d, down_cols=O.down_cols("simd", "prefill"))
    assert O.counter_reads(c) == pred
