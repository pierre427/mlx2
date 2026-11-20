"""aMoEba block-sparse MLP kernels: layout, gates, law emulation, plumbing, offline compile (CPU only)."""

import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.amoeba_blocks import kernels as K
from mlx2.runtime.amoeba_blocks import layout as L
from mlx2.runtime.amoeba_blocks import ops as O
from mlx2.runtime.amoeba_blocks import reference as R


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")))
    yield
    mx.set_default_device(old)


def rand_q(rng, E, N, Kd, bits=4, gs=64, scale=0.02):
    q = rng.integers(0, 1 << bits, size=(E, N, Kd), dtype=np.uint32)
    s = (rng.standard_normal((E, N, Kd // gs)) * scale).astype(np.float32)
    b = (-s * (1 << (bits - 1))).astype(np.float32)
    return R.QArrays(R.pack_codes(q, bits), s, b, bits, gs)


def packed(p: R.QArrays, lb: L.LayerBlocks) -> R.QArrays:
    return R.QArrays(L.pack_rows(p.weight, lb.perm), L.pack_rows(p.scales, lb.perm),
                     L.pack_rows(p.biases, lb.perm), p.bits, p.group_size)


def graph_small():
    # boxes: 0 sci, 1 sci.phys (child of sci), 2 lang, 3 code
    boxes = ["sci", "sci.phys", "lang", "code"]
    tiles = ["core", "box:sci", "box:sci.phys", "box:lang", "pair:lang__code", "joint:sci.phys+lang+code"]
    members = [(), (0,), (1,), (2,), (2, 3), (1, 2, 3)]
    return L.TileGraph(boxes, tiles, members, parent=[-1, 0, -1, -1])


def owners(rng, E, F, tiles, core_share=0.6):
    own = rng.integers(1, tiles, size=(E, F)).astype(np.int32)
    own[rng.random((E, F)) < core_share] = 0
    return own


def dense_wt(rows, per):
    starts = np.arange(0, rows, per)
    return np.stack([starts, np.minimum(per, rows - starts), 0 * starts, 0 * starts], 1)


# ---------------------------------------------------------------------------
# Tile graph
# ---------------------------------------------------------------------------


def test_closure_or_and_lca_rules():
    g = graph_small()
    m = g.tile_box_matrix()
    # box:sci opens when sci or its child sci.phys opens (parent gated in with its subtiles)
    assert m[1].tolist() == [True, True, False, False]
    # the child does not open when only the parent opens
    assert m[2].tolist() == [False, True, False, False]
    # pair and joint tiles: OR of their members (joint member sci.phys has no descendants)
    assert m[4].tolist() == [False, False, True, True]
    assert m[5].tolist() == [False, True, True, True]
    assert not m[0].any()
    raw = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], bool)
    to = g.tile_open(raw)
    assert to[0].tolist() == [1, 1, 1, 0, 0, 1]
    assert to[1].tolist() == [1, 0, 0, 0, 1, 1]
    assert to[2].tolist() == [1, 0, 0, 0, 0, 0]
    bits = g.tile_box_bits()
    for t in range(len(g.tiles)):
        got = [(int(bits[t, c // 32]) >> (c % 32)) & 1 for c in range(len(g.boxes))]
        assert got == m[t].astype(int).tolist()


def test_graph_rejects_bad_structure():
    with pytest.raises(ValueError):
        L.TileGraph(["a"], ["core", "box:a"], [(), ()])            # gated tile without members
    with pytest.raises(ValueError):
        L.TileGraph(["a", "b"], ["core"], [()], parent=[1, 0])     # parent cycle
    with pytest.raises(ValueError):
        L.TileGraph(["a"], ["box:a"], [(0,)])                      # tile 0 must be the core


def test_graph_from_partition_names():
    names = ["core", "box:arb", "box:fra", "pair:arb__fra", "fam:lang"]
    g = L.graph_from_partition_names(names, groups={"fam:lang": ["arb", "fra", "deu"]},
                                     parents={"fra": "deu"})
    assert g.boxes == ["arb", "fra", "deu"]
    assert g.members[3] == (0, 1) and g.members[4] == (0, 1, 2)
    m = g.tile_box_matrix()
    assert m[4].all()                       # family tile: any member opens it
    with pytest.raises(ValueError):
        L.graph_from_partition_names(["core", "fam:x"])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [16, 32])
def test_plan_layer_partition_and_exact_round_trip(chunk):
    rng = np.random.default_rng(0)
    E, F, T = 3, 256, 6
    own = owners(rng, E, F, T)
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=chunk)
    lb.validate()
    assert np.array_equal(lb.owner(), own)
    for e in range(E):
        ch = lb.chunks[lb.chunk_ptr[e]:lb.chunk_ptr[e + 1]]
        assert ch[0, 2] == 0                           # core first
        assert (np.diff(ch[:, 2]) >= 0).all()          # each tile one contiguous run
        assert (ch[:, 1] <= chunk).all()
        # stable: inside a tile the original order is kept
        for t in range(T):
            rows = lb.perm[e][own[e][lb.perm[e]] == t].astype(int)
            assert (np.diff(rows) > 0).all()
    p = rand_q(rng, E, F, 128)
    for a in (p.weight, p.scales, p.biases):
        pk = L.pack_rows(a, lb.perm)
        assert pk.dtype == a.dtype and np.array_equal(L.unpack_rows(pk, lb.perm), a)
        assert not np.array_equal(pk, a)
    inv = lb.inverse()
    for e in range(E):
        assert np.array_equal(lb.perm[e][inv[e]], np.arange(F))
    # a dense layer ([F] owner) is one expert
    lbd = L.plan_layer(own[0], tiles=T, group_size=64, chunk=chunk)
    assert lbd.experts == 1 and np.array_equal(L.unpack_rows(L.pack_rows(p.weight[0], lbd.perm), lbd.perm), p.weight[0])
    # tables survive a save/load through their arrays and digest
    again = L.LayerBlocks.from_arrays(lb.meta(), lb.arrays())
    assert again.digest() == lb.digest()


def test_plan_layer_rejects():
    with pytest.raises(ValueError):
        L.plan_layer(np.zeros((1, 100), np.int32), tiles=1, group_size=64)
    with pytest.raises(ValueError):
        L.plan_layer(np.full((1, 64), 3, np.int32), tiles=2, group_size=64)
    lb = L.plan_layer(np.zeros((1, 64), np.int32), tiles=1, group_size=64)
    lb.gsig = lb.gsig + 1
    lb.tiles = 2
    with pytest.raises(ValueError):
        lb.validate()


def test_owner_from_masks():
    a = np.array([[1, 0, 0], [0, 1, 0]], bool)
    b = ~a
    own = L.owner_from_masks({"core": a, "box:x": b}, ["core", "box:x"])
    assert own.tolist() == [[0, 1, 1], [1, 0, 1]]
    with pytest.raises(ValueError):
        L.owner_from_masks({"core": a, "box:x": a}, ["core", "box:x"])


# ---------------------------------------------------------------------------
# Law emulation: packed == unpacked, skipped == masked, closed blocks never read
# ---------------------------------------------------------------------------


def _case(rng, E=2, F=256, d=128, T=6, bits=4):
    g, u, dn = rand_q(rng, E, F, d, bits), rand_q(rng, E, F, d, bits), rand_q(rng, E, d, F, bits)
    own = owners(rng, E, F, T)
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    return g, u, dn, own, lb


LAW_TOL = {"lane": (2e-4, 2e-5), "prefill": (2e-2, 1e-2)}     # prefill: weights rounded to bf16


@pytest.mark.parametrize("law", ["lane", "prefill"])
@pytest.mark.parametrize("bits", [4, 8])
def test_packed_all_open_equals_unpacked_bitwise(bits, law):
    rng = np.random.default_rng(1)
    g, u, dn, own, lb = _case(rng, E=1, bits=bits)
    x = rng.standard_normal((5, 128)).astype(np.float32)
    wt = dense_wt(5, 16 if law == "lane" else 64)
    topen = np.ones((5, lb.tiles), np.uint8)
    ident = L.identity_layer(1, lb.width, group_size=64, chunk=32, tiles=lb.tiles)
    a0, y0 = R.emulate(x, g, u, dn, ident, wt, topen, law=law)
    a1, y1 = R.emulate(x, packed(g, lb), packed(u, lb), dn, lb, wt, topen, law=law)
    assert np.array_equal(a0, a1) and np.array_equal(y0, y1)
    ref = R.masked_reference(x, g, u, dn, own, topen)
    rt, at = LAW_TOL[law]
    assert np.allclose(y1, ref, rtol=rt, atol=at * np.abs(ref).max())


@pytest.mark.parametrize("law", ["lane", "prefill"])
@pytest.mark.parametrize("per_row", [True, False])
def test_skipped_equals_masked_and_never_reads_closed(per_row, law):
    rng = np.random.default_rng(2)
    g, u, dn, own, lb = _case(rng)
    gp, up = packed(g, lb), packed(u, lb)
    rows = 6
    x = rng.standard_normal((rows, 128)).astype(np.float32)
    nrow = rows if per_row else 1
    topen = (rng.random((nrow, lb.tiles)) < 0.3).astype(np.uint8)
    topen[:, 0] = 1
    if per_row:
        topen[3, 1:] = 0                     # one row with only the core
    experts = np.array([0, 1, 1, 0, 1, 0])
    # sorted routes: rows of one expert contiguous, work row r -> token rmap[r], output omap[r]
    order = np.argsort(experts, kind="stable")
    wt = np.array([[0, 3, 0, 0], [3, 3, 1, 0]])
    kw = dict(rmap=order, omap=order, rows_out=rows, per_row=per_row, law=law)
    reads = R.Reads()
    a1, y1 = R.emulate(x, gp, up, dn, lb, wt, topen, skip=True, reads=reads, **kw)
    a0, y0 = R.emulate(x, gp, up, dn, lb, wt, topen, skip=False, **kw)
    assert np.array_equal(a1, a0) and np.array_equal(y1, y0)
    assert not np.isnan(y1).any()
    # nothing closed for every row of a work tile was read
    tok_rows = [order[0:3], order[3:6]]
    for ti, (row0, n, e, _) in enumerate(wt):
        toks = tok_rows[ti] if per_row else [0]
        for off, ln, t in lb.chunks[lb.chunk_ptr[e]:lb.chunk_ptr[e + 1]]:
            opened = any(topen[k, t] for k in toks)
            touched = any((e, p) in reads.gate_up_rows for p in range(off, off + ln))
            assert touched == bool(opened)
        G = lb.groups
        for gi in range(G):
            sig = lb.gsig[lb.gsig_ptr[e * G + gi]:lb.gsig_ptr[e * G + gi + 1]]
            opened = any(topen[k, t] for k in toks for t in sig)
            assert ((e, gi, ti) in reads.down_groups) == opened
    pred = L.predicted_reads(lb, wt, topen, rmap=order, per_row=per_row, out_cols=128)
    assert pred == {"chunks": reads.chunk_reads, "gate_up_rows": reads.rows_read,
                    "down_group_tiles": reads.down_group_tiles}
    # each work row equals its own masked reference (zeroed neurons), whatever its tile-mates are
    toks = order if per_row else np.zeros(rows, int)
    ref = R.masked_reference(x[order], g, u, dn, own, topen, experts=experts[order], tokens=toks)
    rt, at = LAW_TOL[law]
    assert np.allclose(y1[order], ref, rtol=rt, atol=at * np.abs(ref).max())


def test_expert_with_no_open_block_is_not_read():
    rng = np.random.default_rng(3)
    E, F, d, T = 2, 128, 128, 3
    own = np.zeros((E, F), np.int32)
    own[1] = rng.integers(1, T, size=F)              # expert 1 has no core neurons
    lb = L.plan_layer(own, tiles=T, group_size=64, chunk=32)
    g, u, dn = rand_q(rng, E, F, d), rand_q(rng, E, F, d), rand_q(rng, E, d, F)
    x = rng.standard_normal((1, d)).astype(np.float32)
    topen = np.array([[1, 0, 0]], np.uint8)
    reads = R.Reads()
    _, y = R.emulate(x, packed(g, lb), packed(u, lb), dn, lb, np.array([[0, 1, 1, 0]]), topen,
                     rmap=np.array([0]), reads=reads)
    assert reads.chunk_reads == 0 and reads.down_group_tiles == 0
    assert np.array_equal(y, np.zeros_like(y))


def test_predicted_bytes():
    reads = {"chunks": 3, "gate_up_rows": 80, "down_group_tiles": 10}
    b = L.bytes_from_reads(reads, d=2560, bits=4, group_size=64)
    row = 2560 // 2 + 40 * 4
    assert b["gate_up_bytes"] == 2 * 80 * row
    assert b["down_bytes"] == 10 * 32 * (32 + 4)


# ---------------------------------------------------------------------------
# Device-op helpers on the CPU stream (no kernels)
# ---------------------------------------------------------------------------


def test_sorted_and_pair_worktiles():
    rng = np.random.default_rng(4)
    T, k, E = 37, 3, 9
    inds = np.stack([rng.choice(E, k, replace=False) for _ in range(T)]).astype(np.int32)
    wt, rmap, omap, p, bound = O.sorted_worktiles(mx.array(inds), E, 8)
    wt = np.array(wt).reshape(-1, 4)
    rmap, omap = np.array(rmap), np.array(omap)
    flat = inds.reshape(-1)
    assert p == T * k and wt.shape[0] == bound
    seen = np.zeros(p, int)
    for row0, n, e, _ in wt:
        if n == 0:
            continue
        assert 1 <= n <= 8
        rows = np.arange(row0, row0 + n)
        assert (flat[omap[rows]] == e).all()          # one expert per tile
        assert (rmap[rows] == omap[rows] // k).all()
        seen[rows] += 1
    assert (seen == 1).all()
    assert sorted(omap.tolist()) == list(range(p))
    wt2, rmap2, p2 = O.pair_worktiles(mx.array(inds))
    wt2 = np.array(wt2).reshape(-1, 4)
    assert (wt2[:, 1] == 1).all() and (wt2[:, 2] == flat).all() and (np.array(rmap2) == np.arange(p) // k).all()


def test_pack_qweight_round_trip_bf16():
    rng = np.random.default_rng(5)
    E, F, d = 2, 128, 64
    w = mx.array(rng.integers(0, 2**32, size=(E, F, d // 8), dtype=np.uint64).astype(np.uint32))
    s = mx.array(rng.standard_normal((E, F, d // 64)).astype(np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((E, F, d // 64)).astype(np.float32)).astype(mx.bfloat16)
    q = O.QW(w, s, b, 4, 64)
    lb = L.plan_layer(owners(rng, E, F, 4), tiles=4, group_size=64)
    pk = O.pack_qweight(q, lb)
    back = O.unpack_qweight(pk, lb)
    for x0, x1 in ((w, back.weight), (s, back.scales), (b, back.biases)):
        assert mx.array_equal(x0, x1).item()
    assert np.array_equal(np.array(pk.weight), L.pack_rows(np.array(w), lb.perm))


# ---------------------------------------------------------------------------
# Launch plumbing with a recording fake kernel (no Metal): grids, templates, zero copies
# ---------------------------------------------------------------------------


class _Rec:
    def __init__(self, kind):
        self.kind, self.calls = kind, []

    def __call__(self, **kw):
        out = [mx.zeros(s, d) for s, d in zip(kw["output_shapes"], kw["output_dtypes"])]
        self.calls.append(kw)
        self.outputs = out
        return out


@pytest.fixture
def fake(monkeypatch):
    recs = {}

    def build(kind, backend, variant, bits, xt, st):
        return recs.setdefault((kind, backend, variant), _Rec(kind))

    monkeypatch.setattr(K, "build", build)
    return recs


def _mx_q(p: R.QArrays):
    return O.QW(mx.array(p.weight), mx.array(p.scales).astype(mx.bfloat16),
                mx.array(p.biases).astype(mx.bfloat16), p.bits, p.group_size)


@pytest.mark.parametrize("backend", ["mpp", "simd"])
def test_dense_launch_plumbing_reads_weights_in_place(fake, monkeypatch, backend):
    monkeypatch.setattr(O, "backend", lambda name=None: backend)
    rng = np.random.default_rng(6)
    g, u, dn, own, lb = _case(rng, E=1, F=256, d=256)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.array(rng.standard_normal((3, 256)).astype(np.float32)).astype(mx.bfloat16)
    topen = mx.ones((3, lb.tiles), mx.uint8)
    y, ctr = O.block_mlp(x, gq, uq, dq, blocks, topen)
    assert y.shape == (3, 256) and ctr is None
    gu, dnk = fake[("gu", backend, "dense")].calls[0], fake[("dn", backend, "dense")].calls[0]
    # every weight, scale and table is the caller's own array object: nothing gathered or copied
    for got, want in zip(gu["inputs"][1:10], [gq.weight, gq.scales, gq.biases, uq.weight, uq.scales, uq.biases,
                                             blocks.perm, blocks.chunk_ptr, blocks.chunks]):
        assert got is want
    for got, want in zip(dnk["inputs"][1:6], [dq.weight, dq.scales, dq.biases, blocks.gsig_ptr, blocks.gsig]):
        assert got is want
    assert dnk["inputs"][0] is fake[("gu", backend, "dense")].outputs[0]   # the activation, in place
    tm = dict(gu["template"])
    assert tm["SKIP"] == 1 and tm["MS"] == 1 and tm["COUNT"] == 0
    assert gu["init_value"] is None and dnk["init_value"] is None      # no memset of the outputs
    assert gu["grid"][1] == 1 and gu["grid"][0] == lb.max_chunks * gu["threadgroup"][0]
    assert gu["output_shapes"][0] == (3, 256)
    if backend == "mpp":
        assert tm["TMR"] == 1 and dict(dnk["template"])["TMR"] == 1
        assert dnk["grid"][0] == (256 // 32) * dnk["threadgroup"][0]
    else:
        assert tm["MR"] == 3 and dnk["grid"][0] == (256 // 8) * 64


@pytest.mark.parametrize("backend", ["mpp", "simd"])
@pytest.mark.parametrize("sort", [False, True])
def test_moe_launch_plumbing(fake, monkeypatch, backend, sort):
    monkeypatch.setattr(O, "backend", lambda name=None: backend)
    rng = np.random.default_rng(7)
    g, u, dn, own, lb = _case(rng, E=4, F=128, d=128)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    T, k = 5, 2
    x = mx.array(rng.standard_normal((T, 128)).astype(np.float32)).astype(mx.bfloat16)
    inds = mx.array(np.stack([rng.choice(4, k, replace=False) for _ in range(T)]).astype(np.int32))
    y, _ = O.block_switch_glu(x, inds, gq, uq, dq, blocks, mx.ones((T, lb.tiles), mx.uint8), sort=sort)
    assert y.shape == (T, k, 128)
    gu, dnk = fake[("gu", backend, "rowmap")].calls[0], fake[("dn", backend, "rowmap")].calls[0]
    assert gu["inputs"][0].shape == (T, 128)             # x itself, not x gathered per pair
    assert gu["output_shapes"][0] == (T * k, 128)
    assert dnk["output_shapes"][0] == (T * k, 128)       # written in pair order: no unsort copy
    assert dict(dnk["template"])["OUTMAP"] == int(sort)
    if not sort:
        assert gu["grid"][1] == T * k


def test_launch_refuses_bad_inputs(fake, monkeypatch):
    monkeypatch.setattr(O, "backend", lambda name=None: "mpp")
    rng = np.random.default_rng(8)
    g, u, dn, own, lb = _case(rng, E=1)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.zeros((2, 128), mx.bfloat16)
    with pytest.raises(O.BlockSparseUnsupported):
        O.block_mlp(x, gq, uq, dq, blocks, mx.ones((1, lb.tiles), mx.uint8))           # per_row needs 2 rows
    with pytest.raises(O.BlockSparseUnsupported):
        O.block_mlp(x.astype(mx.float32), gq, uq, dq, blocks, mx.ones((2, lb.tiles), mx.uint8))
    bad = O.QW(gq.weight, gq.scales, gq.biases, 3, 64)
    with pytest.raises(O.BlockSparseUnsupported):
        O.block_mlp(x, bad, uq, dq, blocks, mx.ones((2, lb.tiles), mx.uint8))


def test_split_k_is_shape_only_and_within_budget():
    for (n, k, gs, bits) in [(640, 2560, 64, 4), (2560, 640, 64, 4), (17408, 5120, 64, 8), (5120, 17408, 64, 8)]:
        for variant in ("dense", "rowmap"):
            for kind in ("gu", "dn"):
                sk = O.split_k(n, k, gs, bits, variant=variant, kind=kind)
                assert sk in (1, 2, 4, 8)
                assert K.threadgroup_bytes(kind, "mpp", variant, sk=sk, tmr=O.max_tmr(variant), gs=gs,
                                           groups=k // gs) <= K.TG_MEMORY_BUDGET


# ---------------------------------------------------------------------------
# Offline Metal compilation (xcrun; the GPU is never used)
# ---------------------------------------------------------------------------


def _compile_cases():
    out = []
    for b in K.BACKENDS:
        for v in K.VARIANTS:
            for bits in K.BITS:
                base = dict(GS=64, NTILES=416, MS=1, SKIP=1, COUNT=1)
                out.append(("gu", b, v, bits, dict(base, K=2560, F=640, SK=4, TMR=2, MR=8)))
                out.append(("dn", b, v, bits, dict(base, N=2560, F=640, SK=2, TMR=2, MR=8, OUTMAP=1)))
    out.append(("gate", "simd", "dense", 4, dict(D=2560, NB=79, NTILES=416, NW=3, MODE=0, UNION=1)))
    for b in K.BACKENDS:
        for v in K.VARIANTS:
            for bits in K.BITS:
                base = dict(GS=64, NTILES=416, MS=1, SKIP=1, COUNT=1)
                out.append(("gup", b, v, bits, dict(base, K=2560, F=640)))
                out.append(("dnp", b, v, bits, dict(base, N=2560, F=640, OUTMAP=1)))
    for b in K.BACKENDS:
        for gs in (32, 128):
            c = dict(GS=gs, NTILES=2, MS=0, SKIP=0, COUNT=0)
            out.append(("dnp", b, "dense", 8, dict(c, N=5120, F=17408, OUTMAP=0)))
            out.append(("gup", b, "rowmap", 4, dict(c, K=5120, F=17408)))
    for mode in (0, 1):
        out.append(("gbox", "simd", "dense", 4, dict(D=2560, NB=79, NW=3, MODE=mode, TT=K.GATE_TT)))
    out.append(("gtile", "simd", "dense", 4, dict(NTILES=416, NW=3)))
    return out


@pytest.mark.parametrize("kind,backend,variant,bits,consts", _compile_cases())
def test_offline_metal_compile(tmp_path, kind, backend, variant, bits, consts):
    if not shutil.which("xcrun"):
        pytest.skip("compiler unavailable")
    for xt in ("bfloat", "half"):
        src = tmp_path / f"{kind}_{xt}.metal"
        src.write_text(K.compilation_source(kind, backend, variant, bits, consts, xt=xt, st=xt))
        r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-I",
                            str(Path(mx.__file__).parent / "include"), "-c", str(src), "-o",
                            str(tmp_path / f"{kind}_{xt}.air")], capture_output=True, text=True, timeout=120)
        if r.returncode and "unable to find utility" in r.stderr:
            pytest.skip("metal toolchain unavailable")
        assert r.returncode == 0, r.stderr[-4000:]


def test_kernel_names_carry_the_source_hash():
    a = K.kernel_name("gu", "mpp", "dense", 4, "bfloat", "bfloat")
    b = K.kernel_name("gu", "mpp", "rowmap", 4, "bfloat", "bfloat")
    assert a != b and a.startswith("mlx2_amoeba_blocks_gu_mpp_dense_q4")


def test_gate_tables_and_probe_law_reference():
    g = graph_small()
    rng = np.random.default_rng(9)
    d = 64
    pw = rng.standard_normal((4, d)).astype(np.float32)
    pb = rng.standard_normal(4).astype(np.float32)
    tables = O.GateTables.from_graph(g, pw, pb)
    assert tables.words == 1 and tables.tiles == len(g.tiles)
    h = rng.standard_normal((7, d)).astype(np.float32)
    raw = (h.astype(np.float64) @ pw.T.astype(np.float64) + pb) > 0
    ref = g.tile_open(raw)
    # the kernel's open rule: always | (bits & raw) != 0
    bits = np.array(tables.tile_bits).reshape(len(g.tiles), 1)
    word = (raw.astype(np.uint32) << np.arange(4, dtype=np.uint32)).sum(1).astype(np.uint32)
    got = ((bits[None, :, 0] & word[:, None]) != 0) | (np.array(tables.always)[None] != 0)
    assert np.array_equal(got.astype(np.uint8), ref)
    with pytest.raises(O.BlockSparseUnsupported):
        O.GateTables.from_graph(g, pw[:3], pb[:3])


# ---------------------------------------------------------------------------
# Memory-copy audit: static tables cached, router ids reinterpreted in place, no needless memsets
# ---------------------------------------------------------------------------


def _ptr(a):
    mx.eval(a)
    return np.asarray(memoryview(a)).ctypes.data


def _old_sorted_worktiles(indices, num_experts, per_tile):
    # the pre-audit construction, kept as the bit-exact reference
    k = int(indices.shape[-1])
    flat = indices.reshape(-1).astype(mx.int32)
    p = int(flat.shape[0])
    order = mx.argsort(flat).astype(mx.int32)
    s = flat[order]
    idx = mx.arange(p, dtype=mx.int32)
    start = mx.concatenate([mx.array([True]), s[1:] != s[:-1]]) if p > 1 else mx.array([True])
    run0 = mx.cummax(mx.where(start, idx, mx.zeros_like(idx)))
    tstart = mx.logical_or(start, ((idx - run0) % per_tile) == 0)
    tid = mx.cumsum(tstart.astype(mx.int32)) - 1
    bound = -(-p // per_tile) + min(num_experts, p)
    key = mx.where(tstart, tid, mx.full(tid.shape, bound, mx.int32))
    zeros = mx.zeros((bound + 1,), mx.int32)
    row0 = zeros.at[key].add(mx.where(tstart, idx, mx.zeros_like(idx)))
    nrows = zeros.at[tid].add(mx.ones_like(idx))
    expert = zeros.at[key].add(mx.where(tstart, s, mx.zeros_like(s)))
    wt = mx.stack([row0, nrows, expert, zeros], axis=1)[:bound].reshape(-1)
    return wt, order // k, order, p, bound


def test_dense_worktiles_uploaded_once(fake, monkeypatch):
    monkeypatch.setattr(O, "backend", lambda name=None: "mpp")
    assert O.dense_worktiles(5, 4) is O.dense_worktiles(5, 4)
    assert np.array_equal(np.array(O.dense_worktiles(70, 64)).reshape(-1, 4), dense_wt(70, 64))
    rng = np.random.default_rng(10)
    g, u, dn, own, lb = _case(rng, E=1, F=256, d=256)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    x = mx.zeros((3, 256), mx.bfloat16)
    topen = mx.ones((3, lb.tiles), mx.uint8)
    O.block_mlp(x, gq, uq, dq, blocks, topen)
    O.block_mlp(x, gq, uq, dq, blocks, topen)
    gu, dnk = fake[("gu", "mpp", "dense")].calls, fake[("dn", "mpp", "dense")].calls
    assert gu[0]["inputs"][10] is gu[1]["inputs"][10] is dnk[0]["inputs"][6] is dnk[1]["inputs"][6]


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32])
def test_pair_worktiles_no_id_copy_and_static_columns_reused(dtype):
    rng = np.random.default_rng(11)
    inds = mx.array(np.stack([rng.choice(512, 10, replace=False) for _ in range(4)]).astype(np.int32)).astype(dtype)
    mx.eval(inds)
    flat = O._flat_i32(inds)
    assert flat.dtype == mx.int32 and _ptr(flat) == _ptr(inds)       # reinterpreted, not converted
    wt, rmap, p = O.pair_worktiles(inds)
    wt2, rmap2, _ = O.pair_worktiles(inds)
    assert rmap is rmap2                                               # cached per (pairs, k)
    want = np.stack([np.arange(p), np.ones(p), np.array(inds).reshape(-1), np.zeros(p)], 1).astype(np.int32)
    assert np.array_equal(np.array(wt).reshape(-1, 4), want)
    assert np.array_equal(np.array(rmap), np.arange(p) // 10)


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32])
@pytest.mark.parametrize("T,per", [(1, 8), (37, 8), (205, 64)])
def test_sorted_worktiles_bit_exact_after_copy_removal(dtype, T, per):
    rng = np.random.default_rng(12)
    k, E = 10, 64
    inds = mx.array(np.stack([rng.choice(E, k, replace=False) for _ in range(T)]).astype(np.int32)).astype(dtype)
    new, old = O.sorted_worktiles(inds, E, per), _old_sorted_worktiles(inds, E, per)
    assert new[3:] == old[3:]
    for a, b in zip(new[:3], old[:3]):
        assert a.dtype == b.dtype == mx.int32 and mx.array_equal(a, b).item()


def test_tile_gates_memset_only_for_the_union(fake):
    g = graph_small()
    rng = np.random.default_rng(13)
    tables = O.GateTables.from_graph(g, rng.standard_normal((4, 64)), rng.standard_normal(4))
    h = mx.zeros((3, 64), mx.bfloat16)
    topen, un = O.tile_gates(h, tables, impl="fused")
    assert un is None and topen.shape == (3, tables.tiles)
    _, un = O.tile_gates(h, tables, union=True, impl="fused")
    calls = fake[("gate", "simd", "dense")].calls
    assert calls[0]["init_value"] is None and calls[1]["init_value"] == 0 and un is not None


def test_rowmap_staging_zeroes_pad_rows_once():
    src = K.source("gu", "mpp", "rowmap", 4)
    loop = src.index("for (int g = g_begin; g < g_end; g++)")
    pad = src.index("mya[i] = static_cast<XT>(0.0f)")
    assert pad < loop                                  # pad rows zeroed before the group loop
    assert "i < nst" in src[loop:] and "i < RT * GS" not in src[loop:]


def test_worktiles_bit_exact_on_a_strided_router_output():
    # argpartition(...)[..., :k] is uint32 and not row-contiguous: the real router output
    logits = mx.array(np.random.default_rng(14).standard_normal((300, 64)).astype(np.float32))
    inds = mx.argpartition(-logits, kth=9, axis=-1)[..., :10]
    assert inds.dtype == mx.uint32
    new, old = O.sorted_worktiles(inds, 64, 32), _old_sorted_worktiles(inds, 64, 32)
    for a, b in zip(new[:3], old[:3]):
        assert mx.array_equal(a, b).item()
    wt, _, p = O.pair_worktiles(inds)
    assert np.array_equal(np.array(wt).reshape(-1, 4)[:, 2], np.array(inds).reshape(-1).astype(np.int32))


# ---------------------------------------------------------------------------
# Review fixes: refusals, graph parents, tables on disk
# ---------------------------------------------------------------------------


def test_refuses_chunks_wider_than_a_threadgroup_and_mixed_gate_up(fake, monkeypatch):
    monkeypatch.setattr(O, "backend", lambda name=None: "mpp")
    rng = np.random.default_rng(20)
    g, u, dn = rand_q(rng, 1, 256, 128), rand_q(rng, 1, 256, 128), rand_q(rng, 1, 128, 256)
    own = owners(rng, 1, 256, 4)
    x = mx.zeros((2, 128), mx.bfloat16)
    topen = mx.ones((2, 4), mx.uint8)
    wide = L.plan_layer(own, tiles=4, group_size=64, chunk=64)
    with pytest.raises(O.BlockSparseUnsupported, match="chunks"):
        O.block_mlp(x, _mx_q(g), _mx_q(u), _mx_q(dn), O.DeviceBlocks.from_layout(wide), topen)
    lb = L.plan_layer(own, tiles=4, group_size=64, chunk=32)
    u8 = rand_q(rng, 1, 256, 128, bits=8)
    with pytest.raises(O.BlockSparseUnsupported, match="share bits"):
        O.block_mlp(x, _mx_q(g), _mx_q(u8), _mx_q(dn), O.DeviceBlocks.from_layout(lb), topen)
    assert not fake                                   # refused before any kernel was built


def test_graph_parents_may_name_new_boxes_and_chains():
    g = L.graph_from_partition_names(["core", "box:phys"], parents={"phys": "sci", "sci": "root"})
    assert g.boxes == ["phys", "sci", "root"] and g.parent == [1, 2, -1]
    m = g.tile_box_matrix()
    assert m[1].tolist() == [True, False, False]       # box:phys opens only with phys
    g2 = L.graph_from_partition_names(["core", "box:a"], parents={"newchild": "a"})
    assert g2.tile_box_matrix()[1].tolist() == [True, True]   # a child of a opens a's tile


def test_tables_round_trip_on_disk(tmp_path):
    rng = np.random.default_rng(21)
    g = graph_small()
    layers = {3: L.plan_layer(owners(rng, 4, 128, len(g.tiles)), tiles=len(g.tiles), group_size=64),
              7: L.plan_layer(owners(rng, 4, 128, len(g.tiles)), tiles=len(g.tiles), group_size=64)}
    L.save_tables(tmp_path, layers, g, source={"checkpoint": "test@0"})
    back, g2, man = L.load_tables(tmp_path)
    assert g2.digest() == g.digest() and set(back) == {3, 7}
    for l in layers:
        assert back[l].digest() == layers[l].digest()
        assert np.array_equal(back[l].perm, layers[l].perm)
    man["graph"]["boxes"][0] = "other"
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(man))
    with pytest.raises(ValueError):
        L.load_tables(tmp_path)


def test_output_width_must_be_a_multiple_of_four():
    rng = np.random.default_rng(22)
    q = _mx_q(rand_q(rng, 1, 64, 128))
    O.QW(q.weight, q.scales, q.biases, 4, 64).check()
    bad = O.QW(q.weight[:, :62], q.scales[:, :62], q.biases[:, :62], 4, 64)
    with pytest.raises(O.BlockSparseUnsupported, match="multiple of 4"):
        bad.check()



# ---------------------------------------------------------------------------
# Prefill law (2026-11-19): law selection, plumbing, budget
# ---------------------------------------------------------------------------


def test_select_law_is_shape_only():
    assert O.select_law("mpp", 1, 2560) == "lane"
    assert O.select_law("mpp", 64, 2560) == "lane"           # verify widths keep the lane law
    assert O.select_law("mpp", 65, 2560) == "prefill"
    assert O.select_law("simd", 2048, 5120) == "prefill"
    assert O.select_law("mpp", 2048, 2560, sorted_rows=False) == "lane"   # pair tiles: one row each
    assert O.select_law("mpp", 2048, 96) == "lane"           # K not a multiple of the mpp K step
    assert O.select_law("simd", 2048, 96) == "prefill"
    assert O.select_law("mpp", 3, 2560, law="prefill") == "prefill"
    with pytest.raises(O.BlockSparseUnsupported):
        O.select_law("mpp", 2048, 96, law="prefill")       # pinned, cannot run: refused, not replaced
    with pytest.raises(O.BlockSparseUnsupported):
        O.select_law("mpp", 2048, 2560, law="nope")
    # MoE: rows per expert (pairs / experts) must reach PREFILL_MIN_MEAN_ROWS
    assert O.select_law("mpp", 2048, 2560, mean_rows=2048 * 10 / 512) == "prefill"
    assert O.select_law("mpp", 128, 2560, mean_rows=128 * 10 / 512) == "lane"
    assert O.select_law("mpp", 128, 2560, mean_rows=2.5, law="prefill") == "prefill"   # pinned


@pytest.mark.parametrize("backend", ["mpp", "simd"])
def test_dense_prefill_plumbing(fake, monkeypatch, backend):
    monkeypatch.setattr(O, "backend", lambda name=None: backend)
    rng = np.random.default_rng(31)
    g, u, dn, own, lb = _case(rng, E=1, F=256, d=256)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    rows = 130
    x = mx.zeros((rows, 256), mx.bfloat16)
    y, _ = O.block_mlp(x, gq, uq, dq, blocks, mx.ones((rows, lb.tiles), mx.uint8))
    assert y.shape == (rows, 256)
    gu, dnk = fake[("gup", backend, "dense")].calls[0], fake[("dnp", backend, "dense")].calls[0]
    for got, want in zip(gu["inputs"][1:10], [gq.weight, gq.scales, gq.biases, uq.weight, uq.scales, uq.biases,
                                             blocks.perm, blocks.chunk_ptr, blocks.chunks]):
        assert got is want                                  # weights read in place
    assert dnk["inputs"][0] is fake[("gup", backend, "dense")].outputs[0]
    assert gu["threadgroup"] == (128, 1, 1) and dnk["threadgroup"] == (128, 1, 1)
    assert gu["grid"] == (lb.max_chunks * 128, 3, 1)       # 130 rows -> three 64-row work tiles
    assert dnk["grid"] == ((256 // 64) * 128, 3, 1)
    assert gu["init_value"] is None and dnk["init_value"] is None
    wt = np.array(gu["inputs"][10]).reshape(-1, 4)
    assert wt[:, 1].tolist() == [64, 64, 2]
    tm = dict(gu["template"])
    assert "SK" not in tm and "TMR" not in tm and "MR" not in tm
    # 64 rows stays on the lane law
    O.block_mlp(x[:64], gq, uq, dq, blocks, mx.ones((64, lb.tiles), mx.uint8))
    assert fake[("gu", backend, "dense")].calls


@pytest.mark.parametrize("backend", ["mpp", "simd"])
def test_moe_prefill_plumbing(fake, monkeypatch, backend):
    monkeypatch.setattr(O, "backend", lambda name=None: backend)
    rng = np.random.default_rng(32)
    g, u, dn, own, lb = _case(rng, E=4, F=128, d=128)
    gq, uq, dq = _mx_q(g), _mx_q(u), _mx_q(dn)
    blocks = O.DeviceBlocks.from_layout(lb)
    T, k = 70, 2                                            # 140 pairs / 4 experts: 35 rows per expert
    x = mx.zeros((T, 128), mx.bfloat16)
    inds = mx.array(np.stack([rng.choice(4, k, replace=False) for _ in range(T)]).astype(np.int32))
    y, _ = O.block_switch_glu(x, inds, gq, uq, dq, blocks, mx.ones((T, lb.tiles), mx.uint8))
    assert y.shape == (T, k, 128)
    gu, dnk = fake[("gup", backend, "rowmap")].calls[0], fake[("dnp", backend, "rowmap")].calls[0]
    assert gu["inputs"][0].shape == (T, 128)                # x itself: rows staged through RMAP
    assert dict(dnk["template"])["OUTMAP"] == 1
    bound = -(-(T * k) // 64) + 4
    assert gu["grid"][1] == bound and dnk["grid"][1] == bound
    with pytest.raises(O.BlockSparseUnsupported):
        O.block_switch_glu(x, inds, gq, uq, dq, blocks, mx.ones((T, lb.tiles), mx.uint8), sort=False,
                           law="prefill")


def test_prefill_threadgroup_memory_within_the_hardware_limit():
    for backend in K.BACKENDS:
        for variant in K.VARIANTS:
            assert K.threadgroup_bytes("gup", backend, variant, sk=1, tmr=1, gs=64) <= 32 * 1024
            for gs in K.GROUP_SIZES:
                assert K.threadgroup_bytes("dnp", backend, variant, sk=1, tmr=1, gs=gs, groups=17408 // gs) <= 32 * 1024


def test_prefill_sorted_worktiles_cover_every_pair_once():
    rng = np.random.default_rng(33)
    T, k, E = 300, 10, 64
    inds = mx.array(np.stack([rng.choice(E, k, replace=False) for _ in range(T)]).astype(np.int32))
    wt, rmap, omap, p, bound = O.sorted_worktiles(inds, E, K.PF_BM)
    wt = np.array(wt).reshape(-1, 4)
    assert wt[:, 1].max() <= 64 and wt[:, 1].sum() == T * k
    flat = np.array(inds).reshape(-1)
    om = np.array(omap)
    for row0, n, e, _ in wt:
        if n:
            assert (flat[om[row0:row0 + n]] == e).all()


# ---------------------------------------------------------------------------
# Tile gates: split kernel plumbing and the host gate (2026-11-19)
# ---------------------------------------------------------------------------


def test_split_gate_plumbing(fake):
    g = graph_small()
    rng = np.random.default_rng(34)
    tables = O.GateTables.from_graph(g, rng.standard_normal((4, 64)), rng.standard_normal(4))
    h = mx.zeros((19, 64), mx.bfloat16)
    O.tile_gates(h, tables, union=True)                        # auto, one word: fused
    assert fake[("gate", "simd", "dense")].calls and ("gbox", "simd", "dense") not in fake
    topen, un = O.tile_gates(h, tables, union=True, impl="split")
    box, tile = fake[("gbox", "simd", "dense")].calls[0], fake[("gtile", "simd", "dense")].calls[0]
    assert O.DEFAULT_GATE_IMPL == "auto"
    assert O.gate_impl(1, 1) == "fused" and O.gate_impl(1, 4) == "split" and O.gate_impl(65, 4) == "fused"
    assert O.gate_impl(2048, 13, "split") == "split"
    assert box["grid"] == (256 * tables.words, -(-19 // K.GATE_TT), 1) and box["init_value"] is None
    assert box["output_shapes"] == [(19, tables.words)]
    assert tile["inputs"][0] is fake[("gbox", "simd", "dense")].outputs[0]
    assert tile["inputs"][1] is tables.tile_bits and tile["inputs"][2] is tables.always
    assert tile["grid"] == (19 * tables.tiles, 1, 1) and tile["init_value"] is None
    assert un is not None and topen.shape == (19, tables.tiles)
    with pytest.raises(O.BlockSparseUnsupported):
        O.tile_gates(h, tables, box_open=mx.zeros((19, 3), mx.uint8))


def _graph_many(rng, boxes=79, tiles=60):
    names = [f"b{i}" for i in range(boxes)]
    members = [()] + [tuple(sorted(rng.choice(boxes, rng.integers(1, 4), replace=False).tolist()))
                      for _ in range(tiles - 1)]
    parent = [-1] * boxes
    for i in range(1, boxes, 7):
        parent[i] = i - 1
    return L.TileGraph(names, ["core"] + [f"t{i}" for i in range(1, tiles)], members, parent=parent)


def test_box_words_bit_order():
    raw = np.zeros((2, 70), bool)
    raw[0, [0, 31, 32, 69]] = True
    raw[1, 5] = True
    w = O.box_words(raw, 3)
    assert w.dtype == np.uint32 and w.shape == (2, 3)
    assert w[0].tolist() == [1 | (1 << 31), 1, 1 << 5] and w[1].tolist() == [1 << 5, 0, 0]


def test_host_gate_masks_equal_the_reference_law():
    from mlx2.runtime.amoeba_blocks import nexttok as NT
    rng = np.random.default_rng(35)
    g = _graph_many(rng)
    d = 96
    pw = rng.standard_normal((len(g.boxes), d)).astype(np.float32)
    pb = rng.standard_normal(len(g.boxes)).astype(np.float32)
    tables = O.GateTables.from_graph(g, pw, pb)
    prior = (rng.random((50, len(g.boxes))) < 0.05).astype(np.uint8)
    host = O.HostGateTables.from_tables(tables, prior)
    # router law: bit for bit
    box = (rng.random((17, len(g.boxes))) < 0.1).astype(np.uint8)
    t_ref, u_ref = NT.reference_tile_gates(mx.zeros((17, d), mx.bfloat16), tables, box_open=mx.array(box), union=True)
    t_host, u_host = O.host_tile_gates(host, box_open=box, union=True)
    assert np.array_equal(t_host, np.array(t_ref)) and np.array_equal(u_host, np.array(u_ref))
    assert np.array_equal(t_host, g.tile_open(box.astype(bool)))
    # token law: a row gather of the prior, then the same closure
    toks = rng.integers(0, 50, size=9)
    t_tok, _ = O.host_tile_gates(host, tokens=toks)
    assert np.array_equal(t_tok, g.tile_open(prior[toks].astype(bool)))
    # probe law: equal wherever no logit is within rounding of zero
    h = rng.standard_normal((11, d)).astype(np.float32)
    t_p, _ = O.host_tile_gates(host, h=h)
    logit = h.astype(np.float64) @ pw.T.astype(np.float64) + pb
    clear = (np.abs(logit) > 1e-3).all(1)
    assert clear.sum() >= 5
    assert np.array_equal(t_p[clear], g.tile_open(logit > 0)[clear])
    with pytest.raises(O.BlockSparseUnsupported):
        O.host_tile_gates(host, tokens=toks, box_open=box)
