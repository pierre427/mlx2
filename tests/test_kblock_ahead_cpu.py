# SPDX-License-Identifier: Apache-2.0
"""K-block decode load-ahead (glm53_row_kernel=kblock_rows_ahead2, D2/WP2).

AHEAD=2 issues chunk c+32's block loads (ka_load) before chunk c's arithmetic
(ka_dot, kblock_dot32_v2's expressions) with each lane's ascending chunk
order unchanged, so every launch must equal its AHEAD=1 launch bit for bit.

CPU tests (always): the ``ahead`` keyword on all six entry points, the
effective AHEAD template per law/format, refusals, the generated sources and
an offline Metal compile, and the provider policy's propagation to every
K-block call site including memo keys. Metal tests run only with
MLX2_RUN_GPU_TESTS=1 (under the GPU lease): random packed blocks of Q4_0,
Q3_K, Q4_K, Q5_K and Q6_K with negative, subnormal, zero, large and
non-finite fp16 scales; dense (ROWS 1/2/4, padded rows), grouped, multi,
routed MoE, shared-fold MoE and union, each AHEAD=2 vs AHEAD=1 bitwise.
"""

import os
import shutil
import subprocess
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import kblock_metal as km
from mlx2.runtime.gguf import weights as weights_module
from mlx2.runtime.gguf.k_blocks import SIZES
from mlx2.runtime.gguf.packed_metal import PackedMatrix
from mlx2.runtime.gguf.reader import GGUFError, GGUFTensor
from mlx2.runtime.gguf.weights import GGUFWeights

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
Q4_0, Q3_K, Q4_K, Q5_K, Q6_K = 2, 11, 12, 13, 14
K_QUANTS = (Q3_K, Q4_K, Q5_K, Q6_K)
D_FIELDS = {Q4_0: (0,), Q3_K: (108,), Q4_K: (0, 2), Q5_K: (0, 2), Q6_K: (208,)}


def _blocks(rng, ggml_type, rows, k, nonfinite=False):
    block, nbytes = SIZES[ggml_type]
    count = rows * (k // block)
    raw = rng.integers(0, 256, size=(count, nbytes), dtype=np.uint8)
    for offset in D_FIELDS[ggml_type]:
        d = rng.choice([1e-3, 2e-2, 0.5, 7.0], size=count) * rng.choice([-1.0, 1.0], size=count)
        d[::17] = 6e-8  # fp16 subnormal
        d[5::23] = 6.0e4
        d[3::29] = 0.0
        if nonfinite:
            d[7::31] = np.inf
            d[11::37] = -np.inf
        raw[:, offset:offset + 2] = d.astype(np.float16).view(np.uint8).reshape(count, 2)
    return raw.reshape(-1)


def _packed(rng, ggml_type, shape, nonfinite=False):
    """PackedMatrix of numpy shape ``shape`` ([experts,] out, in)."""
    rows = int(np.prod(shape[:-1]))
    data = _blocks(rng, ggml_type, rows, shape[-1], nonfinite)
    tensor = GGUFTensor(name="synthetic", shape=tuple(shape[::-1]), ggml_type=ggml_type, offset=0,
                        nbytes=data.size)
    return PackedMatrix(mx.array(data), tensor)


def _spec(ggml_type, shape):
    """A zero-filled PackedMatrix of numpy shape ``shape`` (CPU wiring tests)."""
    block, nbytes = SIZES[ggml_type]
    size = int(np.prod(shape[:-1])) * (shape[-1] // block) * nbytes
    tensor = GGUFTensor(name="spec", shape=tuple(shape[::-1]), ggml_type=ggml_type, offset=0, nbytes=size)
    return PackedMatrix(mx.zeros((size,), dtype=mx.uint8), tensor)


@pytest.fixture(autouse=True)
def device(request):
    """CPU for every wiring test; the Metal tests keep the default GPU device."""
    if request.node.name.startswith("test_metal_"):
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# ---- CPU: kernel entry points -------------------------------------------------


class _Capture:
    """Stands in for a compiled metal_kernel: records templates, returns zeros."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        self.calls.append(dict(template))
        return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes)]


@pytest.fixture
def capture(monkeypatch):
    fake = _Capture()
    monkeypatch.setattr(km, "_kernel", lambda kind: fake)
    monkeypatch.setattr(km, "_multi_kernel", lambda count: fake)
    km._dense_launch.cache_clear()
    km._moe_shared_launch.cache_clear()
    km._multi_launch.cache_clear()
    yield fake
    km._dense_launch.cache_clear()
    km._moe_shared_launch.cache_clear()
    km._multi_launch.cache_clear()


def _aheads(fake):
    return [c.get("AHEAD") for c in fake.calls]


@pytest.mark.parametrize("fmt", K_QUANTS)
def test_every_entry_point_accepts_ahead_and_sets_the_template(capture, fmt):
    x = mx.zeros((1, 256))
    w = _spec(fmt, (64, 256))
    km.kblock_dense(x, w, ahead=2)
    km.kblock_grouped(mx.zeros((1, 2, 256)), _spec(fmt, (2, 32, 256)), ahead=2)
    km.kblock_multi([x, x], [w, w], ahead=2)
    experts = (4, 256, 256)
    ids = mx.array([[0, 3]], dtype=mx.uint32)
    scores = mx.ones((1, 2))
    gate, down = _spec(fmt, experts), _spec(fmt, (4, 256, 256))
    km.kblock_moe(x, ids, scores, gate, gate, down, ahead=2)
    km.kblock_moe_shared(x, ids, scores, gate, gate, down, _spec(fmt, (256, 256)), _spec(fmt, (256, 256)),
                         _spec(fmt, (256, 256)), ahead=2)
    before = len(capture.calls)
    km.kblock_moe_union(mx.zeros((2, 256)), mx.array([[0, 3], [3, 1]], dtype=mx.uint32), mx.ones((2, 2)),
                        gate, gate, down, ahead=2)
    # dense, grouped, multi, moe gate_up/down, shared gate_up/down carry AHEAD=2;
    # combines carry none; the union body has no load-ahead form (no AHEAD).
    assert _aheads(capture)[:before] == [2, 2, 2, 2, 2, None, 2, 2, None]
    assert _aheads(capture)[before:] == [None, None, None]


def test_default_is_ahead_one_everywhere(capture):
    x = mx.zeros((1, 256))
    w = _spec(Q3_K, (64, 256))
    km.kblock_dense(x, w)
    km.kblock_grouped(mx.zeros((1, 2, 256)), _spec(Q3_K, (2, 32, 256)))
    km.kblock_multi([x], [w])
    gate, down = _spec(Q3_K, (4, 256, 256)), _spec(Q4_K, (4, 256, 256))
    ids, scores = mx.array([[0, 3]], dtype=mx.uint32), mx.ones((1, 2))
    km.kblock_moe(x, ids, scores, gate, gate, down)
    assert [a for a in _aheads(capture) if a is not None] == [1] * 5


def test_exact_law_token_shared_and_research_formats_keep_ahead_one(capture):
    x = mx.zeros((1, 256))
    km.kblock_dense(x, _spec(Q4_0, (64, 256)), ahead=2)  # Q4_0: exact_law
    km.kblock_dense(x, _spec(Q3_K, (64, 256)), exact=1, ahead=2)
    km.kblock_dense(mx.zeros((4, 256)), _spec(Q3_K, (64, 256)), tok=4, ahead=2)
    q3n = SimpleNamespace(shape=(64, 256), ggml_type=km.Q3N_FORMAT, row_bytes=148, data=mx.zeros((1,)))
    km.kblock_dense(x, q3n, ahead=2)
    km.kblock_multi([x, mx.zeros((1, 128))], [_spec(Q3_K, (64, 256)), _spec(Q4_0, (64, 128))], ahead=2)
    km.kblock_multi([x, x], [_spec(Q3_K, (64, 256)), q3n], ahead=2)
    assert _aheads(capture) == [1, 1, 1, 1, 2, 1]
    # Q4_0 experts under the chunk law (explicit exact=0) do have a load-ahead body.
    gate, down = _spec(Q4_0, (4, 256, 256)), _spec(Q4_0, (4, 256, 256))
    km.kblock_moe(x, mx.array([[1, 2]], dtype=mx.uint32), mx.ones((1, 2)), gate, gate, down, ahead=2)
    assert _aheads(capture)[-3:] == [2, 2, None]


def test_ahead_refusals(capture):
    x = mx.zeros((1, 256))
    w = _spec(Q3_K, (64, 256))
    for bad in (0, 3, 4):
        with pytest.raises(GGUFError):
            km.kblock_dense(x, w, ahead=bad)
        with pytest.raises(GGUFError):
            km.kblock_multi([x], [w], ahead=bad)
        with pytest.raises(GGUFError):
            km.kblock_moe_union(mx.zeros((2, 256)), mx.array([[0, 1], [1, 2]], dtype=mx.uint32),
                                mx.ones((2, 2)), _spec(Q3_K, (4, 256, 256)), _spec(Q3_K, (4, 256, 256)),
                                _spec(Q3_K, (4, 256, 256)), ahead=bad)
    # Load-ahead is the VEC=1 chunk law; other decoders are refused, not substituted.
    for vec in (0, 2, 3):
        with pytest.raises(GGUFError):
            km.kblock_dense(x, w, vec=vec, ahead=2)
    km.kblock_dense(x, w, vec=2, ahead=1)


def test_generated_sources_carry_the_ahead_branch_and_shared_pointers():
    for name in ("DENSE", "GATE_UP", "DOWN", "GATE_UP_SHARED", "DOWN_SHARED"):
        source = getattr(km, name)
        assert "else if(AHEAD>1)" in source and "ka_load<FORMAT>(" in source and "ka_dot<FORMAT>(" in source
    assert "wp_g[r]=(is_shared ? gate_sh : gate)+" in km.GATE_UP_SHARED
    assert "wp_u[r]=(is_shared ? up_sh : up)+" in km.GATE_UP_SHARED
    assert "wp_d[r]=(is_shared ? w_sh : w)+" in km.DOWN_SHARED
    assert "wp_g[r]=gate+" not in km.GATE_UP_SHARED and "wp_d[r]=w+" not in km.DOWN_SHARED
    multi = km._multi_source(3)
    assert multi.count("else if(AHEAD>1)") == 3 and "ka_dot<F2>(ring[" in multi


def test_ahead_loop_keeps_each_lanes_ascending_chunk_order():
    loop = km._ahead_loop([("g", "sums_g", "G"), ("u", "sums_u", "U")], "K/32", "x", "0", "row0+r<M")
    # One loop step covers chunk then chunk+32; the next step starts at chunk+64.
    assert "for(uint chunk=lane;chunk<NCH;chunk+=64)" in loop
    first = loop.index("sums_g[r]+=ka_dot<FORMAT>(ring[(0*ROWS*2+r*2+0)%2],chunk,a)")
    second = loop.index("sums_g[r]+=ka_dot<FORMAT>(ring[(1*ROWS*2+r*2+0)%2],c1,a)")
    assert first < loop.index("if(c1>=NCH) break;") < second
    # Item i+1's load is issued before item i's arithmetic: gate(r) -> up(r)
    # -> gate(r+1), and after the last row the next chunk's first item.
    assert loop.index("=ka_load<FORMAT>(wp_u[r],min(chunk,NCH-1));") < first
    assert "ka_load<FORMAT>(wp_g[r+1<ROWS ? r+1 : 0],min((r+1<ROWS ? c1 : c1+32),NCH-1))" in loop
    assert "ring[0]=ka_load<FORMAT>(wp_g[0],min(lane,NCH-1));" in loop


def _metal_compiler():
    return shutil.which("xcrun") and not subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, check=False).returncode


@pytest.mark.parametrize("fmt", (Q4_0,) + K_QUANTS)
@pytest.mark.parametrize("kind", ("dense", "gate_up_shared", "down_shared"))
def test_ahead_sources_compile_offline(tmp_path, kind, fmt):
    if not _metal_compiler():
        pytest.skip("CPU Metal compiler unavailable")
    block, nbytes = SIZES[fmt]
    rb = 4096 // block * nbytes
    source, inputs = {
        "dense": (km.DENSE.replace("WBASE", "size_t(0)"), ["x", "w"]),
        "gate_up_shared": (km.GATE_UP_SHARED, ["x", "ids", "gate", "up", "gate_sh", "up_sh"]),
        "down_shared": (km.DOWN_SHARED, ["mid", "ids", "w", "w_sh"]),
    }[kind]
    out = {"dense": "y", "gate_up_shared": "mid", "down_shared": "out"}[kind]
    types = {"x": "float", "mid": "float", "ids": "uint"}
    bindings = "".join(f"device const {types.get(n, 'uchar')}* {n} [[buffer({i})]],"
                       for i, n in enumerate(inputs)) + f"device float* {out} [[buffer({len(inputs)})]],"
    constants = ("constant int K=4096,M=2048,N=2048,TOPK=8,EXPERTS=144,ROWS=4,VEC=1,EXACT=0,TOK=1,TOKENS=1,"
                 f"AHEAD=2,FORMAT={fmt},ROW_BYTES={rb};\n")
    attrs = ("uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],"
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],"
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]")
    path = tmp_path / f"{kind}.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + km.KBLOCK_HEADER + constants
                    + f"kernel void k_{kind}(" + bindings + attrs + ") {\n" + source + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o",
                             str(tmp_path / f"{kind}.air")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ---- CPU: provider policy -----------------------------------------------------


def test_provider_accepts_the_policy_and_rejects_unknown_values(tmp_path):
    with pytest.raises(GGUFError):
        GGUFWeights(tmp_path / "missing.gguf", row_policy="kblock_rows_ahead3")
    assert "kblock_rows_ahead2" in weights_module._KBLOCK_ROW_POLICIES


def _provider(policy, tensors):
    w = object.__new__(GGUFWeights)
    w.row_policy = policy
    w.compute_policy = "reference"
    w.dense_prefill_policy = "reference"
    w.expert_prefill_policy = "reference"
    w.tensors = {n: p.tensor for n, p in tensors.items()}
    w._packed = dict(tensors)
    w._grouped = {}
    w.scheduled_expert_calls, w.scheduled_grouped_calls, w.scheduled_dense_calls = {}, {}, {}
    w.packed = lambda name: w._packed[name]
    return w


@pytest.mark.parametrize("policy,ahead", [("kblock_rows", 1), ("kblock_rows_ahead2", 2), ("kblock_exact", 1)])
def test_policy_reaches_every_kblock_call_site(monkeypatch, policy, ahead):
    seen = []

    def record(label):
        def fake(*args, **kwargs):
            seen.append((label, kwargs.get("ahead")))
            return mx.zeros((1, 512))
        return fake

    for name in ("kblock_dense", "kblock_grouped", "kblock_multi", "kblock_moe_shared"):
        monkeypatch.setattr(weights_module, name, record(name))
    monkeypatch.setattr(km, "kblock_moe", record("kblock_moe"))
    monkeypatch.setattr(km, "kblock_moe_union", record("kblock_moe_union"))
    tensors = {
        "dense.weight": _spec(Q3_K, (64, 256)),
        "grouped.weight": _spec(Q3_K, (64, 256)),
        "blk.3.ffn_gate_exps.weight": _spec(Q3_K, (4, 256, 256)),
        "blk.3.ffn_up_exps.weight": _spec(Q3_K, (4, 256, 256)),
        "blk.3.ffn_down_exps.weight": _spec(Q4_K, (4, 256, 256)),
        "blk.3.ffn_gate_shexp.weight": _spec(Q3_K, (256, 256)),
        "blk.3.ffn_up_shexp.weight": _spec(Q3_K, (256, 256)),
        "blk.3.ffn_down_shexp.weight": _spec(Q4_K, (256, 256)),
    }
    w = _provider(policy, tensors)
    x = mx.zeros((1, 256))
    w._dense_rows(x, tensors["dense.weight"])
    w.row_token_share = True
    w._dense_rows(mx.zeros((2, 256)), tensors["dense.weight"])
    w.grouped(mx.zeros((1, 2, 256)), "grouped.weight", groups=2)
    w.multi_linear([x, x], ["dense.weight", "grouped.weight"])
    ids, scores = mx.array([[0, 1]], dtype=mx.uint32), mx.ones((1, 2))
    w.moe(x, ids, scores, "blk.3")
    w.expert_union = True
    w.moe(mx.zeros((2, 256)), mx.array([[0, 1], [1, 2]], dtype=mx.uint32), mx.ones((2, 2)), "blk.3")
    w.moe_with_shared(x, ids, scores, "blk.3", "blk.3.ffn_{}_shexp.weight")
    labels = ["kblock_dense", "kblock_dense", "kblock_grouped", "kblock_multi", "kblock_moe",
              "kblock_moe_union", "kblock_moe_shared"]
    if policy == "kblock_exact":
        labels[1] = "kblock_dense"  # exact keeps the per-token dense call; union is chunk-law only
        labels[5] = "kblock_moe"
    assert seen == [(label, ahead) for label in labels]


def test_memo_keys_carry_the_policy_so_a_switch_rebuilds_them():
    tensors = {"a.weight": _spec(Q3_K, (64, 256)), "b.weight": _spec(Q3_K, (64, 256))}
    w = _provider("kblock_rows", tensors)
    w._dispatch_cache = {}
    first = w._multi_plan(["a.weight", "b.weight"], 1)
    w.row_policy = "kblock_rows_ahead2"
    assert w._kblock_ahead == 2
    w._multi_plan(["a.weight", "b.weight"], 1)
    keys = {k[-1] for k in w._dispatch_cache if k[0] == "multi"}
    assert keys == {"kblock_rows", "kblock_rows_ahead2"} and first is not None
    w.row_policy = "kblock_rows"
    assert w._kblock_ahead == 1 and w._kblock_exact == 0
    # The exact law follows the policy too (no stale flag after a switch).
    w.row_policy = "kblock_exact"
    assert w._kblock_exact == 1 and w._kblock_ahead == 1
    # Providers built without a policy attribute read as AHEAD=1.
    bare = object.__new__(GGUFWeights)
    assert bare._kblock_ahead == 1 and bare._kblock_exact == 0


# ---- Metal (MLX2_RUN_GPU_TESTS=1 only) ---------------------------------------

gpu = pytest.mark.skipif(not GPU, reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


def _x(rng, b, k):
    x = rng.normal(size=(b, k)).astype(np.float32)
    x[:, 32:64] = 0.0  # one all-zero chunk per row
    x[:, 100:300:3] = -0.0
    return mx.array(x)


def bits(a):
    return np.array(a).view(np.uint32)


def _same(f):
    a, b = f(1), f(2)
    mx.eval(a, b)
    np.testing.assert_array_equal(bits(a), bits(b))
    return a


@gpu
@pytest.mark.parametrize("nonfinite", (False, True))
@pytest.mark.parametrize("fmt", K_QUANTS)
@pytest.mark.parametrize("rows", (1, 2, 4))
@pytest.mark.parametrize("n,k", ((40, 4096), (2048, 4096), (64, 256), (32, 2048), (16, 8192),
                                 (38, 2304), (6, 2304), (1, 256)))
def test_metal_dense_ahead_bitwise(fmt, rows, n, k, nonfinite):
    rng = np.random.default_rng(fmt * 1000 + rows * 10 + n + k)
    w = _packed(rng, fmt, (n, k), nonfinite)
    x = _x(rng, 1, k)
    _same(lambda a: km.kblock_dense(x, w, rows=rows, ahead=a))


@gpu
@pytest.mark.parametrize("fmt", K_QUANTS)
def test_metal_grouped_and_multi_ahead_bitwise(fmt):
    rng = np.random.default_rng(77 + fmt)
    g = _packed(rng, fmt, (8, 64, 512))
    _same(lambda a: km.kblock_grouped(_x(np.random.default_rng(1), 8, 512).reshape(1, 8, 512), g, ahead=a))
    w0, w1 = _packed(rng, fmt, (256, 4096)), _packed(rng, Q4_K, (128, 4096))
    w2 = _packed(rng, Q4_0, (64, 128))  # exact law inside the same launch
    x0, x2 = _x(rng, 1, 4096), _x(rng, 1, 128)
    y = _same(lambda a: km.kblock_multi([x0, x0, x2], [w0, w1, w2], ahead=a))
    # Each multi output still equals its own dense call (the multi contract).
    ref = mx.concatenate([km.kblock_dense(x0, w0, ahead=2), km.kblock_dense(x0, w1, ahead=2),
                          km.kblock_dense(x2, w2)], axis=1)
    np.testing.assert_array_equal(bits(y), bits(ref))


@gpu
@pytest.mark.parametrize("gate_fmt,down_fmt", ((Q3_K, Q4_K), (Q4_K, Q5_K), (Q5_K, Q6_K), (Q6_K, Q3_K),
                                               (Q4_0, Q4_0)))
@pytest.mark.parametrize("rows", (1, 2, 4))
def test_metal_moe_shared_union_ahead_bitwise(gate_fmt, down_fmt, rows):
    rng = np.random.default_rng(gate_fmt * 31 + down_fmt + rows)
    experts, m, k = 12, 512, 1024
    gate, up = _packed(rng, gate_fmt, (experts, m, k)), _packed(rng, gate_fmt, (experts, m, k))
    down = _packed(rng, down_fmt, (experts, k, m))
    x = _x(rng, 1, k)
    ids = mx.array(rng.permutation(experts)[:8].reshape(1, 8).astype(np.uint32))
    scores = mx.array(rng.uniform(size=(1, 8)).astype(np.float32))
    _same(lambda a: km.kblock_moe(x, ids, scores, gate, up, down, rows=rows, ahead=a))
    gs, us, ds = (_packed(rng, gate_fmt, (m, k)), _packed(rng, gate_fmt, (m, k)), _packed(rng, down_fmt, (k, m)))
    _same(lambda a: km.kblock_moe_shared(x, ids, scores, gate, up, down, gs, us, ds, rows=rows, ahead=a))
    if gate_fmt == Q4_0:
        # 72 gate/up chunks: lanes 0-7 run a full two-chunk ring step then an
        # odd tail (partial ROWS blocks are covered by the dense n=38/6 cases).
        g4, u4 = _packed(rng, Q4_0, (4, 64, 2304)), _packed(rng, Q4_0, (4, 64, 2304))
        d4 = _packed(rng, Q4_0, (4, 2304, 64))
        x4 = _x(rng, 1, 2304)
        ids4 = mx.array([[3, 0, 2]], dtype=mx.uint32)
        _same(lambda a: km.kblock_moe(x4, ids4, scores[:, :3], g4, u4, d4, rows=rows, ahead=a))
    if gate_fmt in K_QUANTS and down_fmt in K_QUANTS:
        x3 = _x(rng, 3, k)
        ids3 = mx.array(np.stack([rng.permutation(experts)[:8] for _ in range(3)]).astype(np.uint32))
        scores3 = mx.array(rng.uniform(size=(3, 8)).astype(np.float32))
        union = _same(lambda a: km.kblock_moe_union(x3, ids3, scores3, gate, up, down, rows=rows, ahead=a))
        # The union equals per-row kblock_moe, here under AHEAD=2.
        rows_ahead = mx.concatenate([km.kblock_moe(x3[i:i + 1], ids3[i:i + 1], scores3[i:i + 1], gate, up, down,
                                                   rows=rows, ahead=2) for i in range(3)], axis=0)
        np.testing.assert_array_equal(bits(union), bits(rows_ahead))
