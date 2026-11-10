"""Grouped W4A8 routed-expert GEMM for sorted MoE prefill (experimental).

The int8 prefill Q4 in-place GEMM (``int8_prefill.q45_gemm``, omlx Q4 A8
port) run per expert tile: sorted rows (each expert's rows one contiguous
run, as the sorted ``gather_qmm`` path produces them) are cut into
``(row_start, expert, rows)`` tiles by the segmented tile pre-pass of
``models.moe_nax_gather`` (omlx #3995 scan), and each threadgroup computes
one tile x 32*WN output columns with int8 tensor ops on the 4-bit gs64
affine expert table read in place.

Differences from the dense kernel:

- metadata: the dense path reads a group-major ``[G, N]`` scale/bias copy;
  for a 512-expert table that copy would be ~9% of the expert bytes, so each
  threadgroup stages its own ``[G, BN]`` scale/bias block (from the
  checkpoint's ``[E, N, G]`` arrays) in threadgroup memory instead;
- rows: activation rows are read through ``row_map`` (token row of each
  sorted row) when given, so Stage A quantizes the T token rows once, not
  the T * top_k replicated rows.

Not carried: a gate+up kernel with SwiGLU in its epilogue.  It was correct
but slower than two plain calls plus an elementwise SwiGLU at every measured
size >= 4K tokens (M5 Max, FN shapes, 2026-11-09: 8K tokens 28.3 vs 24.4
ms), most likely from the doubled accumulators and staging.

Numerics are approximate (int8 activations, per-64-group absmax scales, as
the dense Q4 kernel's ``group64``): exactly ``x_hat @ dequant(w).T`` where
``x_hat`` is the Stage A fake-quantized activation, up to fp32 summation
order.  ``reference`` computes that.
"""

from __future__ import annotations

import hashlib
import threading

from . import int8_prefill as _ip

GROUP_SIZE = 64
BITS = 4
# (WM, WN): BM = 32 * WM sorted rows (the tile height cut by the pre-pass),
# BN = 32 * WN columns.  Tiles do not change numerics.
TILES = ((1, 2), (2, 2), (1, 4), (2, 4), (4, 2))
# Measured on Flash-Next shapes (M5 Max, 2026-11-09): gate/up [640 x 2560]
# best at (2, 2), down [2560 x 640] at (1, 2).
GATE_UP_TILE = (2, 2)
DOWN_TILE = (1, 2)

_GEMM_SRC = """
    constexpr int TM = 2;
    constexpr int BN = 32 * WN;
    constexpr int G = K / 64;
    constexpr int WORDS = 8;            // 64 Q4 codes per group in uint32 words
    constexpr int W_ROW = G * WORDS;
    constexpr int NT = 32 * WM * WN;

    const uint tile_id = threadgroup_position_in_grid.y;
    if (tile_id >= tile_count[0]) return;   // uniform per threadgroup
    const uint4 tile = reinterpret_cast<const device uint4*>(tiles)[tile_id];
    const int row0 = int(tile.x);
    const int expert = int(tile.y);
    const int nrows = int(tile.z);
    const int col_tile = int(threadgroup_position_in_grid.x) * BN;
    const int R = dims[1];

    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int lid = simd_gid * 32 + lane;
    const int sg_m = simd_gid % WM;
    const int sg_n = simd_gid / WM;

    // Stage this tile's scale/bias block [G][BN] from [E, N, G].
    alignas(16) threadgroup TS tg_s[G * BN];
    alignas(16) threadgroup TS tg_b[G * BN];
    {
        const size_t meta0 = (size_t(expert) * N + size_t(col_tile)) * G;
        for (int i = lid; i < BN * G; i += NT) {
            const int nn = i / G;
            const int gg = i - nn * G;
            tg_s[gg * BN + nn] = s[meta0 + i];
            tg_b[gg * BN + nn] = b[meta0 + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short2 coord = mlx2_q8_frag_coord(uint(lane));

    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_set = matmul2d_descriptor(
        16, 32, 16, /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroup> op;
    matmul2d<desc_set, metal::execution_simdgroup> op_set;

    auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto acc0 = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, int32_t>();
    auto acc1 = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, int32_t>();

    float Cf[TM][16];
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) Cf[i][e] = 0.0f;
    }

    // Source activation row of each of this lane's four fragment rows
    // (rows past the tile repeat its last row; never stored).
    const int local_base = sg_m * (TM * 16) + int(coord.y);
    int src[TM][2];
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; ++r) {
            const int m = row0 + min(local_base + i * 16 + r * 8, nrows - 1);
#if MLX2_MAPPED
            src[i][r] = int(row_map[m]);
#else
            src[i][r] = m;
#endif
        }
    }

    const int n_lane = col_tile + sg_n * 32 + int(coord.y);
    const device uint32_t* wbase = w + (size_t(expert) * N + size_t(n_lane)) * size_t(W_ROW);
    const int n_loc0 = sg_n * 32 + int(coord.x);
    const int cx = int(coord.x) >> 2;

    for (int g = 0; g < G; ++g) {
        uint2 wg[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; ++q) {
            const size_t off = (q & 1) * (8 * W_ROW) + (q >> 1) * (16 * W_ROW) + size_t(g) * WORDS;
            wg[q] = reinterpret_cast<const device uint2*>(wbase + off)[cx];
        }
#pragma clang loop unroll(full)
        for (int t = 0; t < 4; ++t) {
            // Step t: codes 16cx + 8*(t>>1) + 2j + (t&1) (Stage A v8 order).
#pragma clang loop unroll(full)
            for (int q = 0; q < 4; ++q) {
                const int base = (q >> 1) * 8 + (q & 1) * 4;
                const uint32_t word = wg[q][t >> 1];
                const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
                ct_b[base + 0] = quad.x;
                ct_b[base + 1] = quad.y;
                ct_b[base + 2] = quad.z;
                ct_b[base + 3] = quad.w;
            }
#pragma clang loop unroll(full)
            for (int hf = 0; hf < 2; ++hf) {
#pragma clang loop unroll(full)
                for (int r = 0; r < 2; ++r) {
                    const char4 quad = as_type<char4>(*reinterpret_cast<const device uint32_t*>(
                        qa + size_t(src[hf][r]) * K + size_t(g) * 64 + size_t(cx) * 16 + size_t(t) * 4));
                    ct_a[r * 4 + 0] = quad.x;
                    ct_a[r * 4 + 1] = quad.y;
                    ct_a[r * 4 + 2] = quad.z;
                    ct_a[r * 4 + 3] = quad.w;
                }
                if (t == 0) {
                    if (hf == 0) op_set.run(ct_a, ct_b, acc0);
                    else op_set.run(ct_a, ct_b, acc1);
                } else {
                    if (hf == 0) op.run(ct_a, ct_b, acc0);
                    else op.run(ct_a, ct_b, acc1);
                }
            }
        }

        // Per GS64 group: sum_k a_k w_k = sa * (s * acc + b * ra).
        float r_g[TM][2];
        float s_g[TM][2];
#pragma clang loop unroll(full)
        for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
            for (int r = 0; r < 2; ++r) {
                r_g[i][r] = float(ra[size_t(g) * R + src[i][r]]);
                s_g[i][r] = sa[size_t(g) * R + src[i][r]];
            }
        }
        vec<TS, 4> sv[2];
        vec<TS, 4> bv[2];
#pragma clang loop unroll(full)
        for (int h = 0; h < 2; ++h) {
            const int n0 = g * BN + n_loc0 + h * 16;
            sv[h] = *reinterpret_cast<const threadgroup vec<TS, 4>*>(tg_s + n0);
            bv[h] = *reinterpret_cast<const threadgroup vec<TS, 4>*>(tg_b + n0);
        }
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) {
            const int r = (e & 7) >> 2;
            const float swc = float(sv[e >> 3][e & 3]);
            const float bwc = float(bv[e >> 3][e & 3]);
            Cf[0][e] = metal::fma(s_g[0][r],
                metal::fma(swc, float(acc0[e]), bwc * r_g[0][r]), Cf[0][e]);
            Cf[1][e] = metal::fma(s_g[1][r],
                metal::fma(swc, float(acc1[e]), bwc * r_g[1][r]), Cf[1][e]);
        }
    }

#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) {
            const int ee = e & 7;
            const int r = ee >> 2;
            const int local = local_base + i * 16 + r * 8;
            if (local < nrows) {
                const int m = row0 + local;
                const int n = col_tile + sg_n * 32 + (e >> 3) * 16 + int(coord.x) + (ee & 3);
                out[size_t(m) * N + n] = static_cast<TO>(Cf[i][e]);
            }
        }
    }
"""

# The expert-block law beside the kernels: gate and up grouped at
# GATE_UP_TILE, the block's SwiGLU on their rounded outputs, Stage A
# (group64) on the hidden, down grouped.  Bump on any change to
# ``expert_block``'s composition (the tiles do not change numerics).
BLOCK_LAW = "gate,up:grouped;act:block_swiglu(up,gate);hidden:stage_a_group64;down:grouped;v1"

KERNEL_REVISION = hashlib.sha256(
    (_ip._numerics_kernel_source(
        _ip._Q45_HEADER + _ip._Q45_STAGE_A_SRC + _GEMM_SRC
    ) + BLOCK_LAW).encode()
).hexdigest()[:16]

_lock = threading.Lock()
_kernels: dict = {}


def expert_tiles(indices, bm: int, num_experts: int):
    """``(tiles, tile_count, max_tiles)`` of the sorted ``indices`` (uint32
    ``[M]``, each expert one contiguous run) cut into ``bm``-row tiles."""
    import mlx.core as mx

    from .models import moe_nax_gather as nax

    scan = nax._get_kernel("scan")
    if scan is None:
        raise RuntimeError("NAX tile pre-pass kernel unavailable")
    if num_experts > nax._MAX_EXPERTS:
        raise ValueError(f"{num_experts} experts exceed the tile pre-pass limit")
    m = int(indices.shape[0])
    max_tiles = (m + bm - 1) // bm + min(num_experts, m)
    tiles, count = scan(
        inputs=[indices, mx.array([m, num_experts, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return tiles, count, max_tiles


def table_refusal(w, s, b, k=None) -> str | None:
    """Why ``(w, s, b)`` is not a Q4 gs64 affine ``[E, N, K]`` expert table
    this kernel takes, or None.  Shapes and dtypes only."""
    import mlx.core as mx

    if b is None:
        return "no affine biases"
    if w.ndim != 3 or w.dtype != mx.uint32:
        return "weight is not a packed [E, N, K/8] uint32 table"
    e, n, wpr = w.shape
    kk = wpr * 32 // BITS
    if k is not None and kk != k:
        return f"input dim {kk} != {k}"
    if kk % GROUP_SIZE:
        return f"input dim {kk} is not a multiple of {GROUP_SIZE}"
    g = kk // GROUP_SIZE
    if tuple(s.shape) != (e, n, g) or tuple(b.shape) != (e, n, g):
        return "scales/biases are not [E, N, K/64] (not 4-bit gs64)"
    if s.dtype != b.dtype or s.dtype not in (mx.bfloat16, mx.float16):
        return "scales/biases are not one bf16/fp16 dtype"
    if n % 64:
        return f"output dim {n} is not a multiple of 64"
    return None


def grouped_q4_gemm(qa, sa, ra, indices, table, *, row_map=None, tile=None, out_dtype=None):
    """Sorted rows x routed Q4 gs64 expert table -> ``[M, N]``.

    ``qa``/``sa``/``ra``: ``int8_prefill.q45_stage_a(x, "group64")`` of the
    activation rows (``[R, K]``; with ``row_map`` the token rows, else the
    sorted rows themselves).  ``indices``: sorted uint32 expert id of each of
    the ``M`` sorted rows.  ``table``: ``(weight, scales, biases)`` of an
    ``[E, N, K]`` table."""
    import mlx.core as mx

    wm, wn = tuple(tile) if tile is not None else DOWN_TILE
    if (wm, wn) not in TILES:
        raise ValueError(f"tile {(wm, wn)} is not one of {TILES}")
    r_rows, k = qa.shape
    w, s, b = table
    refusal = table_refusal(w, s, b, k)
    if refusal is not None:
        raise ValueError(f"expert table: {refusal}")
    e, n = int(w.shape[0]), int(w.shape[1])
    bn = 32 * wn
    if n % bn:
        raise ValueError(f"output dim {n} is not a multiple of BN={bn}")
    if indices.dtype != mx.uint32 or indices.ndim != 1:
        raise ValueError("indices must be a flat uint32 array")
    m = int(indices.shape[0])
    if row_map is not None and (row_map.dtype != mx.uint32 or row_map.shape != (m,)):
        raise ValueError("row_map must be uint32 [M]")
    if row_map is None and r_rows != m:
        raise ValueError("without row_map the activation rows are the sorted rows")
    if 2 * (k // GROUP_SIZE) * bn * 2 > 28 * 1024:
        raise ValueError("scale/bias staging exceeds threadgroup memory")
    out_dtype = out_dtype or s.dtype
    ts, to = _ip._tname(s.dtype), _ip._tname(out_dtype)
    mapped = row_map is not None
    key = (n, k, ts, to, wm, wn, mapped)
    kernel = _kernels.get(key)
    if kernel is None:
        with _lock:
            kernel = _kernels.get(key)
            if kernel is None:
                names = ["qa", "sa", "ra", "w", "s", "b", "tiles", "tile_count", "dims"]
                if mapped:
                    names.append("row_map")
                kernel = _kernels[key] = mx.fast.metal_kernel(
                    name=(
                        f"mlx2_i8x_q4_grouped_{n}x{k}_{ts}_{to}_wm{wm}_wn{wn}"
                        f"{'_map' if mapped else ''}"
                    ),
                    input_names=names,
                    output_names=["out"],
                    header=_ip._Q45_HEADER,
                    source=f"#define MLX2_MAPPED {int(mapped)}\n"
                    + _ip._q8_prefix(K=k, N=n, WM=wm, WN=wn, TS=ts, TO=to)
                    + _GEMM_SRC,
                )
    tiles, count, max_tiles = expert_tiles(indices, 32 * wm, e)
    inputs = [qa, sa, ra, w, s, b, tiles, count, mx.array([m, r_rows], dtype=mx.int32)]
    if mapped:
        inputs.append(row_map)
    nt = 32 * wm * wn
    return kernel(
        inputs=inputs,
        grid=(n // bn * nt, max_tiles, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
        init_value=0,
    )[0]


# Weak reference to the owning ``Int8PrefillHandle`` on a bound expert block.
STATE_ATTR = "_mlx2_int8_experts"


def module_tables(module):
    """``(gate, up, down)`` ``(weight, scales, biases)`` tables of a split
    SwitchGLU expert block."""
    return tuple(
        (proj["weight"], proj["scales"], proj.get("biases"))
        for proj in (module.gate_proj, module.up_proj, module.down_proj)
    )


def module_refusal(module) -> str | None:
    """Why the grouped W4A8 expert block cannot stand in for ``module``'s
    routed body, or None.  Structural (types, formats, shapes); no device
    work.  Only the split-table ``FusedDownSwitchGLU`` carries the hook."""
    from .models.qwen3_next import FusedDownSwitchGLU
    from .models.switch_layers import QuantizedSwitchLinear, SwiGLU

    if type(module) is not FusedDownSwitchGLU:
        return f"{type(module).__name__} has no int8 expert hook (split FusedDownSwitchGLU only)"
    if type(module.activation) is not SwiGLU:
        return "activation is not SwiGLU"
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = module.get(name)
        if type(proj) is not QuantizedSwitchLinear:
            return f"{name} is not a resident QuantizedSwitchLinear"
        if "bias" in proj:
            return f"{name} has a linear bias"
        if (getattr(proj, "mode", "affine"), proj.bits, proj.group_size) != ("affine", BITS, GROUP_SIZE):
            return f"{name} is {proj.mode} q{proj.bits} gs{proj.group_size}, not affine q4 gs64"
    (gw, gs, gb), (uw, us, ub), (dw, ds, db) = module_tables(module)
    for name, (w, s, b) in (("gate_proj", (gw, gs, gb)), ("up_proj", (uw, us, ub)),
                            ("down_proj", (dw, ds, db))):
        why = table_refusal(w, s, b)
        if why is not None:
            return f"{name}: {why}"
    if gw.shape != uw.shape or gs.dtype != us.dtype:
        return "gate and up tables differ"
    e, inter, wpr = gw.shape
    if dw.shape[0] != e or dw.shape[2] * 32 // BITS != inter:
        return "down table does not take the gate/up width"
    if ds.dtype != gs.dtype:
        return "down metadata dtype differs from gate/up"
    hidden = wpr * 32 // BITS
    if dw.shape[1] != hidden:
        return "down output width differs from the gate/up input width"
    for n, k in ((inter, hidden), (hidden, inter)):
        if 2 * (k // GROUP_SIZE) * 64 * 2 > 28 * 1024:
            return f"[{n} x {k}] scale/bias staging exceeds threadgroup memory"
    return None


def expert_block(x_tok, indices, row_map, gate, up, down, activation=None):
    """``down(silu(gate(x)) * up(x))`` for the sorted rows, all W4A8.

    ``x_tok``: token rows ``[T, K]``; ``indices``/``row_map``: the sorted
    expert id and token row of each of the ``M`` sorted rows (uint32).
    Returns the sorted down output ``[M, H]`` (the caller unsorts and
    weights it).  ``activation(up, gate)`` is the block's own SwiGLU module
    (the stock body's call, on the gate/up outputs in their dtype); without
    it the same ``silu(gate) * up`` as MLX ops."""
    import mlx.core as mx

    qa, sa, ra = _ip.q45_stage_a(x_tok, "group64")
    g = grouped_q4_gemm(qa, sa, ra, indices, gate, row_map=row_map, tile=GATE_UP_TILE)
    u = grouped_q4_gemm(qa, sa, ra, indices, up, row_map=row_map, tile=GATE_UP_TILE)
    hidden = activation(u, g) if activation is not None else mx.sigmoid(g) * g * u
    qh, sh, rh = _ip.q45_stage_a(hidden, "group64")
    return grouped_q4_gemm(qh, sh, rh, indices, down, tile=DOWN_TILE)


_probe_lock = threading.Lock()
_probed: dict = {}


def probe(module) -> tuple[bool, str]:
    """Build the gate/up and down kernels for ``module``'s shapes and check
    one small sorted problem against :func:`reference` (cached per shape).
    A Metal device fault propagates as itself and caches nothing."""
    import mlx.core as mx

    (gw, gs, gb), _, (dw, ds, db) = module_tables(module)
    inter, hidden = int(gw.shape[1]), int(gw.shape[2]) * 32 // BITS
    key = (inter, hidden, gs.dtype, ds.dtype)
    with _probe_lock:
        if key in _probed:
            return _probed[key]
        try:
            mx.random.seed(0)
            experts, rows = 3, 70
            idx = mx.sort(mx.array([0] * 33 + [2] * 37, dtype=mx.uint32))
            rmap = mx.arange(rows, dtype=mx.uint32)[::-1] % 17
            result = (True, "")
            for n, k, dt, tile, mapped in ((inter, hidden, gs.dtype, GATE_UP_TILE, True),
                                           (hidden, inter, ds.dtype, DOWN_TILE, False)):
                w = mx.random.normal((experts, n, k), dtype=dt) * 0.02
                table = tuple(mx.quantize(w, group_size=GROUP_SIZE, bits=BITS))
                x = mx.random.normal((17 if mapped else rows, k), dtype=dt)
                qa, sa, ra = _ip.q45_stage_a(x, "group64")
                got = grouped_q4_gemm(qa, sa, ra, idx, table,
                                      row_map=rmap if mapped else None, tile=tile)
                ref = reference(x, idx, table, row_map=rmap if mapped else None)
                err = float(mx.sqrt(mx.mean((got.astype(mx.float32) - ref) ** 2))
                            / mx.sqrt(mx.mean(ref ** 2)))
                if not (err < 1e-2):
                    result = (False, f"[{n} x {k}] rel RMSE {err:.3g} vs reference")
                    break
        except Exception as error:  # noqa: BLE001 - refusal, not a crash
            if "fault" in str(error).lower():
                raise
            result = (False, f"{type(error).__name__}: {error}")
        _probed[key] = result
        return result


def fake_quantize(x):
    """Stage A's activation law as MLX ops: per-64-group absmax int8 codes
    times their scale, in fp32 (``x_hat`` of the module docstring)."""
    import mlx.core as mx

    rows, k = x.shape
    xg = x.astype(mx.float32).reshape(rows, k // GROUP_SIZE, GROUP_SIZE)
    amax = mx.max(mx.abs(xg), axis=-1, keepdims=True)
    inv = mx.where(amax > 0, 127.0 / amax, 0.0)
    q = mx.clip(mx.round(xg * inv), -127, 127)
    scale = mx.where(amax > 0, amax / 127.0, 0.0)
    return (q * scale).reshape(rows, k)


def reference(x, indices, table, *, row_map=None):
    """fp32 ``x_hat[rows] @ dequant(w[e]).T`` per sorted row."""
    import mlx.core as mx

    xh = fake_quantize(x)
    rows = xh if row_map is None else xh[row_map]
    w, s, b = table
    wd = mx.dequantize(w, s, b, group_size=GROUP_SIZE, bits=BITS).astype(mx.float32)[indices]
    return (wd @ rows[:, :, None]).squeeze(-1)
