"""Multi-row 4-bit matvec for external drafters (affine q4 and NVFP4), opt-in.

Original mlx2 code.  Stock MLX 0.32.4 ``quantized_matmul`` is cheap at one
row but at the drafter's block of 2..16 rows takes per-row ``qmv`` or the
wide / split-K kernels: on the M5 Max an affine q4 ``131072 x 4096`` head
costs 0.54 ms at one row and 1.2 ms at seven or eight, so a quantized
LiLiCorr drafter lost end to end to the BF16 one (q4 58.9, NVFP4 61.1, BF16
63.0 tok/s).

Here the weight is read and dequantized once for all rows and the rows are
a matrix operand:

* ``nax`` (M5-class GPUs, ``applegpu_g17`` and newer): one threadgroup owns
  32 output columns; each simdgroup takes every ``SGK``-th 128-wide K chunk
  and runs eight ``matmul2d(16, 32, 16)`` tensor ops per chunk on the GPU
  neural accelerators, X[16 rows] . W[32 cols]^T, with dequantized half
  weights and float32 accumulation; the ``SGK`` partial tiles are summed
  through threadgroup memory.  The call costs about one stock row.
* ``simdgroup`` (other GPUs, or ``N % 32 != 0``): the same scheme on 8x8
  ``simdgroup_float8x8`` MMA tiles (``ceil(rows / 8)`` row tiles).

Both permute K inside a chunk so that each lane's packed weights are one
``uint4`` and its inputs are contiguous (a dot product does not depend on the
order of its terms; float rounding does, so this is not row-exact).  Affine
dequantizes ``s * q + b``; NVFP4 ``e2m1 * e4m3`` with the per-tensor global
scale and the linear bias applied in the same launch.

Not row-exact and not bit-identical to stock: drafter proposals are verified
by the target, so any numerically sane matvec is admissible.  Covered:
4-bit, ``K % 128 == 0``, ``N % 8 == 0``, affine group 32/64/128 or NVFP4
group 16, bf16/fp16 input, 2..16 rows.  Anything else takes the stock call.

Selection: ``MLX2_DRAFTER_ROWS_QMV=on`` (default ``off``) at drafter load;
:func:`install` swaps the class of compatible ``nn.QuantizedLinear`` and
``GlobalScaledNVFP4Linear`` modules to subclasses whose call routes 2..16
rows here and everything else to the stock path.  Provenance:
provenance/drafter-rows-qmv.json.
"""

from __future__ import annotations

import functools
import os
import threading

import mlx.core as mx
from mlx import nn

ENV = "MLX2_DRAFTER_ROWS_QMV"
MIN_ROWS = 2
MAX_ROWS = 16
CHUNK = 128  # K per simdgroup iteration

_LOCK = threading.Lock()
_KERNELS: dict = {}
_COUNTS = {"kernel_calls": 0, "stock_calls": 0}

_HEADER = r"""
// E2M1 nibble -> half bits whose value is the E2M1 value * 2^-14 (the sign,
// exponent and mantissa land in the half fields; e = 0 is the half
// subnormal), so the 2^14 is folded into the group scale.
inline float mlx2_e2m1(uint n) {
  return float(as_type<half>(ushort(((n << 12) & 0x8000u) | ((n << 9) & 0x0E00u))));
}

inline float mlx2_e4m3(uint v) {
  const uint e = (v >> 3) & 15u;
  const uint m = v & 7u;
  float r = e == 0u ? float(m) * 0.001953125f  // m * 2^-9
                    : as_type<float>(((e + 120u) << 23) | (m << 20));
  return (v & 128u) ? -r : r;
}
"""

# Y^T = W . X^T on 8x8 simdgroup MMA tiles.  Per 128-wide K chunk the MMA
# k index p (0..7) of tile i (0..15) is K offset p * 16 + i: the lane that
# holds A columns (fn, fn + 1) then needs 16 contiguous K for each, i.e. the
# four consecutive packed words at k0 / 8 + 2 fn (one uint4), and the lane
# that holds B row fm needs the 16 contiguous inputs at k0 + 16 fm of its two
# rows.  The dot product is order-free, so the permuted K is exact algebra.
# Each simdgroup owns NT column tiles for all rows and every SGK-th chunk;
# the SGK partial tiles are summed through threadgroup memory.
_SOURCE = r"""
  constexpr int MT = (ROWS + 7) / 8;
  const int K = params[0];
  const int N = params[1];
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int n_base = int(threadgroup_position_in_grid.y) * (8 * NT);
  const int KW = K / 8;
  const int KG = K / GS;
  const int iters = K / 128;

  simdgroup_float8x8 acc[NT][MT];
  _Pragma("clang loop unroll(full)")
  for (int nt = 0; nt < NT; ++nt)
    _Pragma("clang loop unroll(full)")
    for (int mt = 0; mt < MT; ++mt)
      acc[nt][mt] = simdgroup_float8x8(0.0f);

  for (int it = int(sg); it < iters; it += SGK) {
    const int k0 = it * 128;
    float xa[MT][16];
    float xb[MT][16];
    _Pragma("clang loop unroll(full)")
    for (int mt = 0; mt < MT; ++mt) {
      const int ra = mt * 8 + fn;
      const device T* pa = x + size_t(ra) * K + k0 + fm * 16;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 16; ++i) {
        xa[mt][i] = ra < ROWS ? float(pa[i]) : 0.0f;
        xb[mt][i] = ra + 1 < ROWS ? float(pa[K + i]) : 0.0f;
      }
    }
    uint4 wv[NT];
    float sa[NT], sb[NT];
#if !MLX2_NVFP4
    float ba[NT], bb[NT];
#endif
    _Pragma("clang loop unroll(full)")
    for (int nt = 0; nt < NT; ++nt) {
      const int row = n_base + nt * 8 + fm;
      wv[nt] = *((const device uint4*)(w + size_t(row) * KW + k0 / 8 + 2 * fn));
      const size_t ga = size_t(row) * KG + (k0 + fn * 16) / GS;
      const size_t gb = size_t(row) * KG + (k0 + fn * 16 + 16) / GS;
#if MLX2_NVFP4
      sa[nt] = mlx2_e4m3(uint(scales[ga])) * 16384.0f;
      sb[nt] = mlx2_e4m3(uint(scales[gb])) * 16384.0f;
#else
      sa[nt] = float(scales[ga]);
      sb[nt] = float(scales[gb]);
      ba[nt] = float(biases[ga]);
      bb[nt] = float(biases[gb]);
#endif
    }
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 16; ++i) {
      MMA_IN bm[MT];
      _Pragma("clang loop unroll(full)")
      for (int mt = 0; mt < MT; ++mt) {
        bm[mt].thread_elements()[0] = MMA_T(xa[mt][i]);
        bm[mt].thread_elements()[1] = MMA_T(xb[mt][i]);
      }
      _Pragma("clang loop unroll(full)")
      for (int nt = 0; nt < NT; ++nt) {
        const uint qa = ((i < 8 ? wv[nt].x : wv[nt].y) >> (4 * (i & 7))) & 15u;
        const uint qb = ((i < 8 ? wv[nt].z : wv[nt].w) >> (4 * (i & 7))) & 15u;
        MMA_IN am;
#if MLX2_NVFP4
        am.thread_elements()[0] = MMA_T(mlx2_e2m1(qa) * sa[nt]);
        am.thread_elements()[1] = MMA_T(mlx2_e2m1(qb) * sb[nt]);
#else
        am.thread_elements()[0] = MMA_T(fma(sa[nt], float(qa), ba[nt]));
        am.thread_elements()[1] = MMA_T(fma(sb[nt], float(qb), bb[nt]));
#endif
        _Pragma("clang loop unroll(full)")
        for (int mt = 0; mt < MT; ++mt)
          simdgroup_multiply_accumulate(acc[nt][mt], am, bm[mt], acc[nt][mt]);
      }
    }
  }

  if (SGK > 1) {
    threadgroup float red[(SGK > 1 ? SGK - 1 : 1) * NT * MT * 64];
    if (sg > 0) {
      _Pragma("clang loop unroll(full)")
      for (int nt = 0; nt < NT; ++nt)
        _Pragma("clang loop unroll(full)")
        for (int mt = 0; mt < MT; ++mt) {
          const int o = (((int(sg) - 1) * NT + nt) * MT + mt) * 64 + int(lane) * 2;
          red[o] = acc[nt][mt].thread_elements()[0];
          red[o + 1] = acc[nt][mt].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg > 0)
      return;
    _Pragma("clang loop unroll(full)")
    for (int s = 0; s < SGK - 1; ++s)
      _Pragma("clang loop unroll(full)")
      for (int nt = 0; nt < NT; ++nt)
        _Pragma("clang loop unroll(full)")
        for (int mt = 0; mt < MT; ++mt) {
          const int o = ((s * NT + nt) * MT + mt) * 64 + int(lane) * 2;
          acc[nt][mt].thread_elements()[0] += red[o];
          acc[nt][mt].thread_elements()[1] += red[o + 1];
        }
  }

  _Pragma("clang loop unroll(full)")

  for (int nt = 0; nt < NT; ++nt) {
    const int col = n_base + nt * 8 + fm;
    float add = 0.0f;
#if MLX2_HAS_BIAS
    add = float(bias[col]);
#endif
    float mul = 1.0f;
#if MLX2_NVFP4
    mul = gscale[0];
#endif
    _Pragma("clang loop unroll(full)")
    for (int mt = 0; mt < MT; ++mt) {
      const int ra = mt * 8 + fn;
      if (ra < ROWS)
        y[size_t(ra) * N + col] =
            static_cast<T>(fma(acc[nt][mt].thread_elements()[0], mul, add));
      if (ra + 1 < ROWS)
        y[size_t(ra + 1) * N + col] =
            static_cast<T>(fma(acc[nt][mt].thread_elements()[1], mul, add));
    }
  }
"""

# M5 (applegpu_g17+) variant on the GPU neural accelerators: one
# ``matmul2d(16, 32, 16)`` per 16-wide K step, Y[16 rows x 32 cols] +=
# X[16 x 16] . W[32 x 16]^T.  Cooperative-tensor lanes hold columns fn..fn+3
# (fn in 0, 4, 8, 12) of rows fm and fm + 8; within a 128-wide K chunk the
# MMA column fn + j (j < 4) of step s (0..7) is K offset 8 fn + 4 s + j, so a
# lane's weights for all eight steps are the four packed words at k0 / 8 + fn
# of each of its rows (one uint4; step s is a 16-bit field of word s / 2) and
# its inputs are the 32 contiguous values at k0 + 8 fn.  Same exact-algebra
# K permutation as above.
_NAX_HEADER = _HEADER + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_NAX_SOURCE = r"""
  const int K = params[0];
  const int N = params[1];
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const short qid = short(lane >> 2);
  const short fm = short((qid & 4) | ((lane >> 1) & 3));
  const short fn = short(((qid & 2) | (lane & 1)) * 4);
  const int n_base = int(threadgroup_position_in_grid.y) * 32;
  const int KW = K / 8;
  const int KG = K / GS;
  const int iters = K / 128;

  constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto xa = op.template get_left_input_cooperative_tensor<T, half, float>();
  auto wb = op.template get_right_input_cooperative_tensor<T, half, float>();
  using XA = metal::remove_addrspace_t<decltype(xa)>;
  using WB = metal::remove_addrspace_t<decltype(wb)>;
  auto acc = op.template get_destination_cooperative_tensor<XA, WB, float>();
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 16; ++e) acc[e] = 0.0f;

  // Lane rows: q = tn * 2 + i -> column n_base + 16 tn + fm + 8 i.
  const device uint32_t* wrow[4];
  size_t grow[4];
  _Pragma("clang loop unroll(full)")
  for (short q = 0; q < 4; ++q) {
    const int row = n_base + (q >> 1) * 16 + fm + (q & 1) * 8;
    wrow[q] = w + size_t(row) * KW + fn;
    grow[q] = size_t(row) * KG;
  }
  const bool live0 = fm < ROWS;
  const bool live1 = fm + 8 < ROWS;
  const device T* xrow0 = x + size_t(fm) * K + fn * 8;
  const device T* xrow1 = x + size_t(fm + 8) * K + fn * 8;

  for (int it = int(sg); it < iters; it += SGK) {
    const int k0 = it * 128;
    uint4 wv[4];
    vec<T, 4> xr[2][8];
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < 4; ++q)
      wv[q] = *((const device uint4*)(wrow[q] + k0 / 8));
    const device vec<T, 4>* p0 = (const device vec<T, 4>*)(xrow0 + k0);
    const device vec<T, 4>* p1 = (const device vec<T, 4>*)(xrow1 + k0);
    _Pragma("clang loop unroll(full)")
    for (short t = 0; t < 8; ++t) {
      xr[0][t] = live0 ? p0[t] : vec<T, 4>(0);
      xr[1][t] = live1 ? p1[t] : vec<T, 4>(0);
    }
#if MLX2_NVFP4
    half sc[4][2];
#else
    float sc[4];
    float bi[4];
#endif
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < 4; ++q) {
      const size_t g0 = grow[q] + (k0 + fn * 8) / GS;
#if MLX2_NVFP4
      // GS == 16: steps 0..3 read group g0, steps 4..7 the next one.
      sc[q][0] = half(mlx2_e4m3(uint(scales[g0])));
      sc[q][1] = half(mlx2_e4m3(uint(scales[g0 + 1])));
#else
      sc[q] = float(scales[g0]);
      bi[q] = float(biases[g0]);
#endif
    }
    _Pragma("clang loop unroll(full)")
    for (short s = 0; s < 8; ++s) {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; ++j) {
        xa[j] = xr[0][s][j];
        xa[4 + j] = xr[1][s][j];
      }
      _Pragma("clang loop unroll(full)")
      for (short q = 0; q < 4; ++q) {
        // Columns fn..fn+3 of step s are nibbles 4s..4s+3 of the lane's
        // 32 contiguous K: one 16-bit field.
        const uint h = (wv[q][s >> 1] >> (16 * (s & 1))) & 0xFFFFu;
        const uint u01 = (h & 0xFu) | ((h & 0xF0u) << 12);
        const uint u23 = ((h >> 8) & 0xFu) | ((h & 0xF000u) << 4);
#if MLX2_NVFP4
        const half2 v01 = as_type<half2>(((u01 << 12) & 0x80008000u) |
                                         ((u01 << 9) & 0x0E000E00u));
        const half2 v23 = as_type<half2>(((u23 << 12) & 0x80008000u) |
                                         ((u23 << 9) & 0x0E000E00u));
        // The 2^14 unfold in fp32: an E4M3 group scale reaches 448, and
        // 448 * 2^14 overflows half (real exports carry such scales; the
        // dequantized weight, at most 6 * 448, fits half).
        const float sh = float(sc[q][s >> 2]) * 16384.0f;
        const half2 w01 = half2(float2(v01) * sh);
        const half2 w23 = half2(float2(v23) * sh);
#else
        // Affine weights are dequantized to half: |s * 15 + b| above 65504
        // overflows and weights below ~6e-5 lose precision (1.5% at 1e-6 in
        // review); real drafter weights are ~1e-2.  Drafter-only numerics.
        // 0x6400 | n is the half 1024 + n: exact integer nibbles.
        const half2 q01 = as_type<half2>(u01 | 0x64006400u) - half2(1024.0h);
        const half2 q23 = as_type<half2>(u23 | 0x64006400u) - half2(1024.0h);
        const half2 w01 = half2(fma(float2(q01), float2(sc[q]), float2(bi[q])));
        const half2 w23 = half2(fma(float2(q23), float2(sc[q]), float2(bi[q])));
#endif
        wb[q * 4 + 0] = w01.x;
        wb[q * 4 + 1] = w01.y;
        wb[q * 4 + 2] = w23.x;
        wb[q * 4 + 3] = w23.y;
      }
      op.run(xa, wb, acc);
    }
  }

  if (SGK > 1) {
    threadgroup float red[(SGK > 1 ? SGK - 1 : 1) * 512];
    if (sg > 0) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 16; ++e)
        red[(int(sg) - 1) * 512 + int(lane) * 16 + e] = acc[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg > 0)
      return;
    for (int g = 0; g < SGK - 1; ++g) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 16; ++e)
        acc[e] += red[g * 512 + int(lane) * 16 + e];
    }
  }

  float mul = 1.0f;
#if MLX2_NVFP4
  mul = gscale[0];
#endif
  _Pragma("clang loop unroll(full)")
  for (short tn = 0; tn < 2; ++tn)
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; ++j) {
      const int col = n_base + tn * 16 + fn + j;
      float add = 0.0f;
#if MLX2_HAS_BIAS
      add = float(bias[col]);
#endif
      if (live0)
        y[size_t(fm) * N + col] = static_cast<T>(fma(acc[tn * 8 + j], mul, add));
      if (live1)
        y[size_t(fm + 8) * N + col] = static_cast<T>(fma(acc[tn * 8 + 4 + j], mul, add));
    }
"""


@functools.lru_cache(maxsize=1)
def nax_available() -> bool:
    """M5-class GPU (``applegpu_g17`` or newer): MLX's own NAX gate."""
    try:
        arch = str(mx.device_info().get("architecture", ""))
    except Exception:  # pragma: no cover - no Metal device
        return False
    digits = "".join(ch for ch in arch.removeprefix("applegpu_g")[:3] if ch.isdigit())
    return arch.startswith("applegpu_g") and digits != "" and int(digits) >= 17



def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    value = (env.get(ENV, "off") or "off").strip().lower()
    if value not in ("on", "off"):
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def _kernel(nvfp4: bool, has_bias: bool, engine: str = "simdgroup"):
    key = (nvfp4, has_bias, engine)
    with _LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            names = ["x", "w", "scales"]
            names += ["gscale"] if nvfp4 else ["biases"]
            names += ["bias"] if has_bias else []
            names += ["params"]
            defines = (
                "#define MMA_T float\n#define MMA_IN simdgroup_float8x8\n"
                f"#define MLX2_NVFP4 {int(nvfp4)}\n"
                f"#define MLX2_HAS_BIAS {int(has_bias)}\n"
            )
            kernel = mx.fast.metal_kernel(
                name=f"mlx2_drafter_rows_qmv_{engine}_{'nvfp4' if nvfp4 else 'affine'}"
                + ("_bias" if has_bias else ""),
                input_names=names,
                output_names=["y"],
                header=_NAX_HEADER if engine == "nax" else _HEADER,
                source=defines + (_NAX_SOURCE if engine == "nax" else _SOURCE),
            )
            _KERNELS[key] = kernel
        return kernel


def layout_reason(k: int, n: int, *, mode: str, bits: int, group_size: int):
    """``None`` when ``(k, n, mode, bits, group_size)`` is covered."""
    if bits != 4:
        return "bits"
    if mode == "affine":
        if group_size not in (32, 64, 128):
            return "group_size"
    elif mode == "nvfp4":
        if group_size != 16:
            return "group_size"
    else:
        return "mode"
    if k % CHUNK or n % 8:
        return "shape"
    return None


def _tile(n: int, k: int, rows: int, engine: str) -> tuple[int, int]:
    """``(NT, SGK)``: column tiles per simdgroup, simdgroups splitting K."""
    if engine == "nax":
        return 1, (4 if k // CHUNK >= 8 else 1)
    nt = 2 if n % 16 == 0 else 1
    return nt, 4 if k // CHUNK >= 4 else 1


@functools.lru_cache(maxsize=256)
def _params(k: int, n: int):
    return mx.array([k, n], dtype=mx.int32)


def _engine(n: int) -> str:
    return "nax" if n % 32 == 0 and nax_available() else "simdgroup"


def rows_qmv(x, weight, scales, biases=None, *, mode: str, group_size: int,
             bias=None, global_scale=None, tile=None, engine=None):
    """``x [..., R, K]`` with ``prod(leading) * R`` in 2..16 -> ``[..., N]``.

    Affine: ``biases`` are the quantization biases.  NVFP4: ``global_scale``
    is the FP32 per-tensor scale (or ``None`` for 1).  ``bias`` is the
    linear bias.  The caller checks :func:`layout_reason` first.
    """
    k = int(x.shape[-1])
    n = int(weight.shape[0])
    rows = x.size // k
    if not MIN_ROWS <= rows <= MAX_ROWS:
        raise ValueError("drafter rows qmv takes 2..16 rows")
    nvfp4 = mode == "nvfp4"
    flat = mx.contiguous(x.reshape(rows, k))
    inputs = [flat, weight, scales]
    if nvfp4:
        gs = mx.array(1.0, dtype=mx.float32) if global_scale is None else global_scale
        inputs.append(gs.astype(mx.float32).reshape(1))
    else:
        inputs.append(biases)
    if bias is not None:
        inputs.append(bias)
    inputs.append(_params(k, n))
    if engine is None:
        engine = _engine(n)
    nt, sgk = _tile(n, k, rows, engine) if tile is None else tile
    cols = (32 if engine == "nax" else 8) * nt
    (y,) = _kernel(nvfp4, bias is not None, engine)(
        inputs=inputs,
        template=[("T", x.dtype), ("GS", group_size), ("ROWS", rows),
                  ("NT", nt), ("SGK", sgk)],
        grid=(32 * sgk, n // cols, 1),
        threadgroup=(32 * sgk, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*x.shape[:-1], n)


def _admit(x, n):
    """Route this call to the kernel?  2..16 rows on the GPU, half inputs.

    The ``simdgroup`` engine (no NAX, or ``N % 32``) is admitted only up to
    one 8-row tile: two row tiles double its MMA work, and on the M5 Max it
    then lost to stock at some shapes.
    """
    if x.dtype not in (mx.bfloat16, mx.float16) or x.ndim < 2:
        return False
    rows = x.size // int(x.shape[-1])
    if not MIN_ROWS <= rows <= MAX_ROWS or mx.default_device() != mx.gpu:
        return False
    return rows <= 8 or _engine(n) == "nax"


class DrafterRowsQuantizedLinear(nn.QuantizedLinear):
    """``nn.QuantizedLinear`` routing 2..16 rows to :func:`rows_qmv`."""

    def __call__(self, x):
        if _admit(x, int(self["weight"].shape[0])) and "biases" in self:
            _COUNTS["kernel_calls"] += 1
            return rows_qmv(
                x, self["weight"], self["scales"], self["biases"], mode="affine",
                group_size=self.group_size,
                bias=self["bias"] if "bias" in self else None,
            )
        _COUNTS["stock_calls"] += 1
        return super().__call__(x)


def _nvfp4_class():
    from ..nvfp4_linear import GROUP_SIZE, GlobalScaledNVFP4Linear

    class DrafterRowsNVFP4Linear(GlobalScaledNVFP4Linear):
        """``GlobalScaledNVFP4Linear`` routing 2..16 rows to :func:`rows_qmv`."""

        def __call__(self, x):
            if _admit(x, int(self["weight"].shape[0])):
                _COUNTS["kernel_calls"] += 1
                return rows_qmv(
                    x, self["weight"], self["scales"], mode="nvfp4",
                    group_size=GROUP_SIZE, global_scale=self["global_scale"],
                    bias=self["bias"] if "bias" in self else None,
                )
            _COUNTS["stock_calls"] += 1
            return super().__call__(x)

    return DrafterRowsNVFP4Linear


_NVFP4_CLASS = None


def nvfp4_class():
    global _NVFP4_CLASS
    if _NVFP4_CLASS is None:
        _NVFP4_CLASS = _nvfp4_class()
    return _NVFP4_CLASS


def install(model) -> dict:
    """Swap compatible 4-bit linears of ``model`` in place; return a receipt.

    Only the class changes (parameters, paths and ``isinstance`` checks stay),
    so the weights and their identity are untouched.  Modules whose layout
    is not covered keep their class and the stock call.
    """
    from ..nvfp4_linear import GROUP_SIZE, GlobalScaledNVFP4Linear

    nvfp4 = nvfp4_class()
    installed, declined = [], {}
    for path, module in model.named_modules():
        if type(module) is nn.QuantizedLinear:
            n, packed = module["weight"].shape
            k = packed * 32 // module.bits
            # The QuantizedLinear route is the affine kernel; an nvfp4-mode
            # QuantizedLinear (no biases) would only ever take the stock call.
            why = (
                f"QuantizedLinear mode {module.mode}" if module.mode != "affine"
                else layout_reason(k, n, mode=module.mode, bits=module.bits,
                                   group_size=module.group_size)
            )
            target = DrafterRowsQuantizedLinear
        elif type(module) is GlobalScaledNVFP4Linear:
            n, packed = module["weight"].shape
            k = packed * 8
            why = layout_reason(k, n, mode="nvfp4", bits=4, group_size=GROUP_SIZE)
            target = nvfp4
        else:
            continue
        if why is None:
            module.__class__ = target
            installed.append(path)
        else:
            declined[path] = why
    return {
        "schema": "mlx2.drafter-rows-qmv.v1",
        "switch": ENV,
        "rows": [MIN_ROWS, MAX_ROWS],
        "modules": sorted(installed),
        "module_count": len(installed),
        "declined": dict(sorted(declined.items())),
        "row_exact": False,
        "qualified": False,
    }


def install_if_enabled(model, environ=None):
    """:func:`install` when ``MLX2_DRAFTER_ROWS_QMV=on``; else ``None``."""
    receipt = install(model) if enabled(environ) else None
    model.drafter_rows_qmv = receipt
    return receipt


def status() -> dict:
    return dict(_COUNTS)


__all__ = [
    "DrafterRowsQuantizedLinear",
    "ENV",
    "MAX_ROWS",
    "MIN_ROWS",
    "enabled",
    "install",
    "install_if_enabled",
    "layout_reason",
    "nvfp4_class",
    "rows_qmv",
    "status",
]
