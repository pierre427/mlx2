# SPDX-License-Identifier: MIT
"""Exact multi-row (2..8) Source dense products with on-chip coefficient reuse.

Each threadgroup loads one coefficient chunk once and applies it to every
activation row. Per row, the floating-point operation sequence (per-lane
partition, accumulation order, SIMD reduction) is the installed MLX M=1
kernel's sequence, so row r of an R-row call is intended to equal the
ordinary one-row product bitwise. That equality is a GPU gate checked by
tests/test_deepseek_v41_source_multirow_dense_gpu.py, not an assumption.

Laws (the stock M=1 dispatch they mirror):
- ``mxfp8_fast``: ``fp_qmv_fast`` (packs_per_thread=2, block 256 values,
  partial last tile), which the installed MLX selects whenever K % 256 == 0.
- ``mxfp8``: ``fp_qmv`` (packs_per_thread=1, block 128, tail-shift tile).
- ``gemv``: non-transposed ``GEMVKernel`` BN=1/SM=1/SN=32/TM=4/TN=4 with
  F32 or BF16 coefficients and F32 accumulation (the Source head/WoA M1 law).

Reimplemented from installed MLX 0.32.4 ``fp_quantized.h``/``gemv.h``
(Apple, MIT). Provenance: provenance/deepseek-v41-source-multirow-dense.json.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

PROFILE = "source_multirow_dense_exact_v1"
MAX_ROWS = 8

_FP8 = r"""
inline float mlx2_mr_e4m3(uint8_t b) {
  // Exact E4M3 -> float, same value as MLX fp8_e4m3::operator float().
  ushort v = ushort(b & 127);
  ushort sign = ushort(((b >> 7) & 1)) << 15;
  ushort u = ushort((v << 7) | (((v + 1) >> 7) << 14) | sign);
  half converted = as_type<half>(u);
  return static_cast<float>(static_cast<half>(converted * 256.0h));
}
inline float mlx2_mr_e8m0(uint8_t b) {
  uint32_t out = (b == 0 ? 0x400000u : (uint32_t(b) << 23));
  return as_type<float>(out);
}
"""


def _qmv_source(rows, fast):
    packs = 2 if fast else 1
    return f"""
constexpr int R = {rows};
constexpr int SIMD = 32, NSG = 2, RPS = 4;
constexpr int PACKS = {packs};
constexpr int VPT = 4 * PACKS;            // values per thread (8-bit pack factor 4)
constexpr int BLOCK = VPT * SIMD;
constexpr int SSTEP = 32 / VPT;           // group_size / values_per_thread
const int K = int(dims[0]), N = int(dims[1]);
const int in_w = K;                       // bytes per coefficient row
const int in_g = K / 32;
const uint tgy = threadgroup_position_in_grid.y;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const device uint8_t* wb = (const device uint8_t*)w;
int out_row = int(tgy) * (NSG * RPS) + int(sg) * RPS;
if (out_row >= N) return;
int last_row = RPS - 1;
{"last_row = min(last_row, N - 1 - out_row);" if fast else "out_row = min(N - RPS, out_row);"}
const device uint8_t* ws = wb + out_row * in_w + lane * PACKS * 4;
const device uint8_t* sc = scales + out_row * in_g + lane / SSTEP;
const device float* xb = x + lane * VPT;
float result[R][RPS];
for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) result[r][j] = 0.0f;
{"for (int k = 0; k < K; k += BLOCK) {" if fast else "int k = 0; for (; k < K - BLOCK; k += BLOCK) {"}
  float wv[RPS][VPT];
  float s[RPS];
  for (int j = 0; j < RPS; j++) {{
    const int src = min(j, last_row);
    const device uint8_t* wl = ws + src * in_w;
    for (int i = 0; i < VPT; i++) wv[j][i] = mlx2_mr_e4m3(wl[i]);
    s[j] = mlx2_mr_e8m0(sc[src * in_g]);
  }}
  for (int r = 0; r < R; r++) {{
    float xt[VPT];
    const device float* xr = xb + r * K;
    for (int i = 0; i < VPT; i++) xt[i] = xr[i];
    for (int j = 0; j < RPS; j++) {{
      float accum = 0.0f;
      for (int i = 0; i < VPT; i++) accum += xt[i] * wv[j][i];
      result[r][j] += s[j] * accum;
    }}
  }}
  ws += BLOCK; sc += BLOCK / 32; xb += BLOCK;
}}
{"" if fast else '''
{
  const int remaining = clamp(int(K - k - int(lane) * VPT), 0, VPT);
  if (remaining > 0) {
    float wv[RPS][VPT];
    float s[RPS];
    for (int j = 0; j < RPS; j++) {
      const device uint8_t* wl = ws + j * in_w;
      for (int i = 0; i < remaining; i++) wv[j][i] = mlx2_mr_e4m3(wl[i]);
      s[j] = mlx2_mr_e8m0(sc[j * in_g]);
    }
    for (int r = 0; r < R; r++) {
      float xt[VPT];
      const device float* xr = xb + r * K;
      for (int i = 0; i < remaining; i++) xt[i] = xr[i];
      for (int i = remaining; i < VPT; i++) xt[i] = 0.0f;
      for (int j = 0; j < RPS; j++) {
        float accum = 0.0f;
        for (int i = 0; i < remaining; i++) accum += xt[i] * wv[j][i];
        result[r][j] += s[j] * accum;
      }
    }
  }
}
'''}
for (int r = 0; r < R; r++) {{
  for (int j = 0; j < RPS; j++) {{
    float v = simd_sum(result[r][j]);
    if (lane == 0 && j <= last_row) y[r * N + out_row + j] = v;
  }}
}}
"""


def _gemv_source(rows, bm):
    return f"""
constexpr int R = {rows};
constexpr int BM = {bm}, SN = 32, TM = 4, TN = 4;
constexpr int blockM = BM * TM;
constexpr int blockN = SN * TN;
const int K = int(dims[0]), N = int(dims[1]);
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint group = threadgroup_position_in_grid.z;
int bn = int(lane) * TN;
int out_row = int(threadgroup_position_in_grid.x) * blockM + int(sg) * TM;
if (out_row >= N) return;
out_row = out_row + TM <= N ? out_row : N - TM;
const device auto* mat = weight + size_t(group) * N * K + size_t(out_row) * K;
const device float* vec = x + size_t(group) * K;
const int vstride = int(dims[2]);          // row stride of x (G*K)
float result[R][TM];
for (int r = 0; r < R; r++) for (int t = 0; t < TM; t++) result[r][t] = 0.0f;
const int n_iter = K / blockN;
const int leftover = K - blockN * n_iter;
for (int i = 0; i < n_iter; ++i) {{
  float vc[R][TN];
  for (int r = 0; r < R; r++)
    for (int t = 0; t < TN; t++) vc[r][t] = vec[r * vstride + bn + t];
  int mat_offset = 0;
  for (int tm = 0; tm < TM; tm++) {{
    float inter[TN];
    for (int t = 0; t < TN; t++) inter[t] = static_cast<float>(mat[mat_offset + bn + t]);
    for (int r = 0; r < R; r++)
      for (int t = 0; t < TN; t++) result[r][tm] += inter[t] * vc[r][t];
    mat_offset += K;
  }}
  bn += blockN;
}}
if (leftover > 0) {{
  float vc[R][TN];
  for (int r = 0; r < R; r++)
    for (int t = 0; t < TN; t++) vc[r][t] = bn + t < K ? vec[r * vstride + bn + t] : 0.0f;
  for (int tm = 0; tm < TM; tm++) {{
    float inter[TN];
    for (int t = 0; t < TN; t++) inter[t] = bn + t < K ? static_cast<float>(mat[tm * K + bn + t]) : 0.0f;
    for (int r = 0; r < R; r++)
      for (int t = 0; t < TN; t++) result[r][tm] += inter[t] * vc[r][t];
  }}
}}
for (int r = 0; r < R; r++)
  for (int tm = 0; tm < TM; tm++)
    for (ushort s = SN / 2; s >= 1; s >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], s);
if (lane == 0) {{
  const int ostride = int(dims[3]);        // row stride of out (G*N)
  for (int r = 0; r < R; r++)
    for (int tm = 0; tm < TM; tm++) out[r * ostride + group * N + out_row + tm] = result[r][tm];
}}
"""


def qmv_law(n, k):
    """Stock M=1 MXFP8 law selected by installed MLX for this geometry."""
    if type(n) is not int or type(k) is not int or n < 8 or k < 256 or k % 32:
        raise ValueError("Source multirow MXFP8 requires N>=8, K>=256 and K%32==0")
    return "mxfp8_fast" if k % 256 == 0 else "mxfp8"


def gemv_bm(n, k):
    """Stock non-transposed GEMV block rows; only the BN=1 domain is admitted."""
    if type(n) is not int or type(k) is not int or n < 4 or k <= 64 or k >= 16 * n:
        raise ValueError("Source multirow GEMV admits only stock BN=1 geometry")
    return 8 if n >= 4096 else 4


@lru_cache(maxsize=64)
def _qmv_kernel(rows, fast):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx2_source_multirow_qmv_{'fast' if fast else 'tail'}_r{rows}",
        input_names=["x", "w", "scales", "dims"], output_names=["y"],
        header="#include <metal_simdgroup>\n" + _FP8, source=_qmv_source(rows, fast),
        ensure_row_contiguous=True)


@lru_cache(maxsize=64)
def _gemv_kernel(rows, bm, coefficient):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx2_source_multirow_gemv_{coefficient}_bm{bm}_r{rows}",
        input_names=["x", "weight", "dims"], output_names=["out"],
        header="#include <metal_simdgroup>\n", source=_gemv_source(rows, bm),
        ensure_row_contiguous=True)


def _require_gpu():
    import mlx.core as mx
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("Source multirow dense requires the owned Metal GPU")


def qmv(x, q, scales):
    """x[R,K] F32 (already QAT-prepared) times original MXFP8 q/scales -> F32[R,N]."""
    import mlx.core as mx
    if (not isinstance(x, mx.array) or x.ndim != 2 or x.dtype != mx.float32
            or not 1 <= x.shape[0] <= MAX_ROWS):
        raise ValueError("Source multirow MXFP8 activation must be F32[R<=8,K]")
    rows, k = x.shape
    if (q.ndim != 2 or q.dtype != mx.uint32 or q.shape[1] * 4 != k
            or scales.ndim != 2 or scales.dtype != mx.uint8 or scales.shape != (q.shape[0], k // 32)):
        raise ValueError("Source multirow MXFP8 requires original group32 MXFP8")
    n = q.shape[0]
    law = qmv_law(n, k)
    _require_gpu()
    groups = (n + 7) // 8
    return _qmv_kernel(rows, law == "mxfp8_fast")(
        inputs=[x, q, scales, mx.array([k, n], mx.uint32)],
        grid=(32, 2 * groups, 1), threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]


def gemv(x, weight, *, groups=1):
    """x[R,G,K] F32 times F32/BF16 weight[G*N,K] -> F32[R,G,N] (stock BN=1 law)."""
    import mlx.core as mx
    if (not isinstance(x, mx.array) or x.ndim != 3 or x.dtype != mx.float32
            or not 1 <= x.shape[0] <= MAX_ROWS or x.shape[1] != groups):
        raise ValueError("Source multirow GEMV activation must be F32[R<=8,G,K]")
    rows, _, k = x.shape
    if (weight.ndim != 2 or weight.dtype not in (mx.float32, mx.bfloat16)
            or weight.shape[1] != k or weight.shape[0] % groups):
        raise ValueError("Source multirow GEMV coefficient geometry mismatch")
    n = weight.shape[0] // groups
    bm = gemv_bm(n, k)
    _require_gpu()
    block_m = bm * 4
    coefficient = "bf16" if weight.dtype == mx.bfloat16 else "f32"
    return _gemv_kernel(rows, bm, coefficient)(
        inputs=[x, weight, mx.array([k, n, groups * k, groups * n], mx.uint32)],
        grid=((n + block_m - 1) // block_m * bm * 32, 1, groups), threadgroup=(bm * 32, 1, 1),
        output_shapes=[(rows, groups, n)], output_dtypes=[mx.float32])[0]


def plan(kind, rows, n, k):
    """CPU inventory of a multirow call: coefficient bytes read once, not R times."""
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("Source multirow dense requires rows1..8")
    law = qmv_law(n, k) if kind == "mxfp8" else f"gemv_bm{gemv_bm(n, k)}"
    coefficient = {"mxfp8": n * k + n * k // 32, "f32": 4 * n * k, "bf16": 2 * n * k}[kind]
    return dict(profile=PROFILE, kind=kind, law=law, rows=rows, n=n, k=k,
                coefficient_bytes_per_call=coefficient, ordinary_row_calls_replaced=rows,
                coefficient_bytes_avoided=(rows - 1) * coefficient,
                exactness="per-row bitwise vs stock M=1 (GPU gate)", qualified=False)


@lru_cache(maxsize=1)
def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
