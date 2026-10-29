# SPDX-License-Identifier: MIT
"""Source WoA width-one GEMV that reads the original MXFP8 codes directly.

The selected Source profile expands each ``attn_output_a`` MXFP8 tensor to BF16
at load and consumes it with the ``source_dense`` BF16-coefficient F32 GEMV.
This candidate keeps the original E4M3 codes (uint32-packed, four per word) and
the row-repeated E8M0 group-32 scales resident instead, and decodes each
coefficient in registers to the same F32 value the BF16 expansion would load.

Numerical law (decode ``M=1`` only):

* coefficient ``w = float(e8m0) * float(e4m3)`` -- the exact product MLX
  ``fp_dequantize`` computes before its BF16 store. With admitted scales
  (``SCALE_MIN..SCALE_MAX``) every product is a normal F32 with at most four
  significant bits, so the BF16 store is exact and ``float(bf16(w)) == w``;
* lane/loop/shuffle law copied from the pinned installed MLX ``GEMVKernel``
  instantiation used by ``source_dense`` for ``wo_a`` (BM4 BN1 SM1 SN32 TM4 TN4,
  K blocks of 128, simd shuffle-down tree 16..1, no threadgroup reduction).

The adapter refuses scale bytes outside the admitted range at load, so the
candidate is bitwise identical to the BF16-expansion GEMV by construction;
``tests/test_deepseek_v41_source_dense_mxfp8_metal.py`` checks that claim on
real Metal with synthetic tensors. Every other row count is caller-owned.
Provenance: provenance/deepseek-v41-source-woa-mxfp8.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY = "mxfp8_gemv_v1"
GROUPS, N, K = 8, 1024, 4096
GROUP_SIZE = 32
# E8M0 byte b encodes 2**(b-127). b>=10 keeps the smallest nonzero E4M3
# magnitude (2**-9) normal in F32 and BF16; b<=246 keeps 448*2**(b-127) finite.
SCALE_MIN, SCALE_MAX = 10, 246
GEOMETRY = dict(bm=4, bn=1, sm=1, sn=32, tm=4, tn=4)
LIBRARY_SHA256 = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"
HEADER_PINS = {
    "gemv.h": "296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65",
    "fp8.h": "9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae",
}

_HEADER = r"""// Original project kernel; decode law mirrors installed MLX fp8.h
// (fp8_e4m3::operator float16_t, fp8_e8m0::operator float), MIT,
// provenance/source-mxfp4-installed-MLX-LICENSE.txt.
namespace mlx2_source_mxfp8 {
using namespace metal;
inline float e4m3(uint bits) {
  uint16_t v = uint16_t(bits & 127u);
  uint16_t sign_bit = uint16_t(((bits >> 7) & 1u) << 15);
  uint16_t u = uint16_t((v << 7) | (((v + 1) >> 7) << 14) | sign_bit);
  half converted = as_type<half>(u);
  half value = converted * 256.0;
  return static_cast<float>(value);
}
inline float e8m0(uchar bits) {
  uint out = (bits == 0 ? 0x400000u : (uint(bits) << 23));
  return as_type<float>(out);
}
} // namespace mlx2_source_mxfp8
"""

_SOURCE = r"""
constexpr int G=__G__, N=__N__, K=__K__, KQ=K/4, KS=K/__GS__;
constexpr int TM=4, TN=4, SN=32, BLOCK_M=16, BLOCK_N=128;
const uint group=threadgroup_position_in_grid.z;
const int simd_gid=int(simdgroup_index_in_threadgroup);
const int simd_lid=int(thread_index_in_simdgroup);
const device float* in_vec=x+group*K;
const device uint* codes=q+size_t(group)*N*KQ;
const device uchar* sc=s+size_t(group)*N*KS;
device float* out_vec=out+group*N;
thread float result[TM]={0};
thread float inter[TN];
thread float v_coeff[TN];
const int bm=simd_gid*TM;
int bn=simd_lid*TN;
int out_row=int(threadgroup_position_in_grid.x)*BLOCK_M+bm;
if(out_row>=N) return;
out_row=out_row+TM<=N?out_row:N-TM;
codes+=size_t(out_row)*KQ;
sc+=size_t(out_row)*KS;
for(int i=0;i<K/BLOCK_N;++i){
  _Pragma("clang loop unroll(full)")
  for(int tn=0;tn<TN;tn++){ v_coeff[tn]=static_cast<float>(in_vec[bn+tn]); }
  int code_offset=0, scale_offset=0;
  _Pragma("clang loop unroll(full)")
  for(int tm=0;tm<TM;tm++){
    const uint packed=codes[code_offset+bn/4];
    const float scale=mlx2_source_mxfp8::e8m0(sc[scale_offset+bn/__GS__]);
    _Pragma("clang loop unroll(full)")
    for(int tn=0;tn<TN;tn++){
      inter[tn]=scale*mlx2_source_mxfp8::e4m3((packed>>(8*tn))&255u);
    }
    _Pragma("clang loop unroll(full)")
    for(int tn=0;tn<TN;tn++){ result[tm]+=inter[tn]*v_coeff[tn]; }
    code_offset+=KQ;
    scale_offset+=KS;
  }
  bn+=BLOCK_N;
}
_Pragma("clang loop unroll(full)")
for(int tm=0;tm<TM;tm++){
  _Pragma("clang loop unroll(full)")
  for(ushort sn=(SN/2);sn>=1;sn>>=1){ result[tm]+=simd_shuffle_down(result[tm],sn); }
}
if(simd_lid==0){
  _Pragma("clang loop unroll(full)")
  for(int tm=0;tm<TM;tm++){ out_vec[out_row+tm]=static_cast<float>(result[tm]); }
}
"""


def plan(rows=1):
    """CPU-only geometry; one decode row of the eight-group WoA projection."""
    if type(rows) is not int or rows != 1:
        raise ValueError("Source MXFP8 WoA GEMV admits one token row only")
    threads = GEOMETRY["bm"] * 32
    block_m = GEOMETRY["bm"] * GEOMETRY["tm"]
    return {"kind": "wo_a", "algorithm": "gemv_mxfp8", "rows": 1, "groups": GROUPS, "n": N, "k": K,
            "grid": (N // block_m * threads, 1, GROUPS), "threadgroup": (threads, 1, 1),
            "output_shape": (1, 1, GROUPS, N), "input_dtype": "float32", "output_dtype": "float32",
            "code_storage_bytes": GROUPS * N * K, "scale_storage_bytes": GROUPS * N * K // GROUP_SIZE,
            "bf16_expansion_bytes": 0, "gemv_geometry": dict(GEOMETRY), "profile": "candidate-unqualified"}


def _geometry(x, q, scales):
    import mlx.core as mx
    if (getattr(x, "dtype", None) != mx.float32 or tuple(getattr(x, "shape", ())) != (1, 1, GROUPS, K)):
        raise ValueError("Source MXFP8 WoA activation geometry/dtype mismatch")
    if (getattr(q, "dtype", None) != mx.uint32 or tuple(getattr(q, "shape", ())) != (GROUPS * N, K // 4)):
        raise ValueError("Source MXFP8 WoA code geometry/dtype mismatch")
    if (getattr(scales, "dtype", None) != mx.uint8
            or tuple(getattr(scales, "shape", ())) != (GROUPS * N, K // GROUP_SIZE)):
        raise ValueError("Source MXFP8 WoA scale geometry/dtype mismatch")
    return plan(1)


def supports(x, q, scales, *, check_device=True):
    """Structural/device gate only; False lets the caller keep the expansion path."""
    try:
        _geometry(x, q, scales)
    except (ValueError, AttributeError, TypeError):
        return False
    if not check_device:
        return True
    from .source_dense import device_supported
    return device_supported()


def validate_scale_bytes(values):
    """Host-side load guard: refuse scale bytes whose decode is not exact BF16."""
    import numpy as np
    raw = np.asarray(values)
    if raw.dtype != np.uint8 or raw.size == 0:
        raise ValueError("Source MXFP8 WoA scales must be nonempty E8M0 bytes")
    low, high = int(raw.min()), int(raw.max())
    if low < SCALE_MIN or high > SCALE_MAX:
        raise ValueError("Source MXFP8 WoA scale byte outside exact BF16 envelope: "
                         f"{low}..{high} not within {SCALE_MIN}..{SCALE_MAX}")
    return low, high


@lru_cache(maxsize=1)
def implementation_identity():
    """Bind the installed MLX build, pinned headers and this module's source."""
    import importlib.metadata
    import mlx.core as mx
    if importlib.metadata.version("mlx") != "0.32.4":
        raise RuntimeError("Source MXFP8 WoA MLX version changed; re-audit required")
    root = Path(mx.__file__).resolve().parent
    if hashlib.sha256((root / "lib/libmlx.dylib").read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise RuntimeError("Source MXFP8 WoA MLX build changed; re-audit required")
    headers = root / "include/mlx/backend/metal/kernels"
    for name in HEADER_PINS:
        if hashlib.sha256((headers / name).read_bytes()).hexdigest() != HEADER_PINS[name]:
            raise RuntimeError("Source MXFP8 WoA pinned MLX header changed; re-audit required")
    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pins = repr(sorted(HEADER_PINS.items()))
    return hashlib.sha256((POLICY + ":" + LIBRARY_SHA256 + ":" + own + ":" + pins).encode()).hexdigest()


def kernel_spec():
    """Generated Metal source without compiling or scheduling GPU work."""
    p = plan(1)
    source = (_SOURCE.replace("__G__", str(GROUPS)).replace("__N__", str(N))
              .replace("__K__", str(K)).replace("__GS__", str(GROUP_SIZE)))
    return {**p, "header": _HEADER, "source": source,
            "input_names": ["x", "q", "s"], "output_names": ["out"]}


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    spec = kernel_spec()
    return mx.fast.metal_kernel(name="mlx2_source_woa_mxfp8_gemv_r1", input_names=spec["input_names"],
        output_names=spec["output_names"], header=spec["header"], source=spec["source"],
        ensure_row_contiguous=True)


def project(x, q, scales):
    """x F32[1,1,8,4096], codes U32[8192,1024], scales U8[8192,128] -> F32[1,1,8,1024]."""
    import mlx.core as mx
    p = _geometry(x, q, scales)
    from .source_dense import device_supported
    if not device_supported():
        raise ValueError("Source MXFP8 WoA candidate requires owned M5 Max Metal device")
    implementation_identity()
    return _kernel()(inputs=[x, q, scales], grid=p["grid"], threadgroup=p["threadgroup"],
                     output_shapes=[p["output_shape"]], output_dtypes=[mx.float32])[0]


def maybe_project(x, q, scales):
    """Return None on a structural/device refusal; ordinary fallback is caller-owned."""
    if not supports(x, q, scales):
        return None
    return project(x, q, scales)


def expand_bf16(q, scales):
    """The official convert.py expansion, used for every non-decode row count."""
    import mlx.core as mx
    return mx.dequantize(q, scales=scales, mode="mxfp8", dtype=mx.bfloat16)
