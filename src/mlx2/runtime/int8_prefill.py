"""Instance-scoped W8A8 int8 prefill on Apple M5 GPU neural accelerators (NAX).

Mined from ``mlx-lm-unified`` (``mlx_lm/int8_prefill.py``, branch ``unified``,
see docs/PROVENANCE.md).  For prefill-sized calls on selected projections of
*one* model, the stock (quantized or bf16) matmul is replaced by an
int8 x int8 -> int32 GEMM that runs on the M5 GPU neural accelerators through
Metal Performance Primitives ``matmul2d``:

- activations: per-row dynamic symmetric int8 (custom kernel);
- weights: per-output-channel symmetric int8 with the exact per-channel absmax,
  built by a fused kernel straight from the resident packed affine weights
  (any of 2/3/4/5/6/8 bits, group size 32/64/128) or from a bf16
  ``nn.Linear`` weight -- never through a bf16 intermediate;
- int32 accumulation, scales (and bias) applied in-register, bf16 output.

What differs from the source overlay:

- **No global monkeypatch and no env knobs.**  ``apply(model, policy)`` swaps
  the ``__class__`` of the selected module *instances* of that model to a thin
  subclass and returns a handle; ``remove(handle)`` restores them.  A draft
  model (or any other model) hosted in the same process is untouched.
  Parameter names and ``isinstance`` checks are unchanged.
- **Explicit policy.**  :class:`Int8PrefillPolicy` is default off; enabling it
  on a device without Metal 4 tensor ops fails closed at ``apply`` time with a
  compile-and-verify probe, never a silent no-op.
- **Eligibility** extends beyond 4-bit ``QuantizedLinear`` to every affine
  bit-width that packs as a little-endian bitstream (6-bit and 8-bit are the
  served Xing formats) and to plain bf16 ``nn.Linear``.
- **Model-neutral scope.**  ``"mlp"`` / ``"all"`` are resolved from module
  paths (or an adapter-provided selector), not from hard-coded Qwen shapes.
- Per-handle caches keyed by live module identity (the handle holds the
  modules), so model swaps cannot resurrect stale scales.

Numerics: int8 prefill is **approximate** relative to the stock kernels
(``Fidelity.APPROXIMATE``).  Only calls with ``rows >= row_threshold`` take the
int8 path; decode and speculative-verify blocks (bounded far below the
threshold, see :func:`validate_decode_row_bound`) keep the stock kernels and
bit-exact numerics.  Prefix state produced under int8 prefill must live in its
own APCv2 namespace (:func:`apc_semantic_fingerprint`).

Routed MoE experts (``SwitchGLU`` / ``gather_qmm``) stay on the stock kernels;
see ``EXPERTS_NOTE``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "mlx2.int8-prefill.v1"
SCOPES = ("mlp", "all")
CACHE_MODES = ("auto", "none", "ttl", "resident")
DEFAULT_ROW_THRESHOLD = 512
_SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
_TM, _TN, _NSIMD = 128, 128, 8
# lm_head-sized outputs are excluded (vocab projections dominate logits and
# are served by the stock kernels); embeddings are never Linear modules.
MAX_OUT = 32768
MIN_DIM = 256

EXPERTS_NOTE = (
    "routed MoE experts (SwitchGLU / gather_qmm / gather_mm) stay on stock "
    "kernels: at the default 2048-token prefill chunk the per-call int8 "
    "requantization of all experts (~3 ms/layer at 64x3584x1024 6-bit) eats "
    "the projected grouped-GEMM gain, and a resident int8 expert copy costs "
    "~0.7 GB/layer; not implemented"
)

# Path segments that classify a projection.  ``mlp`` scope: dense MLPs and
# shared experts.  ``all`` adds attention / linear-attention / mixer
# projections.  Excluded segments are never touched in any scope.
MLP_SEGMENTS = frozenset(
    {
        "mlp",
        "feed_forward",
        "ffn",
        "shared_expert",
        "shared_experts",
        "shared_mlp",
        "dense_mlp",
    }
)
EXCLUDED_SEGMENTS = frozenset(
    {
        "lm_head",
        "embed_tokens",
        "embedding",
        "embeddings",
        "wte",
        "vision_tower",
        "vision_model",
        "visual",
        "audio_tower",
        "multi_modal_projector",
        "mtp",
        "mtp_layers",
        "router",
        "gate",  # MoE router (``mlp.gate``); not the ``gate_proj`` MLP input
        "switch_mlp",
        "experts",
    }
)


class Int8PrefillError(RuntimeError):
    """A requested int8 prefill configuration cannot be honoured (fail closed)."""


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Int8PrefillPolicy:
    """Server/adapter-selected int8 prefill policy.  Default off."""

    enabled: bool = False
    scope: str = "mlp"
    row_threshold: int = DEFAULT_ROW_THRESHOLD
    cache: str = "auto"
    ttl_s: float = 120.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("int8 prefill enabled must be a boolean")
        if self.scope not in SCOPES:
            raise ValueError(f"int8 prefill scope must be one of {SCOPES}")
        if (
            isinstance(self.row_threshold, bool)
            or not isinstance(self.row_threshold, int)
            or self.row_threshold < 2
        ):
            raise ValueError("int8 prefill row_threshold must be an integer >= 2")
        if self.cache not in CACHE_MODES:
            raise ValueError(f"int8 prefill cache must be one of {CACHE_MODES}")
        if (
            isinstance(self.ttl_s, bool)
            or not isinstance(self.ttl_s, (int, float))
            or not math.isfinite(self.ttl_s)
            or self.ttl_s <= 0
        ):
            raise ValueError("int8 prefill ttl_s must be finite and positive")

    @classmethod
    def from_value(cls, value) -> Int8PrefillPolicy:
        """``None``/``False``/``"off"`` -> disabled; ``"mlp"``/``"all"`` ->
        enabled with that scope; a mapping of the dataclass fields."""
        if value is None or value is False or value == "off":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value not in SCOPES:
                raise ValueError(
                    f"int8 prefill must be 'off' or one of {SCOPES}, got {value!r}"
                )
            return cls(enabled=True, scope=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {
                "enabled",
                "scope",
                "row_threshold",
                "cache",
                "ttl_s",
            }
            if unknown:
                raise ValueError(f"unknown int8 prefill policy keys: {sorted(unknown)}")
            return cls(**dict(value))
        raise ValueError("int8 prefill policy must be 'off', a scope, or a mapping")

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "scope": self.scope,
            "row_threshold": self.row_threshold,
            "cache": self.cache,
            "ttl_s": float(self.ttl_s),
        }

    @property
    def revision(self) -> str:
        """Numerics identity: policy fields that change outputs + kernel source.

        ``cache``/``ttl_s`` only change memory lifetime, not numerics, so they
        are not part of the revision (entries stay reusable across them)."""
        payload = json.dumps(
            {
                "schema": SCHEMA,
                "enabled": self.enabled,
                "scope": self.scope,
                "row_threshold": self.row_threshold,
                "kernels": KERNEL_REVISION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def fidelity(self):
        from ..contracts import Fidelity

        return Fidelity.APPROXIMATE if self.enabled else Fidelity.EXACT


def apc_semantic_fingerprint(base, policy: Int8PrefillPolicy | None):
    """APCv2 semantic namespace for prefix state produced under ``policy``.

    Disabled: ``base`` unchanged (exact namespaces keep their identity).
    Enabled: a distinct namespace bound to the numerics revision, so entries
    (in memory and on disk -- the persistent block signature folds the key)
    produced under int8 prefill are never served to an exact route, nor to a
    route with a different scope/threshold/kernel revision."""
    if policy is None or not policy.enabled:
        return base
    return (base, "int8-prefill", policy.revision)


def validate_decode_row_bound(policy: Int8PrefillPolicy, max_decode_rows: int):
    """Fail closed unless every decode / verify block stays on stock kernels.

    ``max_decode_rows`` is the largest row count one decode or speculative
    verify forward can present (lanes x (draft length + slack)).  If it could
    reach the threshold, verify blocks would silently turn approximate."""
    if not policy.enabled:
        return
    if max_decode_rows >= policy.row_threshold:
        raise Int8PrefillError(
            f"int8 prefill row_threshold {policy.row_threshold} does not exceed "
            f"the largest decode/verify block ({max_decode_rows} rows); raise the "
            "threshold or reduce lanes/draft length"
        )


def adapter_scopes(adapter) -> frozenset:
    """Scopes an adapter declares via ``int8_prefill_supported()``.  Default: none."""
    declared = getattr(adapter, "int8_prefill_supported", None)
    if declared is None:
        return frozenset()
    scopes = declared() if callable(declared) else declared
    if isinstance(scopes, str):
        scopes = (scopes,)
    scopes = frozenset(scopes or ())
    unknown = scopes - set(SCOPES)
    if unknown:
        raise ValueError(f"adapter declares unknown int8 prefill scopes: {sorted(unknown)}")
    return scopes


# --------------------------------------------------------------------------
# Device gating
# --------------------------------------------------------------------------


def device_support() -> tuple[bool, str]:
    """(supported, reason) for Metal 4 tensor-op int8 GEMM on this process."""
    try:
        import mlx.core as mx
    except Exception as error:  # pragma: no cover - mlx is a hard dependency
        return False, f"mlx unavailable: {error}"
    try:
        if not mx.metal.is_available():
            return False, "Metal is unavailable"
        if mx.default_device() != mx.gpu:
            return False, "default device is not the GPU"
        info = mx.device_info()
    except Exception as error:
        return False, f"device query failed: {error}"
    arch = str(info.get("architecture", ""))
    name = str(info.get("device_name", ""))
    match = re.match(r"applegpu_g(\d+)", arch)
    if match is not None:
        if int(match.group(1)) >= 17:
            return True, f"{name} ({arch})"
        return False, f"{name or 'unknown'} ({arch}) lacks Metal 4 tensor ops"
    if re.search(r"\bM([5-9]|\d{2,})\b", name):
        return True, name
    return False, f"{name or 'unknown device'} ({arch or 'unknown arch'}) is not M5-class"


_probe_lock = threading.Lock()
_probe_result: tuple[bool, str] | None = None


def require_supported_device():
    """Raise :class:`Int8PrefillError` unless the NAX int8 GEMM compiles and
    reproduces an exact integer reference on this device (cached)."""
    global _probe_result
    ok, reason = device_support()
    if not ok:
        raise Int8PrefillError(
            f"int8 NAX prefill requires an Apple M5-class GPU with Metal 4 "
            f"tensor ops; {reason}"
        )
    with _probe_lock:
        if _probe_result is None:
            _probe_result = _probe_kernels()
        ok, detail = _probe_result
    if not ok:
        raise Int8PrefillError(f"int8 NAX prefill kernel probe failed: {detail}")
    return reason


def _probe_kernels() -> tuple[bool, str]:
    import mlx.core as mx

    try:
        # Small integer-valued operands: every product and sum is exact in
        # fp32, so the kernel must reproduce the reference bit-for-bit.
        m, n, k = 130, 128, 64
        xq = (mx.arange(m * k) % 7 - 3).astype(mx.int8).reshape(m, k)
        wq = (mx.arange(n * k) % 5 - 2).astype(mx.int8).reshape(n, k)
        ones_m = mx.ones((m,), dtype=mx.float32)
        ones_n = mx.ones((n,), dtype=mx.float32)
        got = _int8_gemm(xq, ones_m, wq, ones_n).astype(mx.float32)
        ref = xq.astype(mx.float32) @ wq.astype(mx.float32).T
        mx.eval(got, ref)
        if not bool(mx.array_equal(got, ref).item()):
            return False, "int8 GEMM disagrees with the integer reference"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "ok"


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup (256 threads) per row: absmax reduce, then quantize.  Used
# for activations and for bf16 nn.Linear weights (per-output-channel rows).
_QUANT_SRC = """
    constexpr int K = {K};
    constexpr int NTH = 256;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    const device {T}* xrow = x + size_t(row) * K;

    float amax = 0.0f;
    for (int i = tid; i < K; i += NTH) {{
        amax = max(amax, fabs(float(xrow[i])));
    }}
    amax = simd_max(amax);

    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);

    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) xs[row] = scale;

    device int8_t* qrow = xq + size_t(row) * K;
    for (int i = tid; i < K; i += NTH) {{
        qrow[i] = int8_t(clamp(rint(float(xrow[i]) * inv), -127.0f, 127.0f));
    }}
"""

# Threadgroup computes a 128x128 output tile with 8 simdgroups; matmul2d
# loops over K internally.  Edge M tiles are bounds-checked by the tensor
# extents and the epilogue guard, so any M works; N must be a multiple of 128.
_GEMM_SRC = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = 128;
    constexpr int TN = 128;

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<8>> op;

    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));

    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);

    auto cT = op.get_destination_cooperative_tensor<
        decltype(tA), decltype(tB), int32_t>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

    op.run(tA, tB, cT);

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            if (m < M && n < N) {{
                float v = float(cT[i]) * xs[m] * ws[n];
                {BIAS_LINE}
                out[size_t(m) * N + n] = bfloat(v);
            }}
        }}
    }}
"""

# Fused requantization from packed affine weights (little-endian bitstream,
# any bit width) to per-channel symmetric int8 with the exact per-channel
# absmax.  One threadgroup (256 threads) per output channel, two passes over
# the packed row (absmax, then quantize); no bf16 intermediate.  A pack unit
# of UW words holds UV values and never straddles a quantization group.
_REQUANT_SRC = """
    constexpr int K = {K};
    constexpr int BITS = {BITS};
    constexpr int GS = {GS};
    constexpr int UW = {UW};
    constexpr int UV = {UV};
    constexpr int NU = K / UV;
    constexpr int WPR = K * BITS / 32;
    constexpr uint MASK = (BITS == 32) ? 0xFFFFFFFFu : ((1u << BITS) - 1u);
    constexpr int NTH = 256;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    const device uint32_t* prow = packed + size_t(row) * WPR;
    const device {T}* srow = scales + size_t(row) * (K / GS);
    const device {T}* brow = biases + size_t(row) * (K / GS);

    float amax = 0.0f;
    for (int u = tid; u < NU; u += NTH) {{
        uint32_t w[UW + 1];
#pragma unroll
        for (int i = 0; i < UW; ++i) w[i] = prow[u * UW + i];
        w[UW] = 0;
        int g = (u * UV) / GS;
        float s = float(srow[g]);
        float b = float(brow[g]);
#pragma unroll
        for (int j = 0; j < UV; ++j) {{
            int off = j * BITS;
            int wi = off >> 5;
            int sh = off & 31;
            uint32_t v = w[wi] >> sh;
            if (sh + BITS > 32) v |= w[wi + 1] << (32 - sh);
            v &= MASK;
            amax = max(amax, fabs(float(v) * s + b));
        }}
    }}
    amax = simd_max(amax);
    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);

    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) ws[row] = scale;

    device int8_t* orow = out + size_t(row) * K;
    for (int u = tid; u < NU; u += NTH) {{
        uint32_t w[UW + 1];
#pragma unroll
        for (int i = 0; i < UW; ++i) w[i] = prow[u * UW + i];
        w[UW] = 0;
        int g = (u * UV) / GS;
        float s = float(srow[g]);
        float b = float(brow[g]);
        device int8_t* o = orow + u * UV;
#pragma unroll
        for (int j = 0; j < UV; ++j) {{
            int off = j * BITS;
            int wi = off >> 5;
            int sh = off & 31;
            uint32_t v = w[wi] >> sh;
            if (sh + BITS > 32) v |= w[wi + 1] << (32 - sh);
            v &= MASK;
            o[j] = int8_t(clamp(rint((float(v) * s + b) * inv), -127.0f, 127.0f));
        }}
    }}
"""

KERNEL_REVISION = hashlib.sha256(
    (_HEADER + _QUANT_SRC + _GEMM_SRC + _REQUANT_SRC).encode()
).hexdigest()[:16]

_kernel_lock = threading.Lock()
_quant_kernels: dict = {}
_gemm_kernels: dict = {}
_requant_kernels: dict = {}


def _tname(dtype):
    import mlx.core as mx

    return {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[dtype]


def _pack_unit(bits: int) -> tuple[int, int]:
    """(words, values) of the smallest whole-word pack unit for ``bits``."""
    words = bits // math.gcd(bits, 32)
    return words, words * 32 // bits


def _quantize_rows(x):
    """Per-row symmetric int8 of a 2-D float array -> (int8 [M,K], fp32 [M])."""
    import mlx.core as mx

    m, k = x.shape
    key = (k, _tname(x.dtype))
    kernel = _quant_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _quant_kernels.get(key)
            if kernel is None:
                kernel = _quant_kernels[key] = mx.fast.metal_kernel(
                    name=f"mlx2_i8p_rowquant_{k}_{key[1]}",
                    input_names=["x"],
                    output_names=["xq", "xs"],
                    header=_HEADER,
                    source=_QUANT_SRC.format(K=k, T=key[1]),
                )
    xq, xs = kernel(
        inputs=[x],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return xq, xs


def _int8_gemm(xq, xs, wq, ws, bias=None):
    import mlx.core as mx

    m, k = xq.shape
    n = wq.shape[0]
    key = (n, k, bias is not None)
    kernel = _gemm_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _gemm_kernels.get(key)
            if kernel is None:
                names = ["xq", "wq", "xs", "ws", "m_dim"]
                bias_line = ""
                if bias is not None:
                    names.append("bias")
                    bias_line = "v += float(bias[n]);"
                kernel = _gemm_kernels[key] = mx.fast.metal_kernel(
                    name=f"mlx2_i8p_gemm_{n}x{k}{'_b' if bias is not None else ''}",
                    input_names=names,
                    output_names=["out"],
                    header=_HEADER,
                    source=_GEMM_SRC.format(N=n, K=k, BIAS_LINE=bias_line),
                )
    inputs = [xq, wq, xs, ws, mx.array([m], dtype=mx.int32)]
    if bias is not None:
        inputs.append(bias)
    return kernel(
        inputs=inputs,
        grid=(n // _TN * 32 * _NSIMD, (m + _TM - 1) // _TM, 1),
        threadgroup=(32 * _NSIMD, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _requant_packed(weight, scales, biases, *, bits, group_size):
    """(int8 [N,K], fp32 [N]) from resident packed affine weights, fused."""
    import mlx.core as mx

    n, wpr = weight.shape
    k = wpr * 32 // bits
    uw, uv = _pack_unit(bits)
    tname = _tname(scales.dtype)
    key = (k, bits, group_size, tname)
    kernel = _requant_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _requant_kernels.get(key)
            if kernel is None:
                kernel = _requant_kernels[key] = mx.fast.metal_kernel(
                    name=f"mlx2_i8p_requant_{k}_b{bits}_g{group_size}_{tname}",
                    input_names=["packed", "scales", "biases"],
                    output_names=["out", "ws"],
                    header=_HEADER,
                    source=_REQUANT_SRC.format(
                        K=k, BITS=bits, GS=group_size, UW=uw, UV=uv, T=tname
                    ),
                )
    wq, ws = kernel(
        inputs=[weight, scales, biases],
        grid=(n * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, k), (n,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return wq, ws


# --------------------------------------------------------------------------
# Eligibility and selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModuleSpec:
    path: str
    kind: str  # "quantized" | "linear"
    n: int
    k: int
    bits: int | None
    group_size: int | None


def module_spec(path: str, module) -> tuple[_ModuleSpec | None, str]:
    """(spec, "") when ``module`` is structurally int8-prefill eligible, else
    (None, reason).  Pure shape/layout arithmetic: no device work."""
    import mlx.core as mx
    from mlx import nn

    cls = type(module)
    if cls is nn.QuantizedLinear:
        if getattr(module, "mode", "affine") != "affine":
            return None, "non-affine quantization mode"
        if "biases" not in module:
            return None, "missing affine biases"
        bits, gs = int(module.bits), int(module.group_size)
        if bits not in _SUPPORTED_BITS:
            return None, f"unsupported bits {bits}"
        n, wpr = module["weight"].shape
        k = wpr * 32 // bits
        _, uv = _pack_unit(bits)
        if gs % uv or k % gs:
            return None, "group size does not align with the pack unit"
        if module["scales"].dtype not in (mx.bfloat16, mx.float16, mx.float32):
            return None, "unsupported scale dtype"
        kind = "quantized"
    elif cls is nn.Linear:
        weight = module["weight"]
        if weight.dtype != mx.bfloat16:
            return None, f"linear weight dtype {weight.dtype} is not bfloat16"
        n, k = weight.shape
        bits = gs = None
        kind = "linear"
    else:
        return None, f"unsupported module type {cls.__name__}"
    if "bias" in module and module["bias"].dtype not in (
        mx.bfloat16,
        mx.float16,
        mx.float32,
    ):
        return None, "unsupported bias dtype"
    if n % _TN:
        return None, f"output dim {n} is not a multiple of {_TN}"
    if k % 32:
        return None, f"input dim {k} is not a multiple of 32"
    if n > MAX_OUT:
        return None, f"output dim {n} exceeds {MAX_OUT} (vocab-sized)"
    if min(n, k) < MIN_DIM:
        return None, f"min dim {min(n, k)} below {MIN_DIM}"
    return _ModuleSpec(path, kind, int(n), int(k), bits, gs), ""


def default_select(scope: str) -> Callable[[str, Any], bool]:
    """Path-based projection classifier for ``scope``."""

    def select(path: str, module) -> bool:
        segments = path.split(".")
        if any(seg in EXCLUDED_SEGMENTS for seg in segments):
            return False
        if scope == "all":
            return True
        return any(seg in MLP_SEGMENTS for seg in segments)

    return select


# --------------------------------------------------------------------------
# Instance-scoped installation
# --------------------------------------------------------------------------

_STATE_ATTR = "_mlx2_int8_prefill"
_subclasses: dict = {}


def _int8_subclass(base):
    sub = _subclasses.get(base)
    if sub is not None:
        return sub

    base_call = base.__call__

    def __call__(self, x):
        ref = self.__dict__.get(_STATE_ATTR)
        handle = ref() if ref is not None else None
        if handle is None:
            return base_call(self, x)
        return handle._call(self, x, base_call)

    sub = type(f"Int8Prefill{base.__name__}", (base,), {"__call__": __call__})
    sub._mlx2_int8_prefill_base = base
    _subclasses[base] = sub
    return sub


class _Bound:
    __slots__ = ("base", "module", "nbytes", "spec", "wq", "ws")

    def __init__(self, module, base, spec):
        self.module = module
        self.base = base
        self.spec = spec
        self.wq = None
        self.ws = None
        self.nbytes = 0


class Int8PrefillHandle:
    """Installed int8 prefill on one model.  Counters are host ints only."""

    def __init__(self, policy: Int8PrefillPolicy, *, device: str = ""):
        self.policy = policy
        self.device = device
        self.active = False
        self._bound: dict[int, _Bound] = {}
        self._skipped: dict[str, str] = {}
        self._lock = threading.Lock()
        self._act_entry = None
        self._last_use = time.monotonic()
        self._reaper_stop: threading.Event | None = None
        self.counts = {
            "engaged_calls": 0,
            "engaged_rows": 0,
            "fallback_rows": 0,
            "fallback_dtype": 0,
            "fallback_unbound": 0,
            "weight_builds": 0,
            "weight_evictions": 0,
            "activation_reuse": 0,
        }

    # -- hot path ---------------------------------------------------------
    def _call(self, module, x, base_call):
        bound = self._bound.get(id(module))
        if bound is None or bound.module is not module or not self.active:
            self.counts["fallback_unbound"] += 1
            return base_call(module, x)
        k = x.shape[-1]
        rows = x.size // k if k else 0
        if rows < self.policy.row_threshold:
            self.counts["fallback_rows"] += 1
            return base_call(module, x)
        import mlx.core as mx

        if x.dtype != mx.bfloat16:
            # The epilogue writes bf16; never change fp16/fp32 semantics.
            self.counts["fallback_dtype"] += 1
            return base_call(module, x)
        self._last_use = time.monotonic()
        wq, ws = self._weights(bound)
        xq, xs = self._activation(x, k)
        bias = module["bias"] if "bias" in module else None
        y = _int8_gemm(xq, xs, wq, ws, bias=bias)
        self.counts["engaged_calls"] += 1
        self.counts["engaged_rows"] += rows
        return y.reshape(*x.shape[:-1], bound.spec.n)

    def _build(self, bound):
        module, spec = bound.module, bound.spec
        self.counts["weight_builds"] += 1
        if spec.kind == "linear":
            return _quantize_rows(module["weight"])
        return _requant_packed(
            module["weight"],
            module["scales"],
            module["biases"],
            bits=spec.bits,
            group_size=spec.group_size,
        )

    def _cache_mode(self, bound) -> str:
        cache = self.policy.cache
        if cache == "auto":
            # bf16 weights: requantizing reads 2 B/elt per call, so keep the
            # int8 copy (half the bf16 footprint, accounted in weight_bytes).
            # Packed weights: the fused requant is cheap; build per call.
            return "resident" if bound.spec.kind == "linear" else "none"
        return cache

    def _weights(self, bound):
        if self._cache_mode(bound) == "none":
            # Built per call; the executor frees it after the GEMM consumes it.
            return self._build(bound)
        with self._lock:
            if bound.wq is None:
                import mlx.core as mx

                wq, ws = self._build(bound)
                mx.eval(wq, ws)
                bound.wq, bound.ws = wq, ws
                bound.nbytes = int(wq.nbytes + ws.nbytes)
            return bound.wq, bound.ws

    def _activation(self, x, k):
        # q/k/v and gate/up share one input tensor: quantize it once.  The
        # entry holds a strong reference, so the id() key cannot be reused
        # while it is cached; one entry bounds the retained tail.
        entry = self._act_entry  # single read: the reaper may clear it
        if entry is not None and entry[0] is x:
            self.counts["activation_reuse"] += 1
            return entry[1], entry[2]
        xq, xs = _quantize_rows(x.reshape(-1, k))
        self._act_entry = (x, xq, xs)
        return xq, xs

    # -- lifecycle --------------------------------------------------------
    def release_activation(self):
        self._act_entry = None

    def evict_weights(self) -> int:
        with self._lock:
            evicted = 0
            for bound in self._bound.values():
                if bound.wq is not None:
                    bound.wq = bound.ws = None
                    bound.nbytes = 0
                    evicted += 1
            self.counts["weight_evictions"] += evicted
        self.release_activation()
        return evicted

    def warmup(self) -> int:
        """Build the cached int8 weights now (modules whose cache mode is not none)."""
        built = 0
        for bound in self._bound.values():
            if self._cache_mode(bound) != "none":
                self._weights(bound)
                built += 1
        return built

    def _reaper(self, stop):
        interval = max(self.policy.ttl_s / 4.0, 1.0)
        while not stop.wait(interval):
            if time.monotonic() - self._last_use > self.policy.ttl_s:
                if self.evict_weights():
                    import mlx.core as mx

                    mx.clear_cache()

    def _start_reaper(self):
        if self.policy.cache == "ttl" and self._reaper_stop is None:
            self._reaper_stop = threading.Event()
            threading.Thread(
                target=self._reaper,
                args=(self._reaper_stop,),
                name="mlx2-int8-prefill-reaper",
                daemon=True,
            ).start()

    # -- reporting --------------------------------------------------------
    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(b.spec.path for b in tuple(self._bound.values())))

    def weight_bytes(self) -> int:
        return sum(b.nbytes for b in tuple(self._bound.values()))

    def resident_estimate_bytes(self) -> int:
        """Bytes the cached int8 weight copies occupy once all are built."""
        return sum(
            b.spec.n * (b.spec.k + 4)
            for b in tuple(self._bound.values())
            if self._cache_mode(b) != "none"
        )

    def transient_weight_bytes_max(self) -> int:
        """Largest int8 weight tensor one call builds and drops (cache none).

        This is memory that exists only during a prefill call and is not
        otherwise visible to admission."""
        return max(
            (
                b.spec.n * (b.spec.k + 4)
                for b in tuple(self._bound.values())
                if self._cache_mode(b) == "none"
            ),
            default=0,
        )

    def weight_copy_bytes(self) -> int:
        """Admission-facing int8 weight-copy bytes: resident copies once built
        plus the worst per-call transient copy."""
        return self.resident_estimate_bytes() + self.transient_weight_bytes_max()

    def status(self) -> dict:
        kinds: dict = {}
        for bound in tuple(self._bound.values()):
            label = (
                "bf16" if bound.spec.kind == "linear" else f"q{bound.spec.bits}"
            )
            kinds[label] = kinds.get(label, 0) + 1
        return {
            "schema": SCHEMA,
            **self.policy.as_dict(),
            "active": self.active,
            "fidelity": self.policy.fidelity.value,
            "revision": self.policy.revision if self.policy.enabled else None,
            "kernel_revision": KERNEL_REVISION,
            "device": self.device,
            "modules": len(self._bound),
            "module_kinds": kinds,
            "skipped": len(self._skipped),
            "experts": "stock",
            "weight_bytes": self.weight_bytes(),
            "resident_estimate_bytes": self.resident_estimate_bytes(),
            "transient_weight_bytes_max": self.transient_weight_bytes_max(),
            "weight_copy_bytes": self.weight_copy_bytes(),
            "counts": dict(self.counts),
        }

    def receipt(self) -> dict | None:
        """Route-level receipt fragment (``None`` when disabled)."""
        if not self.policy.enabled:
            return None
        return {
            "schema": SCHEMA,
            "enabled": True,
            "fidelity": self.policy.fidelity.value,
            "scope": self.policy.scope,
            "row_threshold": self.policy.row_threshold,
            "revision": self.policy.revision,
            "applies_to": "forward calls with rows >= row_threshold",
            "modules": len(self._bound),
            "weight_copy_bytes": self.weight_copy_bytes(),
        }


def apply(
    model,
    policy: Int8PrefillPolicy,
    *,
    select: Callable[[str, Any], bool] | None = None,
) -> Int8PrefillHandle:
    """Install int8 prefill on the eligible projections of ``model`` only.

    A disabled policy returns an inert handle (no modules touched).  An
    enabled policy fails closed on unsupported devices, when no module is
    eligible, or when ``model`` already carries an int8 prefill install."""
    policy = Int8PrefillPolicy.from_value(policy)
    handle = Int8PrefillHandle(policy)
    if not policy.enabled:
        return handle
    handle.device = require_supported_device()
    chooser = select or default_select(policy.scope)
    candidates = []
    for path, module in model.named_modules():
        if _STATE_ATTR in getattr(module, "__dict__", {}) or hasattr(
            type(module), "_mlx2_int8_prefill_base"
        ):
            raise Int8PrefillError(
                f"module {path!r} already carries an int8 prefill install"
            )
        if not chooser(path, module):
            continue
        spec, reason = module_spec(path, module)
        if spec is None:
            if reason and not reason.startswith("unsupported module type"):
                handle._skipped[path] = reason
            continue
        candidates.append((module, spec))
    if not candidates:
        raise Int8PrefillError(
            f"int8 prefill scope {policy.scope!r} selected no eligible projection"
        )
    ref = weakref.ref(handle)
    for module, spec in candidates:
        base = type(module)
        handle._bound[id(module)] = _Bound(module, base, spec)
        module.__dict__[_STATE_ATTR] = ref
        module.__class__ = _int8_subclass(base)
    handle.active = True
    handle._start_reaper()
    logger.info(
        "int8 NAX prefill installed on %d projections (scope %s, threshold %d, "
        "cache %s, %d skipped)",
        len(candidates),
        policy.scope,
        policy.row_threshold,
        policy.cache,
        len(handle._skipped),
    )
    return handle


def remove(handle: Int8PrefillHandle | None) -> bool:
    """Restore every module the handle installed on and drop its caches."""
    if handle is None or not handle.active:
        return False
    handle.active = False
    if handle._reaper_stop is not None:
        handle._reaper_stop.set()
        handle._reaper_stop = None
    for bound in handle._bound.values():
        module = bound.module
        module.__dict__.pop(_STATE_ATTR, None)
        if type(module) is _subclasses.get(bound.base):
            module.__class__ = bound.base
    handle.evict_weights()
    handle._bound.clear()
    return True


def apply_for_adapter(adapter, policy) -> Int8PrefillHandle:
    """Adapter-gated install: the adapter must declare the selected scope.

    The adapter opts in with ``int8_prefill_supported()`` (iterable of scopes)
    and may refine selection with ``int8_prefill_select(scope)`` returning a
    ``(path, module) -> bool`` predicate.  Only ``adapter.model`` is touched."""
    policy = Int8PrefillPolicy.from_value(policy)
    if not policy.enabled:
        return Int8PrefillHandle(policy)
    scopes = adapter_scopes(adapter)
    if policy.scope not in scopes:
        raise Int8PrefillError(
            f"model adapter does not declare int8 prefill scope {policy.scope!r}"
            + (f" (declares {sorted(scopes)})" if scopes else "")
        )
    selector = getattr(adapter, "int8_prefill_select", None)
    select = selector(policy.scope) if callable(selector) else None
    return apply(adapter.model, policy, select=select)



def max_decode_rows(*, max_lanes, config, speculation, prompt_lookup_policy=None):
    """Upper bound on rows one decode / speculative-verify forward presents.

    Verify blocks carry at most ``num_draft + 1`` rows per lane; one extra row
    of slack covers bonus/rollback tokens.  Ordinary decode is one row/lane."""
    draft = 0
    if speculation in ("self_mtp", "external_draft"):
        draft = int((config or {}).get("num_draft", 0) or 0)
    elif speculation == "prompt_lookup":
        draft = int((prompt_lookup_policy or {}).get("num_draft", 8) or 8)
    return int(max_lanes) * (draft + 2)


def bind_for_serving(
    adapter,
    policy,
    *,
    max_lanes,
    config,
    speculation,
    prompt_lookup_policy=None,
):
    """Serving-engine entry: validate, install on ``adapter.model`` and return
    ``(handle, settings fragment)``.  Raises (fail closed) on any refusal."""
    policy = Int8PrefillPolicy.from_value(policy)
    if not policy.enabled:
        raise ValueError("bind_for_serving requires an enabled policy")
    bound = max_decode_rows(
        max_lanes=max_lanes,
        config=config,
        speculation=speculation,
        prompt_lookup_policy=prompt_lookup_policy,
    )
    validate_decode_row_bound(policy, bound)
    handle = apply_for_adapter(adapter, policy)
    warmed = handle.warmup()
    settings = {
        **policy.as_dict(),
        "revision": policy.revision,
        "fidelity": policy.fidelity.value,
        "max_decode_rows": bound,
        "weight_copy_bytes": handle.weight_copy_bytes(),
    }
    logger.info(
        "int8 NAX prefill bound: %d projections, %d cached copies (%.1f MiB), "
        "decode/verify bound %d rows < threshold %d",
        len(handle.modules),
        warmed,
        handle.weight_bytes() / (1 << 20),
        bound,
        policy.row_threshold,
    )
    return handle, settings


def engine_status(engine) -> dict:
    """Host-only status for a serving engine (no device work, no syncs)."""
    handle = getattr(engine, "int8_prefill_handle", None)
    if handle is not None:
        return handle.status()
    policy = getattr(engine, "int8_prefill_policy", None) or Int8PrefillPolicy()
    return {
        "schema": SCHEMA,
        **policy.as_dict(),
        "active": False,
        "fidelity": policy.fidelity.value,
        "modules": 0,
        "counts": {},
    }
