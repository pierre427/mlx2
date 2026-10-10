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

Q8 in place (``q8_inplace``, default off; port of jundot/omlx #4350, see
provenance/omlx-4350-q8-a8-inplace.json): 8-bit / group-size-64 affine
``QuantizedLinear`` projections skip the requantization and read the packed
checkpoint words directly (byte XOR 0x80 -> q - 128, per-group fp32 affine
fold).  Activations get a Stage A with per-row or per-64-group scales
(``act_scale``) and per-group code sums.  The only copy is the group-major
scale/bias metadata (``metadata_copy_bytes``); ``weight_copy_bytes`` counts
no Q8 in-place module.  The mode, its activation scaling and its kernel
source enter the numerics revision (and so the APCv2 namespace) only when on.

Q4/Q5 in place (``q45_inplace``, default off; port of the omlx Q4/Q5 A8
path, see provenance/omlx-q45-a8-inplace.json): 4-bit and 5-bit /
group-size-64 affine projections read MLX's own packed bitstream in place
(unsigned codes straight into int8 fragments, fold ``s*acc + b*r``), with a
Stage A that writes activation codes in omlx's "v8" within-group K order.
Same metadata copy and accounting as Q8; ``inplace_only`` binds only the
projections an in-place kernel takes.  Not covered by ``auto``.
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
# ``--int8-prefill auto``: resolved per loaded model by ``resolve_auto``.
AUTO = "auto"
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
    # omlx #4350 port: 8-bit / GS64 affine projections read their packed
    # weights in place (no int8 weight copy) with Stage A activation scaling
    # ``act_scale``.  Other modules keep the requantization path.
    q8_inplace: bool = False
    act_scale: str = "group64"
    # Bind only the projections the in-place kernel takes; every other
    # projection stays on stock kernels instead of the requantization path
    # (whose per-channel requant costs ~10% top-1 agreement on prose).
    q8_only: bool = False
    # 4-bit / 5-bit GS64 affine projections read their packed weights in
    # place (omlx Q4/Q5 A8 kernels, Stage A v8 K order), with the same
    # ``act_scale``.  Default off; enters as_dict/revision only when on.
    q45_inplace: bool = False
    # With q45_inplace: bind only projections an in-place kernel takes (Q4/Q5,
    # plus Q8 when q8_inplace is also on); every other projection stays on
    # stock kernels, never the requantization path.
    inplace_only: bool = False

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
        if not isinstance(self.q8_inplace, bool):
            raise ValueError("int8 prefill q8_inplace must be a boolean")
        if self.act_scale not in Q8_ACT_SCALES:
            raise ValueError(f"int8 prefill act_scale must be one of {Q8_ACT_SCALES}")
        if self.q8_inplace and not self.enabled:
            raise ValueError("int8 prefill q8_inplace requires an enabled policy")
        if not isinstance(self.q8_only, bool):
            raise ValueError("int8 prefill q8_only must be a boolean")
        if self.q8_only and not self.q8_inplace:
            raise ValueError("int8 prefill q8_only requires q8_inplace")
        if not isinstance(self.q45_inplace, bool):
            raise ValueError("int8 prefill q45_inplace must be a boolean")
        if self.q45_inplace and not self.enabled:
            raise ValueError("int8 prefill q45_inplace requires an enabled policy")
        if not isinstance(self.inplace_only, bool):
            raise ValueError("int8 prefill inplace_only must be a boolean")
        if self.inplace_only and not self.q45_inplace:
            raise ValueError(
                "int8 prefill inplace_only requires q45_inplace (use q8_only for Q8 alone)"
            )
        if self.q8_only and self.q45_inplace:
            # One spelling per behaviour, so one revision per behaviour.
            raise ValueError(
                "int8 prefill q8_only binds Q8 alone; with q45_inplace use inplace_only"
            )

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
                "q8_inplace",
                "act_scale",
                "q8_only",
                "q45_inplace",
                "inplace_only",
            }
            if unknown:
                raise ValueError(f"unknown int8 prefill policy keys: {sorted(unknown)}")
            return cls(**dict(value))
        raise ValueError("int8 prefill policy must be 'off', a scope, or a mapping")

    def as_dict(self) -> dict:
        out = {
            "enabled": self.enabled,
            "scope": self.scope,
            "row_threshold": self.row_threshold,
            "cache": self.cache,
            "ttl_s": float(self.ttl_s),
        }
        if self.q8_inplace:
            # Only when on, so records of every existing policy are unchanged.
            out["q8_inplace"] = True
            out["act_scale"] = self.act_scale
            if self.q8_only:
                out["q8_only"] = True
        if self.q45_inplace:
            out["q45_inplace"] = True
            out["act_scale"] = self.act_scale
            if self.inplace_only:
                out["inplace_only"] = True
        return out

    @property
    def revision(self) -> str:
        """Numerics identity: policy fields that change outputs + kernel source.

        ``cache``/``ttl_s`` only change memory lifetime, not numerics, so they
        are not part of the revision (entries stay reusable across them).
        ``q8_inplace`` (with its activation scaling and kernel source) enters
        only when on, so every pre-existing policy keeps its revision."""
        fields = {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "scope": self.scope,
            "row_threshold": self.row_threshold,
            "kernels": KERNEL_REVISION,
        }
        if self.q8_inplace:
            fields["q8_inplace"] = {
                "act_scale": self.act_scale,
                "group_size": Q8_GROUP_SIZE,
                "kernels": Q8_KERNEL_REVISION,
            }
            if self.q8_only:
                # Which projections run int8 changes numerics.
                fields["q8_inplace"]["q8_only"] = True
        if self.q45_inplace:
            fields["q45_inplace"] = {
                "act_scale": self.act_scale,
                "group_size": Q8_GROUP_SIZE,
                "kernels": Q45_KERNEL_REVISION,
            }
            if self.inplace_only:
                fields["q45_inplace"]["inplace_only"] = True
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def fidelity(self):
        from ..contracts import Fidelity

        return Fidelity.APPROXIMATE if self.enabled else Fidelity.EXACT


def cli_value(scope: str, q8: str = "off"):
    """``--int8-prefill`` / ``--int8-prefill-q8`` -> a value for
    :meth:`Int8PrefillPolicy.from_value` (the scope string when q8 is off, so
    every existing invocation is unchanged)."""
    if scope == AUTO:
        if q8 not in (None, "off"):
            raise ValueError(
                "--int8-prefill-q8 applies to --int8-prefill mlp|all; "
                "'auto' always uses group64 on 8-bit gs64 projections"
            )
        return AUTO
    if q8 in (None, "off"):
        return scope
    if q8 not in Q8_ACT_SCALES:
        raise ValueError(f"--int8-prefill-q8 must be 'off' or one of {Q8_ACT_SCALES}")
    if scope in (None, "off"):
        raise ValueError("--int8-prefill-q8 requires --int8-prefill mlp or all")
    return {"enabled": True, "scope": scope, "q8_inplace": True, "act_scale": q8}


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
            "threshold or reduce lanes/draft length/self-MTP copy-draft span"
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
    reproduces an exact integer reference on this device (cached).  A Metal
    device fault during the probe propagates as itself and caches nothing."""
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
        from .models.served_exp import is_device_fault

        if is_device_fault(error):
            # A Metal out-of-memory or GPU timeout, not a kernel refusal:
            # nothing is cached, the caller recovers and probes again.
            raise
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

# --------------------------------------------------------------------------
# Q8 in-place kernels (omlx #4350 port; provenance/omlx-4350-q8-a8-inplace.json)
# --------------------------------------------------------------------------
#
# For 8-bit / group-size-64 affine QuantizedLinear the GEMM reads the
# checkpoint's packed uint32 words in place.  MLX affine dequantization is
# w = s * q + b with q in [0, 255]; XOR 0x80 on each byte gives q - 128 as a
# two's-complement int8, so per GS64 group
#
#     sum_k a_k w_k = s * (acc + 128 * r) + b * r
#
# where acc = sum a_k (q_k - 128) is the int32 tensor-op result and r is the
# group sum of the int8 activation codes (Stage A).  |acc| and 128 * |r| stay
# far below 2^24, so the fp32 add is exact.  Activations stay in checkpoint K
# order: a lane's 16 codes of a group are four consecutive words and every
# micro-K step uses the same K permutation for both operands, which a group
# sum does not see.
#
# Kept separate from KERNEL_REVISION so a policy with q8_inplace off keeps its
# exact revision; the q8 revision enters the policy revision only when on.

Q8_GROUP_SIZE = 64
Q8_ACT_SCALES = ("per_row", "group64")
_Q8_ACT_MODE = {"per_row": 0, "group64": 1}
# (WM, WN) simdgroup grids: BM = 32 * WM rows, BN = 32 * WN columns.  omlx
# instantiates the same seven (its variants 800-806) and defaults Q8 to 806,
# i.e. (1, 2).  Tiles do not change numerics: every output element folds the
# same groups in the same order.
Q8_TILES = ((2, 2), (4, 2), (2, 4), (4, 4), (1, 4), (8, 2), (1, 2))
Q8_DEFAULT_TILE = (1, 2)

_Q8_HEADER = _HEADER + """
// NAX 16x16 fragment coordinate of this lane: (column, row) of its first
// element; the lane owns rows {fm, fm + 8} x columns {fn .. fn + 3}.  Same
// mapping as mlx::steel::BaseNAXFrag::get_coord.
inline short2 mlx2_q8_frag_coord(uint lane) {
    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    return short2(fn, fm);
}
"""

# Stage A: one threadgroup (8 simdgroups) per row; each simdgroup owns whole
# GS64 groups (lane i holds codes i and i + 32).  Emits int8 codes [M, K] in
# checkpoint order, group sums Ra [G, M] int16 (|Ra| <= 64 * 127), and scales
# Sa ([M] per row, or [G, M] per group) in fp32.  Group-major Ra/Sa are
# written directly (omlx transposes them in a second op).
_Q8_STAGE_A_SRC = """
    constexpr int G = K / 64;
    constexpr int NSG = 8;
    const int M = m_dim[0];
    const int row = int(threadgroup_position_in_grid.x);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lid = int(thread_index_in_simdgroup);

    const device TX* xr = x + size_t(row) * K;
    device int8_t* qr = qa + size_t(row) * K;

    threadgroup float partial[NSG];
    float row_inv = 0.0f;
    if (ACT_MODE == 0) {
        float amax = 0.0f;
        for (int g = sg; g < G; g += NSG) {
            const int base = g * 64 + lid;
            amax = max(amax, fabs(float(xr[base])));
            amax = max(amax, fabs(float(xr[base + 32])));
        }
        amax = simd_max(amax);
        if (lid == 0) partial[sg] = amax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ramax = 0.0f;
        for (int i = 0; i < NSG; ++i) ramax = max(ramax, partial[i]);
        // An all-zero row has no scale: zero codes, the bias term carries it.
        row_inv = ramax > 0.0f ? (127.0f / ramax) : 0.0f;
        if (sg == 0 && lid == 0) sa[row] = ramax > 0.0f ? (ramax / 127.0f) : 0.0f;
    }

    for (int g = sg; g < G; g += NSG) {
        const int base = g * 64 + lid;
        const float x0 = float(xr[base]);
        const float x1 = float(xr[base + 32]);
        float inv = row_inv;
        if (ACT_MODE == 1) {
            const float amax = simd_max(max(fabs(x0), fabs(x1)));
            inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
            if (lid == 0) sa[size_t(g) * M + row] = amax > 0.0f ? (amax / 127.0f) : 0.0f;
        }
        const int q0 = int(clamp(rint(x0 * inv), -127.0f, 127.0f));
        const int q1 = int(clamp(rint(x1 * inv), -127.0f, 127.0f));
        qr[base] = int8_t(q0);
        qr[base + 32] = int8_t(q1);
        // Ra from the rounded codes, or the affine correction stops being exact.
        const int gsum = simd_sum(q0 + q1);
        if (lid == 0) ra[size_t(g) * M + row] = short(gsum);
    }
"""

# GEMM: WM x WN simdgroups, each a 32 x 32 output block (two 16 x 32 NAX
# fragments along M), one int32 tensor op per 16-wide micro-K step, fp32
# affine fold per GS64 group.  Packed weights [N, K/4] uint32 are read in
# place; scales/biases are read group-major [G, N] (the transposed metadata
# copy omlx keeps; reading [N, G] directly measured ~30% slower there).
# Inner loops must unroll fully (``clang loop unroll(full)``, as MLX's
# STEEL_PRAGMA_UNROLL): with a plain ``#pragma unroll`` the cooperative
# tensors and accumulators were indexed dynamically and the GEMM ran 3-5x
# slower (measured 2026-10-09, M5 Max).
_Q8_GEMM_SRC = """
    constexpr int TM = 2;
    constexpr int BM = TM * 16 * WM;
    constexpr int BN = 32 * WN;
    constexpr int G = K / 64;
    constexpr int WORDS = 16;           // 64 codes per group in uint32 words
    constexpr uint32_t CENTER = 0x80808080u;

    const int M = m_dim[0];
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int sg_m = simd_gid % WM;
    const int sg_n = simd_gid / WM;
    const int row_base = int(threadgroup_position_in_grid.y) * BM + sg_m * (TM * 16);
    const int col_base = int(threadgroup_position_in_grid.x) * BN + sg_n * 32;

    const short2 coord = mlx2_q8_frag_coord(thread_index_in_simdgroup);

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
        decltype(ct_a), decltype(ct_b), int32_t>();
    auto acc1 = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), int32_t>();

    float Cf[TM][16];
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) Cf[i][e] = 0.0f;
    }

    const int n_lane = col_base + int(coord.y);
    const device uint32_t* wbase = w + size_t(n_lane) * size_t(G * WORDS);
    constexpr int W_ROW = G * WORDS;
    const int n_run0 = col_base + int(coord.x);
    const int m_base = row_base + int(coord.y);
    // Which 16-code run of the group this lane owns (0..3).
    const int cx = int(coord.x) >> 2;

    for (int g = 0; g < G; ++g) {
        // Four operand rows (n_lane + {0, 8, 16, 24}), 16 codes each: one
        // aligned 16-byte load per row (group bases are 64-byte aligned).
        uint4 wg[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; ++q) {
            const device uint32_t* wr = wbase + (q & 1) * (8 * W_ROW)
                + (q >> 1) * (16 * W_ROW) + size_t(g) * WORDS;
            wg[q] = reinterpret_cast<const device uint4*>(wr)[cx];
        }
#pragma clang loop unroll(full)
        for (int t = 0; t < 4; ++t) {
#pragma clang loop unroll(full)
            for (int q = 0; q < 4; ++q) {
                const int base = (q >> 1) * 8 + (q & 1) * 4;
                const char4 quad = as_type<char4>(wg[q][t] ^ CENTER);
                ct_b[base + 0] = quad.x;
                ct_b[base + 1] = quad.y;
                ct_b[base + 2] = quad.z;
                ct_b[base + 3] = quad.w;
            }
#pragma clang loop unroll(full)
            for (int hf = 0; hf < 2; ++hf) {
#pragma clang loop unroll(full)
                for (int r = 0; r < 2; ++r) {
                    // Rows past M read row M - 1; those results are never stored.
                    const int m = min(m_base + hf * 16 + r * 8, M - 1);
                    const char4 quad = as_type<char4>(*reinterpret_cast<const device uint32_t*>(
                        qa + size_t(m) * K + size_t(g) * 64 + size_t(cx) * 16 + size_t(t) * 4));
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

        // st/bt hold >= 128 values (device); Ra/Sa may hold one value, which
        // MLX can bind in the constant space, hence auto.
        const device TS* srow = st + size_t(g) * N;
        const device TS* brow = bt + size_t(g) * N;
        const auto rrow = ra + size_t(g) * M;
        vec<TS, 4> sv[2];
        vec<TS, 4> bv[2];
#pragma clang loop unroll(full)
        for (int h = 0; h < 2; ++h) {
            const int n0 = n_run0 + h * 16;
            sv[h] = *reinterpret_cast<const device vec<TS, 4>*>(srow + n0);
            bv[h] = *reinterpret_cast<const device vec<TS, 4>*>(brow + n0);
        }
        float r_g[TM][2];
        float r_c[TM][2];
#pragma clang loop unroll(full)
        for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
            for (int r = 0; r < 2; ++r) {
                const int m = min(m_base + i * 16 + r * 8, M - 1);
                r_g[i][r] = float(rrow[m]);
                r_c[i][r] = 128.0f * r_g[i][r];
            }
        }
        if (ACT_MODE == 0) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 16; ++e) {
                const int r = (e & 7) >> 2;
                const float swc = float(sv[e >> 3][e & 3]);
                const float bwc = float(bv[e >> 3][e & 3]);
                Cf[0][e] = metal::fma(swc, float(acc0[e]) + r_c[0][r],
                                      metal::fma(bwc, r_g[0][r], Cf[0][e]));
                Cf[1][e] = metal::fma(swc, float(acc1[e]) + r_c[1][r],
                                      metal::fma(bwc, r_g[1][r], Cf[1][e]));
            }
        } else {
            const auto arow = sa + size_t(g) * M;
            float s_g[TM][2];
#pragma clang loop unroll(full)
            for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
                for (int r = 0; r < 2; ++r) {
                    const int m = min(m_base + i * 16 + r * 8, M - 1);
                    s_g[i][r] = arow[m];
                }
            }
#pragma clang loop unroll(full)
            for (int e = 0; e < 16; ++e) {
                const int r = (e & 7) >> 2;
                const float swc = float(sv[e >> 3][e & 3]);
                const float bwc = float(bv[e >> 3][e & 3]);
                Cf[0][e] = metal::fma(s_g[0][r],
                    metal::fma(swc, float(acc0[e]) + r_c[0][r], bwc * r_g[0][r]), Cf[0][e]);
                Cf[1][e] = metal::fma(s_g[1][r],
                    metal::fma(swc, float(acc1[e]) + r_c[1][r], bwc * r_g[1][r]), Cf[1][e]);
            }
        }
    }

#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) {
            const int ee = e & 7;
            const int r = ee >> 2;
            const int m = row_base + i * 16 + int(coord.y) + r * 8;
            if (m < M) {
                const int n = col_base + (e >> 3) * 16 + int(coord.x) + (ee & 3);
                float v = ACT_MODE == 0 ? sa[m] * Cf[i][e] : Cf[i][e];
                /*BIAS*/
                out[size_t(m) * N + n] = static_cast<TO>(v);
            }
        }
    }
"""

Q8_KERNEL_REVISION = hashlib.sha256(
    (_Q8_HEADER + _Q8_STAGE_A_SRC + _Q8_GEMM_SRC).encode()
).hexdigest()[:16]

# --------------------------------------------------------------------------
# Q4/Q5 in-place kernels (omlx Q4/Q5 A8 path, #3548/#3952; see
# provenance/omlx-q45-a8-inplace.json)
# --------------------------------------------------------------------------
#
# 4-bit and 5-bit / group-size-64 affine QuantizedLinear weights are read in
# MLX's own packed layout (a little-endian bitstream per row, K * BITS / 32
# uint32 words).  At GS64 a group is 8 (Q4) or 10 (Q5) whole words, so no
# code crosses a group boundary.  The codes are unsigned (0..15, 0..31) and
# land in int8 unchanged, so per GS64 group
#
#     sum_k a_k w_k = s * acc + b * r
#
# with acc the int32 tensor-op result and r the group sum of the int8
# activation codes.  Decoding into the tensor-op fragment wants step t of a
# lane's 16-code run to be codes 16c + 8*(t>>1) + 2j + (t&1) (Q4: the even or
# odd nibbles of one word; Q5: 5-bit fields 10 bits apart in a normalized
# 96-bit window), so Stage A writes the activation codes in that permuted
# order ("Stage A v8" in omlx) and the GEMM reads four contiguous bytes per
# step exactly as the Q8 kernel does.  The permutation stays inside a group,
# so r is unchanged.  Q4 and Q5 modules share one Stage A; Q8 (checkpoint K
# order) needs its own.
#
# Kept separate from KERNEL_REVISION and Q8_KERNEL_REVISION; enters the
# policy revision only when q45_inplace is on.

Q45_BITS = (4, 5)
Q45_TILES = Q8_TILES
# omlx's tuned defaults: variant 806 (WM 1 x WN 2) for Q4, 800 (2 x 2) for Q5.
Q45_DEFAULT_TILES = {4: (1, 2), 5: (2, 2)}

_Q45_HEADER = _Q8_HEADER + """
// One 5-bit code at compile-time bit offset OFF of a lane's normalized
// 96-bit Q5 window: one extract unless the field straddles a word.
template <int OFF>
inline int8_t mlx2_q5_window_code(uint3 w) {
    constexpr int wi = OFF >> 5;
    constexpr int sh = OFF & 31;
    const uint32_t lo = (wi == 0) ? w.x : ((wi == 1) ? w.y : w.z);
    if (sh <= 27) {
        return static_cast<int8_t>(metal::extract_bits(lo, uint(sh), 5u));
    }
    const uint32_t hi = (wi == 0) ? w.y : w.z;
    return static_cast<int8_t>(((lo >> sh) | (hi << (32 - sh))) & 0x1fu);
}
"""

_Q8_STAGE_A_STORE = """        qr[base] = int8_t(q0);
        qr[base + 32] = int8_t(q1);
"""
# Stage A v8: in-group code k = 16c + 8a + 2b + d goes to slot
# 16c + 8a + 4d + b.  Lane lid holds k = lid and lid + 32 (c + 2), whose
# slots are 32 apart.  Written directly (omlx quantizes, then permutes in a
# second contiguous copy).
_Q45_STAGE_A_STORE = """        const int slot = g * 64 + ((lid & 24) | ((lid & 1) << 2) | ((lid >> 1) & 3));
        qr[slot] = int8_t(q0);
        qr[slot + 32] = int8_t(q1);
"""
assert _Q8_STAGE_A_SRC.count(_Q8_STAGE_A_STORE) == 1
_Q45_STAGE_A_SRC = _Q8_STAGE_A_SRC.replace(_Q8_STAGE_A_STORE, _Q45_STAGE_A_STORE)

# Same tiling, tensor ops and epilogue as the Q8 GEMM; differs in the weight
# decode (packed Q4/Q5 words read in place) and in the fold (no centering).
_Q45_GEMM_SRC = """
    constexpr int TM = 2;
    constexpr int BM = TM * 16 * WM;
    constexpr int BN = 32 * WN;
    constexpr int G = K / 64;
    constexpr int WORDS = 2 * BITS;     // 64 codes per group in uint32 words

    const int M = m_dim[0];
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int sg_m = simd_gid % WM;
    const int sg_n = simd_gid / WM;
    const int row_base = int(threadgroup_position_in_grid.y) * BM + sg_m * (TM * 16);
    const int col_base = int(threadgroup_position_in_grid.x) * BN + sg_n * 32;

    const short2 coord = mlx2_q8_frag_coord(thread_index_in_simdgroup);

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
        decltype(ct_a), decltype(ct_b), int32_t>();
    auto acc1 = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), int32_t>();

    float Cf[TM][16];
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) Cf[i][e] = 0.0f;
    }

    const int n_lane = col_base + int(coord.y);
    const device uint32_t* wbase = w + size_t(n_lane) * size_t(G * WORDS);
    constexpr int W_ROW = G * WORDS;
    const int n_run0 = col_base + int(coord.x);
    const int m_base = row_base + int(coord.y);
    // Which 16-code run of the group this lane owns (0..3).
    const int cx = int(coord.x) >> 2;
    // Q5: the run's 80 bits start at bit 80 * cx of the group (word 5cx/2,
    // offset 0 or 16); three words always cover them.
    const int w5_bit = BITS == 5 ? 80 * cx : 0;
    const int w5_word = w5_bit >> 5;
    const int w5_sh = w5_bit & 31;

    for (int g = 0; g < G; ++g) {
        // Four operand rows (n_lane + {0, 8, 16, 24}).  Q4: the run is words
        // 2cx, 2cx + 1 of the group, one aligned uint2 load.  Q5: three words
        // normalized once so bit 0 is code 16cx (shift in two steps: a shift
        // by 32 is undefined and w5_sh is 0 for even cx).
        uint2 wg[4];
        uint3 wv[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; ++q) {
            const device uint32_t* wr = wbase + (q & 1) * (8 * W_ROW)
                + (q >> 1) * (16 * W_ROW) + size_t(g) * WORDS;
            if (BITS == 4) {
                wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
            } else {
                const uint32_t a0 = wr[w5_word];
                const uint32_t a1 = wr[w5_word + 1];
                const uint32_t a2 = wr[w5_word + 2];
                wv[q].x = (a0 >> w5_sh) | ((a1 << (31 - w5_sh)) << 1);
                wv[q].y = (a1 >> w5_sh) | ((a2 << (31 - w5_sh)) << 1);
                wv[q].z = a2 >> w5_sh;
            }
        }
#pragma clang loop unroll(full)
        for (int t = 0; t < 4; ++t) {
            // Step t: codes 16cx + 8*(t>>1) + 2j + (t&1), j = 0..3.
#pragma clang loop unroll(full)
            for (int q = 0; q < 4; ++q) {
                const int base = (q >> 1) * 8 + (q & 1) * 4;
                if (BITS == 4) {
                    const uint32_t word = wg[q][t >> 1];
                    const char4 quad = as_type<char4>(
                        ((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
                    ct_b[base + 0] = quad.x;
                    ct_b[base + 1] = quad.y;
                    ct_b[base + 2] = quad.z;
                    ct_b[base + 3] = quad.w;
                } else {
                    // Field j at bit 40*(t>>1) + 5*(t&1) + 10j of the window.
                    const uint3 v = wv[q];
                    if (t == 0) {
                        ct_b[base + 0] = mlx2_q5_window_code<0>(v);
                        ct_b[base + 1] = mlx2_q5_window_code<10>(v);
                        ct_b[base + 2] = mlx2_q5_window_code<20>(v);
                        ct_b[base + 3] = mlx2_q5_window_code<30>(v);
                    } else if (t == 1) {
                        ct_b[base + 0] = mlx2_q5_window_code<5>(v);
                        ct_b[base + 1] = mlx2_q5_window_code<15>(v);
                        ct_b[base + 2] = mlx2_q5_window_code<25>(v);
                        ct_b[base + 3] = mlx2_q5_window_code<35>(v);
                    } else if (t == 2) {
                        ct_b[base + 0] = mlx2_q5_window_code<40>(v);
                        ct_b[base + 1] = mlx2_q5_window_code<50>(v);
                        ct_b[base + 2] = mlx2_q5_window_code<60>(v);
                        ct_b[base + 3] = mlx2_q5_window_code<70>(v);
                    } else {
                        ct_b[base + 0] = mlx2_q5_window_code<45>(v);
                        ct_b[base + 1] = mlx2_q5_window_code<55>(v);
                        ct_b[base + 2] = mlx2_q5_window_code<65>(v);
                        ct_b[base + 3] = mlx2_q5_window_code<75>(v);
                    }
                }
            }
#pragma clang loop unroll(full)
            for (int hf = 0; hf < 2; ++hf) {
#pragma clang loop unroll(full)
                for (int r = 0; r < 2; ++r) {
                    // Rows past M read row M - 1; those results are never stored.
                    const int m = min(m_base + hf * 16 + r * 8, M - 1);
                    const char4 quad = as_type<char4>(*reinterpret_cast<const device uint32_t*>(
                        qa + size_t(m) * K + size_t(g) * 64 + size_t(cx) * 16 + size_t(t) * 4));
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

        const device TS* srow = st + size_t(g) * N;
        const device TS* brow = bt + size_t(g) * N;
        const auto rrow = ra + size_t(g) * M;
        vec<TS, 4> sv[2];
        vec<TS, 4> bv[2];
#pragma clang loop unroll(full)
        for (int h = 0; h < 2; ++h) {
            const int n0 = n_run0 + h * 16;
            sv[h] = *reinterpret_cast<const device vec<TS, 4>*>(srow + n0);
            bv[h] = *reinterpret_cast<const device vec<TS, 4>*>(brow + n0);
        }
        float r_g[TM][2];
#pragma clang loop unroll(full)
        for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
            for (int r = 0; r < 2; ++r) {
                const int m = min(m_base + i * 16 + r * 8, M - 1);
                r_g[i][r] = float(rrow[m]);
            }
        }
        if (ACT_MODE == 0) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 16; ++e) {
                const int r = (e & 7) >> 2;
                const float swc = float(sv[e >> 3][e & 3]);
                const float bwc = float(bv[e >> 3][e & 3]);
                Cf[0][e] = metal::fma(swc, float(acc0[e]),
                                      metal::fma(bwc, r_g[0][r], Cf[0][e]));
                Cf[1][e] = metal::fma(swc, float(acc1[e]),
                                      metal::fma(bwc, r_g[1][r], Cf[1][e]));
            }
        } else {
            const auto arow = sa + size_t(g) * M;
            float s_g[TM][2];
#pragma clang loop unroll(full)
            for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
                for (int r = 0; r < 2; ++r) {
                    const int m = min(m_base + i * 16 + r * 8, M - 1);
                    s_g[i][r] = arow[m];
                }
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
    }

#pragma clang loop unroll(full)
    for (int i = 0; i < TM; ++i) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 16; ++e) {
            const int ee = e & 7;
            const int r = ee >> 2;
            const int m = row_base + i * 16 + int(coord.y) + r * 8;
            if (m < M) {
                const int n = col_base + (e >> 3) * 16 + int(coord.x) + (ee & 3);
                float v = ACT_MODE == 0 ? sa[m] * Cf[i][e] : Cf[i][e];
                /*BIAS*/
                out[size_t(m) * N + n] = static_cast<TO>(v);
            }
        }
    }
"""

Q45_KERNEL_REVISION = hashlib.sha256(
    (_Q45_HEADER + _Q45_STAGE_A_SRC + _Q45_GEMM_SRC).encode()
).hexdigest()[:16]

_kernel_lock = threading.Lock()
_quant_kernels: dict = {}
_gemm_kernels: dict = {}
_requant_kernels: dict = {}
_q8_stage_kernels: dict = {}
_q8_gemm_kernels: dict = {}
_q45_stage_kernels: dict = {}
_q45_gemm_kernels: dict = {}


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


def _q8_prefix(**constants) -> str:
    lines = []
    for name, value in constants.items():
        if isinstance(value, str):
            lines.append(f"    typedef {value} {name};")
        else:
            lines.append(f"    constexpr int {name} = {int(value)};")
    return "\n".join(lines) + "\n"


def q8_stage_a(x, act_scale: str = "group64", m_dim=None):
    """Stage A for the Q8 in-place GEMM.

    ``x`` is a 2-D bf16/fp16 array [M, K] with K % 64 == 0.  Returns
    ``(qa int8 [M, K], sa fp32 [M] or [K/64, M], ra int16 [K/64, M])``."""
    import mlx.core as mx

    act_mode = _Q8_ACT_MODE[act_scale]
    m, k = x.shape
    if k % Q8_GROUP_SIZE:
        raise ValueError(f"q8 stage A needs K % 64 == 0, got {k}")
    tname = _tname(x.dtype)
    key = (k, tname, act_mode)
    kernel = _q8_stage_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _q8_stage_kernels.get(key)
            if kernel is None:
                kernel = _q8_stage_kernels[key] = mx.fast.metal_kernel(
                    name=f"mlx2_i8p_q8_stage_a_{k}_{tname}_am{act_mode}",
                    input_names=["x", "m_dim"],
                    output_names=["qa", "sa", "ra"],
                    header=_Q8_HEADER,
                    source=_q8_prefix(K=k, ACT_MODE=act_mode, TX=tname)
                    + _Q8_STAGE_A_SRC,
                )
    if m_dim is None:
        m_dim = mx.array([m], dtype=mx.int32)
    groups = k // Q8_GROUP_SIZE
    sa_shape = (m,) if act_mode == 0 else (groups, m)
    qa, sa, ra = kernel(
        inputs=[x, m_dim],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), sa_shape, (groups, m)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    )
    return qa, sa, ra


def q8_metadata(scales, biases):
    """Group-major [K/64, N] copies of the checkpoint's [N, K/64] metadata."""
    import mlx.core as mx

    return mx.contiguous(scales.T), mx.contiguous(biases.T)


def q8_gemm(
    qa,
    sa,
    ra,
    weight,
    scales_t,
    biases_t,
    *,
    act_scale: str = "group64",
    out_dtype=None,
    bias=None,
    tile=None,
    m_dim=None,
):
    """(Stage A) x packed Q8 GS64 affine weight [N, K/4] uint32 -> [M, N].

    ``scales_t``/``biases_t`` are group-major [K/64, N] (:func:`q8_metadata`).
    ``out_dtype`` defaults to the metadata dtype (fp32 is accepted for tests).
    No unpacked or requantized weight is ever materialized."""
    import mlx.core as mx

    act_mode = _Q8_ACT_MODE[act_scale]
    wm, wn = tuple(tile) if tile is not None else Q8_DEFAULT_TILE
    if (wm, wn) not in Q8_TILES:
        raise ValueError(f"q8 tile {(wm, wn)} is not one of {Q8_TILES}")
    m, k = qa.shape
    n, wpr = weight.shape
    if weight.dtype != mx.uint32 or wpr * 4 != k:
        raise ValueError(f"packed q8 weight {weight.shape} does not match K={k}")
    bn = 32 * wn
    if n % bn:
        raise ValueError(f"q8 output dim {n} is not a multiple of tile BN={bn}")
    groups = k // Q8_GROUP_SIZE
    if tuple(scales_t.shape) != (groups, n) or tuple(biases_t.shape) != (groups, n):
        raise ValueError("q8 metadata must be group-major [K/64, N]")
    out_dtype = out_dtype or scales_t.dtype
    ts, to = _tname(scales_t.dtype), _tname(out_dtype)
    key = (n, k, act_mode, ts, to, wm, wn, bias is not None)
    kernel = _q8_gemm_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _q8_gemm_kernels.get(key)
            if kernel is None:
                names = ["qa", "sa", "ra", "w", "st", "bt", "m_dim"]
                source = _Q8_GEMM_SRC
                if bias is not None:
                    names.append("bias")
                    source = source.replace("/*BIAS*/", "v += float(bias[n]);")
                kernel = _q8_gemm_kernels[key] = mx.fast.metal_kernel(
                    name=(
                        f"mlx2_i8p_q8_gemm_{n}x{k}_am{act_mode}_{ts}_{to}"
                        f"_wm{wm}_wn{wn}{'_b' if bias is not None else ''}"
                    ),
                    input_names=names,
                    output_names=["out"],
                    header=_Q8_HEADER,
                    source=_q8_prefix(
                        K=k, N=n, ACT_MODE=act_mode, WM=wm, WN=wn, TS=ts, TO=to
                    )
                    + source,
                )
    if m_dim is None:
        m_dim = mx.array([m], dtype=mx.int32)
    inputs = [qa, sa, ra, weight, scales_t, biases_t, m_dim]
    if bias is not None:
        inputs.append(bias)
    bm = 32 * wm
    return kernel(
        inputs=inputs,
        grid=(n // bn * 32 * wm * wn, (m + bm - 1) // bm, 1),
        threadgroup=(32 * wm * wn, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )[0]


def q45_stage_a(x, act_scale: str = "group64", m_dim=None):
    """Stage A for the Q4/Q5 in-place GEMM (Stage A v8 K order).

    Same contract as :func:`q8_stage_a` except that within each GS64 group
    slot ``16c + 4t + j`` of ``qa`` holds code ``16c + 8*(t>>1) + 2j + (t&1)``.
    ``ra``/``sa`` are unchanged by the permutation."""
    import mlx.core as mx

    act_mode = _Q8_ACT_MODE[act_scale]
    m, k = x.shape
    if k % Q8_GROUP_SIZE:
        raise ValueError(f"q45 stage A needs K % 64 == 0, got {k}")
    tname = _tname(x.dtype)
    key = (k, tname, act_mode)
    kernel = _q45_stage_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _q45_stage_kernels.get(key)
            if kernel is None:
                kernel = _q45_stage_kernels[key] = mx.fast.metal_kernel(
                    name=f"mlx2_i8p_q45_stage_a_{k}_{tname}_am{act_mode}",
                    input_names=["x", "m_dim"],
                    output_names=["qa", "sa", "ra"],
                    header=_Q45_HEADER,
                    source=_q8_prefix(K=k, ACT_MODE=act_mode, TX=tname)
                    + _Q45_STAGE_A_SRC,
                )
    if m_dim is None:
        m_dim = mx.array([m], dtype=mx.int32)
    groups = k // Q8_GROUP_SIZE
    sa_shape = (m,) if act_mode == 0 else (groups, m)
    qa, sa, ra = kernel(
        inputs=[x, m_dim],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), sa_shape, (groups, m)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    )
    return qa, sa, ra


def q45_gemm(
    qa,
    sa,
    ra,
    weight,
    scales_t,
    biases_t,
    *,
    bits: int,
    act_scale: str = "group64",
    out_dtype=None,
    bias=None,
    tile=None,
    m_dim=None,
):
    """(Stage A v8) x packed Q4/Q5 GS64 affine weight [N, K*bits/32] -> [M, N].

    ``weight`` is the checkpoint's own packed uint32 array, read in place;
    ``scales_t``/``biases_t`` are group-major [K/64, N] (:func:`q8_metadata`).
    ``tile`` defaults to :data:`Q45_DEFAULT_TILES` for ``bits``."""
    import mlx.core as mx

    if bits not in Q45_BITS:
        raise ValueError(f"q45 bits must be one of {Q45_BITS}, got {bits}")
    act_mode = _Q8_ACT_MODE[act_scale]
    wm, wn = tuple(tile) if tile is not None else Q45_DEFAULT_TILES[bits]
    if (wm, wn) not in Q45_TILES:
        raise ValueError(f"q45 tile {(wm, wn)} is not one of {Q45_TILES}")
    m, k = qa.shape
    n, wpr = weight.shape
    if weight.dtype != mx.uint32 or wpr * 32 != k * bits:
        raise ValueError(f"packed q{bits} weight {weight.shape} does not match K={k}")
    bn = 32 * wn
    if n % bn:
        raise ValueError(f"q45 output dim {n} is not a multiple of tile BN={bn}")
    groups = k // Q8_GROUP_SIZE
    if tuple(scales_t.shape) != (groups, n) or tuple(biases_t.shape) != (groups, n):
        raise ValueError("q45 metadata must be group-major [K/64, N]")
    out_dtype = out_dtype or scales_t.dtype
    ts, to = _tname(scales_t.dtype), _tname(out_dtype)
    key = (bits, n, k, act_mode, ts, to, wm, wn, bias is not None)
    kernel = _q45_gemm_kernels.get(key)
    if kernel is None:
        with _kernel_lock:
            kernel = _q45_gemm_kernels.get(key)
            if kernel is None:
                names = ["qa", "sa", "ra", "w", "st", "bt", "m_dim"]
                source = _Q45_GEMM_SRC
                if bias is not None:
                    names.append("bias")
                    source = source.replace("/*BIAS*/", "v += float(bias[n]);")
                kernel = _q45_gemm_kernels[key] = mx.fast.metal_kernel(
                    name=(
                        f"mlx2_i8p_q{bits}_gemm_{n}x{k}_am{act_mode}_{ts}_{to}"
                        f"_wm{wm}_wn{wn}{'_b' if bias is not None else ''}"
                    ),
                    input_names=names,
                    output_names=["out"],
                    header=_Q45_HEADER,
                    source=_q8_prefix(
                        K=k, N=n, BITS=bits, ACT_MODE=act_mode, WM=wm, WN=wn, TS=ts, TO=to
                    )
                    + source,
                )
    if m_dim is None:
        m_dim = mx.array([m], dtype=mx.int32)
    inputs = [qa, sa, ra, weight, scales_t, biases_t, m_dim]
    if bias is not None:
        inputs.append(bias)
    bm = 32 * wm
    return kernel(
        inputs=inputs,
        grid=(n // bn * 32 * wm * wn, (m + bm - 1) // bm, 1),
        threadgroup=(32 * wm * wn, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )[0]


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


def _projection_classes():
    """Classes int8 prefill installs on: the stock projections and their
    lane-matmul swaps.  A lane projection keeps the checkpoint's arrays (lane
    shares them), and the int8 wrapper sends calls below ``row_threshold`` to
    the lane class's own ``__call__``, so lane keeps decode/verify rows while
    int8 takes prefill rows."""
    from mlx import nn

    from .lane.installer import LaneLinear, LaneQuantizedLinear

    return (nn.QuantizedLinear, LaneQuantizedLinear), (nn.Linear, LaneLinear)


def module_spec(path: str, module) -> tuple[_ModuleSpec | None, str]:
    """(spec, "") when ``module`` is structurally int8-prefill eligible, else
    (None, reason).  Pure shape/layout arithmetic: no device work."""
    import mlx.core as mx

    cls = type(module)
    quantized_classes, linear_classes = _projection_classes()
    if cls in quantized_classes:
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
    elif cls in linear_classes:
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


def q8_inplace_eligible(spec: _ModuleSpec | None, module) -> tuple[bool, str]:
    """(True, "") when ``module`` can take the Q8 in-place kernel: 8-bit,
    group size 64, affine, bf16/fp16 scales and biases of one dtype.  Every
    other eligible module keeps the requantization path."""
    import mlx.core as mx

    if spec is None or spec.kind != "quantized":
        return False, "not a quantized projection"
    if spec.bits != 8 or spec.group_size != Q8_GROUP_SIZE:
        return False, f"q{spec.bits} gs{spec.group_size} is not 8-bit gs64"
    scales, biases = module["scales"], module["biases"]
    if scales.dtype not in (mx.bfloat16, mx.float16):
        return False, f"q8 scale dtype {scales.dtype} is not bf16/fp16"
    if biases.dtype != scales.dtype:
        return False, "q8 scales and biases differ in dtype"
    if module["weight"].dtype != mx.uint32:
        return False, "packed weight is not uint32"
    if spec.k % Q8_GROUP_SIZE or spec.n % max(32 * wn for _, wn in Q8_TILES):
        return False, "q8 shape does not tile"
    return True, ""


def q45_inplace_eligible(spec: _ModuleSpec | None, module) -> tuple[bool, str]:
    """(True, "") when ``module`` can take the Q4/Q5 in-place kernel: 4- or
    5-bit, group size 64, affine, bf16/fp16 scales and biases of one dtype,
    the checkpoint's packed uint32 layout."""
    import mlx.core as mx

    if spec is None or spec.kind != "quantized":
        return False, "not a quantized projection"
    if spec.bits not in Q45_BITS or spec.group_size != Q8_GROUP_SIZE:
        return False, f"q{spec.bits} gs{spec.group_size} is not 4/5-bit gs64"
    scales, biases = module["scales"], module["biases"]
    if scales.dtype not in (mx.bfloat16, mx.float16):
        return False, f"q{spec.bits} scale dtype {scales.dtype} is not bf16/fp16"
    if biases.dtype != scales.dtype:
        return False, f"q{spec.bits} scales and biases differ in dtype"
    weight = module["weight"]
    if weight.dtype != mx.uint32:
        return False, "packed weight is not uint32"
    if tuple(weight.shape) != (spec.n, spec.k * spec.bits // 32):
        return False, "packed weight shape does not match K * bits / 32"
    if spec.k % Q8_GROUP_SIZE or spec.n % max(32 * wn for _, wn in Q45_TILES):
        return False, f"q{spec.bits} shape does not tile"
    return True, ""


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


def _projection_candidates(model, chooser):
    """Visit each module once, refusing an in-place edit across scope aliases.

    MLX named_modules() enumerates paths, including multiple paths to the
    same object. Wrapping those paths independently nests our own wrapper
    and loses the original class needed by remove().
    """
    aliases = {}
    for path, module in model.named_modules():
        entry = aliases.setdefault(id(module), (module, [], []))
        entry[1 if chooser(path, module) else 2].append(path)
    for module, selected, excluded in aliases.values():
        if not selected:
            continue
        path = selected[0]
        spec, reason = module_spec(path, module)
        if spec is not None and excluded:
            raise Int8PrefillError(
                f"projection {path!r} has alias {excluded[0]!r} outside the selected scope"
            )
        yield path, module, spec, reason


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
    __slots__ = (
        "base", "bt", "meta_nbytes", "module", "nbytes", "q45", "q8", "spec", "st", "wq",
        "ws",
    )

    def __init__(self, module, base, spec, q8=False, q45=False):
        self.module = module
        self.base = base
        self.spec = spec
        self.wq = None
        self.ws = None
        self.nbytes = 0
        # Q8 in place: no int8 weight copy ever; only the group-major
        # scale/bias metadata (``st``/``bt``) may be cached.
        self.q8 = bool(q8)
        # Q4/Q5 in place: likewise no weight copy, same metadata copy.
        self.q45 = bool(q45)
        self.st = None
        self.bt = None
        self.meta_nbytes = 0

    @property
    def inplace(self) -> bool:
        return self.q8 or self.q45

    def meta_size(self) -> int:
        """Bytes of one group-major scale + bias copy."""
        scales = self.module["scales"]
        return 2 * int(scales.size) * scales.itemsize


class Int8PrefillHandle:
    """Installed int8 prefill on one model.  Counters are host ints only."""

    def __init__(self, policy: Int8PrefillPolicy, *, device: str = ""):
        self.policy = policy
        self.device = device
        self.active = False
        self._bound: dict[int, _Bound] = {}
        self._skipped: dict[str, str] = {}
        self._q8_declined: dict[str, str] = {}
        self._q45_declined: dict[str, str] = {}
        self._lock = threading.Lock()
        self._act_entry = None
        self._q8_act_entry = None
        self._q45_act_entry = None
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
        if policy.q8_inplace:
            # Proof the in-place kernel ran (a silent fallback to stock or to
            # the requant path would leave these at zero).  q8_calls/rows are
            # also counted in engaged_calls/rows.
            self.counts.update(
                {
                    "q8_calls": 0,
                    "q8_rows": 0,
                    "q8_stage_a": 0,
                    "q8_stage_a_reuse": 0,
                    "q8_meta_builds": 0,
                }
            )
        if policy.q45_inplace:
            # Proof the Q4/Q5 in-place kernels ran, per bit width.
            self.counts.update(
                {
                    "q45_calls": 0,
                    "q45_rows": 0,
                    "q4_calls": 0,
                    "q5_calls": 0,
                    "q45_stage_a": 0,
                    "q45_stage_a_reuse": 0,
                    "q45_meta_builds": 0,
                }
            )

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

        if bound.q8:
            return self._call_q8(module, bound, x, k, rows, base_call)
        if bound.q45:
            return self._call_q45(module, bound, x, k, rows, base_call)
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

    def _call_q8(self, module, bound, x, k, rows, base_call):
        scales = module["scales"]
        if x.dtype != scales.dtype:
            # Output dtype follows the stock promotion only when x and the
            # metadata share bf16 or fp16; anything else stays stock.
            self.counts["fallback_dtype"] += 1
            return base_call(module, x)
        self._last_use = time.monotonic()
        st, bt = self._q8_meta(bound)
        qa, sa, ra, m_dim = self._q8_activation(x, k)
        bias = module["bias"] if "bias" in module else None
        y = q8_gemm(
            qa,
            sa,
            ra,
            module["weight"],
            st,
            bt,
            act_scale=self.policy.act_scale,
            out_dtype=x.dtype,
            bias=bias,
            m_dim=m_dim,
        )
        counts = self.counts
        counts["engaged_calls"] += 1
        counts["engaged_rows"] += rows
        counts["q8_calls"] += 1
        counts["q8_rows"] += rows
        return y.reshape(*x.shape[:-1], bound.spec.n)

    def _call_q45(self, module, bound, x, k, rows, base_call):
        scales = module["scales"]
        if x.dtype != scales.dtype:
            self.counts["fallback_dtype"] += 1
            return base_call(module, x)
        self._last_use = time.monotonic()
        st, bt = self._q8_meta(bound)
        qa, sa, ra, m_dim = self._q45_activation(x, k)
        bias = module["bias"] if "bias" in module else None
        bits = bound.spec.bits
        y = q45_gemm(
            qa,
            sa,
            ra,
            module["weight"],
            st,
            bt,
            bits=bits,
            act_scale=self.policy.act_scale,
            out_dtype=x.dtype,
            bias=bias,
            m_dim=m_dim,
        )
        counts = self.counts
        counts["engaged_calls"] += 1
        counts["engaged_rows"] += rows
        counts["q45_calls"] += 1
        counts["q45_rows"] += rows
        counts[f"q{bits}_calls"] += 1
        return y.reshape(*x.shape[:-1], bound.spec.n)

    def _q8_meta(self, bound):
        # Group-major metadata for any in-place module (Q8 or Q4/Q5).
        module = bound.module
        builds = "q8_meta_builds" if bound.q8 else "q45_meta_builds"
        if self._cache_mode(bound) == "none":
            self.counts[builds] += 1
            return q8_metadata(module["scales"], module["biases"])
        with self._lock:
            if bound.st is None:
                import mlx.core as mx

                st, bt = q8_metadata(module["scales"], module["biases"])
                mx.eval(st, bt)
                bound.st, bound.bt = st, bt
                bound.meta_nbytes = int(st.nbytes + bt.nbytes)
                self.counts[builds] += 1
            return bound.st, bound.bt

    def _q45_activation(self, x, k):
        # Q4 and Q5 projections of one input share one Stage A v8 (its K
        # order differs from Q8's, so the entries are separate).
        entry = self._q45_act_entry
        if entry is not None and entry[0] is x:
            self.counts["q45_stage_a_reuse"] += 1
            return entry[1]
        import mlx.core as mx

        x2 = x.reshape(-1, k)
        m_dim = mx.array([x2.shape[0]], dtype=mx.int32)
        qa, sa, ra = q45_stage_a(x2, self.policy.act_scale, m_dim=m_dim)
        self.counts["q45_stage_a"] += 1
        self._q45_act_entry = (x, (qa, sa, ra, m_dim))
        return qa, sa, ra, m_dim

    def _q8_activation(self, x, k):
        # gate/up (and q/k/v, GDN qkv/z) share one input: one Stage A.  Kept
        # apart from the requant path's entry, whose layout differs.
        entry = self._q8_act_entry
        if entry is not None and entry[0] is x:
            self.counts["q8_stage_a_reuse"] += 1
            return entry[1]
        import mlx.core as mx

        x2 = x.reshape(-1, k)
        m_dim = mx.array([x2.shape[0]], dtype=mx.int32)
        qa, sa, ra = q8_stage_a(x2, self.policy.act_scale, m_dim=m_dim)
        self.counts["q8_stage_a"] += 1
        self._q8_act_entry = (x, (qa, sa, ra, m_dim))
        return qa, sa, ra, m_dim

    def _build(self, bound):
        module, spec = bound.module, bound.spec
        if bound.inplace:
            # Defensive: an in-place module never builds an int8 copy.
            raise Int8PrefillError(f"{spec.path}: in-place module has no weight copy")
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
            # Q8 in place: keep the group-major metadata (1/16 of the weight
            # bytes at GS64 bf16), as omlx #4350 does.
            if bound.inplace:
                return "resident"
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
        self._q8_act_entry = None
        self._q45_act_entry = None

    def evict_weights(self) -> int:
        with self._lock:
            evicted = 0
            for bound in self._bound.values():
                if bound.wq is not None:
                    bound.wq = bound.ws = None
                    bound.nbytes = 0
                    evicted += 1
                if bound.st is not None:
                    bound.st = bound.bt = None
                    bound.meta_nbytes = 0
                    evicted += 1
            self.counts["weight_evictions"] += evicted
        self.release_activation()
        return evicted

    def warmup(self) -> int:
        """Build the cached int8 weights now (modules whose cache mode is not none)."""
        built = 0
        for bound in self._bound.values():
            if self._cache_mode(bound) != "none":
                if bound.inplace:
                    self._q8_meta(bound)
                else:
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
        """Bytes the cached int8 weight copies occupy once all are built.

        In-place (Q8, Q4/Q5) modules have no weight copy and contribute nothing."""
        return sum(
            b.spec.n * (b.spec.k + 4)
            for b in tuple(self._bound.values())
            if not b.inplace and self._cache_mode(b) != "none"
        )

    def metadata_bytes(self) -> int:
        """Bytes held by cached group-major in-place scale/bias copies."""
        return sum(b.meta_nbytes for b in tuple(self._bound.values()))

    def metadata_copy_bytes(self, kind: str | None = None) -> int:
        """Admission-facing in-place metadata bytes: cached copies once all
        are built plus the worst per-call transient copy (cache none).
        ``kind`` "q8" / "q45" restricts to one in-place family."""
        bounds = [
            b
            for b in tuple(self._bound.values())
            if (b.q8 if kind == "q8" else b.q45 if kind == "q45" else b.inplace)
        ]
        resident = sum(b.meta_size() for b in bounds if self._cache_mode(b) != "none")
        transient = max(
            (b.meta_size() for b in bounds if self._cache_mode(b) == "none"), default=0
        )
        return resident + transient

    def q8_module_count(self) -> int:
        return sum(1 for b in tuple(self._bound.values()) if b.q8)

    def q45_module_count(self, bits: int | None = None) -> int:
        return sum(
            1
            for b in tuple(self._bound.values())
            if b.q45 and (bits is None or b.spec.bits == bits)
        )

    def transient_weight_bytes_max(self) -> int:
        """Largest int8 weight tensor one call builds and drops (cache none).

        This is memory that exists only during a prefill call and is not
        otherwise visible to admission."""
        return max(
            (
                b.spec.n * (b.spec.k + 4)
                for b in tuple(self._bound.values())
                if not b.inplace and self._cache_mode(b) == "none"
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
            if bound.q8:
                label = "q8_inplace"
            elif bound.q45:
                label = f"q{bound.spec.bits}_inplace"
            kinds[label] = kinds.get(label, 0) + 1
        status = {
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
        if self.policy.q8_inplace:
            status["q8_inplace"] = self._q8_summary()
            status["q8_inplace"]["metadata_bytes"] = (
                sum(b.meta_nbytes for b in tuple(self._bound.values()) if b.q8)
                if self.policy.q45_inplace
                else self.metadata_bytes()
            )
        if self.policy.q45_inplace:
            status["q45_inplace"] = self._q45_summary()
            status["q45_inplace"]["metadata_bytes"] = sum(
                b.meta_nbytes for b in tuple(self._bound.values()) if b.q45
            )
        return status

    def _q8_summary(self) -> dict:
        return {
            "act_scale": self.policy.act_scale,
            "kernel_revision": Q8_KERNEL_REVISION,
            "tile": list(Q8_DEFAULT_TILE),
            "modules": self.q8_module_count(),
            "declined": len(self._q8_declined),
            "weight_copy_bytes": 0,
            "metadata_copy_bytes": self.metadata_copy_bytes(
                "q8" if self.policy.q45_inplace else None
            ),
        }

    def _q45_summary(self) -> dict:
        return {
            "act_scale": self.policy.act_scale,
            "kernel_revision": Q45_KERNEL_REVISION,
            "tiles": {f"q{b}": list(t) for b, t in Q45_DEFAULT_TILES.items()},
            "modules": self.q45_module_count(),
            "modules_by_bits": {f"q{b}": self.q45_module_count(b) for b in Q45_BITS},
            "declined": len(self._q45_declined),
            "inplace_only": self.policy.inplace_only,
            "weight_copy_bytes": 0,
            "metadata_copy_bytes": self.metadata_copy_bytes("q45"),
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
            **({"q8_inplace": self._q8_summary()} if self.policy.q8_inplace else {}),
            **({"q45_inplace": self._q45_summary()} if self.policy.q45_inplace else {}),
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
    for path, module, spec, reason in _projection_candidates(model, chooser):
        if spec is None:
            if reason and not reason.startswith("unsupported module type"):
                handle._skipped[path] = reason
            continue
        q8 = q45 = False
        if policy.q8_inplace:
            q8, why = q8_inplace_eligible(spec, module)
            if not q8:
                handle._q8_declined[path] = why
                if policy.q8_only:
                    handle._skipped[path] = f"q8_only: {why}"
                    continue
        if policy.q45_inplace and not q8:
            q45, why45 = q45_inplace_eligible(spec, module)
            if q45:
                # Taken by the other in-place kernel: not a Q8 decline.
                handle._q8_declined.pop(path, None)
            else:
                handle._q45_declined[path] = why45
                if policy.inplace_only:
                    handle._skipped[path] = f"inplace_only: {why45}"
                    continue
        candidates.append((module, spec, q8, q45))
    if not candidates:
        raise Int8PrefillError(
            f"int8 prefill scope {policy.scope!r} selected no eligible projection"
        )
    if policy.q8_inplace and not any(q8 for _, _, q8, _ in candidates):
        # Selecting the in-place mode and silently running only the requant
        # path would be a false green for anything measuring it.
        raise Int8PrefillError(
            f"int8 prefill q8_inplace: scope {policy.scope!r} selected no "
            "8-bit gs64 affine projection with bf16/fp16 metadata"
        )
    if policy.q45_inplace and not any(q45 for _, _, _, q45 in candidates):
        raise Int8PrefillError(
            f"int8 prefill q45_inplace: scope {policy.scope!r} selected no "
            "4/5-bit gs64 affine projection with bf16/fp16 metadata"
        )
    ref = weakref.ref(handle)
    for module, spec, q8, q45 in candidates:
        base = type(module)
        handle._bound[id(module)] = _Bound(module, base, spec, q8=q8, q45=q45)
        module.__dict__[_STATE_ATTR] = ref
        module.__class__ = _int8_subclass(base)
    handle.active = True
    handle._start_reaper()
    logger.info(
        "int8 NAX prefill installed on %d projections (scope %s, threshold %d, "
        "cache %s, %d skipped, %d q8 / %d q4 / %d q5 in place%s)",
        len(candidates),
        policy.scope,
        policy.row_threshold,
        policy.cache,
        len(handle._skipped),
        handle.q8_module_count(),
        handle.q45_module_count(4),
        handle.q45_module_count(5),
        f" act_scale {policy.act_scale}"
        if policy.q8_inplace or policy.q45_inplace
        else "",
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



def max_decode_rows(
    *, max_lanes, config, speculation, prompt_lookup_policy=None,
    copy_draft_policy=None,
):
    """Upper bound on rows one decode / speculative-verify forward presents.

    Verify blocks carry at most ``num_draft + 1`` rows per lane; one extra row
    of slack covers bonus/rollback tokens.  Ordinary decode is one row/lane.
    A cliff-aware prompt-lookup span can exceed ``num_draft``
    (``prompt_lookup.max_proposal_span``).

    A self-MTP lane under a ``draft_loop`` drafts past ``num_draft`` to the
    loop's ceiling (for cohorts no wider than its ``max_width``), and a
    copy round verifies a copied span in place of the head drafts, with a
    width that depends on the cohort: ``max_span`` for a solo lane, the
    cohort cap once lanes share the forward.  Every cohort size is checked
    (``draft_loop.cohort_proposal_depths``) because the solo span, or a
    solo looped lane, can be wider than a full batch of head drafts."""
    draft = 0
    if speculation in ("self_mtp", "external_draft"):
        draft = int((config or {}).get("num_draft", 0) or 0)
    elif speculation == "prompt_lookup":
        from .prompt_lookup import max_proposal_span

        policy = dict(prompt_lookup_policy or {})
        policy["num_draft"] = int(policy.get("num_draft", 8) or 8)
        draft = max_proposal_span(policy)
    bound = int(max_lanes) * (draft + 2)
    if speculation == "self_mtp":
        from .draft_loop import cohort_proposal_depths

        for lanes, depth in cohort_proposal_depths(
            config, max_lanes=max_lanes, copy_draft_policy=copy_draft_policy
        ).items():
            bound = max(bound, lanes * (depth + 2))
    return bound


def bind_for_serving(
    adapter,
    policy,
    *,
    max_lanes,
    config,
    speculation,
    prompt_lookup_policy=None,
    copy_draft_policy=None,
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
        copy_draft_policy=copy_draft_policy,
    )
    validate_decode_row_bound(policy, bound)
    handle = apply_for_adapter(adapter, policy)
    try:
        warmed = handle.warmup()
    except BaseException:
        # Never leave a half-bound install behind a refusal.
        remove(handle)
        raise
    settings = {
        **policy.as_dict(),
        "revision": policy.revision,
        "fidelity": policy.fidelity.value,
        "max_decode_rows": bound,
        "weight_copy_bytes": handle.weight_copy_bytes(),
    }
    if policy.q8_inplace:
        settings["q8_modules"] = handle.q8_module_count()
        settings["metadata_copy_bytes"] = handle.metadata_copy_bytes()
    if policy.q45_inplace:
        settings["q45_modules"] = {
            f"q{b}": handle.q45_module_count(b) for b in Q45_BITS
        }
        settings["metadata_copy_bytes"] = handle.metadata_copy_bytes()
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


def auto_scope(adapter) -> str | None:
    """Widest int8 prefill scope ``adapter`` declares, or None."""
    scopes = adapter_scopes(adapter)
    return "all" if "all" in scopes else ("mlp" if "mlp" in scopes else None)


def auto_policy(adapter, census=None) -> Int8PrefillPolicy | None:
    """The policy ``--int8-prefill auto`` selects for ``adapter``, or None.

    In-place kernels with per-64-group activation scales, on the projections
    they take and nothing else (every other projection stays on stock
    kernels, never the requantization path).  That combination kept
    teacher-forced agreement with stock prefill on both measured checkpoints
    (2026-10-09): Qwen3.8-27B-8bit top-1 0.976-1.0, Qwen3.8-27B-oQ4e (Q4/Q5)
    top-1 0.974-0.999; the requantization and per-row modes lost ~10% top-1
    on prose.  The census picks the kernels:

    - Q8 only (or no census): Q8 in place, ``q8_only``.  This is the policy
      ``auto`` has selected since it shipped, so its revision -- and every
      qualification record carrying it -- is unchanged.
    - Q4/Q5 only: Q4/Q5 in place, ``inplace_only``.
    - both: Q8 and Q4/Q5 in place, ``inplace_only``.
    """
    scope = auto_scope(adapter)
    if scope is None:
        return None
    q8 = census is None or census.get("q8", 0) > 0
    q45 = census is not None and census.get("q45", 0) > 0
    if not q45:
        return Int8PrefillPolicy(
            enabled=True, scope=scope, q8_inplace=True, act_scale="group64", q8_only=True
        )
    return Int8PrefillPolicy(
        enabled=True, scope=scope, q8_inplace=q8, q45_inplace=True,
        act_scale="group64", inplace_only=True,
    )


def checkpoint_census(adapter, scope: str) -> dict:
    """Count the in-scope projections by what int8 prefill could do with them.

    Host-only (shapes and dtypes, no device work)."""
    selector = getattr(adapter, "int8_prefill_select", None)
    chooser = (selector(scope) if callable(selector) else None) or default_select(scope)
    counts = {"q8": 0, "q45": 0, "other_quantized": 0, "linear": 0}
    for _path, module, spec, _reason in _projection_candidates(adapter.model, chooser):
        if spec is None:
            continue
        if spec.kind == "linear":
            counts["linear"] += 1
        elif q8_inplace_eligible(spec, module)[0]:
            counts["q8"] += 1
        elif q45_inplace_eligible(spec, module)[0]:
            counts["q45"] += 1
        else:
            counts["other_quantized"] += 1
    return counts


def resolve_auto(adapter, *, qualification_mode, qualified_settings, conflicts=()):
    """Resolve ``--int8-prefill auto`` once the model is loaded.

    Returns ``(policy or None, receipt)``.  On only when every condition holds:
    the adapter declares int8 prefill, nothing that owns the same projections
    is selected, the device has Metal 4 tensor ops, the checkpoint has 8-bit
    or 4/5-bit gs64 affine projections in scope (``auto_policy`` picks the
    kernels from that census), and there is evidence: qualification
    mode (the qualifier is producing it), or a qualification record whose
    settings carry this exact int8 prefill revision.  int8 prefill is
    approximate, and approximate state may only be published through a
    qualified approximate operation (AGENTS.md), so an unqualified route
    resolves off.  Off adds nothing to the route settings, so records of
    every route that resolves off still match."""
    receipt = {"requested": AUTO}

    def off(reason):
        receipt.update(resolved="off", reason=reason)
        return None, receipt

    scope = auto_scope(adapter)
    if scope is None:
        return off("model adapter does not declare int8 prefill")
    if conflicts:
        return off("conflicts with " + ", ".join(sorted(conflicts)))
    supported, why = device_support()
    if not supported:
        return off(f"device: {why}")
    try:
        census = checkpoint_census(adapter, scope)
    except Int8PrefillError as error:
        return off(str(error))
    receipt["census"] = census
    if not (census["q8"] or census["q45"]):
        return off("no 8-bit or 4/5-bit gs64 affine projection in scope")
    policy = auto_policy(adapter, census)
    receipt["policy"] = policy.as_dict()
    if qualification_mode:
        evidence = "qualification mode"
    elif qualified_settings is not None:
        recorded = qualified_settings.get("int8_prefill")
        if not (isinstance(recorded, Mapping) and recorded.get("enabled") is True):
            return off("the qualification record does not include int8 prefill")
        if recorded.get("revision") != policy.revision:
            return off("the qualification record carries a different int8 prefill revision")
        evidence = "qualification record"
    else:
        return off(
            "no qualification evidence: serve with a qualification record "
            "that includes int8 prefill"
        )
    receipt.update(resolved="on", evidence=evidence, revision=policy.revision)
    return policy, receipt


def engine_status(engine) -> dict:
    """Host-only status for a serving engine (no device work, no syncs)."""
    handle = getattr(engine, "int8_prefill_handle", None)
    auto = getattr(engine, "int8_prefill_auto", None)
    if handle is not None:
        status = handle.status()
    else:
        policy = getattr(engine, "int8_prefill_policy", None) or Int8PrefillPolicy()
        status = {
            "schema": SCHEMA,
            **policy.as_dict(),
            "active": False,
            "fidelity": policy.fidelity.value,
            "modules": 0,
            "counts": {},
        }
    if auto is not None:
        # What ``auto`` resolved to and why (on: the evidence; off: the reason).
        status = {**status, "auto": dict(auto)}
    return status
