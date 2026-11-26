"""Hyper-connections (mHC): one reference for coefficients, collapse and expand.

A hyper-connection keeps ``hc_mult`` parallel residual streams
``[..., n, d]``. Around every sub-block (attention, FFN) it

1. projects the flattened streams to ``(2 + n) * n`` mixes (``hc_mixes``),
2. turns the mixes into a pre gate ``[..., n]``, a post gate ``[..., n]`` and a
   doubly stochastic combine matrix ``[..., n, n]`` by Sinkhorn projection
   (``hc_coefficients`` / ``hc_gates``),
3. collapses the streams to the sub-block input with the pre gate
   (``hc_collapse``), and
4. expands the sub-block output back to ``n`` streams:
   ``post[i] * branch + sum_j M[j, i] * residual[j]`` (``hc_expand``).

A learned output head (``hc_head``) is a collapse whose gate is
``sigmoid(mixes * scale + base) + eps``.

Every difference between the model copies that were folded here is a
:class:`HCSpec` field named for what it does (see
``docs/technologies/hyper_connections.md``). The reference never branches on
model identity.

The module also owns the Metal *source text* that the DeepSeek-V4 HC4 kernels
and GLM-5.3's mega boundary used to carry as two pasted copies
(:func:`split_sinkhorn_body`, :func:`weighted_sum_lines`, :func:`expand_lines`);
the generated strings are byte-identical to the old ones (CPU string tests).
Kernels whose arithmetic differs from that law stay separate modules and are
registered below as fast paths. Nothing here touches the GPU on import.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable

import mlx.core as mx

from .dispatch import FastPath, KernelRegistry, spec_from_config

NORM_KINDS = ("pre_rms", "post_rsqrt_mean", "post_inv_l2")
SQUARES = ("square", "mul")
AFFINES = ("per_segment", "full_vector")
GATE_ACTIVATIONS = ("sigmoid", "span_aware_sigmoid")
COMB_NORMALIZERS = ("softmax_precise", "softmax", "exp_max_shift")
ORIENTATIONS = ("transpose", "direct")
COEFF_ROUNDS = ("fp32", "stream")
EXPAND_OUTS = ("branch", "residual")


def _clip_from_config(config):
    def get(key):
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)

    lo, hi = get("mhc_h_res_clamp_min"), get("mhc_h_res_clamp_max")
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


@dataclass(frozen=True)
class HCSpec:
    """Every per-model hyper-connection parameter.

    Defaults are the DeepSeek-V4 law (the DwarfStar/DS4 Sinkhorn schedule).

    * ``hc_mult``: number of residual streams ``n``; mixes have
      ``(2 + n) * n`` entries ordered ``pre[n] | post[n] | comb[n*n]``.
    * ``sinkhorn_iters`` / ``softmax_counts_as_first_pass``: with ``True`` the
      row softmax is the first row pass, followed by one column pass and
      ``sinkhorn_iters - 1`` (row, column) passes; with ``False`` the
      normaliser is followed by ``sinkhorn_iters`` full (row, column) passes.
    * ``eps``: added to every Sinkhorn denominator, to the pre gate when
      ``pre_eps`` and to the normalised matrix once when ``comb_seed_eps``.
    * ``norm_kind`` / ``norm_eps`` / ``norm_square`` / ``head_norm_kind``: how
      the flat stream is RMS-scaled around the mixing projection.
      ``pre_rms``: ``rms_norm(flat) @ fn.T``. ``post_rsqrt_mean``:
      ``(flat @ fn.T) * rsqrt(mean(flat^2) + eps)``. ``post_inv_l2``:
      ``1 / (sqrt(sum(flat^2)) / sqrt(K) + eps) * (flat @ fn.T)``.
      ``norm_square`` spells ``flat^2`` as ``mx.square`` or ``flat * flat``.
    * ``coeff_affine``: ``per_segment`` applies ``scale[0..2]`` to each slice;
      ``full_vector`` applies an already expanded ``scale[(2 + n) * n]``.
    * ``gate_activation``: ``mx.sigmoid`` or ``precise_ops.gate_sigmoid``
      (exact sigmoid inside a traced precise span).
    * ``comb_clip``: optional ``(lo, hi)`` clamp of the combine logits.
    * ``comb_normalizer``: row ``softmax`` (precise or not) or the unnormalised
      ``exp(x - max(x))``.
    * ``orientation``: ``transpose`` mixes ``M^T @ residual`` (DS4: column j of
      M feeds stream j); ``direct`` mixes ``M @ residual`` (HF Xing).
    * ``upcast_fp32``: cast every operand to F32 and the outputs back to the
      stream dtype; ``False`` leaves dtypes exactly as the caller passed them.
    * ``coeff_round``: ``stream`` rounds pre/post/comb to the stream dtype.
    * ``expand_out``: the dtype an expand result is cast to (with upcast).
    """

    hc_mult: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    norm_kind: str = "pre_rms"
    norm_eps: float = 1e-6
    norm_square: str = "square"
    head_norm_kind: str | None = None
    coeff_affine: str = "per_segment"
    gate_activation: str = "sigmoid"
    pre_eps: bool = True
    post_scale: float = 2.0
    comb_clip: tuple[float, float] | None = None
    comb_normalizer: str = "softmax_precise"
    comb_seed_eps: bool = True
    softmax_counts_as_first_pass: bool = True
    orientation: str = "transpose"
    upcast_fp32: bool = True
    coeff_round: str = "fp32"
    expand_out: str = "branch"

    def __post_init__(self):
        for name, kind in (("hc_mult", int), ("sinkhorn_iters", int), ("eps", float),
                           ("norm_eps", float), ("post_scale", float)):
            value = getattr(self, name)
            if isinstance(value, bool) or not hasattr(value, "__float__"):
                raise ValueError(f"HCSpec.{name} must be a number")
            if kind is int and int(value) != float(value):
                raise ValueError(f"HCSpec.{name} must be integral")
            object.__setattr__(self, name, kind(value))
        if type(self.hc_mult) is not int or self.hc_mult < 1:
            raise ValueError("HCSpec.hc_mult must be a positive int")
        # 0 is legal: with softmax_counts_as_first_pass the legacy DS4/Maxwell
        # loops run the column pass and no (row, column) pass.
        if type(self.sinkhorn_iters) is not int or self.sinkhorn_iters < 0:
            raise ValueError("HCSpec.sinkhorn_iters must be a nonnegative int")
        for name, allowed in (
            ("norm_kind", NORM_KINDS),
            ("norm_square", SQUARES),
            ("coeff_affine", AFFINES),
            ("gate_activation", GATE_ACTIVATIONS),
            ("comb_normalizer", COMB_NORMALIZERS),
            ("orientation", ORIENTATIONS),
            ("coeff_round", COEFF_ROUNDS),
            ("expand_out", EXPAND_OUTS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"HCSpec.{name} must be one of {allowed}")
        if self.head_norm_kind is not None and self.head_norm_kind not in NORM_KINDS:
            raise ValueError(f"HCSpec.head_norm_kind must be one of {NORM_KINDS}")
        if self.comb_clip is not None:
            lo, hi = self.comb_clip
            object.__setattr__(self, "comb_clip", (float(lo), float(hi)))

    @property
    def mix_width(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "HCSpec":
        """Derive the checkpoint-owned fields; the law fields come as overrides."""
        return spec_from_config(
            cls,
            config,
            {
                "hc_mult": ("hc_mult", "num_residual_streams"),
                "sinkhorn_iters": ("hc_sinkhorn_iters", "mhc_sinkhorn_iterations"),
                "eps": "hc_eps",
                "norm_eps": "rms_norm_eps",
                "comb_clip": _clip_from_config,
            },
            **overrides,
        )


@lru_cache(maxsize=256)
def cached_spec(**fields) -> HCSpec:
    """A shared spec for hot call sites that rebuild one from scalars."""
    return HCSpec(**fields)


@lru_cache(maxsize=256)
def derive(spec: HCSpec, **fields) -> HCSpec:
    """``dataclasses.replace`` memoised for per-call variants (e.g. a norm eps)."""
    return replace(spec, **fields)


_GATE_SIGMOID = None


def _activation(spec):
    if spec.gate_activation == "span_aware_sigmoid":
        global _GATE_SIGMOID
        if _GATE_SIGMOID is None:  # resolved once; models imports this module
            from ..models.precise_ops import gate_sigmoid

            _GATE_SIGMOID = gate_sigmoid
        return _GATE_SIGMOID
    return mx.sigmoid


def _square(spec, x):
    return mx.square(x) if spec.norm_square == "square" else x * x


# ---------------------------------------------------------------------------
# mixes
# ---------------------------------------------------------------------------


def hc_mixes(
    spec: HCSpec,
    flat,
    fn=None,
    *,
    project: Callable | None = None,
    head: bool = False,
):
    """Mixing logits from the flattened streams ``flat [..., n * d]``.

    ``project`` replaces ``x @ fn.T`` (a quantized linear or a fixed-order
    Metal projection); otherwise ``fn`` is cast to F32 and multiplied.
    """
    kind = (spec.head_norm_kind or spec.norm_kind) if head else spec.norm_kind
    flat32 = flat.astype(mx.float32)

    def proj(x):
        if project is not None:
            return project(x)
        return x @ fn.astype(mx.float32).T

    if kind == "pre_rms":
        return proj(mx.fast.rms_norm(flat32, None, spec.norm_eps))
    if kind == "post_rsqrt_mean":
        scale = mx.rsqrt(
            mx.mean(_square(spec, flat32), axis=-1, keepdims=True) + spec.norm_eps
        )
        return proj(flat32) * scale
    norm = mx.sqrt(mx.sum(_square(spec, flat32), axis=-1, keepdims=True))
    r = 1.0 / (norm / math.sqrt(flat.shape[-1]) + spec.norm_eps)
    return r * proj(flat32)


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------


def _sinkhorn(spec, comb_logits):
    eps = spec.eps
    comb = comb_logits
    if spec.comb_clip is not None:
        comb = mx.clip(comb, spec.comb_clip[0], spec.comb_clip[1])
    if spec.comb_normalizer == "softmax_precise":
        comb = mx.softmax(comb, axis=-1, precise=True)
    elif spec.comb_normalizer == "softmax":
        comb = mx.softmax(comb, axis=-1)
    else:
        comb = mx.exp(comb - comb.max(axis=-1, keepdims=True))
    if spec.comb_seed_eps:
        comb = comb + eps
    passes = spec.sinkhorn_iters
    if spec.softmax_counts_as_first_pass:
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        passes -= 1  # range(-1) is empty, as the legacy iterations - 1 loop
    for _ in range(passes):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return comb


def _finish(spec, pre_logits, post_logits, comb_logits, stream_dtype):
    sigmoid = _activation(spec)
    pre = sigmoid(pre_logits)
    if spec.pre_eps:
        pre = pre + spec.eps
    post = spec.post_scale * sigmoid(post_logits)
    comb = _sinkhorn(spec, comb_logits)
    if spec.coeff_round == "stream":
        if stream_dtype is None:
            raise ValueError("HCSpec.coeff_round='stream' needs the stream dtype")
        pre, post, comb = (x.astype(stream_dtype) for x in (pre, post, comb))
    return pre, post, comb


def hc_coefficients(spec: HCSpec, mixes, scale, base, *, stream_dtype=None):
    """``(pre [..., n], post [..., n], comb [..., n, n])`` from flat mixes."""
    n = spec.hc_mult
    if spec.upcast_fp32:
        mixes, scale, base = (x.astype(mx.float32) for x in (mixes, scale, base))
    if spec.coeff_affine == "full_vector":
        h = mixes * scale + base
        pre_l, post_l = h[..., :n], h[..., n : 2 * n]
        comb_l = h[..., 2 * n :].reshape(*h.shape[:-1], n, n)
    else:
        pre_l = mixes[..., :n] * scale[0] + base[:n]
        post_l = mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n]
        comb_l = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).reshape(
            *mixes.shape[:-1], n, n
        )
    return _finish(spec, pre_l, post_l, comb_l, stream_dtype)


def hc_gates(
    spec: HCSpec,
    pre_mix,
    post_mix,
    comb_mix,
    scale,
    pre_base,
    post_base,
    comb_base,
    *,
    stream_dtype=None,
):
    """:func:`hc_coefficients` for callers that hold the mixes pre-split.

    ``comb_mix [..., n, n]`` and ``comb_base [n, n]`` are already shaped as a
    matrix, so the affine runs after the reshape (as HF Xing writes it).
    """
    pre_l = pre_mix * scale[0] + pre_base
    post_l = post_mix * scale[1] + post_base
    comb_l = comb_mix * scale[2] + comb_base
    return _finish(spec, pre_l, post_l, comb_l, stream_dtype)


# ---------------------------------------------------------------------------
# collapse / expand / head
# ---------------------------------------------------------------------------


def hc_collapse(spec: HCSpec, streams, pre, *, out_dtype=None):
    """``sum_i pre[i] * streams[i]`` over the stream axis ``-2``."""
    if not spec.upcast_fp32:
        return mx.sum(pre[..., None] * streams, axis=-2)
    value = mx.sum(
        pre.astype(mx.float32)[..., None] * streams.astype(mx.float32), axis=-2
    )
    return value.astype(streams.dtype if out_dtype is None else out_dtype)


def hc_expand(spec: HCSpec, branch, residual, post, comb):
    """``post[i] * branch + mix(comb, residual)[i]`` for every stream ``i``."""
    if spec.upcast_fp32:
        b, r = branch.astype(mx.float32), residual.astype(mx.float32)
        post, comb = post.astype(mx.float32), comb.astype(mx.float32)
    else:
        b, r = branch, residual
    matrix = mx.swapaxes(comb, -1, -2) if spec.orientation == "transpose" else comb
    value = post[..., None] * b[..., None, :] + mx.matmul(matrix, r)
    if not spec.upcast_fp32:
        return value
    return value.astype(branch.dtype if spec.expand_out == "branch" else residual.dtype)


def hc_head(spec: HCSpec, streams, mixes, scale, base):
    """Learned output head: collapse with ``sigmoid(mixes * scale + base) + eps``."""
    if spec.upcast_fp32:
        mixes, scale, base = (x.astype(mx.float32) for x in (mixes, scale, base))
    pre = _activation(spec)(mixes * scale + base)
    if spec.pre_eps:
        pre = pre + spec.eps
    return hc_collapse(spec, streams, pre)


def hc_block(spec: HCSpec, streams, mixes, scale, base):
    """``(collapsed, post, comb)``: coefficients then the pre-gated collapse."""
    pre, post, comb = hc_coefficients(
        spec, mixes, scale, base, stream_dtype=streams.dtype
    )
    return hc_collapse(spec, streams, pre), post, comb


# ---------------------------------------------------------------------------
# Metal source generation (HC4 DS4 law, shared by deepseek_v4_hc and
# glm53_hc_mega). Pure strings; byte-identical to the former copies.
# ---------------------------------------------------------------------------


def split_sinkhorn_body(*, n=4, row_offset="row*24+", indent=""):
    """Serial pre/post/Sinkhorn body over ``mixes``/``scale``/``base`` into ``out``.

    ``row_offset`` prefixes every ``mixes``/``out`` index (``"row*24+"`` for
    one-thread-per-row grids, ``""`` for a per-row pointer); the epsilon is
    the in-scope ``e``. Lines end in ``\\n`` and carry ``indent``.
    """
    if n != 4:
        # The kernels and their row offsets ("row*24+") are HC4; a generic n
        # would need the caller's offset too, so refuse rather than guess.
        raise ValueError("split_sinkhorn_body is HC4 only")
    square, two = n * n, 2 * n
    o = row_offset
    lines = [
        f"float c[{square}];",
        f"for(uint i=0;i<{n};i++) {{",
        f"  const float pre=mixes[{o}i]*scale[0]+base[i];",
        f"  const float post=mixes[{o}{n}+i]*scale[1]+base[{n}+i];",
        f"  out[{o}i]=1.0f/(1.0f+exp(-pre))+e;",
        f"  out[{o}{n}+i]=2.0f/(1.0f+exp(-post));",
        "}",
        f"for(uint r=0;r<{n};r++) {{",
        "  float maximum=-INFINITY;",
        f"  for(uint s=0;s<{n};s++) {{ c[r*{n}+s]=mixes[{o}{two}+r*{n}+s]*scale[2]+base[{two}+r*{n}+s]; maximum=max(maximum,c[r*{n}+s]); }}",
        "  float sum=0.0f;",
        f"  for(uint s=0;s<{n};s++) {{ c[r*{n}+s]=exp(c[r*{n}+s]-maximum); sum+=c[r*{n}+s]; }}",
        f"  for(uint s=0;s<{n};s++) c[r*{n}+s]=c[r*{n}+s]/sum+e;",
        "}",
        f"for(uint s=0;s<{n};s++) {{",
        "  float sum=0.0f;",
        f"  for(uint r=0;r<{n};r++) sum+=c[r*{n}+s];",
        f"  for(uint r=0;r<{n};r++) c[r*{n}+s]/=(sum+e);",
        "}",
        "for(uint it=1;it<ITERS;it++) {",
        f"  for(uint r=0;r<{n};r++) {{",
        "    float sum=0.0f;",
        f"    for(uint s=0;s<{n};s++) sum+=c[r*{n}+s];",
        f"    for(uint s=0;s<{n};s++) c[r*{n}+s]/=(sum+e);",
        "  }",
        f"  for(uint s=0;s<{n};s++) {{",
        "    float sum=0.0f;",
        f"    for(uint r=0;r<{n};r++) sum+=c[r*{n}+s];",
        f"    for(uint r=0;r<{n};r++) c[r*{n}+s]/=(sum+e);",
        "  }",
        "}",
        f"for(uint i=0;i<{square};i++) out[{o}{two}+i]=c[i];",
    ]
    return "".join(indent + line + "\n" for line in lines)


def weighted_sum_lines(*, value_index, weight_index, n=4, indent=""):
    """``sum = sum_h value[...] * weights[...]`` (stream loop ``h``)."""
    lines = [
        "float sum=0.0f;",
        f"for(uint h=0;h<{n};h++) sum+=value[{value_index}]*weights[{weight_index}];",
    ]
    return "".join(indent + line + "\n" for line in lines)


def expand_lines(*, branch_index, post_index, residual_index, comb_index, n=4, indent=""):
    """``sum = branch * post[h] + sum_s residual[s] * comb[s, h]`` (DS4 orientation)."""
    lines = [
        f"float sum=branch[{branch_index}]*post[{post_index}];",
        f"for(uint s=0;s<{n};s++) sum+=residual[{residual_index}]*comb[{comb_index}];",
    ]
    return "".join(indent + line + "\n" for line in lines)


# ---------------------------------------------------------------------------
# fast-path registry
# ---------------------------------------------------------------------------

#: Coefficient law of the HC4 DS4 split kernel (deepseek_v4_hc SPLIT_SOURCE and
#: the generated copies): sigmoid gates, +eps pre gate, row softmax seeded with
#: +eps that counts as the first row pass, per-segment affine, F32 results.
DS4_KERNEL_LAW = dict(
    hc_mult=4,
    coeff_affine="per_segment",
    pre_eps=True,
    post_scale=2.0,
    comb_clip=None,
    comb_normalizer="softmax_precise",
    comb_seed_eps=True,
    softmax_counts_as_first_pass=True,
    coeff_round="fp32",
)
#: The served specs that call the DS4 kernels: DS4/DS4.1 (mx.sigmoid, F32
#: upcast) and GLM-5.3 (span-aware sigmoid, no casts, F32 operands).
_DS4_SERVED = dict(gate_activation=("sigmoid", "span_aware_sigmoid"), upcast_fp32=(True, False))
#: Exact law of GLM-5.3's mega/fused boundaries (pre-projection RMS).
GLM53_BOUNDARY_LAW = dict(
    DS4_KERNEL_LAW, norm_kind="pre_rms", gate_activation="span_aware_sigmoid",
    upcast_fp32=False, orientation="transpose",
)
#: Exact law of the DS4.1 Source boundaries (post-projection RMS).
DS41_SOURCE_LAW = dict(
    DS4_KERNEL_LAW, norm_kind="post_rsqrt_mean", gate_activation="sigmoid",
    upcast_fp32=True, orientation="transpose",
)
#: Exact law of maxwell_moe_metal.mhc_pre.
MAXWELL_PRE_LAW = dict(
    hc_mult=4, norm_kind="post_inv_l2", norm_square="mul", coeff_affine="full_vector",
    gate_activation="sigmoid", pre_eps=True, post_scale=2.0, comb_clip=None,
    comb_normalizer="softmax", comb_seed_eps=True, softmax_counts_as_first_pass=True,
    upcast_fp32=True, coeff_round="stream",
)
#: Exact law of xing4_0_mhc_metal.mhc_pre (any clamp; fp32 coefficients).
XING_PRE_LAW = dict(
    hc_mult=4, norm_kind="pre_rms", coeff_affine="per_segment", gate_activation="sigmoid",
    pre_eps=False, post_scale=2.0, comb_normalizer="exp_max_shift", comb_seed_eps=False,
    softmax_counts_as_first_pass=False, upcast_fp32=True, coeff_round="fp32",
)


def _law_refusal(spec, law):
    """``None`` when every field of ``spec`` is one of the law's values."""
    n = law.get("hc_mult")
    if n is not None and spec.hc_mult != n:
        return f"geometry: requires HC{n} (spec has {spec.hc_mult})"
    for key, allowed in law.items():
        allowed = allowed if isinstance(allowed, tuple) and key != "comb_clip" else (allowed,)
        if getattr(spec, key) not in allowed:
            return f"spec_law: requires {key} in {allowed!r} (spec has {getattr(spec, key)!r})"
    return None


def _gpu_refusal():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device: requires an owned Metal GPU execution window"
    return None


def _admits_law(law, *, extra=None):
    def admits(spec, **shape):
        reason = _law_refusal(spec, law)
        if reason is None and extra is not None:
            reason = extra(spec, **shape)
        return reason or _gpu_refusal()

    admits.law = law
    return admits


def _module(name):
    """Static (analysable) lazy imports of the kernel modules; no GPU work."""
    if name == "deepseek_v4_hc":
        from ..models import deepseek_v4_hc as module
    elif name == "deepseek_v4_hc_projection":
        from ..models import deepseek_v4_hc_projection as module
    elif name == "deepseek_v41_source_hc":
        from ..models import deepseek_v41_source_hc as module
    elif name == "deepseek_v41_source_hc_mega":
        from ..models import deepseek_v41_source_hc_mega as module
    elif name == "deepseek_v41_source_hc_boundary":
        from ..models import deepseek_v41_source_hc_boundary as module
    elif name == "deepseek_v41_fused":
        from ..models import deepseek_v41_fused as module
    elif name == "glm53_hc_mega":
        from ..models import glm53_hc_mega as module
    elif name == "glm53_hc_fused":
        from ..models import glm53_hc_fused as module
    elif name == "maxwell_moe_metal":
        from ..models import maxwell_moe_metal as module
    elif name == "xing4_0_mhc_metal":
        from ..models import xing4_0_mhc_metal as module
    else:
        raise KeyError(f"unknown hyper-connection kernel module {name!r}")
    return module


def _target(module, attr):
    return getattr(_module(module), attr)


def _lazy(module, attr):
    """A boundary path: the kernel module's own signature (no single reference)."""

    def run(*args, **kwargs):
        return _target(module, attr)(*args, **kwargs)

    run.__qualname__ = f"{module}.{attr}"
    return run


def _out(spec, value, dtype):
    return value.astype(dtype) if spec.upcast_fp32 else value


# Reference-signature wrappers: each fast path is called exactly like the
# reference it replaces and returns the reference's dtypes.
def _coefficients_via(module, attr):
    def run(spec, mixes, scale, base, *, stream_dtype=None):
        return _target(module, attr)(mixes, scale, base, iterations=spec.sinkhorn_iters, eps=spec.eps)

    run.__qualname__ = f"{module}.{attr}"
    return run


def _ds4_collapse(spec, streams, pre, *, out_dtype=None):
    value = _target("deepseek_v4_hc", "weighted_sum")(streams, pre)
    return _out(spec, value, streams.dtype if out_dtype is None else out_dtype)


def _ds4_expand(spec, branch, residual, post, comb):
    value = _target("deepseek_v4_hc", "expand")(branch, residual, post, comb)
    return _out(spec, value, branch.dtype if spec.expand_out == "branch" else residual.dtype)


def _ds4_head(spec, streams, mixes, scale, base):
    value = _target("deepseek_v4_hc", "head_reduce")(streams, mixes, scale, base, eps=spec.eps)
    return _out(spec, value, streams.dtype)


def _ds4_mixes(spec, flat, fn=None, *, project=None, head=False):
    normalized = mx.fast.rms_norm(flat.astype(mx.float32), None, spec.norm_eps)
    return _target("deepseek_v4_hc_projection", "project")(normalized, fn.astype(mx.float32))


def _projection_refusal(spec, *, project=None, head=False, **_):
    if project is not None:
        return "policy_off: a caller-owned projection replaces fn"
    kind = (spec.head_norm_kind or spec.norm_kind) if head else spec.norm_kind
    if kind != "pre_rms":
        return "spec_law: fixed projection requires pre_rms normalization"
    return None


TECHNOLOGY = "hyper_connections"

COEFFICIENTS = KernelRegistry(TECHNOLOGY + ".coefficients", hc_coefficients)
COLLAPSE = KernelRegistry(TECHNOLOGY + ".collapse", hc_collapse)
EXPAND = KernelRegistry(TECHNOLOGY + ".expand", hc_expand)
HEAD = KernelRegistry(TECHNOLOGY + ".head", hc_head)
MIXES = KernelRegistry(TECHNOLOGY + ".mixes", hc_mixes)
#: Fused boundaries (several operations in one launch, or an epilogue the
#: reference does not have); no single reference, so each path keeps its
#: module's signature and its reference is the composition its module names.
BOUNDARY = KernelRegistry(TECHNOLOGY + ".boundary", None)

_ITERS_1_100 = lambda spec, **_: (  # noqa: E731
    None if 1 <= spec.sinkhorn_iters <= 100 else "iterations: requires 1..100 Sinkhorn iterations"
)

for registry, path in (
    (COEFFICIENTS, FastPath("ds4_split_sinkhorn_fp32",
                            _admits_law(dict(DS4_KERNEL_LAW, **_DS4_SERVED), extra=_ITERS_1_100),
                            _coefficients_via("deepseek_v4_hc", "split_sinkhorn"))),
    (COEFFICIENTS, FastPath("ds41_source_split_fp32",
                            _admits_law(DS41_SOURCE_LAW, extra=_ITERS_1_100),
                            _coefficients_via("deepseek_v41_source_hc", "split"))),
    (COLLAPSE, FastPath("ds4_weighted_sum_fp32",
                        _admits_law(dict(hc_mult=4, **_DS4_SERVED)), _ds4_collapse)),
    (EXPAND, FastPath("ds4_expand_fp32",
                      _admits_law(dict(hc_mult=4, orientation="transpose", **_DS4_SERVED)),
                      _ds4_expand)),
    (HEAD, FastPath("ds4_head_reduce_fp32",
                    _admits_law(dict(hc_mult=4, pre_eps=True, gate_activation="sigmoid",
                                     upcast_fp32=True)), _ds4_head)),
    (MIXES, FastPath("ds4_projection_fixed_fp32",
                     _admits_law(dict(hc_mult=4, upcast_fp32=True),
                                 extra=_projection_refusal), _ds4_mixes)),
    (BOUNDARY, FastPath("glm53_hc_mega_post", _admits_law(GLM53_BOUNDARY_LAW),
                        _lazy("glm53_hc_mega", "post"))),
    (BOUNDARY, FastPath("glm53_hc_mega_pre", _admits_law(GLM53_BOUNDARY_LAW),
                        _lazy("glm53_hc_mega", "pre"))),
    (BOUNDARY, FastPath("glm53_hc_fused_boundary", _admits_law(GLM53_BOUNDARY_LAW),
                        _lazy("glm53_hc_fused", "boundary"))),
    (BOUNDARY, FastPath("ds41_source_hc_mega_pre", _admits_law(DS41_SOURCE_LAW),
                        _lazy("deepseek_v41_source_hc_mega", "pre"))),
    (BOUNDARY, FastPath("ds41_source_hc_mega_store", _admits_law(DS41_SOURCE_LAW),
                        _lazy("deepseek_v41_source_hc_mega", "store"))),
    (BOUNDARY, FastPath("ds41_source_hc_expand_collapse", _admits_law(DS41_SOURCE_LAW),
                        _lazy("deepseek_v41_source_hc_boundary", "expand_collapse"))),
    # Expand with DS4.1's activation rounding epilogue (not the reference expand).
    (BOUNDARY, FastPath("ds41_fused_expand", _admits_law(DS41_SOURCE_LAW),
                        _lazy("deepseek_v41_fused", "expand"))),
    (BOUNDARY, FastPath("maxwell_mhc_pre", _admits_law(MAXWELL_PRE_LAW),
                        _lazy("maxwell_moe_metal", "mhc_pre"))),
    (BOUNDARY, FastPath("maxwell_mhc_post",
                        _admits_law(dict(hc_mult=4, orientation="transpose", upcast_fp32=True,
                                         expand_out="residual")),
                        _lazy("maxwell_moe_metal", "mhc_post"))),
    (BOUNDARY, FastPath("xing4_0_mhc_pre", _admits_law(XING_PRE_LAW),
                        _lazy("xing4_0_mhc_metal", "mhc_pre"))),
    (BOUNDARY, FastPath("xing4_0_mhc_update",
                        _admits_law(dict(hc_mult=4, orientation="direct", upcast_fp32=True,
                                         expand_out="residual")),
                        _lazy("xing4_0_mhc_metal", "mhc_update"))),
):
    registry.register(path)

REGISTRIES = {r.op: r for r in (COEFFICIENTS, COLLAPSE, EXPAND, HEAD, MIXES, BOUNDARY)}


def select(operation: str, spec: HCSpec, policy: str | None = None, **shape):
    """``(callable, receipt)`` for one HC operation (``"coefficients"`` or
    ``"hyper_connections.coefficients"``; counted by the shared dispatcher).

    Fast paths are opt-in: ``policy`` ``None``/``"reference"`` returns the
    reference; a fast-path name runs it only if it admits ``spec``/``shape``
    and fails closed otherwise. No path is ever chosen implicitly.
    """
    registry = REGISTRIES[operation.removeprefix(TECHNOLOGY + ".")]
    if policy in (None, "reference"):
        if registry.reference is None:
            raise RuntimeError(f"{operation}: no single reference; name a fast path")
        return registry.select(spec, policy="reference")
    return registry.select(spec, policy=policy, **shape)


__all__ = [
    "BOUNDARY",
    "COEFFICIENTS",
    "COLLAPSE",
    "DS4_KERNEL_LAW",
    "EXPAND",
    "HCSpec",
    "HEAD",
    "TECHNOLOGY",
    "MIXES",
    "cached_spec",
    "expand_lines",
    "hc_block",
    "hc_coefficients",
    "hc_collapse",
    "hc_expand",
    "hc_gates",
    "hc_head",
    "hc_mixes",
    "derive",
    "select",
    "split_sinkhorn_body",
    "weighted_sum_lines",
]
