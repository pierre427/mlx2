"""MoE routing: gate logits -> (expert ids, combine weights).

One reference law, parameterised by :class:`RouterSpec`:

1. *prepare*: optional cast of the logits (``score_dtype``) and a tanh
   soft-cap (``logit_softcap``);
2. *score*: ``softmax`` / ``sigmoid`` / ``sqrtsoftplus`` (``score_func``),
   either before top-k (scores choose the experts) or, for
   ``softmax_position="after_topk"``, on the picked logits only;
3. *selection*: optional widening of the scores (``selection_dtype``), the
   correction bias (``use_bias``/``bias_cast``), group-limited masking
   (``n_group``/``topk_group``/``group_score``/``group_select``/
   ``group_fill``) and the top-k primitive (``topk_method``);
4. *combine*: gather the unbiased or biased scores (``combine_scores``),
   renormalise (``norm_topk_prob``/``norm_eps``/``norm_min_top_k``, or the
   log-domain law ``renorm="logdomain"``), multiply by
   ``routed_scaling_factor`` and an optional per-expert scale, cast.

Every field reproduces one existing model's arithmetic; differences that
change bits on some input are fields named for what they do, never folded
together. Expert execution is out of scope; fused routers are registered in
:data:`ROUTE` as fast paths that declare which specs they cover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn

from .dispatch import FastPath, KernelRegistry, spec_from_config

ScoreFunc = Literal["softmax", "sigmoid", "sqrtsoftplus", "identity"]
TopkMethod = Literal["argpartition_high", "argpartition_neg", "argsort_neg"]

_SCORE_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus", "identity")
_TOPK_METHODS = ("argpartition_high", "argpartition_neg", "argsort_neg")


@dataclass(frozen=True)
class RouterSpec:
    """Every per-model parameter of the routing law.

    Fields (defaults are the plain Qwen/Mixtral softmax router):

    ``top_k``                 experts per token.
    ``score_func``            ``softmax`` | ``sigmoid`` | ``sqrtsoftplus``
                              (``sqrt(softplus(z))``, DeepSeek-V4) |
                              ``identity`` (the caller passes scores).
    ``softmax_position``      ``before_topk`` (scores choose) or
                              ``after_topk`` (top-k on logits, softmax over
                              the picked logits; Gemma4/Granite/gpt-oss).
    ``softmax_precise``       ``mx.softmax(precise=...)``.
    ``sigmoid_impl``          ``mx`` (``mx.sigmoid``) or ``gate_sigmoid``
                              (``precise_ops.gate_sigmoid``: precise inside a
                              traced span; GLM-5.3).
    ``score_dtype``           ``input`` keeps the logits dtype; ``float32``
                              casts the logits before scoring.
    ``logit_softcap``         ``> 0``: ``tanh(z / c) * c`` after the cast.
    ``selection_dtype``       ``scores`` or ``float32``: selection runs on a
                              widened copy while combine weights keep the
                              score dtype (LFM2.5).
    ``use_bias``              a correction bias is added for selection
                              (``e_score_correction_bias``/``expert_bias``);
                              ``route`` requires one exactly when set.
    ``bias_cast``             ``promote`` (``choice + bias``) or
                              ``to_scores`` (``choice + bias.astype(choice.dtype)``,
                              ``choice`` being the scores in ``selection_dtype``).
    ``combine_scores``        ``unbiased`` (conventional) or ``biased`` (the
                              hub ``modeling_kimi.py`` alias law).
    ``n_group``/``topk_group`` group-limited selection; applied when
                              ``n_group > 1`` and ``topk_group < n_group``
                              (otherwise no group is dropped).
    ``group_score``           ``top2_sum`` (noaux_tc) or ``max``
                              (group_limited_greedy).
    ``group_select``          ``drop_argpartition`` (partition out the
                              ``n_group - topk_group`` lowest groups) or
                              ``keep_argsort`` (keep the stable-argsort top
                              ``topk_group``); they differ only on ties.
    ``group_fill``            score written into dropped groups (``-inf``
                              is correct; ``0.0`` can pick a dropped expert
                              when a negative bias pushes kept scores < 0).
    ``topk_method``           ``argpartition_high``
                              (``argpartition(s, -k)[..., -k:]``),
                              ``argpartition_neg``
                              (``argpartition(-s, k-1)[..., :k]``) or
                              ``argsort_neg`` (``argsort(-s)[..., :k]``);
                              they agree on the selected set except at ties,
                              and differ in slot order.
    ``norm_topk_prob``        renormalise the picked weights.
    ``norm_eps``              ``w / (sum + eps)``; ``0.0`` is ``w / sum``.
    ``norm_min_top_k``        renormalise only when ``top_k >=`` this
                              (``2`` for the ``top_k > 1 and norm`` laws).
    ``renorm``                ``linear`` or ``logdomain``
                              (``softmax(log_sigmoid(picked logits))``: no
                              all-picked-sigmoids-underflow zero; sigmoid
                              only; Inkling2 opt-in).
    ``routed_scaling_factor`` multiplies the final weights.
    ``index_dtype``           ``native`` (``uint32`` from the primitives),
                              ``int32`` or ``uint32`` (explicit cast).
    ``hash_layer``            the layer routes by a token-id table
                              (DeepSeek-V4 hash layers); ``route`` then
                              needs ``hash_table`` and ``token_ids``.
    """

    top_k: int
    score_func: ScoreFunc = "softmax"
    softmax_position: Literal["before_topk", "after_topk"] = "before_topk"
    softmax_precise: bool = True
    sigmoid_impl: Literal["mx", "gate_sigmoid"] = "mx"
    score_dtype: Literal["input", "float32"] = "input"
    logit_softcap: float = 0.0
    selection_dtype: Literal["scores", "float32"] = "scores"
    use_bias: bool = False
    bias_cast: Literal["promote", "to_scores"] = "promote"
    combine_scores: Literal["unbiased", "biased"] = "unbiased"
    n_group: int = 1
    topk_group: int = 1
    group_score: Literal["top2_sum", "max"] = "top2_sum"
    group_select: Literal["drop_argpartition", "keep_argsort"] = "drop_argpartition"
    group_fill: float = -math.inf
    topk_method: TopkMethod = "argpartition_high"
    norm_topk_prob: bool = False
    norm_eps: float = 0.0
    norm_min_top_k: int = 1
    renorm: Literal["linear", "logdomain"] = "linear"
    routed_scaling_factor: float = 1.0
    index_dtype: Literal["native", "int32", "uint32"] = "native"
    hash_layer: bool = False

    def __post_init__(self):
        if int(self.top_k) < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.score_func not in _SCORE_FUNCS:
            raise ValueError(f"unsupported MoE score function {self.score_func!r}")
        if self.topk_method not in _TOPK_METHODS:
            raise ValueError(f"unsupported top-k method {self.topk_method!r}")
        if self.softmax_position == "after_topk":
            if self.score_func != "softmax":
                raise ValueError("softmax_position='after_topk' needs score_func='softmax'")
            if self.use_bias or self.grouped:
                raise ValueError("after-top-k softmax takes no bias or group limit")
        if self.renorm == "logdomain" and self.score_func != "sigmoid":
            raise ValueError("log-domain renormalisation is defined for sigmoid scores")
        if self.n_group < 1 or (self.grouped and self.topk_group < 1):
            raise ValueError(
                f"invalid expert groups n_group={self.n_group} topk_group={self.topk_group}"
            )

    @property
    def grouped(self) -> bool:
        """Group limiting changes the choice only when groups are dropped."""
        return self.n_group > 1 and self.topk_group < self.n_group

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "RouterSpec":
        """Derive the law from standard HF MoE config fields.

        ``num_experts_per_tok``, ``scoring_func``, ``topk_method``,
        ``n_group``, ``topk_group``, ``norm_topk_prob``,
        ``routed_scaling_factor``, and the presence of a correction bias
        (``topk_method == "noaux_tc"``, ``moe_router_enable_expert_bias``,
        ``use_expert_bias``). Sigmoid-family laws default to the HF
        DeepSeek-V3 form (fp32 scores, ``argpartition_neg``, ``+1e-20``,
        normalise only when ``top_k > 1``). Numerics fields a config does not
        state (top-k primitive, epsilon, dtypes) are model identity: pass
        them as overrides.
        """

        def get(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        score = (
            get("scoring_func") or get("score_function") or get("moe_router_score_func")
            or "softmax"
        )
        method = get("topk_method")
        sigmoid_family = score in ("sigmoid", "sqrtsoftplus")
        bias = bool(
            method == "noaux_tc"
            or get("moe_router_enable_expert_bias")
            or get("use_expert_bias")
        )
        derived: dict[str, Any] = {
            "score_func": score,
            "use_bias": bias,
            "group_score": "max" if method == "group_limited_greedy" else "top2_sum",
        }
        if sigmoid_family:
            derived.update(
                score_dtype="float32", topk_method="argpartition_neg",
                norm_eps=1e-20, norm_min_top_k=2,
            )
        scale = get("routed_scaling_factor")
        if scale is None:
            scale = get("router_scaling_factor")
        if scale is None:
            scale = get("moe_routed_scaling_factor")
        mapping = {
            "top_k": ("num_experts_per_tok", "moe_topk", "top_k_experts",
                      "num_experts_per_token"),
            "n_group": ("n_group", "num_expert_group"),
            "topk_group": "topk_group",
            "norm_topk_prob": ("norm_topk_prob", "route_norm", "moe_renormalize"),
        }
        derived.update({k: v for k, v in overrides.items()})
        if scale is not None and "routed_scaling_factor" not in overrides:
            derived["routed_scaling_factor"] = float(scale)
        return spec_from_config(cls, config, mapping, **derived)


@lru_cache(maxsize=None)
def cached_spec(**fields) -> RouterSpec:
    """A memoised ``RouterSpec`` for hot call sites that rebuild it per call."""
    return RouterSpec(**fields)


# ----------------------------------------------------------------- reference


def prepare_logits(logits: mx.array, spec: RouterSpec) -> mx.array:
    """The score-dtype cast and soft-cap (shared with fused routers)."""
    if spec.score_dtype == "float32":
        logits = logits.astype(mx.float32)
    if spec.logit_softcap > 0:
        logits = mx.tanh(logits / spec.logit_softcap) * spec.logit_softcap
    return logits


def _sigmoid(x, spec):
    if spec.sigmoid_impl == "gate_sigmoid":
        from ..models.precise_ops import gate_sigmoid

        return gate_sigmoid(x)
    return mx.sigmoid(x)


def _score(logits, spec):
    if spec.score_func == "softmax":
        return mx.softmax(logits, axis=-1, precise=spec.softmax_precise)
    if spec.score_func == "sigmoid":
        return _sigmoid(logits, spec)
    if spec.score_func == "identity":
        return logits
    return mx.sqrt(nn.softplus(logits))


def _topk(values, k, method):
    # Expert ids are discrete: stop_gradient is value-neutral and keeps the
    # combine weights differentiable (take_along_axis has no index VJP).
    if method == "argpartition_high":
        ids = mx.argpartition(values, kth=-k, axis=-1)[..., -k:]
    elif method == "argpartition_neg":
        ids = mx.argpartition(-values, kth=k - 1, axis=-1)[..., :k]
    else:
        ids = mx.argsort(-values, axis=-1)[..., :k]
    return mx.stop_gradient(ids)


def _group_limit(choice, spec):
    grouped = mx.unflatten(choice, axis=-1, shape=(spec.n_group, -1))
    if spec.group_score == "max":
        group_scores = mx.max(grouped, axis=-1, keepdims=True)
    else:
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
    fill = mx.array(spec.group_fill, dtype=choice.dtype)
    if spec.group_select == "drop_argpartition":
        n_drop = spec.n_group - spec.topk_group
        drop = mx.argpartition(group_scores, kth=n_drop - 1, axis=-2)[..., :n_drop, :]
        grouped = mx.put_along_axis(grouped, mx.stop_gradient(drop), fill, axis=-2)
    else:
        keep = mx.argsort(-group_scores[..., 0], axis=-1)[..., : spec.topk_group]
        mask = mx.put_along_axis(
            mx.zeros(group_scores.shape[:-1], mx.bool_), keep, mx.array(True), axis=-1
        )
        grouped = mx.where(mask[..., None], grouped, fill)
    return mx.flatten(grouped, -2, -1)


def route(
    logits: mx.array,
    spec: RouterSpec,
    bias: mx.array | None = None,
    *,
    out_dtype=None,
    expert_scale: mx.array | None = None,
    hash_table: mx.array | None = None,
    token_ids: mx.array | None = None,
    prepared: bool = False,
    with_scores: bool = False,
):
    """Reference router: ``(indices, weights)`` per row of ``logits``.

    ``bias`` is the selection correction (required iff ``spec.use_bias``).
    ``out_dtype`` casts the final weights. ``expert_scale`` multiplies each
    picked weight by a per-expert factor (Gemma4 ``per_expert_scale``).
    ``prepared=True`` skips :func:`prepare_logits` (the caller already
    applied it, e.g. before trying a fused router). ``with_scores=True``
    returns ``(indices, weights, picked)`` where ``picked`` is the gathered
    combine source before renormalisation.
    """
    if spec.use_bias and bias is None and not spec.hash_layer:
        raise ValueError("learned routing requires correction bias")
    if not spec.use_bias and bias is not None:
        raise ValueError("router spec declares no correction bias but one was given")
    if not prepared:
        logits = prepare_logits(logits, spec)
    k = spec.top_k

    if spec.softmax_position == "after_topk":
        indices = _topk(logits, k, spec.topk_method)
        picked = mx.take_along_axis(logits, indices, axis=-1)
        weights = mx.softmax(picked, axis=-1, precise=spec.softmax_precise)
        picked_scores = weights
    else:
        scores = _score(logits, spec)
        choice = scores
        if spec.selection_dtype == "float32":
            choice = choice.astype(mx.float32)
        if spec.hash_layer:
            if hash_table is None or token_ids is None:
                raise ValueError("hash routing requires original token IDs")
            indices = mx.stop_gradient(hash_table[token_ids].astype(mx.int32))
        else:
            if bias is not None:
                b = bias.astype(choice.dtype) if spec.bias_cast == "to_scores" else bias
                choice = choice + b
            if spec.grouped:
                choice = _group_limit(choice, spec)
            indices = _topk(choice, k, spec.topk_method)
        source = choice if spec.combine_scores == "biased" else scores
        if spec.renorm == "logdomain":
            picked = mx.take_along_axis(logits, indices, axis=-1)
            log_w = mx.minimum(picked, 0) - mx.log1p(mx.exp(-mx.abs(picked)))
            log_w = log_w - log_w.max(axis=-1, keepdims=True)
            weights = mx.exp(log_w)
            picked_scores = mx.take_along_axis(source, indices, axis=-1)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        else:
            weights = mx.take_along_axis(source, indices, axis=-1)
            picked_scores = weights
            if spec.norm_topk_prob and k >= spec.norm_min_top_k:
                total = weights.sum(axis=-1, keepdims=True)
                if spec.norm_eps:
                    total = total + spec.norm_eps
                weights = weights / total
    if spec.routed_scaling_factor != 1.0:
        weights = weights * spec.routed_scaling_factor
    if expert_scale is not None:
        weights = weights * expert_scale[indices]
    if out_dtype is not None:
        weights = weights.astype(out_dtype)
    if spec.index_dtype == "uint32":
        indices = indices.astype(mx.uint32)
    elif spec.index_dtype == "int32":
        indices = indices.astype(mx.int32)
    if with_scores:
        return indices, weights, picked_scores
    return indices, weights


# ---------------------------------------------------------------- fast paths

ROUTE = KernelRegistry("moe_router.route", route)

_FIELDS = tuple(f.name for f in fields(RouterSpec))


def _metal_refusal() -> str | None:
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return "device: Metal kernels are unavailable"
    if mx.default_device() != mx.gpu:
        return "device: default device is not GPU"
    return None


def law_refusal(spec: RouterSpec, law: RouterSpec, free: frozenset = frozenset()) -> str | None:
    """``law:`` refusal unless ``spec`` equals ``law`` on every field not in
    ``free`` (the fields a kernel takes as runtime parameters)."""
    for name in _FIELDS:
        if name in free:
            continue
        have, want = getattr(spec, name), getattr(law, name)
        if have != want:
            return f"law: {name}={have!r}; the kernel implements {want!r}"
    return None


# The plain softmax law (Qwen3-MoE/Qwen3.6/Qwen4/Mellum), every field but
# ``top_k``: the Qwen MoeRoutingContract and the softmax fused routers below
# implement exactly this, at their own geometry.
SOFTMAX_TOPK_LAW = {
    name: getattr(RouterSpec(top_k=1, norm_topk_prob=True), name)
    for name in _FIELDS if name != "top_k"
}


@dataclass(frozen=True)
class RouterFastPath:
    """A fused router behind the reference signature.

    ``law`` is the full spec the kernel reproduces bit for bit; ``free``
    names the spec fields the kernel takes as runtime parameters (its own
    admission still bounds them). ``kernel(logits, spec, bias, **kw)``
    returns ``(indices, weights)`` with ``weights`` in the reference dtype
    before ``out_dtype``; ``run`` applies the rest of the reference
    signature (``prepared``, ``expert_scale``, ``out_dtype``,
    ``with_scores``) with the reference's own ops and order.
    """

    name: str
    law: RouterSpec
    free: frozenset
    kernel_admits: Any
    kernel: Any

    def law_refusal(self, spec: RouterSpec) -> str | None:
        return law_refusal(spec, self.law, self.free)

    def admits(self, spec, *, logits=None, bias=None, hash_table=None, token_ids=None, **kw):
        reason = self.law_refusal(spec)
        if reason:
            return reason
        if hash_table is not None or token_ids is not None:
            return "law: fused routers do not route hash layers"
        if logits is None:
            return "shape: no logits given"
        if spec.use_bias != (bias is not None):
            return "shape: correction bias presence differs from the spec"
        return self.kernel_admits(spec, logits=logits, bias=bias, **kw)

    def run(self, logits, spec, bias=None, *, out_dtype=None, expert_scale=None,
            hash_table=None, token_ids=None, prepared=False, with_scores=False, **kw):
        reason = self.law_refusal(spec)
        if reason:
            raise ValueError(f"{self.name}: {reason}")
        if hash_table is not None or token_ids is not None:
            raise ValueError(f"{self.name}: fused routers do not route hash layers")
        if spec.use_bias != (bias is not None):
            raise ValueError(f"{self.name}: correction bias presence differs from the spec")
        if not prepared:
            logits = prepare_logits(logits, spec)
        indices, weights = self.kernel(logits, spec, bias, **kw)
        if expert_scale is not None:
            weights = weights * expert_scale[indices]
        if out_dtype is not None:
            weights = weights.astype(out_dtype)
        if not with_scores:
            return indices, weights
        # The gathered combine source, recomputed with the reference ops.
        source = _score(logits, spec)
        if spec.combine_scores == "biased":
            if spec.selection_dtype == "float32":
                source = source.astype(mx.float32)
            source = source + (bias.astype(source.dtype) if spec.bias_cast == "to_scores" else bias)
        return indices, weights, mx.take_along_axis(source, indices, axis=-1)


def _qwen4_router_admits(spec, *, logits, candidate_token_widths=(), **_):
    from ..models.qwen4_moe_router import admit_qwen4_moe_router

    adm = admit_qwen4_moe_router(
        logits, top_k=spec.top_k, norm_topk_prob=spec.norm_topk_prob,
        candidate_token_widths=candidate_token_widths,
    )
    return None if adm.accepted else f"admission: {adm.reason}"


def _qwen4_router_kernel(logits, spec, bias, *, candidate_token_widths=(), **_):
    from ..models.qwen4_moe_router import qwen4_moe_router

    return qwen4_moe_router(logits, candidate_token_widths=candidate_token_widths)


def _nemotron_admits(spec, *, logits, bias, **_):
    from ..models.nemotron_moe_router import admit

    adm = admit(
        logits, bias, top_k=spec.top_k, n_group=spec.n_group, topk_group=spec.topk_group,
        norm_topk_prob=spec.norm_topk_prob, routed_scaling_factor=spec.routed_scaling_factor,
    )
    return None if adm.accepted else f"admission: {adm.reason}"


def _nemotron_kernel(logits, spec, bias, **_):
    from ..models.nemotron_moe_router import router_top22

    return router_top22(logits, bias)


def _glm53_admits(spec, *, logits, bias, **_):
    from ..models.glm53_glue_metal import router_admits

    if not router_admits(logits, bias, spec.top_k):
        return "admission: glm53 fused router declines these logits/bias (fp32, E<=1024, GPU)"
    return None


def _glm53_kernel(logits, spec, bias, **_):
    from ..models.glm53_glue_metal import fused_router

    return fused_router(logits, bias, topk=spec.top_k, scale=spec.routed_scaling_factor)


def _laguna_admits(spec, *, logits, bias, candidate_token_widths=(), **_):
    from ..models.laguna_fused_moe import admit_laguna_router

    adm = admit_laguna_router(
        logits, bias, top_k=spec.top_k, norm_topk_prob=True, use_sigmoid=True,
        softcap=0.0, candidate_token_widths=candidate_token_widths,
    )
    return None if adm.accepted else f"admission: {adm.reason}"


def _laguna_kernel(logits, spec, bias, *, candidate_token_widths=(), **_):
    from ..models.laguna_fused_moe import laguna_fused_router

    # fp32 weights, as the reference returns before out_dtype.
    return laguna_fused_router(
        logits, bias, top_k=spec.top_k, norm_topk_prob=True, use_sigmoid=True,
        softcap=0.0, output_dtype=mx.float32, candidate_token_widths=candidate_token_widths,
    )


def _deepseek_v41_admits(spec, *, logits, bias, **_):
    if logits.ndim < 1 or not 1 <= spec.top_k <= logits.shape[-1]:
        return "shape: top_k must fit the logits expert dimension"
    if bias.shape != (logits.shape[-1],):
        return "shape: kernel requires one shared correction vector"
    return _metal_refusal()


def _deepseek_v41_kernel(logits, spec, bias, **_):
    from ..models.deepseek_v41_fused import router

    return router(
        logits, top_k=spec.top_k, scale=spec.routed_scaling_factor, correction=bias,
        normalize=spec.norm_topk_prob,
    )


# Each law is the spec its model passes to ``route`` on the path the kernel
# replaces (see the model files); tests pin the equality.
QWEN4_ROUTER_LAW = RouterSpec(top_k=10, **SOFTMAX_TOPK_LAW)
NEMOTRON_ROUTER_LAW = RouterSpec(
    top_k=22, score_func="sigmoid", score_dtype="float32", use_bias=True,
    topk_method="argpartition_neg", norm_topk_prob=True, norm_eps=1e-20,
    norm_min_top_k=2, routed_scaling_factor=5.0,
)
GLM53_ROUTER_LAW = RouterSpec(
    top_k=8, score_func="sigmoid", sigmoid_impl="gate_sigmoid", use_bias=True,
    topk_method="argsort_neg", norm_topk_prob=True, routed_scaling_factor=2.5,
    index_dtype="uint32",
)
# Laguna casts and soft-caps the logits itself (route(..., prepared=True)).
LAGUNA_ROUTER_LAW = RouterSpec(
    top_k=8, score_func="sigmoid", softmax_precise=False, use_bias=True,
    bias_cast="to_scores", topk_method="argpartition_neg", norm_topk_prob=True,
)
DEEPSEEK_V41_ROUTER_LAW = RouterSpec(
    top_k=6, score_func="sqrtsoftplus", score_dtype="float32", use_bias=True,
    topk_method="argpartition_neg", norm_topk_prob=True, norm_eps=1e-20,
    routed_scaling_factor=1.5, index_dtype="uint32",
)

ROUTER_FAST_PATHS: dict[str, RouterFastPath] = {
    p.name: p
    for p in (
        RouterFastPath("qwen4_moe_router", QWEN4_ROUTER_LAW, frozenset(),
                       _qwen4_router_admits, _qwen4_router_kernel),
        RouterFastPath("nemotron_moe_router", NEMOTRON_ROUTER_LAW, frozenset(),
                       _nemotron_admits, _nemotron_kernel),
        RouterFastPath("glm53_fused_router", GLM53_ROUTER_LAW,
                       frozenset({"top_k", "routed_scaling_factor"}),
                       _glm53_admits, _glm53_kernel),
        RouterFastPath("laguna_fused_router", LAGUNA_ROUTER_LAW, frozenset(),
                       _laguna_admits, _laguna_kernel),
        RouterFastPath("deepseek_v41_fused_router", DEEPSEEK_V41_ROUTER_LAW,
                       frozenset({"top_k", "routed_scaling_factor", "norm_topk_prob"}),
                       _deepseek_v41_admits, _deepseek_v41_kernel),
    )
}
for _path in ROUTER_FAST_PATHS.values():
    ROUTE.register(FastPath(_path.name, _path.admits, _path.run))


@dataclass(frozen=True)
class FusedRouterDeclaration:
    """A router fused into an expert/tail kernel, not callable on its own.

    These kernels consume the reference selection (or reproduce it inside a
    larger launch) and are dispatched by their model's own admission; the
    declaration states the full routing law they cover (``free``: fields
    the kernel takes as parameters) so a spec mismatch is a refusal reason
    rather than a silent wrong answer.
    """

    name: str
    module: str
    law: RouterSpec
    free: frozenset = frozenset()

    def law_refusal(self, spec: RouterSpec) -> str | None:
        return law_refusal(spec, self.law, self.free)

    def admits(self, spec: RouterSpec, **_) -> str | None:
        return self.law_refusal(spec)


FUSED_ROUTERS: dict[str, FusedRouterDeclaration] = {
    d.name: d
    for d in (
        # E512/k10 (Qwen4) and E256/k8 (Qwen3.6) through MoeRoutingContract.
        FusedRouterDeclaration(
            "qwen4_moe_window_router_topk", "models.qwen4_moe_window",
            RouterSpec(top_k=10, **SOFTMAX_TOPK_LAW), frozenset({"top_k"}),
        ),
        # lfm25_moe.SparseMoE.routing; the kernel takes K, NORMALIZE, the
        # scale and the (optional) bias as parameters.
        FusedRouterDeclaration(
            "lfm25_moe_decode_route", "models.lfm25_moe_decode",
            RouterSpec(top_k=4, score_func="sigmoid", selection_dtype="float32",
                       use_bias=True, norm_topk_prob=True, norm_eps=1e-6),
            frozenset({"top_k", "use_bias", "norm_topk_prob", "routed_scaling_factor"}),
        ),
        # inkling2.language._route; TOPK and ROUTE_SCALE are kernel template
        # parameters (moe.top_k, moe.route_scale).
        FusedRouterDeclaration(
            "inkling2_moe_decode_kernel", "models.inkling2.moe_decode_kernel",
            RouterSpec(top_k=8, score_func="sigmoid", use_bias=True,
                       topk_method="argpartition_neg", norm_topk_prob=True,
                       norm_eps=1e-20, routed_scaling_factor=8.0),
            frozenset({"top_k", "routed_scaling_factor"}),
        ),
    )
}


def select_router(spec: RouterSpec, *, policy: str | None = None, **shape):
    """``(callable, receipt)`` for ``spec`` over the registered fast paths."""
    return ROUTE.select(spec, policy=policy, **shape)


__all__ = [
    "FUSED_ROUTERS",
    "FusedRouterDeclaration",
    "ROUTE",
    "ROUTER_FAST_PATHS",
    "RouterFastPath",
    "RouterSpec",
    "SOFTMAX_TOPK_LAW",
    "cached_spec",
    "law_refusal",
    "prepare_logits",
    "route",
    "select_router",
]
