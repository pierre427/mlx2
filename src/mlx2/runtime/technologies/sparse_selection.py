# SPDX-License-Identifier: MIT
"""Sparse indexed selection: indexer scores, top-k, gathered attention.

One reference for the lightning-indexer family (DeepSeek V4 / V4.1 CSA,
GLM-5.3 DSA, Kimi K3.1 DSA, HY-V4 group selection) and Qwen4 QSA block
selection, plus the "local + selected + sinks" attention merge their
attention layers share.

Three operations, all parameterised by :class:`SelectionSpec`:

``score(spec, q, keys, weights)``
    ``sum_h w_h * relu(q_h . k)`` (per-head weights) or
    ``sum_h relu(q_h . k)`` (uniform), with the ``1/sqrt(d)`` scale at the
    site the model uses. The models do not agree on op order, layout,
    contraction primitive or where the scale lands, and each of those moves
    bits, so each is a spec field (``score_layout``, ``score_scale_site``,
    ``score_cast``) set to the model's historical value.

``select(spec, scores, valid, count)`` / ``topk(spec, scores, count)``
    Mask invisible columns to ``-inf`` and take the top ``count`` columns
    under the model's tie-break law (``tie_break``). ``stable_chronological``
    is the law of ``deepseek_v4.stable_pool_topk``: strictly greater, then
    exact ties by ascending column, ids ascending. ReLU scores have exact
    zero ties, so the unstable laws (``argsort``, ``argpartition_low``,
    ``argpartition_high``) can change the attended set across replay shapes;
    they are kept only as the historical values of the models that use them.

``gather_attend(q, sources, sinks=..., scale=...)``
    Softmax attention over several key sources merged in log space
    (``logaddexp`` of per-source ``logsumexp``), with learned sinks adding
    denominator mass only. A source is shared across query rows (local
    window, concatenated pool) or gathered per row (selected pool rows); for
    latent MLA sources values are the keys themselves (``values=None``) or a
    leading slice of them (``value_width``), so absorbed K=V caches plug in.

Fast paths (fused Metal score / select / attention kernels) are listed in
the registries at the bottom; each keeps its own call site and admission,
and its ``admits`` here returns the refusal reason the call site uses.
See docs/technologies/sparse_selection.md.
"""
from __future__ import annotations

import importlib
import hashlib
import inspect
import math
from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx

from .dispatch import FastPath, KernelRegistry, spec_from_config

SCORE_LAYOUTS = (
    # q [B,H,L,D], keys [B,S,D], weights [B,L,H] -> [B,L,S]; q @ k^T, head sum axis 1.
    "bhld_matmul",
    # q [B,H,L,D], keys [B,S,D], weights [B,L,H] -> [B,L,S]; einsum bhsd,btd->bsht, head sum axis 2.
    "bhld_einsum",
    # q [B,H,L,D], keys [B,L,C,D] per-row, weights [B,L,H] -> [B,L,C]; einsum bhsd,bstd->bsht.
    "bhld_rows_einsum",
    # q [L,H,D], keys [S,D], weights [L,H] -> [L,S]; einsum thd,sd->ths, head sum axis 1.
    "lhd_einsum",
    # q [B,L,H,D], keys [B,S,D], weights [B,L,H] -> [B,L,S]; q @ k^T per (b,l), head sum axis 2.
    "blhd_matmul",
    # q [B,L,H,D], keys [B,S,D] -> [B,L,S]; einsum blhd,bnd->blnh, head sum last axis.
    "blhd_einsum",
    # q [H,L,D], keys [S,D] -> [L,S]; relu(q[h] @ k^T) accumulated h = 0, 1, ... in order.
    "hld_head_loop",
    # q [H,L,D], keys [S,D] -> [L,S]; relu(q @ k^T).sum(axis=0) (differentiable).
    "hld_sum",
    # q [L,H,D], keys [L,C,D] per-row candidates -> [L,C]; einsum lhd,lcd->lch.
    "lhd_rows_einsum",
)
SCALE_SITES = (
    "post_relu",  # sum_h relu(qk) * w * scale          (DeepSeek V4)
    "pre_relu",  # sum_h relu(qk * scale) * w           (GLM-5.3)
    "weights",  # sum_h w * relu(qk), scale folded in w (V4.1, K3.1, HY-V4)
    "divide",  # sum_h relu(qk) / sqrt(d)              (Qwen4 QSA)
    "none",  # sum_h relu(qk)                       (QSA sidecar / trainer)
)
TIE_BREAKS = (
    "stable_chronological",  # strictly greater, then ascending column; ids ascending
    "argsort",  # argsort(-s)[..., :k]; descending score, ties unspecified
    "argpartition_low",  # argpartition(-s, k-1)[..., :k]; unordered, ties unspecified
    "argpartition_high",  # argpartition(s, n-k)[..., -k:]; unordered, ties unspecified
)
VISIBILITIES = (
    "pool_complete",  # pool j visible iff j < (pos + 1) // ratio
    "before_own_group",  # group j visible iff j < pos // ratio (own group added separately)
    "token_causal",  # key s visible iff s <= pos
)
SCORE_CASTS = ("float32", "none")


@dataclass(frozen=True)
class SelectionSpec:
    """Every parameter in which the indexer/selection copies differ.

    Shape fields (``n_heads``, ``head_dim``, ``topk``, ``ratio``) come from
    the checkpoint config. The remaining fields are laws; each model keeps
    its historical value (see the doc page for the table).
    """

    n_heads: int | None = None
    head_dim: int | None = None
    topk: int | None = None
    ratio: int = 1
    # Pooling the keys are built with: "none" (token keys), "gated_softmax"
    # (DeepSeek/GLM compressor), "mean" (QSA blocks), "group" (HY groups).
    # The compressors live with their model; this field is descriptive and
    # feeds the identity.
    pooling: str = "none"
    score_layout: str = "bhld_matmul"
    score_weighting: str = "per_head"  # "per_head" | "uniform"
    score_scale_site: str = "post_relu"
    score_cast: str = "float32"  # cast q/k to fp32 before the contraction, or not
    tie_break: str = "stable_chronological"
    visibility: str = "pool_complete"
    # HY-V4 group selection: the query's own group is always appended after
    # ``topk - 1`` scored groups before it.
    include_own_group: bool = False

    def __post_init__(self):
        for name, allowed in (
            ("score_layout", SCORE_LAYOUTS),
            ("score_scale_site", SCALE_SITES),
            ("tie_break", TIE_BREAKS),
            ("visibility", VISIBILITIES),
            ("score_cast", SCORE_CASTS),
            ("score_weighting", ("per_head", "uniform")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"SelectionSpec.{name} must be one of {allowed}")
        if self.score_weighting == "uniform" and self.score_scale_site in ("post_relu", "pre_relu", "weights"):
            raise ValueError("uniform head weighting has no weights to scale")
        if type(self.ratio) is not int or self.ratio < 1:
            raise ValueError("SelectionSpec.ratio must be a positive int")

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "SelectionSpec":
        """Derive shape fields from an HF/checkpoint config (dict or object).

        Recognised keys: ``index_n_heads``/``indexer_heads``/``qsa_heads``,
        ``index_head_dim``/``indexer_head_dim``, ``index_topk``/
        ``qsa_block_topk``/``group_topk``, ``compress_ratio``/
        ``qsa_compress_ratio``/``index_group_size``. Laws are not in any
        config; pass them as overrides (each model file does).
        """
        return spec_from_config(
            cls,
            config,
            {
                "n_heads": ("index_n_heads", "indexer_heads", "qsa_heads", "index_heads"),
                "head_dim": ("index_head_dim", "indexer_head_dim", "qsa_head_dim"),
                "topk": ("index_topk", "qsa_block_topk", "block_topk", "group_topk"),
                "ratio": ("compress_ratio", "qsa_compress_ratio", "index_group_size"),
            },
            **overrides,
        )

    def scale(self, head_dim: int | None = None) -> float:
        dim = self.head_dim if head_dim is None else head_dim
        if dim is None:
            raise ValueError("SelectionSpec.scale needs head_dim")
        return dim**-0.5


# --------------------------------------------------------------------------
# score


def score(spec: SelectionSpec, q, keys, weights=None):
    """Indexer scores in ``spec.score_layout``; returns float32 (or q dtype
    when ``score_cast == "none"`` and inputs are low precision)."""
    layout, site = spec.score_layout, spec.score_scale_site
    if spec.score_cast == "float32":
        q, keys = q.astype(mx.float32), keys.astype(mx.float32)
    per_head = spec.score_weighting == "per_head"
    if per_head and weights is None:
        raise ValueError("per-head scores need head weights")
    dim = q.shape[-1] if spec.head_dim is None else spec.head_dim
    if layout == "bhld_matmul":
        relu = mx.maximum(q @ mx.swapaxes(keys[:, None], -1, -2), 0)
        weight = weights.transpose(0, 2, 1)[..., None]
        if site == "post_relu":
            return mx.sum(relu * weight * dim**-0.5, axis=1)
        if site == "weights":
            return mx.sum(relu * weight, axis=1)
    elif layout == "bhld_einsum":
        relu = mx.maximum(mx.einsum("bhsd,btd->bsht", q, keys), 0)
        if site == "weights":
            return mx.sum(relu * weights[:, :, :, None], 2)
    elif layout == "bhld_rows_einsum":
        relu = mx.maximum(mx.einsum("bhsd,bstd->bsht", q, keys), 0)
        if site == "weights":
            return mx.sum(relu * weights[:, :, :, None], 2)
    elif layout == "lhd_einsum":
        relu = mx.maximum(mx.einsum("thd,sd->ths", q, keys), 0)
        if site == "weights":
            return mx.sum(relu * weights[..., None], axis=1)
    elif layout == "blhd_matmul":
        mm = q @ keys[:, None].swapaxes(-1, -2)
        if site == "pre_relu":
            return (mx.maximum(mm * dim**-0.5, 0) * weights[..., None]).sum(axis=2)
    elif layout == "blhd_einsum":
        if not per_head and site == "divide":
            values = mx.einsum("blhd,bnd->blnh", q, keys)
            return mx.sum(mx.maximum(values, 0), axis=-1) / math.sqrt(dim)
    elif layout == "lhd_rows_einsum":
        if not per_head and site == "divide":
            values = mx.einsum("lhd,lcd->lch", q, keys)
            return mx.sum(mx.maximum(values, 0), axis=-1) / math.sqrt(dim)
    elif layout == "hld_head_loop":
        if not per_head and site == "none":
            acc = None
            for h in range(q.shape[0]):
                s = mx.maximum(q[h] @ keys.T, 0.0)
                acc = s if acc is None else acc + s
            return acc
    elif layout == "hld_sum":
        if not per_head and site == "none":
            return mx.maximum(q @ keys.T, 0.0).sum(axis=0)
    raise ValueError(
        f"score layout {layout!r} has no {spec.score_weighting}/{site} law"
    )


# --------------------------------------------------------------------------
# visibility and top-k


def visible(spec: SelectionSpec, positions, count: int, start: int = 0):
    """Bool ``positions.shape + (count,)``: column ``start + j`` visible to a
    query at absolute position ``pos`` (integer array)."""
    columns = mx.arange(start, start + count)
    pos = positions[..., None]
    if spec.visibility == "pool_complete":
        return columns < (pos + 1) // spec.ratio
    if spec.visibility == "before_own_group":
        return columns < pos // spec.ratio
    return columns <= pos


def stable_keep_mask(scores, top_k):
    """Kept-column mask of the stable chronological law (exactly ``top_k`` per row)."""
    count = scores.shape[-1]
    if type(top_k) is not int or not 1 <= top_k <= count:
        raise ValueError("pool top-k must fit the available rows")
    candidates = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    cutoff = mx.min(
        mx.take_along_axis(scores, candidates, axis=-1), axis=-1, keepdims=True
    )
    better = scores > cutoff
    tied = scores == cutoff
    remaining = top_k - mx.sum(better, axis=-1, keepdims=True)
    return better | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= remaining))


def stable_topk(scores, top_k):
    """Select exact scores with a chronological cutoff tie-break and order.

    ReLU indexer scores have exact zero ties. Partition's unspecified ordering
    must not change the attended set or summation order across replay shapes.
    Returns ascending column ids (uint32, argpartition's index dtype).
    See provenance/deepseek-v4-audit-64k.json for the peer mechanism review.
    """
    count = scores.shape[-1]
    keep = stable_keep_mask(scores, top_k)
    positions = mx.arange(count, dtype=mx.int32)
    ordered = mx.where(keep, positions, count)
    selected = mx.argpartition(ordered, kth=top_k - 1, axis=-1)[..., :top_k]
    return mx.sort(selected, axis=-1)


def topk(spec: SelectionSpec, scores, count: int, *, tie_break: str | None = None):
    """Top ``count`` column ids of ``scores`` under the spec's tie-break law."""
    law = spec.tie_break if tie_break is None else tie_break
    if law == "stable_chronological":
        return stable_topk(scores, count)
    if law == "argsort":
        return mx.argsort(-scores, axis=-1)[..., :count]
    if law == "argpartition_low":
        return mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
    if law == "argpartition_high":
        n = scores.shape[-1]
        return mx.argpartition(scores, kth=n - count, axis=-1)[..., -count:]
    raise ValueError(f"unknown tie-break law {law!r}")


def select(spec: SelectionSpec, scores, valid, count: int, *, fill=-mx.inf):
    """``(ids, selected_valid)``: mask invisible columns, take the top ``count``.

    ``selected_valid`` is ``valid`` gathered at the ids (an id past a row's
    visible columns stays in range for gathers and is masked here).
    """
    masked = mx.where(valid, scores, fill)
    ids = topk(spec, masked, count)
    return ids, mx.take_along_axis(mx.broadcast_to(valid, masked.shape), ids, axis=-1)


def select_groups(spec: SelectionSpec, scores, own, *, fill=-1):
    """HY-V4 group law: ``topk - 1`` scored groups before ``own``, then ``own``.

    ``scores`` [..., L, G] already masked to ``-inf`` outside visibility;
    ``own`` [L] int32. Picks at or after the own group become ``fill``.
    """
    groups = scores.shape[-1]
    k = min(spec.topk - 1, groups)
    own_col = mx.broadcast_to(own[None, :, None], (*scores.shape[:-1], 1))
    if k <= 0:
        return own_col
    picked = topk(spec, scores, k).astype(mx.int32)
    valid = mx.arange(groups)[None, None, :] < own[None, :, None]
    picked = mx.where(mx.take_along_axis(valid, picked, axis=-1), picked, fill)
    return mx.concatenate([picked, own_col], axis=-1)


# --------------------------------------------------------------------------
# gathered attention


@dataclass(frozen=True)
class Source:
    """One key/value source of ``gather_attend``.

    ``keys`` is [B, H|1, S, D] (shared by every query row) or, with
    ``per_row=True``, [B, L, K, D] gathered rows (one set per query row).
    ``values=None`` means values are the keys (K=V latent caches); with
    ``value_width`` they are the leading ``value_width`` key dims.
    ``mask`` is bool, broadcastable to the source's score shape
    [B, H, L, S|K], or None (all visible).
    ``cast`` is ``"float32"`` (cast keys/values to fp32 before the products;
    DeepSeek V4) or ``"none"`` (pass them in their own dtype, so a bf16 row
    set reaches a mixed-dtype einsum exactly as the DeepSeek V4.1 selected
    rows and the Source AST ``attention_core`` always did; on Metal the
    pre-cast form could select a different GEMM).
    """

    keys: Any
    mask: Any = None
    values: Any = None
    per_row: bool = False
    value_width: int | None = None
    cast: str = "float32"

    def __post_init__(self):
        if self.cast not in ("float32", "none"):
            raise ValueError("Source.cast must be 'float32' or 'none'")

    def value_rows(self):
        values = self.keys if self.values is None else self.values
        if self.value_width is not None:
            values = values[..., : self.value_width]
        return values


GATHER_CONTRACTIONS = ("matmul", "einsum")


def gather_attend(
    q,
    sources: Sequence[Source],
    *,
    sinks=None,
    scale: float | None = None,
    contraction: str = "matmul",
    out_dtype=None,
):
    """fp32 softmax attention over ``sources`` merged in log space.

    ``q`` [B, H, L, D]. One source: ``logsumexp`` of its scores. Several:
    ``logaddexp`` of each source's ``logsumexp``, in source order. ``sinks``
    [H] adds denominator mass only. ``contraction`` spells the per-row
    gathered products: ``matmul`` (query rows-first, ``@``; DeepSeek V4) or
    ``einsum`` (``bhld,blkd->bhlk``; DeepSeek V4.1); shared sources always
    use ``@``. Returns [B, H, L, Dv] in ``out_dtype`` (default ``q.dtype``).
    """
    if contraction not in GATHER_CONTRACTIONS:
        raise ValueError(f"gather contraction must be one of {GATHER_CONTRACTIONS}")
    if not sources:
        raise ValueError("gather_attend needs at least one source")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    query = q.astype(mx.float32) * scale
    scores = []
    for source in sources:
        keys = source.keys.astype(mx.float32) if source.cast == "float32" else source.keys
        if not source.per_row:
            s = query @ mx.swapaxes(keys, -1, -2)
        elif contraction == "einsum":
            s = mx.einsum("bhld,blkd->bhlk", query, keys)
        else:
            s = (query.transpose(0, 2, 1, 3) @ mx.swapaxes(keys, -1, -2)).transpose(
                0, 2, 1, 3
            )
        if source.mask is not None:
            s = mx.where(source.mask, s, -mx.inf)
        scores.append(s)
    if len(scores) == 1:
        norm = mx.logsumexp(scores[0], axis=-1, keepdims=True)
    else:
        norm = mx.logsumexp(scores[0], -1, keepdims=True)
        for s in scores[1:]:
            norm = mx.logaddexp(norm, mx.logsumexp(s, -1, keepdims=True))
    if sinks is not None:
        norm = mx.logaddexp(norm, sinks[None, :, None, None].astype(mx.float32))
    result = None
    for source, s in zip(sources, scores):
        values = source.value_rows()
        if source.cast == "float32":
            values = values.astype(mx.float32)
        if not source.per_row:
            part = mx.exp(s - norm) @ values
        elif contraction == "einsum":
            part = mx.einsum("bhlk,blkd->bhld", mx.exp(s - norm), values)
        else:
            part = (mx.exp(s - norm).transpose(0, 2, 1, 3) @ values).transpose(
                0, 2, 1, 3
            )
        if result is None:
            result = part
        else:
            result += part
    return result.astype(q.dtype if out_dtype is None else out_dtype)


# --------------------------------------------------------------------------
# identity


_LAW_FUNCTIONS = {
    "score": (score,),
    "visible": (visible,),
    "stable_chronological": (stable_keep_mask, stable_topk, topk, select),
    "argsort": (topk, select),
    "argpartition_low": (topk, select),
    "argpartition_high": (topk, select),
    "select_groups": (select_groups,),
    "gather_attend": (gather_attend, Source),
}


def implementation_identity(spec: SelectionSpec | None = None, *, laws: Sequence[str] = ()) -> str:
    """sha256 of the spec and the source of the functions its laws run.

    This over-binds by design: ``score`` and ``topk`` hold every layout and
    tie-break law, and ``repr(spec)`` includes every field, so an edit to
    another model's branch of ``score``/``topk`` (or a new spec field) also
    moves the identity. It never under-binds: an edit to any law the spec
    runs moves it. Model numerics identities that hash files bind this whole
    module by path (``src/mlx2/runtime/technologies/sparse_selection.py``)
    instead; this function is for identities that are not file lists.
    """
    names = list(laws)
    if spec is not None:
        names += ["score", "visible", spec.tie_break]
        if spec.include_own_group:
            names.append("select_groups")
    seen, parts = set(), [repr(spec)]
    for name in names:
        for fn in _LAW_FUNCTIONS[name]:
            if fn not in seen:
                seen.add(fn)
                parts.append(inspect.getsource(fn))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# --------------------------------------------------------------------------
# fast-path registries
#
# Each fused kernel keeps its call site and admission in its model module
# (they are opt-in or default-on under their own policy keys, with receipts
# there). They are registered here so the technology has one inventory and a
# caller can ask ``registry.select(spec, **shape)`` for the receipt. ``run``
# resolves lazily so importing this module never builds a Metal kernel.


def _lazy(load, name: str):
    """``load`` is a zero-argument literal import, so the source-dependency
    analysis can resolve the kernel module this path runs."""

    def run(*args, **kwargs):
        return getattr(load(), name)(*args, **kwargs)

    run.__qualname__ = name
    run.load = load
    return run


def _qsa_fused_scores_admits(spec, *, q=None, pooled=None, offset=None, **_):
    if spec.score_layout != "blhd_einsum" or spec.score_scale_site != "divide":
        return "spec: not the QSA uniform/divide law"
    if q is None or pooled is None:
        return "shape: needs q and pooled"
    from ..models import qwen4_qsa_scores as m

    if not m.enabled():
        return "policy_off: MLX_QWEN4_QSA_FUSED_SCORES disabled"
    reason = m.supported(q, pooled) or (
        None if offset is None else m.offset_supported(offset, int(q.shape[0]))
    )
    return None if reason is None else "geometry: " + reason


def _glm_score_kernel_admits(spec, *, mm=None, weights=None, **_):
    if spec.score_layout != "blhd_matmul" or spec.score_scale_site != "pre_relu":
        return "spec: not the GLM pre-relu law"
    if mm is None or weights is None:
        return "shape: needs the q x pooled product and weights"
    from ..models import glm53_gguf_indexer as m

    if not m.score_kernel_admits(mm, weights):
        return "geometry: shape not admitted"
    return None if m.score_kernel_ready() else "gate: first-use bitwise gate not passed"


def _glm_select_kernel_admits(spec, **_):
    if spec.tie_break != "stable_chronological":
        return "spec: kernel implements the stable chronological law only"
    from ..models import glm53_gguf_indexer as m

    return None if m.select_kernel_ready() else "gate: first-use bitwise gate not passed"


def _call_site(spec, **_):
    return None


def _requested(name: str, admits):
    """Admit ``name`` only when the caller's own policy chose it.

    Every fast path here is opt-in or default-on under a model policy key
    that lives at the call site, so the registry never picks one implicitly:
    without ``requested=<name>`` it refuses, and with it the path's own
    admission (shape, gate, enablement) decides.
    """

    def gated(spec, *, requested=None, **shape):
        if requested != name:
            return "policy_off: not requested by the call-site policy"
        return admits(spec, **shape)

    return gated


def _register(registry, name, admits, load, fn):
    registry.register(FastPath(name, _requested(name, admits), _lazy(load, fn)))


SCORE_PATHS = KernelRegistry("sparse_selection.score", score)
SELECT_PATHS = KernelRegistry("sparse_selection.select", topk)
GATHER_ATTEND_PATHS = KernelRegistry("sparse_selection.gather_attend", gather_attend)
for _registry, _name, _admits, _load, _fn in (
    (SCORE_PATHS, "qwen4_qsa_fused_scores", _qsa_fused_scores_admits, lambda: importlib.import_module("mlx2.runtime.models.qwen4_qsa_scores"), "block_scores"),
    (SCORE_PATHS, "glm53_pool_scores_kernel_v1", _glm_score_kernel_admits, lambda: importlib.import_module("mlx2.runtime.models.glm53_gguf_indexer"), "score_rows"),
    (SCORE_PATHS, "hy_v4_indexer_fused_scores", _call_site, lambda: importlib.import_module("mlx2.runtime.models.hy_v4_flash_fusions"), "indexer_logits_metal"),
    (SCORE_PATHS, "hy_v4_indexer_fused_scores_simd", _call_site, lambda: importlib.import_module("mlx2.runtime.models.hy_v4_flash_fusions"), "indexer_logits_simd"),
    (SELECT_PATHS, "glm53_select_expand_v1", _glm_select_kernel_admits, lambda: importlib.import_module("mlx2.runtime.models.glm53_gguf_indexer"), "select_expand"),
    (SELECT_PATHS, "qwen4_qsa_stage1_fused", _call_site, lambda: importlib.import_module("mlx2.runtime.models.qwen4_qsa_stage1"), "qsa_stage1_select"),
    (SELECT_PATHS, "qwen4_qsa_onepass_select", _call_site, lambda: importlib.import_module("mlx2.runtime.models.qwen4_qsa_stage1"), "_select_scores_onepass"),
    (SELECT_PATHS, "qsa_sidecar_fused_select", _call_site, lambda: importlib.import_module("mlx2.runtime.models.qsa_select"), "fused_select"),
    (GATHER_ATTEND_PATHS, "deepseek_v4_visible_rows_fp32", _call_site,
     lambda: importlib.import_module("mlx2.runtime.models.deepseek_v4_visible_attention"), "visible_attention"),
    (GATHER_ATTEND_PATHS, "deepseek_v4_packed_visible_rows_fp32", _call_site,
     lambda: importlib.import_module("mlx2.runtime.models.deepseek_v4_packed_visible_attention"), "packed_visible_attention"),
):
    _register(_registry, _name, _admits, _load, _fn)


__all__ = [
    "GATHER_ATTEND_PATHS",
    "SCORE_PATHS",
    "SELECT_PATHS",
    "SelectionSpec",
    "Source",
    "gather_attend",
    "implementation_identity",
    "score",
    "select",
    "select_groups",
    "stable_keep_mask",
    "stable_topk",
    "topk",
    "visible",
]
