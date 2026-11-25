# SPDX-License-Identifier: Apache-2.0
"""MLX implementation of Maxwell-V1-Flash (``maxwell_moe``).

The model-specific numerical law lives here (see ``provenance/maxwell-moe.json``;
the source of truth is the checkpoint's own ``modeling_maxwell_moe.py``):

* mHC residual streams: the hidden state is ``n`` (4) copies of the model width.
  Each sub-block computes ``h_pre`` (sigmoid + eps), ``h_post`` (2 sigmoid) and
  a doubly-stochastic ``h_res`` (softmax, then Sinkhorn column/row passes) from
  an RMS-scaled fp32 projection of all streams, collapses the streams with
  ``h_pre``, runs the sub-block and writes ``h_post * out + h_res^T @ streams``.
  A learned ``hc_head`` contracts the streams before the final RMSNorm.
* Attention: GQA with per-head QK RMSNorm and RoPE (base 2e7).  Every trunk
  layer uses Block Sparse GQA: for each query and KV group the route is the
  fixed prefix blocks (the first 1,024 tokens), the local blocks (the current
  block and the 7 before it) and the 16 remote blocks of 128 tokens that score
  highest under a per-group indexer (block score = max token score of a
  4-head x 128 indexer query against one shared, post-RoPE indexer key).
  Contexts of at most ``(prefix + local + remote) * block`` = 4,096 tokens are
  covered completely, so there the route is dense causal attention and runs
  through the fused SDPA kernel.  Beyond that the route is evaluated exactly:
  decode and verify rows gather the 32 routed blocks per group; long prefill
  chunks attend the union of their routed blocks (or, opt-in, a
  block-membership mask).  Indexer keys are cached for every token, in fp32
  (the reference's ``.float()`` of the bf16 key), as a parallel KV plane per
  layer, so a context can cross 4,096 tokens at any point.
* MoE (layers not in ``mlp_only_layers``): fp32 sigmoid scores, top-k chosen on
  ``scores + expert_bias`` (fp32), weights are the chosen *unbiased* scores
  renormalised to sum to one, plus an ungated shared expert.
* Logits: an fp32 head over the post-norm hidden state.
* Native MTP (one layer): ``enorm(embed(next token))`` and ``hnorm`` of each
  trunk residual stream are projected and summed into a 4-stream input for one
  dense-attention mHC MoE layer; its own ``hc_head`` and final norm feed the
  shared fp32 head.  At draft depth 1 this is the reference; deeper drafts
  feed the layer its own multi-stream output (an extrapolation: the
  checkpoint trains one MTP layer and never chains it), so only depth 1 is
  the trained law and verification keeps every depth exact.

Bit-preserving implementation choices (same arithmetic, fewer kernels): the
fp32 weights the reference casts on every call (router, mHC mappings, lm_head)
are stored fp32 once at load, the mHC and RMSNorm spans run as compiled
graphs, and the head only sees the checkpoint rows the engine will read.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import BatchKVCache, KVCache
from .switch_layers import SwitchGLU

HC_EPS = 1e-6
CACHE_LAYOUT = "maxwell-mhc-bsgqa-kv-indexer-planes-v1"

# Rows at or below this query count gather the routed blocks; longer chunks
# use the block-membership mask over the whole key range.
SPARSE_GATHER_ROWS = int(os.environ.get("MLX2_MAXWELL_SPARSE_GATHER_ROWS", "16"))
# Query rows scored against the indexer keys per chunk (bounds the fp32
# token-score temporary at long context; the reference uses 64).
INDEX_QUERY_CHUNK = int(os.environ.get("MLX2_MAXWELL_INDEX_QUERY_CHUNK", "64"))
# Long-chunk sparse route: "union" gathers, per 64-query sub-chunk, the union
# of the blocks any of its queries routes to (exact; attention cost follows
# the route, not the context); "masked" attends the whole key range through a
# block-membership mask (exact; dense cost).
SPARSE_PREFILL = os.environ.get("MLX2_MAXWELL_SPARSE_PREFILL", "union")
UNION_QUERY_CHUNK = int(os.environ.get("MLX2_MAXWELL_UNION_QUERY_CHUNK", "64"))
if SPARSE_PREFILL not in ("union", "masked"):
    raise ValueError("MLX2_MAXWELL_SPARSE_PREFILL must be union or masked")
# Compiled mHC / norm spans (bit-identical to the eager graph).
COMPILE_SPANS = os.environ.get("MLX2_MAXWELL_COMPILE", "1") != "0"
# Opt-in execution levers (unqualified; summation order differs from the
# compiled graph, arithmetic class unchanged -- see maxwell_moe_metal):
#   MLX2_MAXWELL_MHC_KERNEL=1  fused Metal mHC pre/post kernels
#   MLX2_MAXWELL_HEAD=fp32     fp32-resident head (the default is bf16-resident,
#                              read by an fp32-math GEMV: same products, half the
#                              bytes; token-exact 7/7 cells on the bf16 checkpoint,
#                              qualification/runs/maxwell-full-t7-20261122-r2/head_bf16;
#                              default since 2026-11-22, Pierre)
#   MLX2_MAXWELL_INDEX_KERNEL=1 bf16 indexer-key plane scored by a fused block-max
#                              kernel for decode/verify rows (no fp32 plane)
#   MLX2_MAXWELL_FUSED_QKV=1   one projection for q, k, v and the indexer key
#   MLX2_MAXWELL_CANONICAL_PICK=1 remote-block pick with ties broken toward the
#                              lowest block id and blocks gathered in ascending
#                              order (default: argpartition, whose tie choice and
#                              order depend on kernel and shape).  Same set as
#                              torch.topk when there is no tie.  Off until a GPU
#                              token/speed check (Pierre 2026-11-23: merge default off).
#   MLX2_MAXWELL_ROW_EXACT_VERIFY=layerwise|serial  native-MTP verify rows with
#                              the one-token decode arithmetic (greedy MTP-on output
#                              byte-identical to MTP-off, B=1 lanes only).  layerwise:
#                              projections and experts as canary-gated one-weight-read
#                              strips, everything else per row through the decode code;
#                              serial: one full decode forward per row (reference, slow).
#                              Off by default until a GPU A/B (Pierre 2026-11-23).
MHC_KERNEL = os.environ.get("MLX2_MAXWELL_MHC_KERNEL", "0") == "1"
# The fused mHC kernel runs one threadgroup per token with a serial Sinkhorn,
# which loses to the compiled span on long prefill chunks (32K TTFT 108 s ->
# 240 s, qualification/runs/maxwell-full-t7-20261122-r2/mhc_kernel); it is used
# for at most this many rows (decode and verify).
MHC_KERNEL_MAX_ROWS = int(os.environ.get("MLX2_MAXWELL_MHC_KERNEL_MAX_ROWS", "16"))
HEAD_MODE = os.environ.get("MLX2_MAXWELL_HEAD", "bf16")
INDEX_KERNEL = os.environ.get("MLX2_MAXWELL_INDEX_KERNEL", "0") == "1"
FUSED_QKV = os.environ.get("MLX2_MAXWELL_FUSED_QKV", "0") == "1"
# Prefill expert gathers through the NAX sorted gather (moe_nax_gather):
# "off" (default) or "gather".  The adapter claims the process NAX mode.
MOE_NAX = os.environ.get("MLX2_MAXWELL_MOE_NAX", "off")
if MOE_NAX not in ("off", "gather"):
    raise ValueError("MLX2_MAXWELL_MOE_NAX must be off or gather")


def apply_policy_levers(*, index_kernel=None, sparse_prefill=None) -> dict:
    """Set the index-kernel and long-chunk-prefill levers from an execution policy.

    A route default (the single-lane best-route entry) carries these as policy
    keys.  An operator's environment variable always wins: a lever whose
    variable is set (non-empty) is left alone and reported as ``"environment"``.  An
    applied lever is also written to the process environment, so the APCv2
    execution-numerics identity (read from it after the adapter loads) binds
    the value the model actually runs.
    """
    global INDEX_KERNEL, SPARSE_PREFILL
    applied = {}
    if index_kernel is not None:
        if type(index_kernel) is not bool:
            raise ValueError("Maxwell index_kernel policy must be true or false")
        if os.environ.get("MLX2_MAXWELL_INDEX_KERNEL"):
            applied["index_kernel"] = "environment"
        else:
            INDEX_KERNEL = index_kernel
            os.environ["MLX2_MAXWELL_INDEX_KERNEL"] = "1" if index_kernel else "0"
            applied["index_kernel"] = "policy"
    if sparse_prefill is not None:
        if sparse_prefill not in ("union", "masked"):
            raise ValueError("Maxwell sparse_prefill policy must be union or masked")
        if os.environ.get("MLX2_MAXWELL_SPARSE_PREFILL"):
            applied["sparse_prefill"] = "environment"
        else:
            SPARSE_PREFILL = sparse_prefill
            os.environ["MLX2_MAXWELL_SPARSE_PREFILL"] = sparse_prefill
            applied["sparse_prefill"] = "policy"
    return applied


# Up to this many rows a multi-row call of the fused table runs each row with
# its one-row law (canary-gated matvec strips; see Attention._linears).
FUSED_QKV_EXACT_ROWS = 16
CANONICAL_PICK = os.environ.get("MLX2_MAXWELL_CANONICAL_PICK", "0") == "1"
ROW_EXACT_VERIFY = os.environ.get("MLX2_MAXWELL_ROW_EXACT_VERIFY", "off")
if ROW_EXACT_VERIFY not in ("off", "layerwise", "serial"):
    raise ValueError("MLX2_MAXWELL_ROW_EXACT_VERIFY must be off, layerwise or serial")
if HEAD_MODE not in ("fp32", "bf16"):
    raise ValueError("MLX2_MAXWELL_HEAD must be fp32 or bf16")

STATS = {"dense_calls": 0, "sparse_gather_calls": 0, "sparse_masked_calls": 0,
         "sparse_union_calls": 0, "union_blocks": 0, "union_subchunks": 0,
         "sparse_rows": 0, "mtp_steps": 0, "kernel_calls": 0, "kernel_fallbacks": 0,
         "last_kernel_fallback": None, "head_kernel_calls": 0, "index_kernel_calls": 0,
         # MLX2_MAXWELL_FUSED_QKV: calls of the fused q/k/v/indexer-key table, and
         # of those the 2..FUSED_QKV_EXACT_ROWS-row calls by row-exact path
         # (matvec strip under its canary, or one call per row).
         "fused_qkv_calls": 0, "fused_qkv_row_exact_strip": 0, "fused_qkv_row_exact_per_row": 0}
_KERNEL_OK: set = set()
_KERNEL_DISABLED: set = set()


def _on_gpu() -> bool:
    return mx.default_device() == mx.gpu


def _kernel_call(key, fn, *args, **kwargs):
    """Run an opt-in Metal kernel, or return None to fall back.

    A specialization's first use is evaluated inside the guard (kernel output
    is lazy, so a build/dispatch failure would otherwise surface later); a
    failure disables that kernel family for the process."""
    if key[0] in _KERNEL_DISABLED:
        return None
    try:
        out = fn(*args, **kwargs)
        if key not in _KERNEL_OK:
            mx.eval(out)
            _KERNEL_OK.add(key)
    except Exception as exc:  # fail closed to the compiled graph
        _KERNEL_DISABLED.add(key[0])
        STATS["kernel_fallbacks"] += 1
        STATS["last_kernel_fallback"] = f"{key[0]}: {type(exc).__name__}: {exc}"
        return None
    STATS["kernel_calls"] += 1
    return out


def maxwell_stats(reset: bool = False) -> dict:
    out = dict(STATS, levers={"compile": COMPILE_SPANS, "mhc_kernel": MHC_KERNEL, "head": HEAD_MODE,
                              "fused_qkv": FUSED_QKV, "index_kernel": INDEX_KERNEL,
                              "canonical_pick": CANONICAL_PICK, "row_exact_verify": ROW_EXACT_VERIFY,
                              "gather_rows": SPARSE_GATHER_ROWS, "moe_nax": MOE_NAX,
                              "sparse_prefill": SPARSE_PREFILL})
    from . import gemv_rows_exact

    try:
        out["levers"]["gemv_rows_kernel"] = gemv_rows_exact.kernel_mode()
    except ValueError:  # an invalid setting fails at its first use, not in a stats read
        out["levers"]["gemv_rows_kernel"] = "invalid"
    out["gemv_rows_kernel_calls"] = gemv_rows_exact.STATS["kernel_calls"]
    out["gemv_rows_stock_fallbacks"] = gemv_rows_exact.STATS["stock_fallbacks"]
    if ROW_EXACT_VERIFY != "off":
        from . import maxwell_row_exact

        out["row_exact_verify"] = maxwell_row_exact.stats(reset=reset)
    if reset:
        for k in STATS:
            STATS[k] = None if k == "last_kernel_fallback" else 0
    return out


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_size: int
    rms_norm_eps: float
    rope_theta: float
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    mlp_only_layers: list = field(default_factory=list)
    max_position_embeddings: int = 524288
    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    block_sparse_gqa_block_size: int = 128
    block_sparse_gqa_prefix_window_size: int = 1024
    block_sparse_gqa_local_window_size: int = 1024
    block_sparse_gqa_num_remote_blocks: int = 16
    block_sparse_gqa_indexer_n_heads: int = 4
    block_sparse_gqa_indexer_head_dim: int = 128
    tie_word_embeddings: bool = False
    qk_layernorm: bool = True
    enable_hyper_connections: bool = True
    moe_router_enable_expert_bias: bool = True
    router_score_function: str = "sigmoid"
    use_shared_expert_gate: bool = False
    hidden_act: str = "silu"
    rope_scaling: dict | None = None
    mtp_num_layers: int = 0
    pad_token_id: int = 3

    def __post_init__(self):
        if self.model_type != "maxwell_moe":
            raise ValueError("Maxwell model_type must be 'maxwell_moe'")
        if not (self.qk_layernorm and self.enable_hyper_connections
                and self.moe_router_enable_expert_bias):
            raise ValueError("Maxwell requires QK norm, mHC and expert bias")
        if self.router_score_function != "sigmoid" or self.use_shared_expert_gate:
            raise ValueError("Maxwell requires sigmoid routing and an ungated shared expert")
        if self.hidden_act != "silu" or self.rope_scaling is not None:
            raise ValueError("Maxwell requires silu and unscaled RoPE")
        if self.tie_word_embeddings:
            raise ValueError("Maxwell has an untied lm_head")
        if self.mtp_num_layers not in (0, 1):
            raise ValueError("Maxwell supports zero or one MTP layer")
        b = self.block_sparse_gqa_block_size
        if (self.block_sparse_gqa_prefix_window_size % b
                or self.block_sparse_gqa_local_window_size % b):
            raise ValueError("Maxwell prefix/local windows must be whole blocks")
        if self.block_sparse_gqa_indexer_n_heads != self.num_key_value_heads:
            raise ValueError("Maxwell indexer groups must equal the KV heads")
        if self.block_sparse_gqa_indexer_head_dim != self.attention_head_size:
            raise ValueError("Maxwell indexer reuses the main RoPE; head dims must agree")

    @property
    def dense_context_limit(self) -> int:
        """Largest context where the Block Sparse GQA route is dense causal."""
        b = self.block_sparse_gqa_block_size
        return (self.block_sparse_gqa_prefix_window_size // b
                + self.block_sparse_gqa_local_window_size // b
                + self.block_sparse_gqa_num_remote_blocks) * b


# --------------------------------------------------------------------------
# compiled spans
# --------------------------------------------------------------------------

_COMPILED: dict = {}


def _compiled(key, builder):
    if not COMPILE_SPANS:
        return builder()
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = mx.compile(builder())
    return fn


def _rms_norm_body(x, weight, eps):
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (weight.astype(mx.float32) * xf).astype(x.dtype)


def rms_norm(x, weight, eps):
    """HF Maxwell RMSNorm: fp32 normalise, fp32 weight multiply, one rounding."""
    return _compiled(("rms", float(eps)), lambda: lambda a, w: _rms_norm_body(a, w, eps))(x, weight)


def _hc_pre_body(x, w, alpha_pre, alpha_post, alpha_res, bias, *, n, iters):
    """mHC mappings + stream collapse.  Returns (aggregated, h_post, h_res)."""
    b, t, nc = x.shape
    c = nc // n
    xf = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    r = 1.0 / (norm / math.sqrt(nc) + HC_EPS)
    proj = xf @ w.astype(mx.float32).T
    alpha = mx.concatenate([
        mx.broadcast_to(alpha_pre, (n,)), mx.broadcast_to(alpha_post, (n,)),
        mx.broadcast_to(alpha_res, (n * n,)),
    ]).astype(mx.float32)
    h = r * proj * alpha + bias.astype(mx.float32)
    h_pre = (mx.sigmoid(h[..., :n]) + HC_EPS).astype(x.dtype)
    h_post = (mx.sigmoid(h[..., n:2 * n]) * 2).astype(x.dtype)
    m = mx.softmax(h[..., 2 * n:].reshape(b, t, n, n), axis=-1)
    m = m + HC_EPS
    m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
    for _ in range(iters - 1):
        m = m / (mx.sum(m, axis=-1, keepdims=True) + HC_EPS)
        m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
    s = x.reshape(b, t, n, c).astype(mx.float32)
    agg = (s * h_pre.astype(mx.float32)[..., None]).sum(axis=2).astype(x.dtype)
    return agg, h_post, m.astype(x.dtype)


def _hc_post_body(out, residual, h_post, h_res, *, n):
    b, t, c = out.shape
    res = residual.reshape(b, t, n, c).astype(mx.float32)
    mixed = mx.swapaxes(h_res.astype(mx.float32), -1, -2) @ res
    expanded = h_post.astype(mx.float32)[..., None] * out.astype(mx.float32)[:, :, None, :]
    return (expanded + mixed).astype(residual.dtype).reshape(b, t, n * c)


def _hc_head_body(x, fn, base, scale, *, n):
    b, t, nc = x.shape
    c = nc // n
    xf = x.astype(mx.float32)
    rs = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + HC_EPS)
    mixes = (xf @ fn.astype(mx.float32).T) * rs
    w = mx.sigmoid(mixes * scale.astype(mx.float32) + base.astype(mx.float32)) + HC_EPS
    s = x.reshape(b, t, n, c).astype(mx.float32)
    return (s * w[..., None]).sum(axis=2).astype(x.dtype)


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    """SwiGLU; checkpoint w1/w3/w2 are renamed gate/up/down at load."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Indexer(nn.Module):
    """Block Sparse GQA index branch: per-group queries, one shared key."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.groups = g = args.block_sparse_gqa_indexer_n_heads
        self.head_dim = d = args.block_sparse_gqa_indexer_head_dim
        self.scale = d ** -0.5
        self.linear_q_proj = nn.Linear(args.hidden_size, g * d, bias=False)
        self.linear_k_proj = nn.Linear(args.hidden_size, d, bias=False)
        self.q_layernorm = RMSNorm(d, args.rms_norm_eps)
        self.k_layernorm = RMSNorm(d, args.rms_norm_eps)

    def queries(self, x):
        return self.queries_from(self.linear_q_proj(x))

    def queries_from(self, raw):
        """Indexer queries from the ``linear_q_proj`` output ``[b, t, G * d]``."""
        b, t, _ = raw.shape
        q = raw.reshape(b, t, self.groups, self.head_dim)
        return self.q_layernorm(q).transpose(0, 2, 1, 3)


def _row_padding(cache, width, dense_limit):
    """Exact per-row (left, right) padding when this forward can leave the dense regime.

    ``None`` for an unpadded plain ``KVCache`` or a forward that stays dense.
    One host read per forward (the trunk's first plane; every plane shares the
    layout).  Right padding is the current chunk's pending tail (``prepare``
    sets it, ``finalize`` rolls it into left padding)."""
    if type(cache) is not BatchKVCache:
        return None              # unpadded, or refused by the attention when sparse
    if cache._idx + width <= dense_limit:
        return None
    left = [int(v) for v in cache.left_padding.tolist()]
    right = [0] * len(left)
    pending = getattr(cache, "_right_padding", None)
    if pending is not None:
        host = getattr(cache, "_host_right_padding", None)
        right = list(host[1]) if host is not None and host[0] is pending else [int(v) for v in pending.tolist()]
    return list(zip(left, right))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, *, sparse: bool = True):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.attention_head_size
        self.scale = self.head_dim ** -0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
        if sparse:
            self.indexer = Indexer(args)
        self.sparse = sparse
        self.block = args.block_sparse_gqa_block_size
        self.n_prefix = args.block_sparse_gqa_prefix_window_size // self.block
        self.n_local = args.block_sparse_gqa_local_window_size // self.block
        self.n_remote = args.block_sparse_gqa_num_remote_blocks
        self.dense_limit = args.dense_context_limit

    def fuse_projections(self):
        """Replace q/k/v (+ indexer key) with one projection (same per-column dot products)."""
        if "qkv_proj" in self:
            return
        projections = [self.q_proj, self.k_proj, self.v_proj]
        if self.sparse:
            projections.append(self.indexer.linear_k_proj)
        if any(type(proj) is not nn.Linear for proj in projections):
            return  # A packed projection has its own scales and cannot join the dense table.
        parts = [proj.weight for proj in projections]
        weight = mx.concatenate(parts, axis=0)
        self.qkv_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.qkv_proj.weight = weight
        self._split = [p.shape[0] for p in parts]
        for name in ("q_proj", "k_proj", "v_proj"):
            delattr(self, name)
        if self.sparse:
            delattr(self.indexer, "linear_k_proj")

    def _linears(self, x):
        """The weight-bearing q/k/v (+ indexer key) projections, unreshaped."""
        if "qkv_proj" in self:
            rows = x.size // x.shape[-1]
            if x.ndim == 3 and x.shape[1] > 1 and rows <= FUSED_QKV_EXACT_ROWS:
                # Verify-width calls (an MTP verify, a short prefill tail) get
                # the one-row decode arithmetic of the fused table.  Its plain
                # multi-row product does not keep it: at 16K, MTP verify through
                # the fused table flipped a greedy token against MTP off, and
                # with separate q/k/v (or MTP off) it did not
                # (qualification/runs/maxwell-attr-16k-night-20261123).
                from .. import row_exact_verify as REV
                from ..row_exact import ops as row_exact_ops

                strip = x.reshape(1, rows, x.shape[-1])
                outer, record = REV.current(), REV.Window(rows)
                with REV.window(record):
                    y = row_exact_ops.dense_linear(self.qkv_proj, strip, stage="maxwell_fused_qkv")
                routes = record.stages.get("maxwell_fused_qkv", {})
                if outer is not None:
                    for route, count in routes.items():
                        outer.note("maxwell_fused_qkv", route, count)
                STATS["fused_qkv_row_exact_strip" if "matvec_rows" in routes
                      else "fused_qkv_row_exact_per_row"] += 1
                y = y.reshape(*x.shape[:-1], y.shape[-1])
            else:
                y = self.qkv_proj(x)
            STATS["fused_qkv_calls"] += 1
            cuts, acc = [], 0
            for size in self._split[:-1]:
                acc += size
                cuts.append(acc)
            parts = mx.split(y, cuts, axis=-1)
            q, k, v = parts[:3]
            ik = parts[3] if self.sparse else None
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            ik = self.indexer.linear_k_proj(x) if self.sparse else None
        return q, k, v, ik

    def __call__(self, x, mask=None, cache=None, index_cache=None, padding=None):
        q, k, v, ik = self._linears(x)
        y = self._attend(q, k, v, ik, lambda: self.indexer.linear_q_proj(x), mask, cache, index_cache, padding)
        return self.o_proj(y)

    def _attend(self, q, k, v, ik, index_q, mask, cache, index_cache, padding):
        """Everything between the input projections and ``o_proj``.

        ``q``/``k``/``v``/``ik`` are the raw projection outputs ``[b, t, *]``;
        ``index_q()`` returns the indexer query projection (called only on the
        sparse route).  Returns the attention output ``[b, t, heads * dim]``.
        The row-exact verify path (``maxwell_row_exact``) runs this once per
        row with batched projections, so it must stay the decode code."""
        b, t, _ = q.shape
        if ik is not None:
            ik = self.indexer.k_layernorm(ik.reshape(b, t, 1, self.indexer.head_dim)).transpose(0, 2, 1, 3)
        q = self.q_layernorm(q.reshape(b, t, self.n_heads, self.head_dim))
        k = self.k_layernorm(k.reshape(b, t, self.n_kv_heads, self.head_dim))
        v = v.reshape(b, t, self.n_kv_heads, self.head_dim)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        offset = cache.offset if cache is not None else 0
        if isinstance(offset, mx.array):
            # BatchKVCache advances its offset array in place on update; the
            # indexer queries are rotated after the update, so hold a copy.
            offset = offset + 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if self.sparse:
            if cache is not None and index_cache is None:
                raise ValueError("Maxwell sparse attention needs its indexer-key cache plane")
            # The reference scores ``index_key.float()`` of the bf16 key.  Default:
            # store that fp32 widening.  INDEX_KERNEL: store the bf16 key (half the
            # bytes) and widen inside the scoring kernel.  A 1-wide value slice
            # keeps the plane an ordinary KVCache without a copy.
            ik = self.rope(ik, offset=offset)
            if not INDEX_KERNEL:
                ik = ik.astype(mx.float32)
            if index_cache is not None:
                ik, _ = index_cache.update_and_fetch(ik, ik[..., :1])
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            if self.sparse and ik.shape[2] != k.shape[2]:
                raise RuntimeError(f"Maxwell trunk KV ({k.shape[2]}) and indexer plane ({ik.shape[2]}) "
                                   "lengths diverged")
        if not self.sparse or k.shape[2] <= self.dense_limit:
            STATS["dense_calls"] += 1
            y = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        else:
            if cache is not None and type(cache) not in (KVCache, BatchKVCache):
                raise ValueError(
                    "Maxwell Block Sparse GQA beyond the dense regime supports KVCache and "
                    f"BatchKVCache only, not {type(cache).__name__}")
            if b > 1 and padding is None and type(cache) is BatchKVCache:
                raise ValueError("Maxwell sparse batch attention needs the row padding")
            iq = self.rope(self.indexer.queries_from(index_q()), offset=offset)
            y = self._sparse(q, k, v, iq, ik, padding)
        return y.transpose(0, 2, 1, 3).reshape(b, t, -1)

    # ------------------------------------------------------------ sparse route

    def _sparse(self, q, k, v, iq, ik, padding):
        if q.shape[0] == 1 and not (padding and any(padding[0])):
            return self._sparse_row(q, k, v, iq, ik)
        if padding is None:
            padding = [(0, 0)] * q.shape[0]
        if len(padding) != q.shape[0]:
            raise ValueError("Maxwell sparse batch padding does not match the batch")
        rows = []
        t, width = q.shape[2], k.shape[2]
        heads, dim = q.shape[1], v.shape[3]
        for r, (left, right) in enumerate(padding):
            if left < 0 or right < 0 or right > t or left + right > width:
                raise ValueError(f"Maxwell sparse row {r} padding ({left}, {right}) is outside the cache")
            length, end = width - left - right, t - right                   # real keys; real queries end
            valid = min(end, length)
            parts = []
            if end - valid > 0:
                parts.append(mx.zeros((1, heads, end - valid, dim), v.dtype))
            if valid > 0:
                parts.append(self._sparse_row(q[r:r + 1, :, end - valid:end], k[r:r + 1, :, left:width - right],
                                              v[r:r + 1, :, left:width - right], iq[r:r + 1, :, end - valid:end],
                                              ik[r:r + 1, :, left:width - right]))
            if right:
                parts.append(mx.zeros((1, heads, right, dim), v.dtype))
            rows.append(parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2))
        return mx.concatenate(rows, axis=0)

    def _route(self, iq, ik, length, t):
        """Per-group routed block ids ``[1, G, t, prefix + remote + local]`` (-1 = none)."""
        bs = self.block
        first = length - t
        cb = mx.arange(first, length) // bs                                  # [t]
        local_start = mx.maximum(cb - self.n_local + 1, 0)                   # [t] block ids
        hi_block = max((length - 1) // bs - self.n_local + 1, 0)             # host: largest local start
        groups = iq.shape[1]
        remote = None
        if hi_block > self.n_prefix:
            keys = ik[:, :, : hi_block * bs]                                 # [1, 1, hi, d] key plane
            nb = hi_block
            if keys.dtype == mx.bfloat16 and _on_gpu() and bs == 128 and t <= 16:
                from .maxwell_moe_metal import index_block_scores

                scores = _kernel_call(("index_scores", t, groups), index_block_scores,
                                      iq[0].astype(mx.float32), keys[0, 0], nb, bs, self.indexer.scale)
                if scores is not None:
                    STATS["index_kernel_calls"] += 1
                    remote = self._pick(scores[None], local_start, nb)
            if remote is None:
                if keys.dtype != mx.float32:
                    keys = keys.astype(mx.float32)
                picks = []
                # Bound the fp32 token-score temporary to ~256 MB at long context.
                step = max(1, min(INDEX_QUERY_CHUNK, (1 << 26) // max(1, groups * hi_block * bs)))
                for s0 in range(0, t, step):
                    s1 = min(t, s0 + step)
                    scores = (iq[:, :, s0:s1].astype(mx.float32) @ mx.swapaxes(keys, -1, -2)) * self.indexer.scale
                    scores = scores.reshape(1, groups, s1 - s0, nb, bs).max(axis=-1)
                    picks.append(self._pick(scores, local_start[s0:s1], nb))
                remote = mx.concatenate(picks, axis=2) if len(picks) > 1 else picks[0]
        if remote is None:
            remote = mx.full((1, groups, t, self.n_remote), -1, mx.int32)
        remote = remote.astype(mx.int32)
        prefix = mx.arange(self.n_prefix)[None, :]
        prefix = mx.where(prefix <= cb[:, None], prefix, -1)
        local = cb[:, None] + mx.arange(1 - self.n_local, 1)[None, :]
        local = mx.where(local >= self.n_prefix, local, -1)
        fixed = lambda a: mx.broadcast_to(a.astype(mx.int32)[None, None], (1, groups, t, a.shape[-1]))
        return mx.concatenate([fixed(prefix), remote, fixed(local)], axis=-1)

    def _pick(self, scores, local_start, nb, canonical=None):
        """Top remote blocks from block scores ``[1, G, t, nb]``: eligible blocks only, -1 pads."""
        blk = mx.arange(nb)
        eligible = (blk[None, :] >= self.n_prefix) & (blk[None, :] < local_start[:, None])
        scores = mx.where(eligible[None, None], scores, -mx.inf)
        kk = min(self.n_remote, nb)
        if canonical is None:
            canonical = CANONICAL_PICK
        if not canonical:
            idx = mx.argpartition(-scores, kth=kk - 1, axis=-1)[..., :kk].astype(mx.int32)
            val = mx.take_along_axis(scores, idx, axis=-1)
            idx = mx.where(mx.isfinite(val), idx, -1)
            if kk < self.n_remote:
                idx = mx.concatenate([idx, mx.full((*idx.shape[:-1], self.n_remote - kk), -1, mx.int32)], -1)
            return idx
        # Canonical selection (opt-in, MLX2_MAXWELL_CANONICAL_PICK): the top-kk set with ties at the threshold broken
        # toward the lowest block id, returned in ascending block order.  An
        # argpartition set and order depend on the kernel and the shape, so two
        # numerically "exact" scorers (fp32 matmul, bf16 block kernel) could
        # pick different tied blocks or gather the same blocks in a different
        # summation order.  Same set as torch.topk whenever there is no tie.
        kth = mx.sort(scores, axis=-1)[..., nb - kk: nb - kk + 1]          # kk-th largest
        finite = mx.isfinite(scores)
        above = (scores > kth) & finite
        tied = (scores == kth) & finite
        room = kk - above.sum(axis=-1, keepdims=True)
        chosen = above | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= room))
        order = mx.argsort(mx.where(chosen, blk, nb + blk), axis=-1)[..., :kk].astype(mx.int32)
        idx = mx.where(mx.take_along_axis(chosen, order, axis=-1), order, -1)
        if kk < self.n_remote:
            idx = mx.concatenate([idx, mx.full((*idx.shape[:-1], self.n_remote - kk), -1, mx.int32)], -1)
        return idx

    def _sparse_row(self, q, k, v, iq, ik):
        """One unpadded row: queries are the last ``t`` of ``length`` positions."""
        t, length = q.shape[2], k.shape[2]
        if length <= self.dense_limit:
            STATS["dense_calls"] += 1
            return scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale,
                                                mask="causal" if t > 1 else None)
        STATS["sparse_rows"] += t
        route = self._route(iq, ik, length, t)                               # [1, G, t, R]
        groups, heads = k.shape[1], q.shape[1]
        per = heads // groups
        bs = self.block
        qpos = mx.arange(length - t, length)
        if t <= SPARSE_GATHER_ROWS:
            STATS["sparse_gather_calls"] += 1
            tok = route[..., None] * bs + mx.arange(bs)                       # [1, G, t, R, bs]
            span = tok.shape[3] * bs
            tok = tok.reshape(1, groups, t, span)
            valid = (mx.repeat(route, bs, axis=-1) >= 0) & (tok <= qpos[None, None, :, None])
            tok = mx.clip(tok, 0, length - 1).reshape(groups, t * span, 1)
            kg = mx.take_along_axis(k[0], tok, axis=1).reshape(groups * t, 1, span, -1)
            vg = mx.take_along_axis(v[0], tok, axis=1).reshape(groups * t, 1, span, -1)
            qq = q[0].reshape(groups, per, t, -1).transpose(0, 2, 1, 3).reshape(groups * t, per, 1, -1)
            mask = valid.reshape(groups * t, 1, 1, span)
            y = mx.fast.scaled_dot_product_attention(qq, kg, vg, scale=self.scale, mask=mask)
            return y.reshape(groups, t, per, -1).transpose(0, 2, 1, 3).reshape(1, heads, t, -1)
        nb = (length + bs - 1) // bs
        slot = mx.where(route >= 0, route, nb)                               # [1, G, t, R]
        member = mx.zeros((1, groups, t, nb + 1), dtype=mx.bool_)
        member = mx.put_along_axis(member, slot, mx.array(True), axis=-1)[..., :nb]
        if SPARSE_PREFILL == "union":
            return self._sparse_union(q, k, v, member, qpos, length)
        STATS["sparse_masked_calls"] += 1
        causal = mx.arange(length)[None, :] <= qpos[:, None]                 # [t, L]
        outs = []
        for g in range(groups):
            mask = mx.repeat(member[:, g:g + 1], bs, axis=-1)[..., :length] & causal[None, None]
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, g * per:(g + 1) * per], k[:, g:g + 1], v[:, g:g + 1], scale=self.scale, mask=mask))
        return mx.concatenate(outs, axis=1)

    def _sparse_union(self, q, k, v, member, qpos, length):
        """Exact block-sparse attention over each sub-chunk's union of routed blocks."""
        STATS["sparse_union_calls"] += 1
        groups, heads, t = k.shape[1], q.shape[1], q.shape[2]
        per, bs, nb = heads // groups, self.block, member.shape[-1]
        step = max(1, UNION_QUERY_CHUNK)
        starts = list(range(0, t, step))
        pad = len(starts) * step - t
        m = member[0]                                                        # [G, t, nb]
        if pad:
            m = mx.concatenate([m, mx.zeros((groups, pad, nb), dtype=mx.bool_)], axis=1)
        union = m.reshape(groups, len(starts), step, nb).any(axis=(0, 2))    # [S, nb]
        counts = union.sum(axis=-1).tolist()                                 # one host read per layer chunk
        ids = mx.arange(nb)
        order = mx.argsort(mx.where(union, ids, nb + ids), axis=-1)          # members first, ascending
        offsets = mx.arange(bs)
        outs = []
        for s_i, s0 in enumerate(starts):
            s1 = min(t, s0 + step)
            u = int(counts[s_i])
            STATS["union_subchunks"] += 1
            STATS["union_blocks"] += u
            blocks = order[s_i, :u]                                          # [U]
            tok = (blocks[:, None] * bs + offsets[None, :]).reshape(-1)      # [U * bs]
            inside = tok < length
            tok_c = mx.minimum(tok, length - 1)
            kg = mx.take(k, tok_c, axis=2)                                   # [1, G, U*bs, D]
            vg = mx.take(v, tok_c, axis=2)
            sel = mx.take(member[0, :, s0:s1], blocks, axis=-1)              # [G, c, U]
            sel = mx.repeat(sel, bs, axis=-1)                                # [G, c, U*bs]
            ok = sel & inside[None, None, :] & (tok[None, None, :] <= qpos[s0:s1][None, :, None])
            mask = mx.broadcast_to(ok[:, None], (groups, per, s1 - s0, ok.shape[-1]))
            mask = mask.reshape(1, heads, s1 - s0, ok.shape[-1])
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, s0:s1], kg, vg, scale=self.scale, mask=mask))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class Router(nn.Module):
    """fp32 sigmoid scores; top-k on scores + expert_bias; renormalised unbiased weights.

    Returns ``(ids, weights)`` using the Laguna router contract.  The weight is stored fp32 at load."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.weight = mx.zeros((args.num_experts, args.hidden_size), dtype=mx.float32)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        biased = scores + self.expert_bias.astype(mx.float32)
        ids = mx.stop_gradient(mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k])
        w = mx.take_along_axis(scores, ids, axis=-1)
        return ids, w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.scale = 1.0           # routed-output scale (none in Maxwell); carve contract
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x):
        ids, w = self.gate(x)
        rows = self.switch_mlp(x, ids)
        w = w.astype(mx.float32) * self.scale
        routed = (rows.astype(mx.float32) * w[..., None]).sum(axis=-2)
        return routed.astype(x.dtype) + self.shared_expert(x)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n = n = args.num_residual_streams
        self.hidden_size = args.hidden_size
        self.iterations = args.mhc_sinkhorn_iterations
        self.mapping_proj = nn.Linear(n * args.hidden_size, n * n + 2 * n, bias=False)
        self.mapping_proj.weight = self.mapping_proj.weight.astype(mx.float32)
        self.alpha_pre = mx.zeros((1,))
        self.alpha_post = mx.zeros((1,))
        self.alpha_res = mx.zeros((1,))
        self.bias = mx.zeros((n * n + 2 * n,))

    def pre(self, x):
        """(collapsed sub-block input, h_post, h_res)."""
        n, iters = self.n, self.iterations
        if (MHC_KERNEL and _on_gpu() and n == 4 and x.shape[0] * x.shape[1] <= MHC_KERNEL_MAX_ROWS
                and x.dtype in (mx.bfloat16, mx.float16, mx.float32)):
            from .maxwell_moe_metal import mhc_pre, pack_params

            parts = (self.alpha_pre, self.alpha_post, self.alpha_res, self.bias)
            cached = getattr(self, "_packed", None)
            # Keyed on the parameter arrays themselves (an id() would outlive a
            # freed array whose address is reused).
            if cached is None or any(a is not b for a, b in zip(cached[0], parts)):
                cached = (parts, pack_params(*parts, HC_EPS))
                object.__setattr__(self, "_packed", cached)
            params = cached[1]
            out = _kernel_call(("mhc_pre", x.shape[-1], x.dtype, iters), mhc_pre,
                               x, self.mapping_proj.weight.astype(mx.float32), params, iters=iters)
            if out is not None:
                return out
        fn = _compiled(("hc_pre", n, iters),
                       lambda: lambda *a: _hc_pre_body(*a, n=n, iters=iters))
        return fn(x, self.mapping_proj.weight, self.alpha_pre, self.alpha_post, self.alpha_res, self.bias)

    def post(self, out, residual, h_post, h_res):
        n = self.n
        if (MHC_KERNEL and _on_gpu() and n == 4 and out.shape[0] * out.shape[1] <= MHC_KERNEL_MAX_ROWS
                and residual.dtype in (mx.bfloat16, mx.float16, mx.float32)):
            from .maxwell_moe_metal import mhc_post

            fused = _kernel_call(("mhc_post", residual.shape[-1], residual.dtype), mhc_post,
                                 out, residual, h_post, h_res)
            if fused is not None:
                return fused
        return _compiled(("hc_post", n), lambda: lambda *a: _hc_post_body(*a, n=n))(
            out, residual, h_post, h_res)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, index: int, *, sparse: bool = True):
        super().__init__()
        self.self_attn = Attention(args, sparse=sparse)
        self.mlp = (MLP(args.hidden_size, args.intermediate_size)
                    if index in set(args.mlp_only_layers) else SparseMoE(args))
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn_hyper_connection = HyperConnection(args)
        self.mlp_hyper_connection = HyperConnection(args)

    def __call__(self, x, mask=None, cache=None, index_cache=None, padding=None):
        hc = self.self_attn_hyper_connection
        a, h_post, h_res = hc.pre(x)
        a = self.self_attn(self.input_layernorm(a), mask, cache, index_cache, padding)
        x = hc.post(a, x, h_post, h_res)
        hc = self.mlp_hyper_connection
        m, h_post, h_res = hc.pre(x)
        m = self.mlp(self.post_attention_layernorm(m))
        return hc.post(m, x, h_post, h_res)


class FinalNorm(nn.Module):
    """The learned mHC head (n streams -> 1) followed by the final RMSNorm.

    One module so that ``norm(multi-stream hidden)`` is the post-norm hidden
    state, as structural callers (layer replay) expect."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n, c = args.num_residual_streams, args.hidden_size
        self.n, self.hidden_size, self.eps = n, c, args.rms_norm_eps
        self.weight = mx.ones((c,))
        self.hc_head_fn = mx.zeros((n, n * c), dtype=mx.float32)
        self.hc_head_base = mx.zeros((n,))
        self.hc_head_scale = mx.ones((1,))

    def contract(self, x):
        n = self.n
        return _compiled(("hc_head", n), lambda: lambda *a: _hc_head_body(*a, n=n))(
            x, self.hc_head_fn, self.hc_head_base, self.hc_head_scale)

    def __call__(self, x):
        return rms_norm(self.contract(x), self.weight, self.eps)


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = FinalNorm(args)

    def expand(self, inputs):
        h = self.embed_tokens(inputs)
        b, t, c = h.shape
        n = self.args.num_residual_streams
        return mx.broadcast_to(h[:, :, None, :], (b, t, n, c)).reshape(b, t, n * c)

    def __call__(self, inputs, cache=None, capture=None, return_streams=False):
        n = len(self.layers)
        if cache is None:
            cache = [None] * (2 * n)
        if len(cache) != 2 * n:
            raise ValueError(f"Maxwell cache must hold {n} trunk and {n} indexer planes, got {len(cache)}")
        main, index = cache[:n], cache[n:]
        first = main[0]
        # Padded width bounds every row's real length (batch offsets are device arrays).
        reach = first.offset if type(first) is KVCache else getattr(first, "_idx", 0) if first is not None else 0
        if int(reach) + inputs.shape[1] > self.args.max_position_embeddings:
            raise ValueError(f"Maxwell context exceeds {self.args.max_position_embeddings} positions")
        h = self.expand(inputs)
        mask = create_attention_mask(h, main[0])
        padding = _row_padding(main[0], inputs.shape[1], self.args.dense_context_limit)
        # Opt-in MLX2_MAXWELL_MOE_NAX=gather: prefill forwards admit the
        # segmented NAX sorted expert gather (bit-identical to the stock sorted
        # kernel, canary-proven per instantiation, M5 only); a no-op while off.
        from . import moe_nax_gather

        with moe_nax_gather.forward_scope(inputs, cache):
            for i, (layer, c, ic) in enumerate(zip(self.layers, main, index, strict=True)):
                h = layer(h, mask, c, ic, padding)
                if capture is not None and i in capture:
                    capture[i] = h          # multi-stream residual after block i
        out = self.norm(h)
        return (out, h) if return_streams else out


class MTP(nn.Module):
    """Native MTP layer (``mtp_layers.0`` in the checkpoint)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        c = args.hidden_size
        self.n = args.num_residual_streams
        self.enorm = RMSNorm(c, args.rms_norm_eps)
        self.hnorm = RMSNorm(c, args.rms_norm_eps)
        self.e_proj = nn.Linear(c, c, bias=False)
        self.h_proj = nn.Linear(c, c, bias=False)
        # The MTP decoder layer uses plain (dense) attention, no indexer.
        self.layer = Layer(args, args.num_hidden_layers, sparse=False)
        self.norm = FinalNorm(args)

    def __call__(self, streams, embeddings, cache):
        b, s, nc = streams.shape
        c = nc // self.n
        e = self.e_proj(self.enorm(embeddings))
        h = self.h_proj(self.hnorm(streams.reshape(b, s, self.n, c)))
        x = (e[:, :, None, :] + h).reshape(b, s, nc)
        mask = create_attention_mask(x, cache)
        return self.layer(x, mask, cache)


class Model(nn.Module):
    apc_v2_layout = CACHE_LAYOUT
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Decoder(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.head_mode = HEAD_MODE
        self.lm_head.weight = self.lm_head.weight.astype(mx.float32 if HEAD_MODE == "fp32" else mx.bfloat16)
        if args.mtp_num_layers:
            self.mtp = MTP(args)
        self.completed_forwards = 0
        if ROW_EXACT_VERIFY != "off":
            from . import maxwell_row_exact

            maxwell_row_exact.install(self, ROW_EXACT_VERIFY)

    @property
    def has_mtp(self) -> bool:
        return "mtp" in self

    def logits_from_hidden(self, h):
        """The reference head: fp32 linear over the post-norm hidden state."""
        if isinstance(self.lm_head, nn.QuantizedLinear):
            # Quantized artifact: dequantized weights, fp32 activations and logits.
            return self.lm_head(h.astype(mx.float32)).astype(mx.float32)
        w = self.lm_head.weight
        if w.dtype == mx.bfloat16 and _on_gpu():
            from .maxwell_moe_metal import HEAD_MAX_ROWS, head_gemv

            rows = 1
            for size in h.shape[:-1]:
                rows *= size
            if rows <= HEAD_MAX_ROWS:
                out = _kernel_call(("head_gemv", rows, h.shape[-1]), head_gemv, h, w)
                if out is not None:
                    STATS["head_kernel_calls"] += 1
                    return out
        return h.astype(mx.float32) @ w.astype(mx.float32).T

    def fuse_projections(self):
        """Opt-in (``MLX2_MAXWELL_FUSED_QKV``): one input projection per attention."""
        for layer in self.model.layers:
            layer.self_attn.fuse_projections()
        if self.has_mtp:
            self.mtp.layer.self_attn.fuse_projections()

    logits = logits_from_hidden

    def __call__(self, inputs, cache=None):
        out = self.logits_from_hidden(select_checkpoint_rows(self.model(inputs, cache)))
        self.completed_forwards += 1
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, max_kv_size=None):
        """Trunk KV planes for every layer, then one indexer-key plane per layer.

        The indexer plane stores the post-RoPE fp32 indexer key as the K of a
        one-head ``KVCache`` (its V is a 1-wide slice of it) so every generic
        cache operation (trim, batch merge/extract, APCv2 capture) treats it as
        ordinary KV."""
        n = len(self.model.layers)
        return [KVCache() for _ in range(2 * n)]

    # ------------------------------------------------------------ native MTP

    def mtp_backbone(self, inputs, cache=None):
        """(post-norm hidden for the head, final multi-stream residual = MTP seed)."""
        hidden, streams = self.model(inputs, cache, return_streams=True)
        self.completed_forwards += 1
        return hidden, streams

    def make_mtp_cache(self):
        if not self.has_mtp:
            raise RuntimeError("Maxwell MTP head is not loaded")
        return [KVCache()]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """Advance the MTP pair stream.

        ``hidden`` is the multi-stream residual at positions p.. (the trunk's
        final streams, or a previous step's MTP streams when chaining) and
        ``tokens`` the tokens at p+1..  The private cache offset counts pairs,
        a uniform shift of absolute positions that cancels inside the layer's
        own attention.  Returns (fp32 logits, MTP multi-stream output)."""
        if not self.has_mtp:
            raise RuntimeError("Maxwell MTP head is not loaded")
        if not isinstance(mtp_cache, list) or len(mtp_cache) != 1:
            raise ValueError("Maxwell MTP requires one private cache")
        STATS["mtp_steps"] += 1
        embeddings = self.model.embed_tokens(tokens)
        # The activation dtype (a quantized embedding's weight is uint32).
        streams = self.mtp(hidden.astype(embeddings.dtype), embeddings, mtp_cache[0])
        return self.logits_from_hidden(self.mtp.norm(streams)), streams

    # ------------------------------------------------------------ weights

    def _pack_moe(self, weights, src, dst, layer, *, dense, eager):
        a = self.args
        proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        parts = ("weight", "scales", "biases")          # affine-quantized artifacts carry all three
        if dense:
            for w, p in proj.items():
                for part in parts:
                    if f"{src}.{w}.{part}" in weights:
                        weights[f"{dst}.{p}.{part}"] = weights.pop(f"{src}.{w}.{part}")
            return
        for key, new in ((f"{src}.gate.weight", f"{dst}.gate.weight"),
                         (f"{src}.expert_bias", f"{dst}.gate.expert_bias")):
            if key in weights:
                # fp32 resident: the router runs in fp32, and the expert bias'
                # per-expert spread is below one bf16 step
                weights[new] = weights.pop(key).astype(mx.float32)
        for w, p in proj.items():
            for part in parts:
                key = f"{src}.shared_expert.{w}.{part}"
                if key in weights:
                    weights[f"{dst}.shared_expert.{p}.{part}"] = weights.pop(key)
        packed = []
        for w, p in proj.items():
            for part in parts:
                names = [f"{src}.experts.{e}.{w}.{part}" for e in range(a.num_experts)]
                present = [n in weights for n in names]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(f"Maxwell layer {layer} has an incomplete {w} expert table")
                value = mx.stack([weights.pop(n) for n in names])
                weights[f"{dst}.switch_mlp.{p}.{part}"] = value
                packed.append(value)
        if eager and packed:
            mx.eval(packed)
            mx.clear_cache()

    def sanitize(self, weights, *, eager=False):
        """Checkpoint names -> this module tree; per-expert tensors -> SwitchGLU tables.

        Keeps the native MTP layer when the model was built with one and drops
        it otherwise.  Stores the weights the reference casts to fp32 on every
        call (router, mHC mappings, hc_head, lm_head) as fp32."""
        a = self.args
        mtp = [k for k in weights if k.startswith("mtp_layers.")]
        if not a.mtp_num_layers:
            for name in mtp:
                weights.pop(name)
        else:
            if any(not k.startswith("mtp_layers.0.") for k in mtp):
                raise ValueError("Maxwell checkpoint carries more than one MTP layer")
            if not mtp:
                raise ValueError("Maxwell MTP model requires mtp_layers.0 tensors")
            for name in mtp:
                rest = name.removeprefix("mtp_layers.0.")
                if rest.startswith("decoder_layer."):
                    rest = "layer." + rest.removeprefix("decoder_layer.")
                elif rest.startswith("hc_head."):
                    rest = "norm." + rest.removeprefix("hc_head.")
                elif rest == "final_layernorm.weight":
                    rest = "norm.weight"
                weights[f"mtp.{rest}"] = weights.pop(name)
            src = "mtp.layer.block_sparse_moe"
            if any(k.startswith(src + ".") for k in weights):
                self._pack_moe(weights, src, "mtp.layer.mlp", "mtp", dense=False, eager=eager)
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            key = f"model.hc_head.{suffix}"
            if key in weights:
                weights[f"model.norm.{suffix}"] = weights.pop(key)
        for layer in range(a.num_hidden_layers):
            self._pack_moe(weights, f"model.layers.{layer}.block_sparse_moe", f"model.layers.{layer}.mlp",
                           layer, dense=layer in set(a.mlp_only_layers), eager=eager)
        for name in [n for n in weights if n.endswith(("mapping_proj.scales", "norm.hc_head_fn.scales"))]:
            raise ValueError(f"Maxwell mHC weights must stay unquantized (fp32 law): {name}")
        for name in list(weights):
            if (name.endswith("hyper_connection.mapping_proj.weight") or name.endswith("norm.hc_head_fn")
                    or (name == "lm_head.weight" and self.head_mode == "fp32"
                        and "lm_head.scales" not in weights)):
                weights[name] = weights[name].astype(mx.float32)
        return weights

    # ------------------------------------------------------------ quantized artifacts

    def hf_module_names(self, path: str) -> list[str]:
        """Checkpoint (HF) module names behind one of this model's module paths."""
        if path.startswith("mtp."):
            rest = path.removeprefix("mtp.")
            if rest.startswith("layer."):
                return self._hf_layer_names(rest.removeprefix("layer."), "mtp_layers.0.decoder_layer.", None)
            if rest == "norm":
                return ["mtp_layers.0.hc_head"]
            return [f"mtp_layers.0.{rest}"]
        if path.startswith("model.layers."):
            index = int(path.split(".")[2])
            prefix = f"model.layers.{index}."
            return self._hf_layer_names(path.removeprefix(prefix), prefix, index)
        return [path]

    def _hf_layer_names(self, rest, prefix, index):
        a = self.args
        proj = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        dense = index is not None and index in set(a.mlp_only_layers)
        if rest.startswith("mlp."):
            leaf = rest.removeprefix("mlp.")
            moe = prefix + "block_sparse_moe."
            if leaf in proj and dense:
                return [moe + proj[leaf]]
            if leaf.startswith("shared_expert."):
                return [moe + "shared_expert." + proj[leaf.removeprefix("shared_expert.")]]
            if leaf.startswith("switch_mlp."):
                w = proj[leaf.removeprefix("switch_mlp.")]
                return [f"{moe}experts.{e}.{w}" for e in range(a.num_experts)]
            if leaf == "gate":
                return [moe + "gate"]
        return [prefix + rest]

    def apply_quantization(self, quantization: dict, quantized_names):
        """Quantize modules the artifact stores quantized (oQ/mlx-lm per-module config).

        ``quantization`` has default ``bits``/``group_size``/``mode`` and
        per-module overrides keyed by checkpoint module name; ``weights`` names
        ``quantized_names`` the checkpoint modules stored quantized (those with
        ``<name>.scales``).  Every
        checkpoint module behind one of this model's modules must agree (the
        256 experts of a layer pack into one table)."""
        default = {k: quantization[k] for k in ("bits", "group_size") if k in quantization}
        default.setdefault("mode", quantization.get("mode", "affine"))
        stored = frozenset(quantized_names)

        def spec(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            names = self.hf_module_names(path)
            hits = [n in stored for n in names]
            if not any(hits):
                return False
            if not all(hits):
                raise ValueError(f"Maxwell {path}: only part of its checkpoint modules are quantized")
            specs = []
            for n in names:
                override = quantization.get(n)
                specs.append({**default, **override} if isinstance(override, dict) else dict(default))
            keys = {(s["bits"], s["group_size"], s.get("mode", "affine")) for s in specs}
            if len(keys) != 1:
                raise ValueError(f"Maxwell {path}: checkpoint modules disagree on quantization {sorted(keys)}")
            bits, group, mode = keys.pop()
            return {"bits": bits, "group_size": group, "mode": mode}

        nn.quantize(self, class_predicate=spec)
        self.quantized = True
