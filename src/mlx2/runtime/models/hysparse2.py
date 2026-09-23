# SPDX-License-Identifier: MIT
"""HySparse2 reference model (arXiv 2609.26368, Xiaomi LLM-Core).

Original mlx2 implementation written from the paper; see
``provenance/hysparse2-prototype.json``. No checkpoint or reference code has
been published, so every tensor-level detail the paper leaves open is a named
``ModelArgs`` field with the assumption recorded next to it.

Architecture:

* **Self-decoder** – sliding-window attention (SWA) layers plus full-attention
  (FA) layers. SWA layers use partial RoPE, sigmoid output gates and per-head
  sink logits; FA layers use NoPE and no gate.
* **Cross-decoder** – hybrid blocks of one FA layer followed by token-sparse
  attention (SA) layers.
* **KV Bridging** – a cross-decoder FA layer projects its K/V from the *input
  hidden state* of a paired self-decoder FA layer, so its cache never depends
  on cross-decoder computation.
* **KV Reuse** – SA layers own no K/V. They reuse their block's FA cache at the
  token indices that FA layer selected from its exact attention
  probabilities: the most recent ``sparse_local_window`` tokens are forced and
  ``sparse_topk`` more are chosen by score.

Two forward paths share the same weights:

* ``Model.__call__`` is the ordinary reference path: every layer runs on every
  input row. Decode and speculative verify use it.
* ``Model.prefill`` is the early-exit path. The cross-decoder runs only on the
  last prompt row (the paper's prefill exit). In addition, the self-decoder
  layers after its last FA layer are all sliding-window, so they need a
  bounded suffix of the prompt: that FA layer runs its queries (and its MLP)
  only on that suffix. This goes beyond the paper, which charges full
  attention for that layer during prefill. Prefill attention then costs
  O(T * suffix) instead of O(T^2), and it is exact:
  ``tests/test_hysparse2_model.py`` pins it against the reference path.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache

SWA = "swa"
FULL = "full"
SPARSE = "sparse"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "hysparse2"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    # Table 1: HySparse2 uses MQA, 64 query heads, head dim 256 for QK and V.
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 256
    # The paper does not give separate SWA head shapes; default to the same.
    swa_num_attention_heads: Optional[int] = None
    swa_num_key_value_heads: Optional[int] = None
    swa_head_dim: Optional[int] = None
    sliding_window: int = 128
    swa_rope_dims: int = 64
    rope_theta: float = 10000.0
    # Figure 2: 12 SWA + FA + 12 SWA, prefill exit, then 4 x (FA + 5 SA).
    self_decoder_layers: Optional[List[str]] = None
    cross_decoder_layers: Optional[List[str]] = None
    # For each cross-decoder FA layer, the ordinal of the self-decoder FA layer
    # whose input hidden state it projects. Default spreads them evenly.
    bridge_sources: Optional[List[int]] = None
    sparse_topk: int = 1024
    sparse_local_window: int = 128
    # Open: how FA scores become one token ranking under MQA. "prob_sum" sums
    # softmax probabilities over query heads; "logit_max" takes the max logit.
    selection_score: str = "prob_sum"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    # Simplified mHC with identity residual mixing (Section 4.1). 1 = plain
    # residual. >1 uses static pre/post stream weights: an approximation until
    # a checkpoint shows the real parameterisation.
    hc_streams: int = 1

    def __post_init__(self):
        if self.self_decoder_layers is None:
            self.self_decoder_layers = [SWA] * 12 + [FULL] + [SWA] * 12
        if self.cross_decoder_layers is None:
            self.cross_decoder_layers = ([FULL] + [SPARSE] * 5) * 4
        self.self_decoder_layers = [str(t) for t in self.self_decoder_layers]
        self.cross_decoder_layers = [str(t) for t in self.cross_decoder_layers]
        if any(t not in (SWA, FULL) for t in self.self_decoder_layers):
            raise ValueError("self-decoder layers must be 'swa' or 'full'")
        if any(t not in (FULL, SPARSE) for t in self.cross_decoder_layers):
            raise ValueError("cross-decoder layers must be 'full' or 'sparse'")
        if self.cross_decoder_layers and self.cross_decoder_layers[0] != FULL:
            raise ValueError("a sparse layer needs a preceding full layer in its block")
        n_self_full = self.self_decoder_layers.count(FULL)
        n_cross_full = self.cross_decoder_layers.count(FULL)
        if n_cross_full and not n_self_full:
            raise ValueError("KV bridging needs a full-attention self-decoder layer")
        if self.bridge_sources is None:
            self.bridge_sources = [
                (j * n_self_full) // n_cross_full for j in range(n_cross_full)
            ]
        if len(self.bridge_sources) != n_cross_full or any(
            not 0 <= s < n_self_full for s in self.bridge_sources
        ):
            raise ValueError("bridge_sources must name one self FA layer per cross FA")
        if self.selection_score not in ("prob_sum", "logit_max"):
            raise ValueError(f"unknown selection_score {self.selection_score!r}")
        if self.sliding_window < 1 or self.sparse_local_window < 0:
            raise ValueError("window sizes must be positive")
        if self.hc_streams < 1:
            raise ValueError("hc_streams must be >= 1")
        self.swa_num_attention_heads = (
            self.swa_num_attention_heads or self.num_attention_heads
        )
        self.swa_num_key_value_heads = (
            self.swa_num_key_value_heads or self.num_key_value_heads
        )
        self.swa_head_dim = self.swa_head_dim or self.head_dim

    @property
    def num_hidden_layers(self) -> int:
        return len(self.self_decoder_layers) + len(self.cross_decoder_layers)

    @property
    def self_full_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.self_decoder_layers) if t == FULL]

    @property
    def prefill_suffix_rows(self) -> int:
        """Rows of the last self FA layer's output that early-exit prefill must
        compute. Each trailing SWA layer's correct outputs start ``window - 1``
        rows after its first input row, and the top one must leave ``window``
        exact rows in its cache (the last of which feeds the cross-decoder)."""
        n_tail = len(self.self_decoder_layers) - 1 - self.self_full_layers[-1]
        w = self.sliding_window
        return w + (n_tail - 1) * (w - 1) if n_tail else 1


# ---------------------------------------------------------------------------
# Token selection (KV Reuse)
# ---------------------------------------------------------------------------


def select_tokens(
    importance: mx.array,
    query_positions: mx.array,
    budget: int,
    window: int,
) -> Tuple[mx.array, mx.array]:
    """Pick ``min(budget + window, T)`` key positions per query row.

    ``importance`` is ``(B, L, T)``; ``query_positions`` is ``(L,)`` absolute
    positions whose keys are ``0..T-1``. Keys within ``window`` of the query
    are always chosen; the rest go by score. Ties at the cut are broken toward
    the lower position, so the chosen set is a pure function of the scores
    (unlike ``argpartition``, which vLLM #56749 found nondeterministic on equal
    scores). Indices come back ascending, so the reduction order in the
    attention that consumes them is canonical too.

    Returns ``(indices, valid)``, both ``(B, L, K)``; ``valid`` masks indices
    that are after the query (only possible when ``T`` is short).
    """
    B, L, T = importance.shape
    K = min(budget + window, T)
    key_pos = mx.arange(T)
    qpos = query_positions.reshape(1, L, 1)
    visible = key_pos[None, None, :] <= qpos
    if K == T:
        indices = mx.broadcast_to(key_pos.astype(mx.int32), (B, L, T))
        return indices, mx.broadcast_to(visible, (B, L, T))
    forced = visible & (key_pos[None, None, :] > qpos - window)
    inf = mx.array(float("inf"), dtype=mx.float32)
    s = mx.where(forced, inf, mx.where(visible, importance.astype(mx.float32), -inf))
    top = mx.argpartition(-s, kth=K - 1, axis=-1)[..., :K]
    threshold = mx.min(mx.take_along_axis(s, top, axis=-1), axis=-1, keepdims=True)
    strict = s > threshold
    n_strict = strict.astype(mx.int32).sum(axis=-1, keepdims=True)
    ties = s == threshold
    tie_rank = mx.cumsum(ties.astype(mx.int32), axis=-1)
    chosen = strict | (ties & (tie_rank <= K - n_strict))
    slot = mx.cumsum(chosen.astype(mx.int32), axis=-1) - 1
    slot = mx.where(chosen, slot, K)
    out = mx.zeros((B, L, K + 1), dtype=mx.int32)
    src = mx.broadcast_to(key_pos.astype(mx.int32), (B, L, T))
    out = mx.put_along_axis(out, slot, src, axis=-1)
    indices = out[..., :K]
    valid = mx.take_along_axis(visible, indices, axis=-1)
    return indices, valid


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _local_window_mask(n: int, window: int) -> mx.array:
    pos = mx.arange(n)
    q = pos[:, None]
    k = pos[None, :]
    return (k <= q) & (q - k < window)


class SWAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.swa_num_attention_heads
        self.n_kv_heads = args.swa_num_key_value_heads
        self.head_dim = args.swa_head_dim
        self.window = args.sliding_window
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.sinks = mx.zeros((self.n_heads,))
        self.rope = nn.RoPE(args.swa_rope_dims, traditional=False, base=args.rope_theta)

    def _qkv(self, x, offset):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return self.rope(q, offset=offset), self.rope(k, offset=offset), v

    def _out(self, x, attn):
        B, L, _ = x.shape
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn * mx.sigmoid(self.gate_proj(x)))

    def __call__(self, x, cache: RotatingKVCache):
        L = x.shape[1]
        q, k, v = self._qkv(x, cache.offset)
        keys, values = cache.update_and_fetch(k, v)
        # A multi-row update returns keys in temporal order ending with these
        # rows; a one-row update returns the ring, which never exceeds the
        # window, so every key is visible.
        mask = _cache_window_mask(L, keys.shape[2], self.window) if L > 1 else None
        attn = scaled_dot_product_attention(
            q, keys, values, cache=None, scale=self.scale, mask=mask, sinks=self.sinks
        )
        return self._out(x, attn)

    def suffix(self, x, cache: RotatingKVCache, start: int):
        """Run on rows at absolute positions ``start..start+L-1`` with no
        history, then leave ``cache`` holding the last ``window`` keys as a
        normal prefill would. Rows before ``window - 1`` see a truncated window;
        the caller discards them."""
        L = x.shape[1]
        q, k, v = self._qkv(x, start)
        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=None,
            scale=self.scale,
            mask=_local_window_mask(L, self.window),
            sinks=self.sinks,
        )
        keep = min(self.window, L)
        cache.keys = k[..., -keep:, :]
        cache.values = v[..., -keep:, :]
        cache.offset = start + L
        cache._idx = keep
        return self._out(x, attn)


def _cache_window_mask(n_q: int, n_k: int, window: int) -> mx.array:
    """Queries are the last ``n_q`` of ``n_k`` cached keys in temporal order."""
    q = mx.arange(n_k - n_q, n_k)[:, None]
    k = mx.arange(n_k)[None, :]
    return (k <= q) & (q - k < window)


class SelfFullAttention(nn.Module):
    """Self-decoder FA: NoPE, no gate, no sink."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def append_kv(self, x, cache: KVCache):
        B, L, _ = x.shape
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return cache.update_and_fetch(k, v)

    def attend(self, x, keys, values):
        """Queries are the last ``x.shape[1]`` positions of ``keys``."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        T = keys.shape[2]
        mask = _cache_window_mask(L, T, T) if L > 1 else None
        attn = scaled_dot_product_attention(
            q, keys, values, cache=None, scale=self.scale, mask=mask
        )
        return self.o_proj(attn.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def __call__(self, x, cache: KVCache):
        keys, values = self.append_kv(x, cache)
        return self.attend(x, keys, values)


class CrossFullAttention(nn.Module):
    """Cross-decoder FA. K/V come from the bridge; queries from the current
    hidden state. Attention is computed explicitly because its probabilities
    also rank tokens for the block's sparse layers."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.selection_score = args.selection_score
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        # Bridge: this layer's own projections of the source layer's input.
        # Open: whether the source is normed by the source layer's norm; we
        # give the bridge its own RMSNorm.
        self.bridge_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)

    def bridge(self, source_hidden, cache: KVCache):
        B, L, _ = source_hidden.shape
        h = self.bridge_norm(source_hidden)
        k = self.k_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cache.update_and_fetch(k, v)

    def __call__(self, x, keys, values, query_positions):
        B, L, _ = x.shape
        T = keys.shape[2]
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        rep = self.n_heads // self.n_kv_heads
        kk = mx.repeat(keys, rep, axis=1) if rep > 1 else keys
        vv = mx.repeat(values, rep, axis=1) if rep > 1 else values
        scores = (q * self.scale).astype(mx.float32) @ kk.astype(
            mx.float32
        ).swapaxes(-1, -2)
        visible = mx.arange(T)[None, :] <= query_positions[:, None]
        scores = mx.where(visible, scores, -mx.inf)
        probs = mx.softmax(scores, axis=-1, precise=True)
        attn = (probs.astype(vv.dtype) @ vv).transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.selection_score == "prob_sum":
            importance = probs.sum(axis=1)
        else:
            importance = scores.max(axis=1)
        return self.o_proj(attn), importance


class SparseAttention(nn.Module):
    """Token-sparse attention over the block FA's cache at selected indices.
    Owns Q/O projections, a sigmoid output gate and per-head sinks; no K/V."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.sinks = mx.zeros((self.n_heads,))

    def __call__(self, x, keys, values, indices, valid):
        B, L, _ = x.shape
        K = indices.shape[-1]
        D = self.head_dim
        # One SDPA row per (batch, query): (B*L, H, 1, D) against its own K keys.
        q = self.q_proj(x).reshape(B * L, 1, self.n_heads, D).transpose(0, 2, 1, 3)
        idx = indices.reshape(B, 1, L * K, 1)
        k_sel = mx.take_along_axis(keys, idx, axis=2)
        v_sel = mx.take_along_axis(values, idx, axis=2)
        k_sel = (
            k_sel.reshape(B, self.n_kv_heads, L, K, D)
            .transpose(0, 2, 1, 3, 4)
            .reshape(B * L, self.n_kv_heads, K, D)
        )
        v_sel = (
            v_sel.reshape(B, self.n_kv_heads, L, K, D)
            .transpose(0, 2, 1, 3, 4)
            .reshape(B * L, self.n_kv_heads, K, D)
        )
        mask = valid.reshape(B * L, 1, 1, K)
        attn = scaled_dot_product_attention(
            q, k_sel, v_sel, cache=None, scale=self.scale, mask=mask, sinks=self.sinks
        )
        attn = attn.reshape(B, L, -1)
        return self.o_proj(attn * mx.sigmoid(self.gate_proj(x)))


class MLP(nn.Module):
    """Dense SwiGLU. The paper's models are MoE (80B-A3B); routing is
    orthogonal to the attention design and is left to a checkpoint port."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HyperConnection(nn.Module):
    """Simplified mHC with the residual mixing matrix fixed to identity:
    ``streams += post * f(sum(pre * streams))``. With one stream this is the
    plain pre-norm residual."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        if n > 1:
            self.pre_logits = mx.zeros((n,))
            self.post_logits = mx.zeros((n,))

    def read(self, streams):
        if self.n == 1:
            return streams
        w = mx.sigmoid(self.pre_logits)
        return (streams * w[:, None]).sum(axis=-2)

    def write(self, streams, out):
        if self.n == 1:
            return streams + out
        w = 2 * mx.sigmoid(self.post_logits)
        return streams + w[:, None] * mx.expand_dims(out, -2)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, kind: str):
        super().__init__()
        self.kind = kind
        d = args.hidden_size
        if kind == SWA:
            self.self_attn = SWAAttention(args)
        elif kind == FULL:
            self.self_attn = None
        self.input_layernorm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.mlp = MLP(d, args.intermediate_size)
        self.attn_hc = HyperConnection(args.hc_streams)
        self.mlp_hc = HyperConnection(args.hc_streams)

    def finish(self, streams, attn_out):
        streams = self.attn_hc.write(streams, attn_out)
        h = self.post_attention_layernorm(self.mlp_hc.read(streams))
        return self.mlp_hc.write(streams, self.mlp(h))


class SelfSWALayer(DecoderLayer):
    def __init__(self, args):
        super().__init__(args, SWA)

    def __call__(self, streams, cache):
        h = self.input_layernorm(self.attn_hc.read(streams))
        return self.finish(streams, self.self_attn(h, cache))

    def suffix(self, streams, cache, start):
        h = self.input_layernorm(self.attn_hc.read(streams))
        return self.finish(streams, self.self_attn.suffix(h, cache, start))


class SelfFullLayer(DecoderLayer):
    def __init__(self, args):
        super().__init__(args, FULL)
        self.self_attn = SelfFullAttention(args)

    def source(self, streams):
        """Bridge source: the (pre-norm) input hidden state of this layer."""
        return self.attn_hc.read(streams)


class CrossFullLayer(DecoderLayer):
    def __init__(self, args):
        super().__init__(args, FULL)
        self.self_attn = CrossFullAttention(args)


class CrossSparseLayer(DecoderLayer):
    def __init__(self, args):
        super().__init__(args, SPARSE)
        self.self_attn = SparseAttention(args)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ForwardStats:
    """Rows each layer group processed in the last call; a falsifier for the
    early exit (a path that silently runs everything still matches outputs)."""

    self_rows: List[int] = field(default_factory=list)
    self_full_query_rows: List[int] = field(default_factory=list)
    cross_rows: int = 0
    bridge_rows: int = 0


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.self_layers = [
            SelfSWALayer(args) if t == SWA else SelfFullLayer(args)
            for t in args.self_decoder_layers
        ]
        self.cross_layers = [
            CrossFullLayer(args) if t == FULL else CrossSparseLayer(args)
            for t in args.cross_decoder_layers
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._self_full = args.self_full_layers
        self._cross_full = [
            j for j, t in enumerate(args.cross_decoder_layers) if t == FULL
        ]
        self.stats = ForwardStats()

    # -- cache ---------------------------------------------------------------

    def make_cache(self) -> List[Any]:
        """One cache per self-decoder layer, then one per cross FA layer.
        Sparse layers own no cache (KV Reuse)."""
        caches: List[Any] = [
            RotatingKVCache(max_size=self.args.sliding_window) if t == SWA else KVCache()
            for t in self.args.self_decoder_layers
        ]
        caches.extend(KVCache() for _ in self._cross_full)
        return caches

    def _cross_caches(self, cache):
        return cache[len(self.self_layers) :]

    # -- helpers -------------------------------------------------------------

    def _embed(self, inputs):
        h = self.embed_tokens(inputs)
        if self.args.hc_streams > 1:
            h = mx.repeat(mx.expand_dims(h, -2), self.args.hc_streams, axis=-2)
        return h

    def _collapse(self, streams):
        return streams.mean(axis=-2) if self.args.hc_streams > 1 else streams

    def _logits(self, streams):
        h = self.norm(self._collapse(streams))
        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def _bridge(self, sources: Dict[int, mx.array], cache):
        cross_caches = self._cross_caches(cache)
        for j, src in enumerate(self.args.bridge_sources):
            layer = self.cross_layers[self._cross_full[j]]
            layer.self_attn.bridge(sources[src], cross_caches[j])
        self.stats.bridge_rows += next(iter(sources.values())).shape[1]

    def _cross_decoder(self, streams, cache, query_positions):
        cross_caches = self._cross_caches(cache)
        keys = values = indices = valid = None
        full_i = 0
        self.stats.cross_rows += streams.shape[1]
        for layer in self.cross_layers:
            h = layer.input_layernorm(layer.attn_hc.read(streams))
            if layer.kind == FULL:
                keys, values = cross_caches[full_i].keys_and_values()
                full_i += 1
                out, importance = layer.self_attn(h, keys, values, query_positions)
                indices, valid = select_tokens(
                    importance,
                    query_positions,
                    self.args.sparse_topk,
                    self.args.sparse_local_window,
                )
            else:
                out = layer.self_attn(h, keys, values, indices, valid)
            streams = layer.finish(streams, out)
        return streams

    # -- ordinary reference path ----------------------------------------------

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None):
        """Every layer on every row. Returns logits for all rows."""
        if cache is None:
            cache = self.make_cache()
        self.stats = ForwardStats()
        L = inputs.shape[1]
        start = cache[self._self_full[0]].offset
        streams = self._embed(inputs)
        sources: Dict[int, mx.array] = {}
        full_ord = 0
        for i, layer in enumerate(self.self_layers):
            if layer.kind == FULL:
                sources[full_ord] = layer.source(streams)
                full_ord += 1
                h = layer.input_layernorm(sources[full_ord - 1])
                streams = layer.finish(streams, layer.self_attn(h, cache[i]))
                self.stats.self_full_query_rows.append(L)
            else:
                streams = layer(streams, cache[i])
            self.stats.self_rows.append(L)
        self._bridge(sources, cache)
        positions = mx.arange(start, start + L)
        streams = self._cross_decoder(streams, cache, positions)
        return self._logits(streams)

    # -- early-exit prefill ---------------------------------------------------

    def prefill(
        self,
        inputs: mx.array,
        cache: List[Any],
        chunk_size: int = 2048,
        suffix_bound: bool = True,
    ) -> mx.array:
        """Prefill ``inputs`` (B=1) and return logits for the last row only.

        The cross-decoder runs on the last row; every cross FA cache is filled
        from the bridge. With ``suffix_bound``, the last self FA layer queries
        only the final ``prefill_suffix_rows`` rows and the trailing SWA layers
        run on that suffix alone."""
        if inputs.shape[0] != 1:
            raise ValueError("early-exit prefill is single-sequence")
        self.stats = ForwardStats()
        n = inputs.shape[1]
        start = cache[self._self_full[0]].offset
        last_full = self._self_full[-1]
        n_tail = len(self.self_layers) - 1 - last_full
        suffix = self.args.prefill_suffix_rows
        bounded = suffix_bound and n_tail > 0 and n > suffix
        cut = n - suffix if bounded else 0
        self.stats.self_rows = [0] * len(self.self_layers)
        self.stats.self_full_query_rows = [0] * len(self._self_full)

        # Phase A, chunked: every self layer, or (bounded) those up to and
        # including the last self FA.
        stop = last_full + 1 if bounded else len(self.self_layers)
        kept: List[mx.array] = []
        for c0 in range(0, n, chunk_size):
            c1 = min(n, c0 + chunk_size)
            streams = self._embed(inputs[:, c0:c1])
            sources: Dict[int, mx.array] = {}
            full_ord = 0
            for i, layer in enumerate(self.self_layers[:stop]):
                self.stats.self_rows[i] += c1 - c0
                if layer.kind != FULL:
                    streams = layer(streams, cache[i])
                    continue
                sources[full_ord] = layer.source(streams)
                h = layer.input_layernorm(sources[full_ord])
                keys, values = layer.self_attn.append_kv(h, cache[i])
                q0 = min(max(c0, cut), c1) - c0 if i == last_full else 0
                if q0 < c1 - c0:
                    out = layer.self_attn.attend(h[:, q0:], keys, values)
                    streams = layer.finish(streams[:, q0:], out)
                else:
                    streams = streams[:, :0]
                self.stats.self_full_query_rows[full_ord] += c1 - c0 - q0
                full_ord += 1
            self._bridge(sources, cache)
            if not bounded:
                streams = streams[:, -1:]
            if streams.shape[1]:
                kept.append(streams)
            mx.eval(streams, [c.state for c in cache if not c.empty()])
        streams = mx.concatenate(kept, axis=1) if bounded else kept[-1]

        # Phase B (bounded): trailing SWA layers on the suffix alone.
        if bounded:
            pos = start + cut
            w = self.args.sliding_window
            for i in range(last_full + 1, len(self.self_layers)):
                layer = self.self_layers[i]
                self.stats.self_rows[i] += streams.shape[1]
                streams = layer.suffix(streams, cache[i], pos)
                if i < len(self.self_layers) - 1:
                    streams = streams[:, w - 1 :]
                    pos += w - 1

        # Phase C: prefill exit — cross-decoder on the last row only.
        streams = streams[:, -1:]
        positions = mx.array([start + n - 1])
        streams = self._cross_decoder(streams, cache, positions)
        return self._logits(streams)

    # -- loading -------------------------------------------------------------

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.self_layers + self.cross_layers
