"""QSAIndexer.__call__ honours fused_query (sweep 2026-10-02 A1).

Attention passes ``fused_query=fused_rows`` to the indexer, but __call__
ignored it, so the fused indexer query (``attn_rows.index_q``) never ran on
the decode/verify path.  CPU only: ``_fused_query`` is spied and returns
None, so the MLX ops still compute the selection.
"""

import mlx.core as mx

from mlx2.runtime.models import qwen4_exp as Q
from qsa_oracle import tiny_args


def _prefilled():
    args = tiny_args(indexer_budget=8)
    attn = Q.Attention(args, 3)
    attn.eval()
    mx.eval(attn.parameters())
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    attn(mx.ones((1, 40, args.hidden_size)), None, cache)
    return args, attn, cache


def _spy(monkeypatch, result=None):
    calls = []

    def spy(self, qk, q_pos):
        calls.append((tuple(qk.shape), tuple(q_pos.shape)))
        return result

    monkeypatch.setattr(Q.QSAIndexer, "_fused_query", spy)
    return calls


def test_indexer_call_runs_fused_query_when_asked(monkeypatch):
    args, attn, cache = _prefilled()
    calls = _spy(monkeypatch)
    sel = attn.indexer(
        mx.ones((1, 1, args.hidden_size)), None, cache,
        dense_shortcircuit=True, fused_query=True,
    )
    assert sel.kind == "explicit"
    assert calls == [((1, 1, calls[0][0][2]), (1, 1))]


def test_indexer_call_skips_fused_query_by_default(monkeypatch):
    args, attn, cache = _prefilled()
    calls = _spy(monkeypatch)
    attn.indexer(mx.ones((1, 1, args.hidden_size)), None, cache)
    assert calls == []


def test_fused_query_result_feeds_selection(monkeypatch):
    """A returned query replaces the composed one (same selection when equal)."""
    args, attn, cache_a = _prefilled()
    cache_b = Q.QSAKVCache(attn.indexer.summary_identity)
    attn(mx.ones((1, 40, args.hidden_size)), None, cache_b)
    hidden = mx.ones((1, 1, args.hidden_size))
    want = attn.indexer(hidden, None, cache_a)
    seen = []

    def composed(self, qk, q_pos):
        q, _ = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = self.q_layernorm(q.reshape(*qk.shape[:2], self.n_heads, self.head_dim))
        seen.append(1)
        return Q._apply_rope_positions(
            q, q_pos[..., None], self.rotary_dim, self.rope_theta, self.rope_scaling
        )

    monkeypatch.setattr(Q.QSAIndexer, "_fused_query", composed)
    got = attn.indexer(hidden, None, cache_b, fused_query=True)
    assert seen == [1]
    assert got.kind == want.kind == "explicit"
    assert mx.array_equal(got.dense_mask(), want.dense_mask()).item()
