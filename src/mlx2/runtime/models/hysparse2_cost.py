# SPDX-License-Identifier: MIT
"""Analytic KV-cache and prefill-FLOP model for HySparse2 and its baselines.

Reproduces Figure 4 of arXiv 2609.26368 from Table 1 alone, so it doubles as a
check that the layer layout was read correctly. The KV numbers need nothing
beyond Table 1 and the 49-layer layout (Figure 2): with FP8 storage,
``1M = 2**20`` tokens and ``GB = 1e9`` bytes they match the paper's 2.69 /
6.72 / 12.09 GB to two decimals.

FLOPs count matmul multiply-adds as 2 and average the causal attention context
over a prefill of ``T`` tokens (row ``t`` sees ``t + 1`` keys). Non-attention
compute is ``2 * active_params_per_layer`` per row per layer; the paper does not
give that split, so it is an input (``DEFAULT_ACTIVE_MLP_PARAMS`` is a rough
80B-A3B share and only moves the FLOP ratios by a few percent at 1M).
"""

from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_ACTIVE_MLP_PARAMS = 30e6


@dataclass(frozen=True)
class AttentionDesign:
    name: str
    full_layers: int
    swa_layers: int
    sparse_layers: int
    q_heads: int
    kv_heads: int
    qk_dim: int
    v_dim: int
    window: int = 128
    sparse_global: int = 0
    # Sparse layers that keep a separate SWA branch with its own KV (HySparse).
    sparse_swa_branch: bool = False
    hidden: int = 2048
    # HySparse2 only: layer structure of the self-decoder.
    self_swa_before_full: int = 0
    self_swa_after_full: int = 0
    self_full_layers: int = 0
    cross_full_layers: int = 0

    @property
    def layers(self) -> int:
        return self.full_layers + self.swa_layers + self.sparse_layers

    @property
    def kv_per_token(self) -> int:
        return self.kv_heads * (self.qk_dim + self.v_dim)

    @property
    def attn_proj_params(self) -> int:
        d = self.hidden
        return d * self.q_heads * self.qk_dim + d * self.kv_per_token + (
            self.q_heads * self.v_dim * d
        )


HYBRID_SWA = AttentionDesign(
    "Hybrid SWA", full_layers=9, swa_layers=40, sparse_layers=0,
    q_heads=64, kv_heads=4, qk_dim=192, v_dim=128,
)
HYSPARSE = AttentionDesign(
    "HySparse", full_layers=5, swa_layers=0, sparse_layers=44,
    q_heads=64, kv_heads=4, qk_dim=192, v_dim=128,
    sparse_global=1024, sparse_swa_branch=True,
)
HYSPARSE2 = AttentionDesign(
    "HySparse2", full_layers=5, swa_layers=24, sparse_layers=20,
    q_heads=64, kv_heads=1, qk_dim=256, v_dim=256,
    sparse_global=1024,
    self_swa_before_full=12, self_swa_after_full=12,
    self_full_layers=1, cross_full_layers=4,
)
DESIGNS = (HYBRID_SWA, HYSPARSE, HYSPARSE2)


def kv_cache_bytes(design: AttentionDesign, tokens: int, bytes_per_value: float = 1.0) -> float:
    """Per-sequence KV cache. Sparse layers reuse FA KV (zero bytes) unless
    they carry a separate SWA branch; window layers hold ``window`` tokens."""
    per_full = design.kv_per_token * tokens
    windowed = design.swa_layers + (
        design.sparse_layers if design.sparse_swa_branch else 0
    )
    per_window = design.kv_per_token * min(tokens, design.window)
    return bytes_per_value * (design.full_layers * per_full + windowed * per_window)


def _full_attn_flops(design: AttentionDesign, context: float) -> float:
    return 2.0 * design.q_heads * (design.qk_dim + design.v_dim) * context


def _layer_linear_flops(design: AttentionDesign, active_mlp: float) -> float:
    return 2.0 * (design.attn_proj_params + active_mlp)


def prefill_flops_per_token(
    design: AttentionDesign,
    tokens: int,
    *,
    active_mlp_params: float = DEFAULT_ACTIVE_MLP_PARAMS,
    suffix_bound: bool = False,
) -> float:
    """Average prefill FLOPs per prompt token for a ``tokens``-long prompt.

    For HySparse2 the cross-decoder is skipped (only its bridge K/V projections
    run). ``suffix_bound`` additionally models ``hysparse2.Model.prefill``: the
    self-decoder layers after its FA run on a bounded suffix only, and that FA
    layer queries only that suffix."""
    T = float(tokens)
    avg_ctx = (T + 1) / 2
    lin = _layer_linear_flops(design, active_mlp_params)
    win_ctx = min(avg_ctx, float(design.window))
    if design.cross_full_layers == 0:
        attn = design.full_layers * _full_attn_flops(design, avg_ctx)
        attn += design.swa_layers * _full_attn_flops(design, win_ctx)
        if design.sparse_layers:
            sel = min(avg_ctx, float(design.sparse_global + design.window))
            attn += design.sparse_layers * _full_attn_flops(design, sel)
            if design.sparse_swa_branch:
                attn += design.sparse_layers * _full_attn_flops(design, win_ctx)
        return design.layers * lin + attn

    # HySparse2 prefill: self-decoder + bridge projections only.
    d = design.hidden
    bridge = 2.0 * design.cross_full_layers * d * design.kv_per_token
    before = design.self_swa_before_full
    after = design.self_swa_after_full
    swa_row = lin + _full_attn_flops(design, win_ctx)
    if not suffix_bound:
        rows_after = 1.0
        fa_query_frac = 1.0
    else:
        w = design.window
        suffix = w + (after - 1) * (w - 1) if after else 1
        fa_query_frac = min(1.0, suffix / T)
        # Tail layer k (0-based) runs suffix - k*(w-1) rows.
        tail_rows = sum(max(1, suffix - k * (w - 1)) for k in range(after))
        rows_after = min(1.0, tail_rows / max(1, after) / T)
    fa_kv = 2.0 * d * design.kv_per_token
    fa_rest = lin - fa_kv
    # A suffix row sits near the end of the prompt and sees ~T keys.
    fa_ctx = T if suffix_bound and fa_query_frac < 1.0 else avg_ctx
    fa = fa_kv + fa_query_frac * (fa_rest + _full_attn_flops(design, fa_ctx))
    return before * swa_row + fa + after * rows_after * swa_row + bridge


def figure4_summary(tokens: int = 2**20, **kw) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for design in DESIGNS:
        out[design.name] = {
            "kv_gb": kv_cache_bytes(design, tokens) / 1e9,
            "prefill_gflops_per_token": prefill_flops_per_token(design, tokens, **kw) / 1e9,
        }
    out["HySparse2 (suffix-bound)"] = {
        "kv_gb": out["HySparse2"]["kv_gb"],
        "prefill_gflops_per_token": prefill_flops_per_token(
            HYSPARSE2, tokens, suffix_bound=True, **kw
        )
        / 1e9,
    }
    return out


if __name__ == "__main__":
    import json
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2**20
    print(json.dumps(figure4_summary(n), indent=2))
