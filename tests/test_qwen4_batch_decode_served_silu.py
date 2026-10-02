"""Flash-Next batched one-token fused GDN decode skips the served-SiLU guard.

Every other fused GDN entry (B=1 decode, verify, batch verify, catch-up,
prefill, Qwen3.6, 27B) declines when MLX serves a different SiLU
(omlx#4122, MLX #4461).  ``qwen4_exp._try_fused_batch_decode`` (default-on
``fused_gdn_batch_decode: row_exact``) never calls ``served_silu_refusal``,
so on such a build every batched decode step launches the kernel with the
old SiLU and forks from the reference.  CPU only: the guard outcome and the
kernel are injected.

Run: PYTHONPATH=src:tests python -m pytest -p no:cacheprovider -q <this file>
"""

import mlx.core as mx

from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn as fused

REFUSAL = "served SiLU uses metal::precise::exp"


class _Cache(list):
    speculating = False
    lengths = None

    def rollback_spans(self, steps, mask):
        return ()

    def record_rollback(self, *args, **kwargs):
        pass

    def advance(self, steps):
        pass


def test_batch_decode_declines_when_served_silu_differs(monkeypatch):
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    model, _vocab = tiny_qwen4_mtp()
    layer = next(
        l.linear_attn
        for l in model.language_model.model.layers
        if getattr(l, "linear_attn", None) is not None
    )
    layer.set_fused_gdn_decode_mode("fused")
    layer.set_fused_gdn_batch_decode_mode("row_exact")
    launched = []
    monkeypatch.setattr(
        qwen4_exp,
        "admit_qwen4_fused_gdn_batch_decode",
        lambda **kw: fused.FusedGdnAdmission(True, "eligible"),
    )
    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: REFUSAL)
    monkeypatch.setattr(qwen4_exp, "probe_qwen4_fused_gdn_decode", lambda *a, **k: 32)

    def kernel(qkv, z, b, a, conv, w, A, dt, state, nw, eps, **kw):
        launched.append(kw)
        return (mx.zeros((2, 1, z.shape[-1]), qkv.dtype), conv, state)

    monkeypatch.setattr(qwen4_exp, "qwen4_fused_gdn_batch_decode", kernel)
    rows = 2
    qkv = mx.zeros((rows, 1, layer.conv_dim), mx.bfloat16)
    z = mx.zeros((rows, 1, layer.value_dim), mx.bfloat16)
    g = mx.zeros((rows, 1, layer.num_v_heads), mx.bfloat16)
    cache = _Cache([
        mx.zeros((rows, layer.conv_kernel_size - 1, layer.conv_dim), mx.bfloat16),
        mx.zeros((rows, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim)),
    ])
    layer._try_fused_decode(qkv, z, g, g, None, cache)
    assert not launched, "batched fused decode launched although the served SiLU refused it"
    assert layer.fused_gdn_batch_decode_last_fallback == REFUSAL
