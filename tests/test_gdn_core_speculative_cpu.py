"""Core-scan dispatch must not change the speculative recurrence law.

Synthetic float32 CPU tensors with CPU stand-ins for both GPU implementations;
no real checkpoint or Metal execution, and no speed/qualification claim.
"""
import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.decisions.candidates import d3_model
from mlx2.runtime.models import gated_delta as gd
from mlx2.runtime.models import qwen3_5, qwen4_exp
from mlx2.runtime.models.cache import ArraysCache


@pytest.mark.parametrize("module", [qwen3_5, qwen4_exp, d3_model])
@pytest.mark.parametrize("array_accept", [False, True])
def test_speculative_verify_and_replay_bypass_core_scan(module, array_accept, monkeypatch):
    mx.random.seed(11303)
    args = module.TextModelArgs(
        hidden_size=32, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    layer = getattr(module, "D3GatedDeltaNet", module.GatedDeltaNet)(args)
    layer.eval()
    if module is qwen4_exp:
        layer.set_fused_gdn_decode_mode("stock")
        layer.set_fused_gdn_verify_mode("stock")
        layer.set_fused_gdn_batch_decode_mode("off")
        layer.set_fused_gdn_batch_verify_mode("off")
        layer.gdn_fused_inproj = False
    monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", array_accept)
    monkeypatch.setattr(gd, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gd, "_CORE_GDN_HEADS", frozenset({(2, 4)}))
    monkeypatch.setattr(gd, "_CORE_ACCEPTS_CHUNK_SIZE", False)
    # The real MLX default device stays CPU. Only dispatch's device query is
    # replaced; both candidates below evaluate the CPU recurrence.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    core_widths = []

    def core(q, k, v, g, beta, *, initial_state, **kwargs):
        core_widths.append(q.shape[1])
        return gd.gated_delta_ops(q, k, v, g, beta, initial_state)

    monkeypatch.setattr(gd, "_core_gated_delta_update", core)
    monkeypatch.setattr(gd, "gated_delta_kernel", gd.gated_delta_ops)
    prefix, verify = mx.random.normal((2, 20, 32)), mx.random.normal((2, 20, 32))
    cache = ArraysCache(2)
    layer(prefix, cache=cache)
    assert core_widths == [20]  # Ordinary prefill retains the selected core.
    before = copy.deepcopy(cache)
    cache.start_speculation()
    layer(verify, cache=cache)
    assert core_widths == [20]
    cache.trim_ragged([2, 12])  # Replays straddle the core's 17-row boundary.
    assert core_widths == [20]
    # Compare all state planes with independent ordinary one-token decode.
    for row, keep in enumerate((18, 8)):
        reference = before.extract(row)
        for token in range(keep):
            layer(verify[row:row + 1, token:token + 1], cache=reference)
        for got, want in zip(cache.extract(row), reference):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                       rtol=2e-5, atol=2e-6)
