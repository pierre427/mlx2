"""Every decode route reclaims the MLX buffer pool periodically.

Ollama v0.34.2: freed KV/intermediate buffers pile up in the MLX pool during
speculative decode unless something clears it. The prompt-lookup route had no
clear in prefill or decode.
"""

import mlx.core as mx

from mlx2.runtime import generate as G
from mlx2.runtime import pld
from mlx2.runtime.sample_utils import LaneRNG


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 16,
    )
    mx.random.seed(7)
    m = TextModel(args)
    m.eval()
    mx.eval(m.parameters())
    return m


def _count_clears(monkeypatch):
    calls = {"n": 0}
    real = mx.clear_cache

    def counted():
        calls["n"] += 1
        real()

    monkeypatch.setattr(mx, "clear_cache", counted)
    return calls


def _drain(gen, n):
    emitted = 0
    while emitted < n:
        _p, rs = gen.next()
        emitted += len(rs)
        if any(r.finish_reason for r in rs):
            break
    return emitted


def test_self_mtp_decode_invokes_periodic_reclaim(monkeypatch):
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 16)
    model = tiny_model()
    gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=64,
                           self_mtp={"num_draft": 2, "persistent": True})
    try:
        gen.insert([list(range(2, 100))], max_tokens=[80], lane_rngs=[LaneRNG(1)],
                   self_mtp_configs=[{"sampling_temp": 0.0}])
        _p, rs = gen.next()  # prefill (clears per chunk)
        calls = _count_clears(monkeypatch)
        _drain(gen, 79)
        steps = gen._steps_counter
    finally:
        gen.close()
    assert calls["n"] >= steps // 16 >= 1


def test_prompt_lookup_prefill_and_decode_reclaim(monkeypatch):
    monkeypatch.setattr(pld, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 16)
    model = tiny_model()
    gen = pld.PromptLookupBatchGenerator(
        model, completion_batch_size=1, prefill_step_size=64,
        prompt_lookup={}, stop_tokens=[],
    )
    try:
        gen.insert([[(i % 17) + 2 for i in range(200)]], max_tokens=[120])
        calls = _count_clears(monkeypatch)
        while gen.next()[0]:
            pass  # prefill rounds
        assert calls["n"] >= 3, "PLD prefill never reclaims per chunk"
        calls["n"] = 0
        _drain(gen, 119)
        steps = gen._steps_counter
    finally:
        gen.close()
    assert steps >= 16
    assert calls["n"] >= steps // 16


def test_self_mtp_reclaim_follows_emitted_tokens(monkeypatch):
    # Steps alone would never reach the interval here; tokens do.
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 1 << 40)
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL", 16)
    model = tiny_model()
    gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=64,
                           self_mtp={"num_draft": 2, "persistent": True})
    try:
        gen.insert([list(range(2, 100))], max_tokens=[80], lane_rngs=[LaneRNG(1)],
                   self_mtp_configs=[{"sampling_temp": 0.0}])
        gen.next()  # prefill
        calls = _count_clears(monkeypatch)
        emitted = _drain(gen, 79)
    finally:
        gen.close()
    assert emitted >= 32
    assert calls["n"] >= emitted // 16 - 1
