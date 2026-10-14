"""A non-finite fp16 GDN state never enters APCv2, and a lane it fails names it.

The fp16 storage class computes a token's readout from the fp32 register
state and rounds only the stored copy.  An overflow on a prefill chunk's last
token therefore leaves finite logits: the prefill checkpoint guard
(``generate._validate_prefill_checkpoint_output``) passes while the cache
holds ``inf``.  Before the store-side check that checkpoint was published,
and every request restoring it failed on its first token.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2 as apc_mod
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.generate import (
    _invalid_output_reason,
    _name_nonfinite_states,
    _validate_prefill_checkpoint_output,
)
from mlx2.runtime.models import gdn_state, qwen3_5
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.recurrent_state_codec import (
    NonfiniteRecurrentState,
    validate_prompt_cache,
)

KEY = APCKey(
    "artifact-a", revision="source-a", adapter="artifact-a",
    tokenizer_fingerprint="tokenizer-a", cache_layout_fingerprint="layout-a",
    semantic_fingerprint=("text-token-v1", "tenant", "tenant-a"),
)


def _kv(n):
    cache = KVCache()
    values = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _gdn(value, dtype=mx.float16, rows=1):
    cache = ArraysCache(2)
    cache[0] = mx.zeros((rows, 3, 8), mx.bfloat16)
    cache[1] = mx.full((rows, 2, 4, 4), value).astype(dtype)
    return cache


def test_validate_refuses_a_nonfinite_fp16_state_and_names_its_entry():
    validate_prompt_cache([_kv(4), _gdn(1.0)], None)
    with pytest.raises(NonfiniteRecurrentState) as raised:
        validate_prompt_cache([_kv(4), _gdn(1.0), _gdn(mx.inf)], None)
    assert raised.value.entries == (2,)
    assert isinstance(raised.value, ValueError)


def test_validate_checks_recorded_checkpoint_snapshots_too():
    cache = _gdn(1.0)
    cache._checkpoints = [[(3, [cache[0], mx.full((1, 2, 4, 4), mx.nan).astype(mx.float16)])]]
    with pytest.raises(NonfiniteRecurrentState):
        validate_prompt_cache([cache], None)


def test_the_fp32_class_gets_no_extra_device_read():
    # Its readout comes from the stored state, so the prefill guard covers it.
    validate_prompt_cache([_gdn(mx.inf, dtype=mx.float32)], None)


def test_apcv2_store_refuses_and_counts_a_nonfinite_fp16_state():
    apc = APCv2(max_size=4, layout_name="layout-a")
    try:
        tokens = list(range(4))
        stored = apc.store(KEY, tokens, [_kv(4), _gdn(mx.inf)])
        assert stored.stored is False
        assert apc.apc_stats["nonfinite_state_rejections"] == 1
        assert not apc.lookup(KEY, tokens + [9]).hit
        assert apc.store(KEY, tokens, [_kv(4), _gdn(1.0)]).stored is not False
        assert apc.apc_stats["nonfinite_state_rejections"] == 1
    finally:
        apc.close()


def test_a_persisted_nonfinite_state_is_dropped_on_disk_restore(tmp_path, monkeypatch):
    """An entry written before the store refused one never comes back."""
    tag = ("tenant-a", "conversation-1")
    apc = APCv2(
        max_size=8, max_bytes=1 << 20, layout_name="layout-a",
        idle_disk_seconds=180, idle_disk_dir=str(tmp_path), persist_dir=str(tmp_path),
        persist_identity=KEY, persist_semantic_namespace="tenant",
    )
    try:
        tokens = list(range(4))
        with monkeypatch.context() as patch:
            patch.setattr(apc_mod, "validate_prompt_cache", lambda *_: 0)
            apc.store(KEY, tokens, [_kv(4), _gdn(mx.inf)], session_tag=tag)
        assert apc.park_session(*tag, ttl_seconds=600)["state"] == "disk"
        assert not apc.lookup(KEY, tokens + [9]).hit
        assert apc._disk_stats["restore_nonfinite_state"] == 1
    finally:
        apc.close()


def _tiny_fp16_model(seed):
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=128,
    )
    mx.random.seed(seed)
    model = TextModel(args)
    model.eval()
    gdn_state.install_state_dtype(model, "float16")
    return model


def _chunk_after_near_limit_state(model, scale):
    """Prefill 3 tokens, push every GDN state to ``scale``, run a 1-token chunk."""
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    for entry in cache:
        state = entry[1] if isinstance(entry, ArraysCache) else None
        if isinstance(state, mx.array) and state.ndim == 4:
            entry[1] = (mx.sign(state.astype(mx.float32)) * scale).astype(mx.float16)
    assert gdn_state.nonfinite_state_layers(cache) == []
    logits = model(mx.array([[4]]), cache=cache)
    mx.eval(logits)
    return cache, logits


def _last_token_overflow():
    for seed in range(40):
        model = _tiny_fp16_model(seed)
        for scale in (64000.0, 65000.0, 65400.0):
            cache, logits = _chunk_after_near_limit_state(model, scale)
            if gdn_state.nonfinite_state_layers(cache):
                return model, cache, logits
    pytest.fail("no last-token fp16 overflow found in the search range")


def test_last_token_overflow_passes_the_prefill_guard_but_not_the_store():
    model, cache, logits = _last_token_overflow()
    bad = gdn_state.nonfinite_state_layers(cache)
    # The gap: the chunk's logits are finite, so the guard authorizes the
    # checkpoint even though the stored state is inf.
    _validate_prefill_checkpoint_output(logits, [1], [7])
    apc = APCv2(max_size=4, layout_name="layout-a")
    try:
        stored = apc.store(KEY, [1, 2, 3, 4], cache)
        assert stored.stored is False
        assert apc.apc_stats["nonfinite_state_rejections"] == 1
    finally:
        apc.close()
    # What a restore would have hit: the next token fails, and its receipt
    # now names the overflowed layers.
    after = model(mx.array([[5]]), cache=cache)[0, -1]
    lp = after - mx.logsumexp(after)
    reason = _invalid_output_reason(int(mx.argmax(lp).item()), lp)
    assert reason is not None and "non-finite log probability" in reason
    # The next token's inf reaches the GDN layers above the overflowed one.
    spread = gdn_state.nonfinite_state_layers(cache)
    assert set(bad) <= set(spread)
    failures = [{"uid": 7, "reason": reason}]
    _name_nonfinite_states(cache, [7], failures)
    assert failures[0]["reason"].endswith(
        f"; non-finite recurrent state in cache layers {spread}"
    )


def test_a_clean_chunk_on_the_same_route_is_stored():
    # Control arm: the store refuses the overflow, not the fp16 route.
    model = _tiny_fp16_model(5)
    cache, logits = _chunk_after_near_limit_state(model, 1.0)
    _validate_prefill_checkpoint_output(logits, [1], [7])
    apc = APCv2(max_size=4, layout_name="layout-a")
    try:
        assert apc.store(KEY, [1, 2, 3, 4], cache).stored is not False
        assert apc.apc_stats["nonfinite_state_rejections"] == 0
    finally:
        apc.close()


def test_state_rows_name_only_the_failed_lane():
    clean, poisoned = _gdn(1.0, rows=2), _gdn(1.0, rows=2)
    poisoned[1] = mx.concatenate(
        [poisoned[1][:1], mx.full((1, 2, 4, 4), mx.inf).astype(mx.float16)]
    )
    caches = [_kv(4), clean, poisoned]
    assert gdn_state.nonfinite_state_rows(caches, [0, 1]) == {1: [2]}
    failures = [
        {"uid": 10, "reason": "sampled token 3 has non-finite log probability"},
        {"uid": 11, "reason": "sampled token 3 has non-finite log probability"},
        {"uid": 12, "reason": "sampled token 999 is outside vocabulary of size 8"},
    ]
    rows = [_gdn(1.0, rows=3), poisoned]
    _name_nonfinite_states(rows, [10, 11, 12], failures)
    assert failures[0]["reason"] == "sampled token 3 has non-finite log probability"
    assert failures[1]["reason"].endswith("non-finite recurrent state in cache layers [1]")
    assert "recurrent state" not in failures[2]["reason"]
