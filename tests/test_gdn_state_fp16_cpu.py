"""fp16 GDN recurrent-state storage class: wiring, numerics law, key separation.

CPU only (the ops reference path).  The Metal kernels' bit equality with the
composed reference is gated in tests/test_gdn_state_fp16_metal.py.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.flash_next_memory import FlashNextCacheBudget
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models import gdn_state, qwen3_5
from mlx2.runtime.models import qwen4_fused_gdn as fused
from mlx2.runtime.models import qwen4_fused_gdn_prefill as fused_prefill
from mlx2.runtime.models import qwen4_fused_gdn_verify as fused_verify
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.gated_delta import gated_delta_ops, gated_delta_update


def _bits(x):
    return np.asarray(x.view(mx.uint16 if x.dtype.size == 2 else mx.uint32))


def _layer(seed=5):
    args = qwen3_5.TextModelArgs(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    mx.random.seed(seed)
    layer = qwen3_5.GatedDeltaNet(args)
    mx.eval(layer.parameters())
    return layer


def _recurrence(T=6, B=1, Hk=2, Hv=4, Dk=16, Dv=8, seed=3):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 5)
    q = mx.random.normal((B, T, Hk, Dk), key=ks[0]) * 0.3
    k = mx.random.normal((B, T, Hk, Dk), key=ks[1]) * 0.3
    v = mx.random.normal((B, T, Hv, Dv), key=ks[2])
    a = mx.random.normal((B, T, Hv), key=ks[3])
    b = mx.random.normal((B, T, Hv), key=ks[4])
    return q, k, v, a, b


def _update(q, k, v, a, b, state, state_dtype):
    A_log = mx.zeros((v.shape[2],))
    dt_bias = mx.zeros((v.shape[2],))
    return gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=False,
        state_dtype=state_dtype,
    )


def test_fresh_state_takes_the_selected_class_and_stays_in_it():
    q, k, v, a, b = _recurrence()
    _, state32 = _update(q, k, v, a, b, None, None)
    _, state16 = _update(q, k, v, a, b, None, mx.float16)
    assert state32.dtype == mx.float32
    assert state16.dtype == mx.float16
    # One more step keeps the class.
    _, nxt = _update(q[:, :1], k[:, :1], v[:, :1], a[:, :1], b[:, :1], state16, mx.float16)
    assert nxt.dtype == mx.float16


def test_fp16_law_rounds_every_token_so_grouping_does_not_matter():
    """Chunked, token-by-token and one-launch runs store identical fp16 bits."""
    q, k, v, a, b = _recurrence(T=7)
    y_all, s_all = _update(q, k, v, a, b, None, mx.float16)
    state, ys = None, []
    for t in range(7):
        y, state = _update(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                           a[:, t:t + 1], b[:, t:t + 1], state, mx.float16)
        ys.append(y)
    y_step = mx.concatenate(ys, axis=1)
    _, s_mid = _update(q[:, :3], k[:, :3], v[:, :3], a[:, :3], b[:, :3], None, mx.float16)
    _, s_split = _update(q[:, 3:], k[:, 3:], v[:, 3:], a[:, 3:], b[:, 3:], s_mid, mx.float16)
    mx.eval(y_all, s_all, y_step, state, s_split)
    assert np.array_equal(_bits(s_all), _bits(state))
    assert np.array_equal(_bits(s_all), _bits(s_split))
    assert np.array_equal(_bits(y_all), _bits(y_step))


def test_fp16_law_is_fp32_math_with_an_fp16_store_per_token():
    """Each step: widen the stored state, run the fp32 step, round to fp16."""
    q, k, v, a, b = _recurrence(T=5)
    _, s16 = _update(q, k, v, a, b, None, mx.float16)
    state = mx.zeros(s16.shape, dtype=mx.float32)
    for t in range(5):
        _, state = _update(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                           a[:, t:t + 1], b[:, t:t + 1], state, None)
        state = state.astype(mx.float16).astype(mx.float32)
    mx.eval(s16, state)
    assert np.array_equal(_bits(s16), _bits(state.astype(mx.float16)))
    # A different numerics class from fp32: rounding once at the end differs.
    _, s32 = _update(q, k, v, a, b, None, None)
    assert not np.array_equal(_bits(s32.astype(mx.float16)), _bits(s16))


def test_classes_never_mix():
    q, k, v, a, b = _recurrence(T=2)
    s16 = mx.zeros((1, 4, 8, 16), dtype=mx.float16)
    s32 = mx.zeros((1, 4, 8, 16), dtype=mx.float32)
    with pytest.raises(ValueError, match="never mix"):
        _update(q, k, v, a, b, s16, None)  # layer did not select fp16
    with pytest.raises(ValueError, match="never mix"):
        _update(q, k, v, a, b, s32, mx.float16)  # fp32 state into an fp16 layer


def test_install_binds_layers_and_the_ordinary_path_honours_it():
    layer = _layer()
    with pytest.raises(ValueError):
        gdn_state.install_state_dtype(layer, "bfloat16")
    assert gdn_state.install_state_dtype(layer, "float32") == {
        "state_dtype": "float32", "layers": 0,
    }
    receipt = gdn_state.install_state_dtype(layer, "float16")
    assert receipt["layers"] == 1 and receipt["qualification"] == "unqualified"
    x = mx.random.normal((1, 5, 32), key=mx.random.key(1))
    cache = ArraysCache(2)
    before = gdn_state.STATS["ops_calls"]
    layer(x, cache=cache)
    assert cache[1].dtype == mx.float16
    assert gdn_state.STATS["ops_calls"] > before  # mechanism counter engaged
    # Prefill split 2 + 3 == one 5-token prefill, bit for bit.
    split = ArraysCache(2)
    layer(x[:, :2], cache=split)
    layer(x[:, 2:], cache=split)
    mx.eval(cache[1], split[1])
    assert np.array_equal(_bits(cache[1]), _bits(split[1]))


def test_rollback_restores_the_sequential_fp16_state():
    layer = _layer(seed=11)
    gdn_state.install_state_dtype(layer, "float16")
    prefix = mx.random.normal((1, 3, 32), key=mx.random.key(2))
    block = mx.random.normal((1, 4, 32), key=mx.random.key(3))
    cache = ArraysCache(2)
    layer(prefix, cache=cache)
    cache.start_speculation()
    layer(block, cache=cache)
    cache.trim(2)  # keep 2 of 4 verify tokens
    cold = ArraysCache(2)
    layer(mx.concatenate([prefix, block[:, :2]], axis=1), cache=cold)
    mx.eval(cache[1], cold[1])
    assert cache[1].dtype == mx.float16
    assert np.array_equal(_bits(cache[1]), _bits(cold[1]))


def test_fp16_overflow_is_inf_never_a_silent_clamp_and_is_located():
    state = mx.full((1, 1, 1, 4), 70000.0).astype(mx.float16)
    assert bool(mx.all(mx.isinf(state)).item())

    class _C(list):
        pass

    caches = [_C([None, mx.zeros((1, 1, 1, 4), mx.float16)]), _C([None, state]), object()]
    assert gdn_state.nonfinite_state_layers(caches) == [1]


def test_an_overflowed_state_fails_the_lane_closed():
    """inf in an fp16 state reaches the per-step log-probability guard."""
    from mlx2.runtime.generate import _invalid_output_reason
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=128,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    receipt = gdn_state.install_state_dtype(model, "float16")
    assert receipt["layers"] == 3
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]]), cache=cache)
    gdn = [c for c in cache if getattr(c, "cache", None) and getattr(c.cache[1], "ndim", 0) == 4]
    assert len(gdn) == 3 and all(c.cache[1].dtype == mx.float16 for c in gdn)
    clean = model(mx.array([[5]]), cache=cache)[0, -1]
    lp = clean - mx.logsumexp(clean)
    assert _invalid_output_reason(int(mx.argmax(lp).item()), lp) is None
    gdn[1].cache[1] = mx.full(gdn[1].cache[1].shape, mx.inf).astype(mx.float16)
    assert gdn_state.nonfinite_state_layers(cache) == [cache.index(gdn[1])]
    bad = model(mx.array([[6]]), cache=cache)[0, -1]
    lp = bad - mx.logsumexp(bad)
    reason = _invalid_output_reason(int(mx.argmax(lp).item()), lp)
    assert reason is not None and "non-finite" in reason


def test_fused_admissions_accept_fp16_and_refuse_other_classes():
    def decode(state_dtype):
        return fused._admit_decode_operands(
            rows=1,
            qkv=mx.zeros((1, 1, fused.CONV_DIM), mx.bfloat16),
            z=mx.zeros((1, 1, fused.VALUE_DIM), mx.bfloat16),
            b=mx.zeros((1, 1, fused.NUM_VALUE_HEADS), mx.bfloat16),
            a=mx.zeros((1, 1, fused.NUM_VALUE_HEADS), mx.bfloat16),
            conv_state=mx.zeros((1, 3, fused.CONV_DIM), mx.bfloat16),
            recurrent_state=mx.zeros((1, 48, 128, 128), state_dtype),
            conv_weight=mx.zeros((fused.CONV_DIM, 4, 1), mx.bfloat16),
            A_log=mx.zeros((48,), mx.float32),
            dt_bias=mx.zeros((48,), mx.bfloat16),
            norm_weight=mx.zeros((128,), mx.bfloat16),
            num_key_heads=16, num_value_heads=48, key_head_dim=128,
            value_head_dim=128, conv_kernel=4, gate_activation="sigmoid",
            architecture="qwen4",
        )

    assert decode(mx.float32).accepted and decode(mx.float16).accepted
    assert decode(mx.bfloat16).reason == "recurrent_state must be float32 or float16"


def test_probe_keys_accept_the_default_none_state_dtype(monkeypatch):
    # mx.Dtype == None raises TypeError; the fp32 default must never reach it.
    assert fused_verify._probe_key(3, None) == 3
    assert fused_verify._probe_key(3, mx.float32) == 3
    assert fused_verify._probe_key(3, mx.float16) == (3, "float16")
    monkeypatch.setattr(fused, "_PROBE_COMPLETE", True)
    monkeypatch.setattr(fused, "_PROBED_THREADGROUP_Y", 16)
    monkeypatch.setattr(fused, "_probe_st16_decode", lambda dtype: 8)
    assert fused.probe_qwen4_fused_gdn_decode(mx.bfloat16) == 16
    assert fused.probe_qwen4_fused_gdn_decode(mx.bfloat16, state_dtype=None) == 16
    assert fused.probe_qwen4_fused_gdn_decode(mx.bfloat16, state_dtype=mx.float16) == 8


def test_fp16_sources_are_derived_and_fp32_sources_untouched():
    for source in (fused._SOURCE, fused_verify._SOURCE, fused_verify._REPLAY_SOURCE,
                   fused_verify._CATCHUP_SOURCE, fused_verify._RECONSTRUCT_SOURCE,
                   fused_verify._RECONSTRUCT_DYNAMIC_SOURCE):
        assert "half" not in source
    for source in (fused._SOURCE_ST16, fused._SOURCE_BATCH_ST16, fused_verify._SOURCE_ST16,
                   fused_verify._REPLAY_SOURCE_ST16, fused_verify._CATCHUP_SOURCE_ST16):
        assert source.count("st[j][i] = float(static_cast<half>(st[j][i]));") == 1
        assert "device const float* si" not in source
    for source in (fused_verify._RECONSTRUCT_SOURCE_ST16,
                   fused_verify._RECONSTRUCT_DYNAMIC_SOURCE_ST16):
        assert source.count("st = float(static_cast<half>(st));") == 1
    # The snapshot verify stores its per-token restore points as half too.
    assert "device half* state_dst" in fused_verify._SOURCE_ST16


def test_policy_field_layout_and_receipts():
    assert "gdn_state_dtype" not in FlashNextPolicy().as_dict()
    on = FlashNextPolicy.from_mapping({"gdn_state_dtype": "float16"})
    assert on.as_dict()["gdn_state_dtype"] == "float16"
    assert "MLX" not in "".join(k for k in on.environment() if "GDN_STATE" in k)
    assert on.environment() == FlashNextPolicy().environment()  # not an env switch
    with pytest.raises(ValueError):
        FlashNextPolicy.from_mapping({"gdn_state_dtype": "bfloat16"})
    base = "qwen4-exp-layer-segments-v1"
    assert gdn_state.layout_with_state_dtype(base, "float32") == base
    assert gdn_state.layout_with_state_dtype(base, "float16") == base + ":gdn-state-fp16-v1"


def test_apc_keys_of_the_two_classes_never_cross():
    def key(layout):
        return APCv2.key(
            "fp", revision="r", adapter="fp", tokenizer_fingerprint="fp",
            cache_layout_fingerprint=layout, semantic_fingerprint="s",
        )

    base = "qwen38-27b-hybrid-layer-segments-v1"
    fp32 = key(gdn_state.layout_with_state_dtype(base, "float32"))
    fp16 = key(gdn_state.layout_with_state_dtype(base, "float16"))
    assert fp32 == key(base)  # default namespace unchanged
    assert fp16 != fp32


def test_adapters_validate_the_policy_before_loading():
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    for cls in (Qwen3635BA3BAdapter, Qwen3827BAdapter):
        with pytest.raises(ValueError, match="gdn_state_dtype"):
            cls("/nonexistent-model", execution_policy={"gdn_state_dtype": "bf16"})


def test_budgets_halve_only_the_recurrent_state():
    flash = {
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_hidden_layers": 4, "mtp_num_hidden_layers": 1,
        "num_key_value_heads": 2, "head_dim": 128, "indexer_head_dim": 64,
        "indexer_compress_ratio": 4, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 16, "linear_conv_kernel_dim": 4,
        "ple_layer_ids": [], "ple_embed_dim": 0, "ple_conv_kernel_size": 0,
    }
    b32 = FlashNextCacheBudget.from_config(flash, mtp=False)
    b16 = FlashNextCacheBudget.from_config(flash, mtp=False, recurrent_state_bytes=2)
    state = 3 * 48 * 128 * 128
    assert b32.fixed_bytes - b16.fixed_bytes == 2 * state * 2  # live + rollback
    assert "recurrent_state_bytes" not in b32.as_dict()
    assert b16.as_dict()["recurrent_state_bytes"] == 2
    qwen = {
        "num_hidden_layers": 4, "full_attention_interval": 4,
        "num_key_value_heads": 2, "head_dim": 128,
        "linear_num_value_heads": 48, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    q32 = Qwen38CacheBudget.from_config(qwen, mtp=False)
    q16 = Qwen38CacheBudget.from_config(qwen, mtp=False, recurrent_state_bytes=2)
    assert q32.recurrent_live_bytes - q16.recurrent_live_bytes == 2 * state
    assert "recurrent_state_bytes" not in q32.as_dict()


def test_prefill_admission_accepts_fp16_state():
    ok = fused_prefill.admit_qwen4_fused_gdn_prefill(
        qkv=mx.zeros((1, 64, fused.CONV_DIM), mx.bfloat16),
        z=mx.zeros((1, 64, fused.VALUE_DIM), mx.bfloat16),
        b=mx.zeros((1, 64, 48), mx.bfloat16),
        a=mx.zeros((1, 64, 48), mx.bfloat16),
        conv_state=None,
        recurrent_state=mx.zeros((1, 48, 128, 128), mx.float16),
        conv_weight=mx.zeros((fused.CONV_DIM, 4, 1), mx.bfloat16),
        norm_weight=mx.zeros((128,), mx.bfloat16),
        mask=None, has_cache=True, lengths=None, left_padding=None,
        speculating=False, training=False, sharded=False,
        num_key_heads=16, num_value_heads=48, key_head_dim=128,
        value_head_dim=128, conv_kernel=4, gate_activation="sigmoid",
    )
    assert ok.accepted


def test_ops_reference_keeps_fp32_widening_for_the_readout():
    q, k, v, a, b = _recurrence(T=3)
    g = mx.exp(-mx.ones((1, 3, 4)) * 0.1)
    beta = mx.sigmoid(b)
    y16, _ = gated_delta_ops(q.astype(mx.float16), k.astype(mx.float16),
                             v.astype(mx.float16), g, beta,
                             mx.zeros((1, 4, 8, 16), mx.float16))
    y32, _ = gated_delta_ops(q.astype(mx.float16), k.astype(mx.float16),
                             v.astype(mx.float16), g, beta,
                             mx.zeros((1, 4, 8, 16), mx.float32))
    assert y16.dtype == y32.dtype == mx.float16
