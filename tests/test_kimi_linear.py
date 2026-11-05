"""Synthetic CPU model and cache oracles; no checkpoint qualification claim."""
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.kimi_linear import KimiLinearAdapter, parse_policy
from mlx2.adapters.kimi_linear_memory import KimiLinearCacheBudget
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.kimi_linear import (
    Gate,
    KimiDeltaAttention,
    Model,
    ModelArgs,
    MultiLatentAttention,
    kda_gates,
    l2_qk,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(118)
    yield
    mx.set_default_device(previous)


def args():
    return ModelArgs(model_type="kimi_linear", vocab_size=64, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=4, max_position_embeddings=1048576,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=72, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=16,
        v_head_dim=16, linear_attn_config={"full_attn_layers": [4], "kda_layers": [1,2,3],
                    "num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4})


def close(a, b, atol=2e-5, rtol=2e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)


def test_kda_l2_and_softplus_match_numpy():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(2,3,2,128)).astype(np.float32)*1e-3
    q, k = l2_qk(mx.array(raw), mx.array(raw))
    expected = raw/np.sqrt((raw*raw).sum(-1,keepdims=True)+1e-6)
    close(k, expected)
    close(q, expected/np.sqrt(128))
    a = np.array([-100, -10, 0, 10, 100], dtype=np.float32).reshape(1,1,1,5)
    g, beta = kda_gates(mx.array(a), mx.zeros((1,1,1)), mx.array([[np.log(2.)]]), mx.zeros((1,5)))
    # The gate returns FP32 log decay; exp() of it is the per-token decay.
    close(g, -2*np.logaddexp(0,a))
    close(mx.exp(g), np.exp(-2*np.logaddexp(0,a)))
    close(beta, np.full((1,1,1),.5))


@pytest.mark.parametrize("batch", [1,2])
def test_grouped_projections_preserve_kda_and_state(batch):
    attn = KimiDeltaAttention(args())
    attn.eval()
    x = mx.random.normal((batch,7,32))
    ca, cb = ArraysCache(size=4), ArraysCache(size=4)
    ref = attn(x, cache=ca)
    attn.prepare_projections()
    attn.fused_projections = True
    got = attn(x, cache=cb)
    close(ref, got)
    for a,b in zip(ca,cb): close(a,b)
    close(attn(x[:,:1], cache=ca), attn(x[:,:1], cache=cb))


def test_kda_prefill_matches_token_recurrence_and_chunk32():
    a = args()
    a = replace(a, linear_attn_config={**a.linear_attn_config, "num_heads": 4})
    attn = KimiDeltaAttention(a)
    attn.eval()
    x = mx.random.normal((1,35,32))
    whole, step, chunk = (ArraysCache(size=4) for _ in range(3))
    expected = attn(x, cache=whole)
    sequential = mx.concatenate([attn(x[:,i:i+1], cache=step) for i in range(35)],axis=1)
    close(expected,sequential)
    attn.chunk32 = True
    close(attn(x,cache=chunk),expected)
    close(chunk[3], whole[3])
    assert attn.calls["chunk32"] == 1
    assert whole[3].shape == step[3].shape == chunk[3].shape == (1,4,128,128)


def test_nope_mla_latent_tiles_match_expanded_with_cached_prefix():
    attn = MultiLatentAttention(args())
    attn.eval()
    attn.key_tile = 3
    x = mx.random.normal((2,7,32))
    ca, cb = KVCache(), KVCache()
    mask = mx.arange(7)[:,None] >= mx.arange(7)[None,:]
    expected = attn(x, mask=mask, cache=ca)
    attn.policy = "latent_tiled"
    close(attn(x, mask=mask, cache=cb), expected)
    attn.policy = "expanded"
    expected = attn(x[:,:1], cache=ca)
    attn.policy = "latent_tiled"
    close(attn(x[:,:1], cache=cb), expected)
    assert ca.keys.shape == cb.keys.shape == (2,1,256,16)
    assert ca.offset == cb.offset == 8


def test_router_preserves_checkpoint_bias_alias_and_fp32_weights():
    gate = Gate(args())
    gate.gate_proj.weight = mx.zeros((4,32))
    gate.expert_bias = mx.array([.1,.2,.3,.4])
    x = mx.zeros((1,1,32),dtype=mx.bfloat16)
    ids, weights = gate(x)
    scores = np.array([.6,.7,.8,.9])[np.array(ids)]
    close(weights,scores/scores.sum(-1,keepdims=True)*2.446)
    assert weights.dtype == mx.float32
    gate.combine = "unbiased"
    _, weights = gate(x)
    close(weights,np.full((1,1,2),2.446/2))


def test_hybrid_model_slice_parity():
    model = Model(args())
    model.eval()
    model.configure(parse_policy({"kda":"reference"}))
    tokens = mx.array([[1,2,3,4,5]])
    a,b = model.make_cache(),model.make_cache()
    whole = model(tokens,cache=a)
    sliced = mx.concatenate([model(tokens[:,:3],cache=b),model(tokens[:,3:],cache=b)],axis=1)
    close(whole,sliced,atol=6e-5)
    assert a[-1].offset == b[-1].offset == 5
    for left,right in zip(a[:-1],b[:-1]):
        close(left[3],right[3])


def test_sanitize_retains_mla_projection_and_maps_every_expert():
    model = Model(args())
    weights = {"model.layers.3.self_attn.kv_b_proj.weight": mx.ones((64,16)),
        "model.layers.0.self_attn.q_conv1d.weight": mx.ones((256,1,4)),
        "model.layers.0.self_attn.A_log": mx.ones((1,1,2,1))}
    for i in range(4):
        for name in ("w1","w2","w3"):
            weights[f"model.layers.1.block_sparse_moe.experts.{i}.{name}.weight"] = mx.full((2,3),i)
    got = model.sanitize(weights)
    assert got["model.layers.3.attention.kv_b_proj.weight"].shape == (64,16)
    assert got["model.layers.0.attention.q_conv1d.conv.weight"].shape == (256,4,1)
    assert got["model.layers.0.attention.A_log"].shape == (2,)
    assert got["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (4,2,3)


def test_policy_rejects_unbound_options():
    for policy in ({"nax":True},{"fused_norm":1},{"mla":"ds41"},{"router_combine":"auto"}):
        with pytest.raises(ValueError): parse_policy(policy)


def test_fused_mla_mines_bounded_glm_kernel_and_declines_cpu():
    from mlx2.runtime.models.kimi_linear_mla_fused import admits, kernel_specs
    partial,combine = kernel_specs()
    assert "shared_q[i] * shared_stage" in partial["source"]
    assert "LAST_ID" not in partial["source"]
    assert "extra_q" in partial["input_names"]
    assert "SPLITS" in combine["source"]
    assert not admits(mx.zeros((1,32,1,512)),mx.zeros((1,32,1,64)),
                      mx.zeros((1,1,7,512)),mx.zeros((1,1,7,64)),None)


def budget(mla="latent_tiled"):
    return KimiLinearCacheBudget(7,20,512,64,32,128,128,128,2304,163840,3,4,mla)


def test_memory_budget_is_compressed_and_session_count_follows_free_memory():
    b = budget()
    assert b.kv_bytes_per_token == 7*576*2
    assert b.recurrent_live_bytes == 20*(32*128*128*4+3*3*32*128*2)
    cost = b.project(1048576)+b.prefill_transient_bytes(1048576,512)
    assert b.full_context_sessions(cost-1) == 0
    assert b.full_context_sessions(cost) == 1
    assert b.full_context_sessions(cost*3) == 3
    assert b.full_context_sessions(cost*3,reserve_bytes=cost) == 2
    assert budget("expanded").prefill_transient_bytes(1048576,512) > b.prefill_transient_bytes(1048576,512)


def test_adapter_context_default_and_operator_override():
    from mlx2.server import apply_adapter_context_default, build_parser
    resolution = SimpleNamespace(adapter_type=KimiLinearAdapter)
    parser = build_parser()
    implicit = parser.parse_args(["--model","/tmp/model"])
    explicit = parser.parse_args(["--model","/tmp/model","--max-context","8192"])
    assert apply_adapter_context_default(implicit,resolution) == 1048576
    assert apply_adapter_context_default(explicit,resolution) == 8192
    other = parser.parse_args(["--model","/tmp/model"])
    assert apply_adapter_context_default(other,SimpleNamespace(adapter_type=object)) == 262144


def test_norm_rounding_is_an_explicit_numerical_policy():
    assert parse_policy()["norm_rounding"] == "fla_fp32"
    a = KimiDeltaAttention(args())
    a.set_dtype(mx.bfloat16)
    a.eval()
    x = mx.random.normal((1,7,32)).astype(mx.bfloat16)
    expected = a(x)
    a.norm_rounding = "qwen_bf16"
    got = a(x)
    assert bool(mx.all(mx.isfinite(got)))
    assert bool(mx.any(expected != got))
    assert a.calls["norm_fla_fp32"] == a.calls["norm_qwen_bf16"] == 1


def test_kda_mla_prometheus_has_bounded_labels_and_no_unknown_events():
    from mlx2.prometheus import PrometheusBuilder, _add_execution
    b = PrometheusBuilder()
    _add_execution(b,{"kda":{"vector_kernel":7,"secret_prompt":99},
                      "mla":{"fused_mla_decode":3,"secret_path":98}})
    text = b.render()
    assert 'mlx2_kda_events_total{event="vector_kernel"} 7' in text
    assert 'mlx2_mla_events_total{event="fused_mla_decode"} 3' in text
    assert 'secret_' not in text


def test_reference_rms_law_rounds_like_kimi_rmsnorm():
    from mlx2.runtime.models.kimi_linear import RMSNorm
    norm = RMSNorm(64, eps=1e-5)
    norm.weight = (mx.random.normal((64,))*0.1+1).astype(mx.bfloat16)
    x = (mx.random.normal((3,64))*4).astype(mx.bfloat16)
    wide = x.astype(mx.float32)
    hf = norm.weight * (wide*mx.rsqrt((wide*wide).mean(-1,keepdims=True)+1e-5)).astype(mx.bfloat16)
    norm.law = "reference"
    got = norm(x)
    assert got.dtype == mx.bfloat16
    close(got.astype(mx.float32), hf.astype(mx.float32), atol=1e-2, rtol=1e-2)
    assert parse_policy({"rms_law": "reference"})["rms_law"] == "reference"
    with pytest.raises(ValueError):
        parse_policy({"rms_law": "bf16"})


def test_latent_single_over_bound_keeps_law_and_receipt():
    attn = MultiLatentAttention(args())
    x = mx.random.normal((1,6,32))
    mask = mx.arange(6)[:,None] >= mx.arange(6)[None,:]
    attn.policy = "latent_single"
    small = attn(x, mask=mask, cache=KVCache())
    attn.single_score_bytes, attn.key_tile = 0, 4
    tiled = attn(x, mask=mask, cache=KVCache())
    close(small, tiled, atol=1e-5, rtol=1e-5)
    assert attn.calls["latent_single"] == 1 and attn.calls["latent_single_tiled"] == 1


def test_auto_mla_splits_prompt_and_decode_phases():
    attn = MultiLatentAttention(args())
    attn.policy, attn.auto_rows = "auto", 4
    cache = KVCache()
    mask = mx.arange(6)[:,None] >= mx.arange(6)[None,:]
    attn(mx.random.normal((1,6,32)), mask=mask, cache=cache)
    attn(mx.random.normal((1,1,32)), cache=cache)
    assert attn.calls["auto_expanded"] == 1 and attn.calls["auto_latent"] == 1
    attn.expanded_bytes = 0
    attn(mx.random.normal((1,6,32)), mask=mx.arange(7,13)[:,None] >= mx.arange(13)[None,:], cache=cache)
    assert attn.calls["auto_latent"] == 2


def test_chunk_wy_matches_recurrence_for_unbounded_gates():
    from mlx2.runtime.models.gated_delta import gated_delta_ops
    from mlx2.runtime.models.kda_chunk_wy import chunk_wy
    b, t, h, d = 2, 37, 2, 32
    q, k = l2_qk(mx.random.normal((b,t,h,d)), mx.random.normal((b,t,h,d)))
    v = mx.random.normal((b,t,h,d))
    log_g = -30*mx.log1p(mx.exp(2*mx.random.normal((b,t,h,d))))  # many near-zero decays
    beta = mx.sigmoid(mx.random.normal((b,t,h)))
    s0 = mx.random.normal((b,h,d,d))*.1
    ref, ref_state = gated_delta_ops(q, k, v, mx.exp(log_g), beta, s0)
    got, state = chunk_wy(q, k, v, log_g, beta, s0, segment=16)
    close(got, ref, atol=1e-5, rtol=1e-4)
    close(state, ref_state, atol=1e-5, rtol=1e-4)
