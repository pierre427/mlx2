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
    assert gate.combine == "unbiased"  # default since 2026-11-02
    gate.combine = "checkpoint"
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


@pytest.mark.parametrize("policy", ["expanded", "latent_tiled", "latent_single", "auto"])
def test_causal_string_mask_matches_explicit_mask_after_prefix(policy):
    attn = MultiLatentAttention(args())
    attn.policy, attn.auto_rows = policy, 2
    prefix, rows = mx.random.normal((1,5,32)), mx.random.normal((1,4,32))
    explicit = mx.arange(5, 9)[:,None] >= mx.arange(9)[None,:]
    outs = []
    for mask in (explicit, "causal"):
        cache = KVCache()
        attn(prefix, mask=mx.arange(5)[:,None] >= mx.arange(5)[None,:], cache=cache)
        outs.append(attn(rows, mask=mask, cache=cache))
    close(outs[1], outs[0], atol=1e-5, rtol=1e-5)


def test_language_model_requests_causal_offset_mode():
    model = Model(args())
    seen = []
    layer = next(l for l in model.layers if l.is_global)
    original = type(layer.attention).__call__
    def spy(self, x, mask=None, cache=None, **kw):
        seen.append(mask)
        return original(self, x, mask=mask, cache=cache, **kw)
    type(layer.attention).__call__ = spy
    try:
        model(mx.array([[1,2,3,4]]), cache=model.make_cache())
    finally:
        type(layer.attention).__call__ = original
    assert seen == ["causal"]


def test_auto_expanded_bound_counts_unfused_scores():
    attn = MultiLatentAttention(args())
    attn.policy, attn.auto_rows = "auto", 2
    # CPU has no fused D256: the bound must include the row x context scores.
    h = attn.args.num_attention_heads
    kv = (attn.args.qk_nope_head_dim+attn.args.qk_rope_head_dim+attn.args.v_head_dim)*4
    attn.expanded_bytes = 6*h*(kv+4*6) - 1
    attn(mx.random.normal((1,6,32)), mask="causal", cache=KVCache())
    assert attn.calls["auto_latent"] == 1 and attn.calls.get("expanded_fused_d256", 0) == 0
    attn.expanded_bytes += 1
    attn(mx.random.normal((1,6,32)), mask="causal", cache=KVCache())
    assert attn.calls["auto_expanded"] == 1


def kimi_mla_args():
    # Served Kimi MLA geometry (32 heads, NoPE 128 + shared 64, v 128,
    # latent 512) on a small hidden size: the only shape the flash admits.
    return replace(args(), num_attention_heads=32, num_key_value_heads=32, kv_lora_rank=512,
                   qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)


def test_mla_flash_policy_switch_is_validated_and_all_by_default():
    assert parse_policy()["mla_flash"] == "all"
    assert parse_policy({"mla_flash": "off"})["mla_flash"] == "off"
    assert parse_policy({"mla": "auto", "mla_flash": "all"})["mla_flash"] == "all"
    for value in ("on", True, "absorbed"):
        with pytest.raises(ValueError):
            parse_policy({"mla_flash": value})


def test_mla_flash_kernels_decline_cpu_and_auto_falls_back():
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    assert not fl.available("expanded") and not fl.available("absorbed")
    attn = MultiLatentAttention(kimi_mla_args())
    attn.set_dtype(mx.bfloat16)
    attn.policy, attn.auto_rows = "auto", 2
    x = mx.random.normal((1,6,32)).astype(mx.bfloat16)
    attn.flash = "off"
    ref = [attn(x, mask="causal", cache=KVCache()), attn(x[:, :1], cache=KVCache())]
    assert "flash_declined" not in attn.calls
    attn.flash = "all"
    got = [attn(x, mask="causal", cache=KVCache()), attn(x[:, :1], cache=KVCache())]
    for r, g in zip(ref, got):
        assert bool(mx.all(r == g))
    # Latent-first (rows <= flash_rows), flash192, then the decode row.
    assert attn.calls["flash_declined"] == 3
    assert "expanded_flash192" not in attn.calls and "latent_flash" not in attn.calls


def test_mla_flash_geometry_refuses_what_the_kernels_cannot_do():
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    q = mx.zeros((1,4,32*192), mx.bfloat16)
    kv = mx.zeros((1,9,32*256), mx.bfloat16)
    sh = mx.zeros((1,1,9,64), mx.bfloat16)
    assert fl.expanded_geometry(q, kv, sh, "causal") == (1, 4, 9, 5, True)
    assert fl.expanded_geometry(q, kv, sh, None) == (1, 4, 9, 0, False)
    explicit = mx.arange(5, 9)[:, None] >= mx.arange(9)[None, :]
    bad = [(q, kv, sh, explicit), (q.astype(mx.float32), kv, sh, "causal"),
           (q[..., :-1], kv, sh, "causal"), (q, kv, sh[:, :, :3], "causal"),
           (mx.zeros((1,12,32*192), mx.bfloat16), kv, sh, "causal")]
    for operands in bad:
        with pytest.raises(ValueError):
            fl.expanded_geometry(*operands)
    qa, qs = mx.zeros((1,32,4,512), mx.bfloat16), mx.zeros((1,32,4,64), mx.bfloat16)
    lat = mx.zeros((1,1,9,512), mx.bfloat16)
    assert fl.absorbed_geometry(qa, qs, lat, sh, "causal") == (1, 4, 9, 5, True)
    with pytest.raises(ValueError):
        fl.absorbed_geometry(qa, qs, lat, sh, explicit)
    with pytest.raises(ValueError):
        fl.absorbed_geometry(qa[:, :16], qs[:, :16], lat, sh, None)
    # Splits of at least 2048 keys (2026-11-02 sweep): partial traffic stays
    # small while a lone decode row still spreads over 64 threadgroups.
    assert fl.split_plan(1, 1, 131072) == (2048, 64)
    assert fl.split_plan(1, 512, 131072) == (131072, 1)


def test_mla_flash_wiring_matches_ordinary_through_the_kernel_laws(monkeypatch):
    # CPU stands in for the GPU kernels with their FP32 ops laws: checks the
    # sub-route selection, the operand layouts and the V un-absorption.
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    calls = []
    def law(name, fn):
        def run(*operands, **options):
            calls.append(name)
            options.pop("split", None)
            result = fn(*operands, **options)
            return result if isinstance(result, tuple) else result.astype(operands[0].dtype)
        return run
    monkeypatch.setattr(fl, "available", lambda kind, dtype=None: True)
    monkeypatch.setattr(fl, "expanded", law("expanded", fl.expanded_reference))
    monkeypatch.setattr(fl, "absorbed", law("absorbed", fl.absorbed_reference))
    attn = MultiLatentAttention(kimi_mla_args())
    attn.set_dtype(mx.bfloat16)
    attn.auto_rows, attn.flash, attn.flash_rows = 2, "all", 2
    prefix = mx.random.normal((1,5,32)).astype(mx.bfloat16)
    rows = mx.random.normal((1,4,32)).astype(mx.bfloat16)
    explicit = mx.arange(5, 9)[:, None] >= mx.arange(9)[None, :]
    def run(policy, x, mask):
        cache = KVCache()
        attn.policy = "expanded"
        attn(prefix, mask=mx.arange(5)[:, None] >= mx.arange(5)[None, :], cache=cache)
        attn.policy = policy
        return attn(x, mask=mask, cache=cache)
    ref_wide = ref = run("expanded", rows, explicit)
    got = run("auto", rows, "causal")
    assert calls == ["expanded"] and attn.calls["expanded_flash192"] == 1
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)
    ref = run("latent_single", rows[:, :1], None)
    got = run("auto", rows[:, :1], None)
    assert calls[-1] == "absorbed" and attn.calls["latent_flash"] == 1
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)
    # Over the expanded transient bound a wide chunk streams key windows.
    attn.expanded_bytes = 0
    got = run("auto", rows, "causal")
    assert calls[-1] == "expanded" and attn.calls["expanded_flash192_windows"] == 1
    close(got.astype(mx.float32), ref_wide.astype(mx.float32), atol=2e-2, rtol=2e-2)
    # Several windows, some hidden from early rows, merge to the one-pass law.
    lat = mx.random.normal((1,1,9,512)).astype(mx.bfloat16)
    sh = mx.random.normal((1,1,9,64)).astype(mx.bfloat16)
    q = mx.random.normal((1,4,32*192)).astype(mx.bfloat16)
    one = fl.expanded_reference(q, attn.kv_b_proj(lat[:, 0]), sh, "causal", scale=attn.scale)
    for window in (2, 3, 9):
        got = fl.expanded_windows(q, lat, sh, attn.kv_b_proj, "causal", scale=attn.scale, window=window)
        close(got.astype(mx.float32), one.astype(mx.float32), atol=2e-2, rtol=2e-2)
    # An explicit visibility array (padded batch) is never admitted.
    attn.expanded_bytes = 4 << 30
    before = len(calls)
    run("auto", rows, explicit)
    assert len(calls) == before and attn.calls["flash_declined"] == 1
    attn.flash = "expanded"
    run("auto", rows[:, :1], None)
    assert len(calls) == before
    # flash="all" keeps chunks up to flash_rows on the latent flash.
    attn.flash, attn.flash_rows = "all", 4
    got = run("auto", rows, "causal")
    assert calls[-1] == "absorbed" and attn.calls["latent_flash"] == 2
    close(got.astype(mx.float32), ref_wide.astype(mx.float32), atol=2e-2, rtol=2e-2)
    # ...unless it declines (padded-batch array mask): flash192 or D256 then.
    before = len(calls)
    run("auto", rows, explicit)
    assert len(calls) == before


def test_mla_flash_counters_reach_prometheus():
    from mlx2.prometheus import PrometheusBuilder, _add_execution
    b = PrometheusBuilder()
    _add_execution(b, {"mla": {"expanded_flash192": 4, "latent_flash": 2, "flash_declined": 1}})
    text = b.render()
    for event, count in (("expanded_flash192", 4), ("latent_flash", 2), ("flash_declined", 1)):
        assert f'mlx2_mla_events_total{{event="{event}"}} {count}' in text


def _mlx_lm_names(params):
    """Our parameter names spelled the way mlx-lm conversions store them."""
    out = {}
    for key, value in params.items():
        key = key.replace("model.word_embeddings.", "model.embed_tokens.").replace(".attention.", ".self_attn.")
        key = key.replace(".mlp.gate.expert_bias", ".mlp.e_score_correction_bias")
        key = key.replace(".mlp.gate.gate_proj.", ".mlp.gate.")
        for name in ("q", "k", "v"):
            key = key.replace(f".self_attn.{name}_conv1d.conv.", f".self_attn.{name}_conv.conv.")
        out[key] = value
    return out


def test_mlx_lm_quantized_layout_round_trips_and_absorbs():
    from mlx import nn
    from mlx.utils import tree_flatten

    def quantize(model):
        nn.quantize(model, group_size=32, bits=8, class_predicate=lambda name, m:
                    hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0)
        return model
    tiny = replace(args(), kv_lora_rank=32)  # group-aligned, so kv_b quantizes
    source = quantize(Model(tiny))
    source.model.layers[1].mlp.gate.expert_bias = mx.random.normal((4,))*0.1
    mx.eval(source.parameters())
    weights = _mlx_lm_names(dict(tree_flatten(source.parameters())))
    assert any(k.endswith(".mlp.gate.scales") for k in weights)
    target = quantize(Model(tiny))
    target.load_weights(list(target.sanitize(weights).items()), strict=True)
    ids = mx.array([[3, 9, 1, 7, 2]])
    close(target(ids), source(ids), atol=1e-6, rtol=1e-6)
    # Quantized router and absorbed latent MLA read a once-dequantized matrix.
    from mlx2.runtime.models.kimi_linear import dense_weight
    gate = target.model.layers[1].mlp.gate
    assert isinstance(gate.gate_proj, nn.QuantizedLinear)
    assert dense_weight(gate.gate_proj, mx.float32) is dense_weight(gate.gate_proj, mx.float32)
    mla = next(l.attention for l in target.model.layers if l.is_global)
    assert isinstance(mla.kv_b_proj, nn.QuantizedLinear)
    x = mx.random.normal((1, 5, 32))
    mask = mx.arange(5)[:, None] >= mx.arange(5)[None, :]
    mla.policy = "expanded"
    ref = mla(x, mask=mask, cache=KVCache())
    mla.policy = "latent_single"
    close(mla(x, mask=mask, cache=KVCache()), ref, atol=1e-4, rtol=1e-4)


def test_artifact_layouts_and_quantized_policy_refusals():
    from mlx2.adapters.kimi_linear import LAYOUTS, validate_config
    base = {"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"],
            "hidden_size": 2304, "vocab_size": 163840, "num_hidden_layers": 27,
            "num_experts_per_token": 8, "num_shared_experts": 1, "num_expert_group": 1,
            "topk_group": 1, "first_k_dense_replace": 1, "moe_layer_freq": 1,
            "num_attention_heads": 32, "num_key_value_heads": 32, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "intermediate_size": 9216, "moe_intermediate_size": 1024, "q_lora_rank": None,
            "mla_use_nope": True, "model_max_length": 1048576, "tie_word_embeddings": False,
            "hidden_act": "silu", "moe_router_activation_func": "sigmoid", "moe_renormalize": True,
            "routed_scaling_factor": 2.446, "rms_norm_eps": 1e-5, "num_nextn_predict_layers": 0,
            "linear_attn_config": {"full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
                "kda_layers": [i for i in range(1, 28) if i not in (4, 8, 12, 16, 20, 24, 27)],
                "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4}}
    assert validate_config({**base, "num_experts": 180}) == "hub_bf16"
    q8 = {"group_size": 64, "bits": 8, "mode": "affine", "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8}}
    assert validate_config({**base, "num_experts": 256, "quantization": q8}) == "mlx_affine"
    for bad in ({**base, "num_experts": 256},  # unquantized 48B is unstaged
                {**base, "num_experts": 180, "quantization": q8},
                {**base, "num_experts": 256, "quantization": {**q8, "bits": 4}},
                {**base, "num_experts": 256, "quantization": {**q8, "model.layers.1.mlp.gate": {"group_size": 32, "bits": 8}}}):
        with pytest.raises(ValueError):
            validate_config(bad)
    assert {v["variant"] for v in LAYOUTS.values()} == {"reap35b-bf16-ordinary", "48b-mlx-q8-ordinary"}
    assert parse_policy({"moe_nax": "gather"})["moe_nax"] == "gather"


def test_mla_flash_routes_take_quantized_kv_b(monkeypatch):
    # The 48B MLX 8-bit artifact quantizes kv_b_proj: flash192 expands
    # through the quantized module, the latent flash absorbs dense_weight().
    from mlx import nn
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    calls = []
    def law(name, fn):
        def run(*operands, **options):
            calls.append((name, operands[1].ndim))
            options.pop("split", None)
            return fn(*operands, **options).astype(operands[0].dtype)
        return run
    monkeypatch.setattr(fl, "available", lambda kind, dtype=None: True)
    monkeypatch.setattr(fl, "expanded", law("expanded", fl.expanded_reference))
    monkeypatch.setattr(fl, "absorbed", law("absorbed", fl.absorbed_reference))
    attn = MultiLatentAttention(kimi_mla_args())
    attn.set_dtype(mx.bfloat16)
    nn.quantize(attn, group_size=64, bits=8, class_predicate=lambda p, m: p == "kv_b_proj")
    assert isinstance(attn.kv_b_proj, nn.QuantizedLinear)
    attn.auto_rows, attn.flash, attn.flash_rows = 2, "all", 2
    x = mx.random.normal((1,6,32)).astype(mx.bfloat16)
    explicit = mx.arange(6)[:, None] >= mx.arange(6)[None, :]
    attn.policy = "expanded"
    ref = attn(x, mask=explicit, cache=KVCache())
    attn.policy = "auto"
    got = attn(x, mask="causal", cache=KVCache())
    assert calls == [("expanded", 3)]  # token-major, from the quantized product
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)
    attn.policy = "latent_single"
    ref = attn(x[:, :1], cache=KVCache())
    attn.policy = "auto"
    got = attn(x[:, :1], cache=KVCache())
    assert calls[-1][0] == "absorbed" and attn.calls["latent_flash"] == 1
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)


def test_mla_flash_reads_batched_kvcache_buffers_in_place(monkeypatch):
    # B > 1: the kernels get the whole KVCache buffers plus the visible
    # width (no slice copy); results match the sliced ordinary laws.
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    seen = []
    def law(name, fn):
        def run(*operands, **options):
            seen.append((name, operands[2].shape[2], options.get("width")))
            options.pop("split", None)
            return fn(*operands, **options).astype(operands[0].dtype)
        return run
    monkeypatch.setattr(fl, "available", lambda kind, dtype=None: True)
    monkeypatch.setattr(fl, "expanded", law("expanded", fl.expanded_reference))
    monkeypatch.setattr(fl, "absorbed", law("absorbed", fl.absorbed_reference))
    attn = MultiLatentAttention(kimi_mla_args())
    attn.set_dtype(mx.bfloat16)
    attn.auto_rows, attn.flash, attn.flash_rows = 2, "all", 2
    prefix = mx.random.normal((2,5,32)).astype(mx.bfloat16)
    rows = mx.random.normal((2,4,32)).astype(mx.bfloat16)
    explicit = mx.arange(5, 9)[:, None] >= mx.arange(9)[None, :]
    def run(policy, x, mask):
        cache = KVCache()
        attn.policy = "expanded"
        attn(prefix, mask=mx.arange(5)[:, None] >= mx.arange(5)[None, :], cache=cache)
        attn.policy = policy
        return attn(x, mask=mask, cache=cache), cache
    ref, _ = run("expanded", rows, explicit)
    got, cache = run("auto", rows, "causal")
    assert cache.keys.shape[2] > cache.offset == 9
    assert seen[-1][0] == "expanded"
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)
    ref, _ = run("latent_single", rows[:, :1], None)
    got, cache = run("auto", rows[:, :1], None)
    assert seen[-1] == ("absorbed", cache.keys.shape[2], 6)
    close(got.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2)


def test_default_policy_is_auto_mla_with_reference_paths_selectable():
    policy = parse_policy()
    assert policy["mla"] == "auto" and policy["kda"] == "kernel"
    assert policy["router_combine"] == "unbiased" and policy["mla_flash"] == "all"
    assert parse_policy({"router_combine": "checkpoint"})["router_combine"] == "checkpoint"
    for mla in ("expanded", "latent_tiled", "latent_single", "latent_fused"):
        assert parse_policy({"mla": mla})["mla"] == mla
    assert KimiLinearCacheBudget.from_config.__kwdefaults__["mla"] == "auto"


def test_every_kimi_counter_reaches_prometheus():
    """A counter the model can emit but /metrics drops is a silent receipt."""
    import re
    from pathlib import Path

    import mlx2.prometheus as prom
    root = Path(prom.__file__).parent/"runtime"/"models"
    source = "".join(p.read_text() for p in root.glob("kimi_linear*.py"))
    emitted = set(re.findall(r'calls\["([a-z0-9_]+)"\]', source))
    for pair in re.findall(r'calls\["([a-z0-9_]+)" if [a-z_]+ else "([a-z0-9_]+)"\]', source):
        emitted |= set(pair)
    assert {"auto_expanded", "auto_latent"} <= emitted
    emitted |= {f"{shape}_{kind}" for shape in ("multirow", "single_row") for kind in ("calls", "tokens")}
    emitted |= {"norm_fla_fp32", "norm_qwen_bf16"}
    emitted -= {"router_calls", "router_tokens"}  # exported through mlx2_moe_*
    text = Path(prom.__file__).read_text()
    allow = text[text.index("mechanism_events = {"):text.index("for mechanism,events in mechanism_events.items()")]
    missing = sorted(e for e in emitted if f'"{e}"' not in allow)
    assert not missing, missing


def test_kimi_execution_policy_artifact_and_moe_render():
    from mlx2.prometheus import PrometheusBuilder, _add_execution
    execution = {"policy": parse_policy({"mla_flash": "all"}),
                 "artifact": {"layout": "mlx_affine", "variant": "48b-mlx-q8-ordinary",
                              "checkpoint": "mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit@553a7a56"},
                 "moe": {"router_calls": 5, "router_tokens": 40, "sort": {"auto_sorted": 3, "auto_unsorted": 2}},
                 "moe_nax_gather": {"calls": {"gather": 4}, "fallbacks": {"shape": 1},
                                    "not_candidates": 2, "not_prefill": 7},
                 "kda": {"vector_kernel": 3}, "mla": {"latent_flash": 2, "expanded_flash192": 1}}
    builder = PrometheusBuilder()
    _add_execution(builder, execution)
    text = builder.render()
    for line in ('mlx2_execution_policy_info{key="mla",value="auto"} 1',
                 'mlx2_execution_policy_info{key="mla_flash",value="all"} 1',
                 'mlx2_execution_policy_info{key="fused_norm",value="false"} 1',
                 'layout="mlx_affine"', 'mlx2_moe_router_calls_total 5', 'mlx2_moe_router_tokens_total 40',
                 'mlx2_moe_sort_decisions_total{decision="auto_sorted"} 3',
                 'mlx2_moe_nax_gather_calls_total{route="gather"} 4',
                 'mlx2_moe_nax_gather_fallbacks_total{reason="shape"} 1',
                 'mlx2_moe_nax_gather_not_prefill_total 7',
                 'mlx2_mla_events_total{event="latent_flash"} 2', 'mlx2_kda_events_total{event="vector_kernel"} 3'):
        assert line in text, line


def test_prompt_chunk_projects_one_checkpoint_row_through_the_head():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    model = Model(args())
    ids = mx.array([[3, 9, 1, 7, 2, 5]])
    full = model(ids, cache=model.make_cache())
    with prefill_checkpoint_rows([6], width=6) as request:
        narrowed = model(ids, cache=model.make_cache())
    assert request.applied == 1 and narrowed.shape == (1, 1, 64)
    close(narrowed[:, 0], full[:, -1], atol=1e-5, rtol=1e-5)
    assert model(ids, cache=model.make_cache()).shape == (1, 6, 64)  # identity outside a request


def test_prefill_defaults_follow_flash_next_and_keep_plan_exact():
    from mlx2.runtime.prefill_plan import depth_bounded_prefill_rows
    adapter = KimiLinearAdapter.__new__(KimiLinearAdapter)
    assert adapter.prefill_step_default() == 8192
    budget = adapter.prefill_depth_budget_default()
    assert depth_bounded_prefill_rows(8192, 131072, budget) == 8192
    assert depth_bounded_prefill_rows(8192, 1048576, budget) == 1024
    assert adapter.execution_config(max_lanes=2, prefill_step=8192)["prefill_plan_exact"] is True


def test_prefill_transient_charges_the_head_once_per_chunk():
    b = budget("auto")
    # Per-row FP32 logits alone would be 8192 x 163840 x 4 bytes (5 GiB).
    per_row_logits = 8192*163840*4
    assert b.prefill_transient_bytes(0, 8192) < per_row_logits
    # One more row costs its activations only, never another vocabulary row.
    assert (b.prefill_transient_bytes(0, 8192) - b.prefill_transient_bytes(0, 8191)
            == (16*b.heads*b.head_dim + 8*b.hidden_size)*4)


def test_unpadded_batch_decode_passes_no_mask_and_padded_keeps_it():
    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.kimi_linear import _proves_unpadded
    assert _proves_unpadded(BatchKVCache([0, 0]))
    assert not _proves_unpadded(BatchKVCache([1, 0]))
    assert not _proves_unpadded(KVCache())
    from mlx2.runtime.pld import _set_ordinary_b1_mask
    lane = KVCache()
    _set_ordinary_b1_mask([lane], True)
    assert _proves_unpadded(lane)
    _set_ordinary_b1_mask([lane], False)
    assert not _proves_unpadded(lane)
    model = Model(args())
    seen = []
    layer = next(l for l in model.layers if l.is_global)
    original = type(layer.attention).__call__
    from mlx2.runtime.models.kimi_linear import LeftPaddedCausal
    starts = []
    def spy(self, x, mask=None, cache=None, **kw):
        if isinstance(mask, LeftPaddedCausal):
            seen.append("padded")
            starts.append((mask.starts is cache.left_padding, mask.starts.tolist()))
        else:
            seen.append(mask if mask is None or isinstance(mask, str) else "array")
        return original(self, x, mask=mask, cache=cache, **kw)
    type(layer.attention).__call__ = spy
    try:
        for padding in ([0], [1, 0]):
            caches = model.make_cache()
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    caches[i] = BatchKVCache(padding)
            b = len(padding)
            model(mx.array([[1, 2, 3, 4]]*b), cache=caches)
            model(mx.array([[5]]*b), cache=caches)
    finally:
        type(layer.attention).__call__ = original
    # A padded batch cache's own mask reaches MLA with that cache's starts.
    assert seen == ["causal", None, "padded", "padded"]
    assert starts == [(True, [1, 0]), (True, [1, 0])]


def test_padded_mask_reaches_non_flash_laws_unchanged(monkeypatch):
    # On CPU (no flash) every law still receives the padded array itself.
    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.kimi_linear import LeftPaddedCausal
    attn = MultiLatentAttention(args())
    attn.eval()
    attn.policy = "auto"
    seen = []
    original = MultiLatentAttention.attend
    def spy(self, q, latent, shared, mask, itemsize, **kw):
        seen.append((type(mask).__name__, kw.get("starts")))
        return original(self, q, latent, shared, mask, itemsize, **kw)
    monkeypatch.setattr(MultiLatentAttention, "attend", spy)
    cache = BatchKVCache([2, 0])
    x = mx.random.normal((2, 4, 32))
    array = cache.make_mask(4)
    ref = MultiLatentAttention.__call__(attn, x, mask=array, cache=BatchKVCache([2, 0]))
    got = attn(x, mask=LeftPaddedCausal(array, cache.left_padding), cache=cache)
    assert np.array_equal(np.array(got), np.array(ref))
    assert seen[0] == ("array", None) and seen[1][0] == "array"
    assert seen[1][1].tolist() == [2, 0]
    assert attn.calls["latent_flash_padded"] == 0


def test_absorbed_reference_starts_equal_the_padded_batch_mask():
    # The kernel's FP32 reference with lane starts is the batch cache's
    # create_causal_mask(N, offset=_idx, left_padding) law.
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    from mlx2.runtime.models.base import create_causal_mask
    rng = np.random.default_rng(1103)
    for rows, width, pads in ((1, 40, [0, 7, 39]), (5, 40, [0, 12, 38])):
        b = len(pads)
        q = mx.array(rng.normal(0, .05, (b, fl.HEADS, rows, fl.LATENT)).astype(np.float32)).astype(mx.bfloat16)
        qs = mx.array(rng.normal(0, .5, (b, fl.HEADS, rows, fl.SHARED)).astype(np.float32)).astype(mx.bfloat16)
        lat = mx.array(rng.normal(0, 1, (b, 1, width, fl.LATENT)).astype(np.float32)).astype(mx.bfloat16)
        sh = mx.array(rng.normal(0, 1, (b, 1, width, fl.SHARED)).astype(np.float32)).astype(mx.bfloat16)
        starts = mx.array(pads, mx.int32)
        got = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=.07, starts=starts)
        mask = create_causal_mask(rows, offset=width-rows, left_padding=starts)
        score = (q.astype(mx.float32) @ lat.astype(mx.float32).swapaxes(-1, -2)
                 + qs.astype(mx.float32) @ sh.astype(mx.float32).swapaxes(-1, -2)) * .07
        ref = fl._safe_softmax(mx.where(mask, score, -float("inf"))) @ lat.astype(mx.float32)
        assert np.array_equal(np.array(got), np.array(ref))
        # A fully padded row (start past its causal limit) is zeros.
        if rows > 1:
            assert not np.any(np.array(got)[2, :, :38-(width-rows)])
            assert np.any(np.array(got)[2, :, 38-(width-rows)])
    with pytest.raises(ValueError, match="one int per batch row"):
        fl._starts(mx.array([0], mx.int32), 2)


def test_latent_tiled_counter_counts_only_the_tiled_loop(monkeypatch):
    from mlx2.runtime.models import kimi_linear_mla_fused as fused

    attn = MultiLatentAttention(args())
    attn.eval()
    x = mx.random.normal((1,5,32))
    mask = mx.arange(5)[:,None] >= mx.arange(5)[None,:]
    attn.policy = "latent_tiled"
    attn(x, mask=mask, cache=KVCache())
    assert attn.calls["latent_tiled"] == 1
    monkeypatch.setattr(fused, "admits", lambda *a: True)
    monkeypatch.setattr(fused, "attend", lambda qn, qs, latent, shared: mx.zeros((*qn.shape[:-1], 16)))
    attn.policy = "latent_fused"
    attn(x[:, :1], cache=KVCache())
    assert attn.calls["fused_mla_decode"] == 1 and attn.calls["latent_tiled"] == 1
    monkeypatch.setattr(fused, "admits", lambda *a: False)
    attn(x[:, :1], cache=KVCache())
    assert attn.calls["fused_mla_declined"] == 1 and attn.calls["latent_tiled"] == 2


def test_mla_forward_keeps_no_query_block_on_the_module():
    attn = MultiLatentAttention(args())
    attn.eval()
    attn(mx.random.normal((1,6,32)), mask="causal", cache=KVCache())
    assert "_q_rows" not in attn.__dict__
    q, latent, shared = attn.project(mx.random.normal((1,2,32)))
    assert q.shape == (1,2,2,24) and latent.shape == (1,1,2,16) and shared.shape == (1,1,2,8)


def test_absorbed_flash_refuses_a_head_count_the_kernel_does_not_tile():
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    qa, qs = mx.zeros((1,16,4,512), mx.bfloat16), mx.zeros((1,16,4,64), mx.bfloat16)
    lat, sh = mx.zeros((1,1,9,512), mx.bfloat16), mx.zeros((1,1,9,64), mx.bfloat16)
    with pytest.raises(ValueError, match="geometry"):
        fl.absorbed_geometry(qa, qs, lat, sh, "causal", heads=16)


def test_dense_weight_cache_follows_scales_and_biases():
    from mlx import nn

    from mlx2.runtime.models.kimi_linear import dense_weight

    linear = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=8)
    first = dense_weight(linear)
    linear.scales = linear.scales * 2
    close(dense_weight(linear), mx.dequantize(linear.weight, linear.scales, linear.biases,
                                              group_size=64, bits=8))
    assert float(mx.abs(dense_weight(linear) - first).max()) > 0
    linear.biases = linear.biases + 1
    close(dense_weight(linear), mx.dequantize(linear.weight, linear.scales, linear.biases,
                                              group_size=64, bits=8))


def _latent_operands(b, prefix, rows):
    attn = MultiLatentAttention(args())
    attn.eval()
    q, latent, shared = attn.project(mx.random.normal((b, prefix+rows, 32)))
    return attn, q[:, :, prefix:], latent, shared


LATENT_LAWS = [("latent_tiled", None, "latent_tiled"),
               ("latent_single", None, "latent_single"),
               ("latent_single", 0, "latent_single_tiled"),
               ("latent_fused", None, "fused_mla_declined")]


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("policy,single_bytes,counter", LATENT_LAWS)
def test_latent_causal_string_is_bit_identical_to_the_explicit_mask(b, policy, single_bytes, counter):
    # 70 rows (not a multiple of the 64-row query tile) after a 13-token
    # prefix; key_tile 17 does not divide the 83-key width.
    prefix, rows = 13, 70
    attn, q, latent, shared = _latent_operands(b, prefix, rows)
    attn.key_tile = 17
    if single_bytes is not None:
        attn.single_score_bytes = single_bytes
    width = prefix + rows
    explicit = mx.arange(width-rows, width)[:, None] >= mx.arange(width)[None, :]
    ref = attn._latent(q, latent, shared, explicit, policy)
    calls = dict(attn.calls)
    got = attn._latent(q, latent, shared, "causal", policy)
    assert attn.calls[counter] == 2 * calls[counter] == 2
    assert got.shape == ref.shape == (b, 2, rows, 16)
    assert np.array_equal(np.array(got), np.array(ref))


@pytest.mark.parametrize("policy,single_bytes,counter", [law for law in LATENT_LAWS
                                                          if law[2] != "latent_single"])
def test_latent_tiled_causal_never_materializes_the_rows_by_context_mask(
        monkeypatch, policy, single_bytes, counter):
    prefix, rows = 300, 70
    attn, q, latent, shared = _latent_operands(1, prefix, rows)
    attn.key_tile = 17
    if single_bytes is not None:
        attn.single_score_bytes = single_bytes
    lengths, arange = [], mx.arange

    def spy(*a, **kw):
        out = arange(*a, **kw)
        lengths.append(out.size)
        return out

    monkeypatch.setattr(mx, "arange", spy)
    mx.eval(attn._latent(q, latent, shared, "causal", policy))
    assert attn.calls[counter] == 1
    # Positions per block only: <= 64 query rows and <= key_tile keys, never
    # the width-long key axis of a [rows, prefix + rows] mask.
    assert lengths and max(lengths) <= 64


def test_latent_estimates_charge_no_whole_chunk_mask():
    # The latent laws build causal visibility per block, so explicit latent
    # routes and auto charge what they did before the 2026-11-02 sweep
    # (estimate at 2fa374ae9^): no [rows, context] mask for prompt chunks.
    cells = ((1048576, 16384), (131072, 8192), (1048576, 512), (1048576, 8192))
    before = {"auto": [22683451392, 12415795200, 17352491008, 19931987968],
              "latent_tiled": [14143848448, 3876192256, 8812888064, 11392385024],
              "latent_fused": [14143848448, 3876192256, 8812888064, 11392385024],
              "latent_single": [14898823168, 4631166976, 9567862784, 12147359744]}
    for mla, values in before.items():
        assert [budget(mla).prefill_transient_bytes(*cell) for cell in cells] == values
    # latent_single's bounded one-pass mask is charged where it exceeds the
    # 128-row term (short contexts): 512 x 4096 is 2 MiB, not 512 KiB. Over
    # latent_tiled (whose tile scratch is flat here), 512 rows add that mask
    # and three more score blocks.
    single, tiled = budget("latent_single"), budget("latent_tiled")
    assert (single.prefill_transient_bytes(4096, 512) - single.prefill_transient_bytes(4096, 128)
            - (tiled.prefill_transient_bytes(4096, 512) - tiled.prefill_transient_bytes(4096, 128))
            == ((2 << 20) - 4096*128) + 3*(512-128)*4096*32*4)


@pytest.mark.parametrize("m5", [True, False])
@pytest.mark.parametrize("mla_flash", ["all", "expanded", "off"])
def test_planning_budget_agrees_with_the_loaded_budget_on_every_host(monkeypatch, m5, mla_flash):
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    monkeypatch.setattr(fl, "_AVAILABLE", {})
    monkeypatch.setattr(fl, "_device_is_m5", lambda: m5)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr("mlx2.adapters.kimi_linear.validate_config", lambda config: None)
    config = {"linear_attn_config": {"full_attn_layers": list(range(7)), "kda_layers": list(range(20)),
                                     "num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4},
              "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
              "v_head_dim": 128, "hidden_size": 2304, "vocab_size": 163840}
    adapter = object.__new__(KimiLinearAdapter)
    adapter.config = config
    adapter.policy = parse_policy({"mla_flash": mla_flash})
    loaded = adapter.cache_budget(mtp=False)
    planned = KimiLinearAdapter.planning_cache_budget(config, mtp=False)
    assert loaded == planned == budget("auto")
    for cell in ((1048576, 16384), (131072, 8192), (4096, 512)):
        assert loaded.prefill_transient_bytes(*cell) == planned.prefill_transient_bytes(*cell)
    assert loaded.project(1048576) == planned.project(1048576)


@pytest.mark.parametrize("plan_exact", [True, False])
def test_kimi_prefill_never_hands_mla_a_padded_multi_row_mask(plan_exact):
    """Pins the estimate's premise: Kimi's plan-exact law prefills one
    unpadded prompt per forward, so multi-row MLA rows see "causal"; the
    unlawful two-prompt batch (control) is what would build an array."""
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    adapter = KimiLinearAdapter.__new__(KimiLinearAdapter)
    assert adapter.execution_config(max_lanes=2, prefill_step=8192)["prefill_plan_exact"] is True
    model = Model(args())
    model.eval()
    layer = next(l for l in model.layers if l.is_global)
    seen, original = [], type(layer.attention).__call__

    def spy(self, x, mask=None, cache=None, **kw):
        if x.shape[1] > 1:
            seen.append(mask if mask is None or isinstance(mask, str) else "array")
        return original(self, x, mask=mask, cache=cache, **kw)

    type(layer.attention).__call__ = spy
    gen = BatchGenerator(model, prefill_step_size=8, completion_batch_size=2,
                         prefill_batch_size=2, prefill_plan_exact=plan_exact)
    try:
        assert gen.prefill_batch_size == (1 if plan_exact else 2)
        # Warm prefixes of different lengths: batching these two prompts in
        # one forward is what left-pads a BatchKVCache.
        caches, histories = [], ([3, 5, 7, 9, 11], [2, 4])
        for history in histories:
            cache = model.make_cache()
            mx.eval(model(mx.array([history]), cache=cache))
            caches.append(cache)
        tails = [[13, 15, 17, 19, 21, 23], [6, 8, 10, 12, 14, 16]]
        gen.insert(tails, max_tokens=[2, 2], caches=caches,
                   all_tokens=[h + t for h, t in zip(histories, tails)],
                   lane_rngs=[LaneRNG(1), LaneRNG(2)])
        finished = set()
        for _ in range(200):
            _, responses = gen.next()
            finished.update(r.uid for r in responses if r.finish_reason)
            if len(finished) == 2:
                break
        assert len(finished) == 2
    finally:
        gen.close()
        type(layer.attention).__call__ = original
    assert seen
    if plan_exact:
        assert set(seen) == {"causal"}
    else:
        assert "array" in seen


def test_plain_segmented_row_masks_are_exactly_bottom_right_causal():
    # The per-lane masks row_views yields for a plain segmented batch equal
    # bottom-right causal over each lane's own keys, so Kimi MLA passes
    # "causal" (admitting latent flash) without changing the law.
    from mlx2.runtime.models.base import create_causal_mask
    from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache
    for bases, lengths in (([5, 0, 9], [3, 1, 2]), ([4, 4], [1, 1]), ([7, 2, 3, 6], [4, 4, 0, 2])):
        rows = []
        for base in bases:
            row = KVCache()
            if base:
                row.update_and_fetch(mx.zeros((1, 1, base, 4)), mx.zeros((1, 1, base, 4)))
            rows.append(row)
        view = SegmentedBatchKVCache(rows)
        width = max(lengths)
        view.prepare(lengths=lengths, right_padding=[width-n for n in lengths])
        prepared = view.make_mask(width)
        view.update_and_fetch(mx.zeros((len(bases), 1, width, 4)), mx.zeros((len(bases), 1, width, 4)))
        for mask in (prepared, None):
            for index, valid, keys, _, row_mask in view.row_views(mask):
                if not valid:
                    continue
                base = bases[index]
                assert keys.shape[2] == base + valid
                causal = create_causal_mask(valid, offset=base)
                assert row_mask.shape[-2:] == causal.shape[-2:] == (valid, base + valid)
                assert np.array_equal(np.broadcast_to(np.array(row_mask), (1, 1, valid, base + valid)),
                                      np.broadcast_to(np.array(causal), (1, 1, valid, base + valid)))
        view.finalize()


def test_segmented_verify_rows_attend_with_causal_and_unchanged_results(monkeypatch):
    from mlx2.runtime.models import kimi_linear as kl
    from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache
    attn = MultiLatentAttention(args())
    attn.eval()
    prefixes = [mx.random.normal((1, 6, 32)), mx.random.normal((1, 3, 32))]
    x = mx.random.normal((2, 2, 32))

    from mlx2.runtime.models.kimi_linear import LanguageModel, SegmentedCausal

    def step(policy, window=None, trusted=True):
        attn.policy = policy
        rows = [KVCache(), KVCache()]
        for row, prefix in zip(rows, prefixes):
            attn(prefix, mask="causal", cache=row)
        view = SegmentedBatchKVCache(rows)
        view.prepare(lengths=[2, 1], right_padding=[0, 1])
        if window is not None:
            mask = view.make_mask(2, window_size=window)
        elif trusted:
            mask = LanguageModel.attention_mask(mx.zeros((2, 2, 32)), view)
            assert isinstance(mask, SegmentedCausal)
        else:
            mask = view.make_mask(2)
        out = attn(x, mask=mask, cache=view)
        view.finalize()
        return out

    masks = []
    original = MultiLatentAttention.attend
    def spy(self, q, latent, shared, mask, itemsize, **kw):
        masks.append(mask if isinstance(mask, str) else "array")
        return original(self, q, latent, shared, mask, itemsize, **kw)
    monkeypatch.setattr(MultiLatentAttention, "attend", spy)
    for policy in ("latent_single", "latent_tiled", "auto"):
        masks.clear()
        got = step(policy)
        assert masks[-2:] == ["causal", "causal"]
        # The previous behaviour: each lane's explicit row mask.
        with monkeypatch.context() as patch:
            patch.setattr(kl, "SegmentedBatchKVCache", type("Other", (), {}))
            masks.clear()
            ref = step(policy, trusted=False)
            assert masks[-2:] == ["array", "array"]
        assert np.array_equal(np.array(got[0]), np.array(ref[0]))
        assert np.array_equal(np.array(got[1, :1]), np.array(ref[1, :1]))
        # An explicit array of unknown provenance, or a windowed mask, keeps
        # each lane's own row mask (a window restricts more than causal).
        for kwargs in ({"trusted": False}, {"window": 1}):
            masks.clear()
            step(policy, **kwargs)
            assert masks[-2:] == ["array", "array"]
