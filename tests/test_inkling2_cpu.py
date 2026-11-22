"""Inkling 2 (inkling2_loop) native model and adapter, CPU only.

The model math was checked bit-exact (fp32, CPU) against mlx-vlm@111ac668 on
this tiny geometry by scripts/research/inkling2_mlxvlm_oracle.py; these tests
pin self-consistency, the checkpoint name map, the expert formats and the
fail-closed gates without the upstream tree.
"""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.models.inkling2 import Model, TextConfig


TINY = dict(
    model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=24, vocab_size=512,
    unpadded_vocab_size=500, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
    sliding_window_size=8, rel_extent=8, d_rel=16,
    local_layer_ids=[i for i in range(24) if (i + 1) % 6],
    intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2, log_scaling_n_floor=4,
    loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=8),
)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _tiny_model(seed=0):
    mx.random.seed(seed)
    model = Model(TextConfig.from_dict(TINY))
    params = tree_flatten(model.parameters())
    weights = []
    for key, value in params:
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        weights.append((key, noise + (1.0 if key.endswith("norm.weight") else 0.0)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _run(model, tokens, plan):
    cache = model.make_cache()
    return mx.concatenate([model(tokens[:, a:b], cache=cache) for a, b in plan], axis=1)


def test_loop_visits_and_cache_planes():
    config = TextConfig.from_dict(TINY)
    assert config.layer_indices == (
        tuple(range(6)) + tuple(range(6, 18)) * 3 + tuple(range(18, 24))
    )
    model = Model(config)
    cache = model.make_cache()
    assert len(cache) == 2 * 48
    assert [c.apcv2_layer_index for c in cache[1::2]] == list(range(48))
    kinds = [type(c).__name__ for c in cache[0::2]]
    assert kinds[5] == kinds[11] == kinds[47] == "KVCache"
    assert kinds[0] == "RotatingKVCache"
    with pytest.raises(ValueError, match="per layer visit"):
        model(mx.zeros((1, 2), mx.int32), cache=cache[:-2])


@pytest.mark.parametrize("bad", [dict(n_shared_experts=1), dict(q_bias=True),
                                 dict(loop_config=dict(core_start=5, core_end=18,
                                                       num_passes=3, gate_rank=8))])
def test_config_refuses_unimplemented_topology(bad):
    with pytest.raises(ValueError):
        TextConfig.from_dict({**TINY, **bad})


def test_decode_and_chunked_prefill_match_one_prefill():
    """Ring wrap (window 8 < 21 tokens) and log scaling (floor 4) engaged."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 21))
    base = _run(model, tokens, [(0, 21)])
    assert base.shape == (1, 21, 500)
    decode = _run(model, tokens, [(0, 16)] + [(i, i + 1) for i in range(16, 21)])
    chunks = _run(model, tokens, [(0, 7), (7, 14), (14, 21)])
    assert mx.abs(decode - base).max().item() < 1e-4
    assert mx.abs(chunks - base).max().item() < 1e-4


def test_second_loop_pass_reads_messages():
    """Zeroing the message controllers' gains changes the output: the
    previous-pass bank really feeds passes two and three."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 6))
    base = model(tokens)
    for layer in model.model.layers[6:18]:
        for controller in layer.controllers.values():
            controller.gain_weight = mx.zeros_like(controller.gain_weight)
            controller.forget_weight = mx.zeros_like(controller.forget_weight)
    assert mx.abs(model(tokens) - base).max().item() > 1e-3


def test_logits_are_fp32_softcapped_and_cropped():
    model = _tiny_model()
    logits = model(mx.random.randint(0, 500, (1, 3)))
    assert logits.dtype == mx.float32 and logits.shape[-1] == 500
    assert mx.abs(logits).max().item() <= 30.0


def test_sanitize_maps_checkpoint_names():
    model = Model(TextConfig.from_dict(TINY))
    w = TINY["hidden_size"]
    raw = {
        "model.llm.embed.weight": mx.zeros((512, w)),
        "model.llm.unembed.weight": mx.zeros((512, w)),
        "model.llm.layers.0.attn.wq_du.weight": mx.zeros((128, w)),
        "model.llm.layers.0.attn.k_sconv.weight": mx.zeros((64, 1, 4)),
        "model.llm.layers.5.attn.k_sconv.static_w": mx.zeros((4, 64)),
        "model.llm.layers.5.attn.k_sconv.weight_proj": mx.zeros((8, w)),
        "model.llm.layers.0.attn_norm.weight": mx.zeros((w,)),
        "model.llm.layers.5.q_input_norm.weight": mx.zeros((w,)),
        "model.llm.layers.0.mlp.gate.bias": mx.zeros((8,)),
        "model.llm.layers.0.mlp.experts.w2_weight": mx.zeros((8, w, 32)),
        "model.llm.attn_res.attn_queries.4": mx.zeros((3, w)),
        "model.llm.attn_res.mlp_queries.0": mx.zeros((w,)),
        "model.llm.attn_res.output_query": mx.zeros((w,)),
        "model.llm.middle_cycle_cross_pass_queries": mx.arange(24 * w).reshape(24, w),
        "model.llm.middle_cycle_cross_pass_kv_queries": -mx.arange(24 * w).reshape(24, w),
    }
    out, experts = model.sanitize(raw)
    assert set(out) == {
        "model.embed_tokens.weight", "lm_head.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_sconv.conv.weight",
        "model.layers.5.self_attn.k_sconv.static_weight",
        "model.layers.5.self_attn.k_sconv.weight_proj.weight",
        "model.layers.0.input_layernorm.weight", "model.layers.5.q_input_layernorm.weight",
        "model.layers.0.mlp.e_score_correction_bias", "model.layers.5.attention_input.query",
        "model.layers.0.mlp_input.query", "model.final_input.query", "model.message_queries",
    }
    assert out["model.layers.0.self_attn.k_sconv.conv.weight"].shape == (64, 4, 1)
    assert set(experts) == {0} and set(experts[0]) == {"w2_weight"}
    queries = out["model.message_queries"]
    assert queries.shape == (12, 4, w)
    # (attn, k, v, mlp) per core layer from the two interleaved banks.
    assert queries[0, 0, 0].item() == 0 and queries[0, 3, 0].item() == w
    assert queries[0, 1, 0].item() == 0 and queries[0, 2, 1].item() == -(w + 1)
    with pytest.raises(ValueError, match="unexpected"):
        model.sanitize({"model.visual.x": mx.zeros((1,))})


def _modelopt_nvfp4(dense, num):
    """Fake ModelOpt NVFP4 experts: MLX nvfp4 bytes plus amax/denominator."""
    weight, scales = mx.quantize(dense, group_size=16, bits=4, mode="nvfp4")
    amax = mx.linspace(1.0, 2.0, num)
    return {
        "weight": weight.view(mx.uint8),
        "scale": scales,
        "amax": amax,
        "denominator": mx.array([4.0]),
        "global": amax / 4.0,
        "deq": mx.dequantize(weight, scales, group_size=16, bits=4, mode="nvfp4"),
    }


def test_install_experts_nvfp4_and_mxfp8_layouts():
    from mlx2.adapters.inkling2 import MXFP8, NVFP4, install_experts

    model = Model(TextConfig.from_dict(TINY))
    num, hidden, inter = 8, 128, 64
    mx.random.seed(3)
    w13 = mx.random.normal((num, 2 * inter, hidden))
    w2 = mx.random.normal((num, hidden, inter))
    a, b = _modelopt_nvfp4(w13, num), _modelopt_nvfp4(w2, num)
    nv = {
        "w13_weight": a["weight"], "w13_weight_scale": a["scale"], "w13_weight_amax": a["amax"],
        "w13_weight_outer_scale_denominator": a["denominator"],
        "w2_weight": b["weight"], "w2_weight_scale": b["scale"], "w2_weight_amax": b["amax"],
        "w2_weight_outer_scale_denominator": b["denominator"],
    }
    q13, s13 = mx.quantize(w13, group_size=32, bits=8, mode="mxfp8")
    q2, s2 = mx.quantize(w2, group_size=32, bits=8, mode="mxfp8")
    fp8 = {"w13_weight": q13.view(mx.uint8), "w13_weight_scale": s13,
           "w2_weight": q2.view(mx.uint8), "w2_weight_scale": s2}
    receipt = install_experts(model, {0: nv, 20: fp8}, {0: NVFP4, 20: MXFP8})
    assert receipt == {"nvfp4_layers": 1, "mxfp8_layers": 1}
    mlp = model.model.layers[0].mlp.switch_mlp
    assert type(mlp.gate_proj).__name__ == "ScaledNVFP4SwitchLinear"
    gate = mx.dequantize(mlp.gate_proj.weight, mlp.gate_proj.scales, group_size=16, bits=4,
                         mode="nvfp4")
    up = mx.dequantize(mlp.up_proj.weight, mlp.up_proj.scales, group_size=16, bits=4,
                       mode="nvfp4")
    assert mx.array_equal(gate, a["deq"][:, 0::2]) and mx.array_equal(up, a["deq"][:, 1::2])
    assert mx.allclose(mlp.gate_proj.weight_scale_2, a["global"])
    assert mx.allclose(mlp.down_proj.weight_scale_2, b["global"])
    fp8_mlp = model.model.layers[20].mlp.switch_mlp
    assert fp8_mlp.up_proj.mode == "mxfp8" and fp8_mlp.up_proj.bits == 8
    deq = mx.dequantize(fp8_mlp.up_proj.weight, fp8_mlp.up_proj.scales, group_size=32, bits=8,
                        mode="mxfp8")
    full = mx.dequantize(q13, s13, group_size=32, bits=8, mode="mxfp8")
    assert mx.array_equal(deq, full[:, 1::2])


def _artifact(tmp_path, *, text_overrides=None, drop=None):
    text = {
        "model_type": "inkling2_loop_text", "hidden_size": 2304, "num_hidden_layers": 24,
        "vocab_size": 201024, "unpadded_vocab_size": 200058, "num_attention_heads": 24,
        "num_key_value_heads": 8, "head_dim": 128, "swa_num_attention_heads": 24,
        "swa_num_key_value_heads": 8, "swa_head_dim": 128, "sliding_window_size": 1024,
        "d_rel": 16, "rel_extent": 1024, "sconv_kernel_size": 4, "n_routed_experts": 128,
        "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 1152,
        "use_gate_bias": True, "q_bias": False, "o_bias": False,
        "use_value_norm_no_gamma": True, "attn_res_normalize_values": True,
        "n_layers_per_attn_res_block": 6, "model_max_length": 1048576,
        "local_layer_ids": [i for i in range(24) if (i + 1) % 6],
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
                        "is_varloop": False},
        **(text_overrides or {}),
    }
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "inkling2_loop", "eos_token_id": 200006, "text_config": text}))
    names = ["model.llm.embed.weight", "model.llm.unembed.weight", "model.llm.norm.weight",
             "model.llm.attn_res.output_query", "model.llm.middle_cycle_cross_pass_queries",
             "model.llm.middle_cycle_cross_pass_kv_queries"]
    groups = {"nvfp4_nearest": [], "mxfp8_nearest": []}
    for i in range(24):
        p = f"model.llm.layers.{i}."
        names += [p + s for s in ("attn.wq_du.weight", "attn.attn_sink_proj.weight",
                                  "attn.rel_logits_proj.proj", "mlp.gate.weight",
                                  "mlp.gate.bias", "attn_sconv.weight", "mlp_sconv.weight")]
        kind = "nvfp4_nearest" if i < 20 else "mxfp8_nearest"
        for n in ("w13_weight", "w2_weight"):
            names += [p + f"mlp.experts.{n}", p + f"mlp.experts.{n}_scale"]
            if kind == "nvfp4_nearest":
                names += [p + f"mlp.experts.{n}_amax",
                          p + f"mlp.experts.{n}_outer_scale_denominator"]
            groups[kind].append(p + f"mlp.experts.{n}")
        if 6 <= i < 18:
            names.append(p + "controllers.attn.down_weight")
    names = [n for n in names if n != drop]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}))
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 16)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "quant_method": "inkling2",
        "model": {"parameter_groups": [{"weight_transform": "cast", "tensors": []}] + [
            {"weight_transform": kind, "tensors": t} for kind, t in groups.items()]},
    }))
    return tmp_path


def test_inspect_artifact_accepts_mini_and_records_formats(tmp_path):
    from mlx2.adapters.inkling2 import inspect_artifact

    artifact = inspect_artifact(_artifact(tmp_path))
    assert artifact["expert_kinds"][0] == "nvfp4_nearest"
    assert artifact["expert_kinds"][23] == "mxfp8_nearest"
    assert artifact["has_mtp"] is False


@pytest.mark.parametrize("change", [
    dict(text_overrides={"n_routed_experts": 384}),
    dict(text_overrides={"loop_config": {"core_start": 6, "core_end": 18, "num_passes": 2,
                                         "gate_rank": 64, "is_varloop": False}}),
    dict(drop="model.llm.layers.3.mlp.experts.w2_weight_amax"),
    dict(drop="model.llm.layers.7.controllers.attn.down_weight"),
])
def test_inspect_artifact_fails_closed(tmp_path, change):
    from mlx2.adapters.inkling2 import inspect_artifact

    with pytest.raises(ValueError):
        inspect_artifact(_artifact(tmp_path, **change))


def test_registry_resolves_inkling2_loop(tmp_path):
    from mlx2.adapters.registry import _RESOLVERS

    resolution = _RESOLVERS["inkling2_loop"](_artifact(tmp_path), {})
    assert resolution.adapter_type.__name__ == "Inkling2Adapter"


def test_prompt_effort_rendering():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    render = Inkling2Adapter._reasoning_effort
    assert render({}) is None
    assert render({"enable_thinking": False}) == "none"
    assert render({"reasoning_effort": "low"}) == "low"
    assert render({"thinking_budget": 0}) == "none"
    assert render({"thinking_budget": 0, "reasoning_effort": "none"}) == "none"
    with pytest.raises(ValueError, match="conflicts"):
        render({"thinking_budget": 0, "reasoning_effort": "high"})
    with pytest.raises(ValueError, match="conflicts"):
        render({"thinking_budget": 0, "enable_thinking": True})
    Inkling2Adapter._messages({"messages": [], "thinking_budget": 0})
    for budget in (64, False, 0.0):
        with pytest.raises(ValueError, match="budgets"):
            Inkling2Adapter._messages({"messages": [], "thinking_budget": budget})
    with pytest.raises(ValueError, match="media"):
        Inkling2Adapter._messages({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}}]}]})
    with pytest.raises(ValueError, match="tool"):
        Inkling2Adapter._messages({"messages": [], "tools": [{"type": "function"}]})


def test_forward_with_taps_matches_ordinary_and_reference_contract():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 9))
    ordinary = model(tokens, cache=model.make_cache())
    logits, features = model.forward_with_taps(tokens, model.make_cache(), [5, 23, 41, 48])
    assert mx.array_equal(logits, ordinary)
    assert features.shape == (1, 9, 4 * TINY["hidden_size"])
    # Tap i is the hidden ENTERING visit i; tap 48 the final normed hidden.
    seen = {}
    original = model.model.layers[0].__class__.__call__

    def spy(layer, x, blocks, cache=None, messages=None):
        seen.setdefault(len(seen), x)
        return original(layer, x, blocks, cache, messages)

    type(model.model.layers[0]).__call__ = spy
    try:
        model(tokens, cache=model.make_cache())
    finally:
        type(model.model.layers[0]).__call__ = original
    width = TINY["hidden_size"]
    assert mx.array_equal(features[..., :width], seen[5])
    assert mx.array_equal(features[..., width:2 * width], seen[23])
    assert mx.array_equal(features[..., 2 * width:3 * width], seen[41])
    _, body = model.forward_with_taps(tokens, model.make_cache(), [48], body_only=True)
    assert mx.array_equal(body, features[..., 3 * width:])
    for bad in ([], [49], [5, 5], [-1]):
        with pytest.raises(ValueError):
            model.forward_with_taps(tokens, model.make_cache(), bad)


def test_fused_projections_match_separate_and_keep_memory():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 13))
    plan = [(0, 9)] + [(i, i + 1) for i in range(9, 13)]
    base = _run(model, tokens, plan)
    before = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    model.configure_kernels(fused_projections=True)
    attn = model.model.layers[5].self_attn
    assert attn.q_proj.weight.shape == (128, 128) and attn.k_proj.weight.shape == (64, 128)
    fused = _run(model, tokens, plan)
    assert model.kernel_receipt()["fused_projection_calls"] == 48 * len(plan)
    assert mx.abs(fused - base).max().item() < 1e-5
    assert sum(v.nbytes for _, v in tree_flatten(model.parameters())) == before


def test_policy_parsing():
    from mlx2.adapters.inkling2 import validate_policy

    assert validate_policy(None) == (
        {"fused_projections": False, "head_kernel": False, "sink_sdpa": False,
         "sink_slot": False, "flash_prefill": False, "verify_splitk": False,
         "moe_fused": True, "moe_weighted_sum": True,
         "moe_nax_gather": None, "hard_no_thinking": False, "dspark_prefill_reclaim": None,
         "prefill_mask_mb": None, "loop_passes": None, "fuse_controllers": False, "fuse_residual": False, "fuse_sconv": False,
         "fuse_attention": False, "fuse_sconv_kernel": False, "fuse_residual_kernel": False,
         "moe_act_quant": False}, {})
    assert validate_policy({"inkling2_fused_projections": True})[0]["fused_projections"] is True
    assert validate_policy({"draft_model": "x"})[1] == {"draft_model": "x"}
    with pytest.raises(TypeError):
        validate_policy({"inkling2_fused_projections": 1})


def test_fused_projection_toggle_round_trip():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 5))
    base = model(tokens)
    for flag in (True, False, True, False):
        model.configure_kernels(fused_projections=flag)
        assert mx.abs(model(tokens) - base).max().item() < 1e-5
    assert model.kernel_policy == {"fused_projections": False, "head_kernel": False,
                                   "sink_sdpa": False, "loop_passes": None,
                                   "prefill_mask_mb": None, "flash_prefill": False,
                                   "moe_fused": False, "moe_weighted_sum": False,
                                   "moe_nax_gather": False, "sink_slot": False,
                                   "fuse_controllers": False, "fuse_residual": False, "fuse_sconv": False,
                                   "fuse_attention": False, "fuse_sconv_kernel": False, "fuse_residual_kernel": False,
                                   "verify_splitk": False, "moe_act_quant": False}


@pytest.mark.parametrize("rows", [1, 2])
def test_sink_sdpa_decode_matches_concatenated_sink(rows):
    """One-token rows through SDPA ``sinks``: the same softmax, summed in a
    different order (FP32 ~1e-7 relative), with the ring wrapped (window 8)
    and log scaling engaged; prefill keeps the concat path."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (rows, 20))
    plan = [(0, 9)] + [(i, i + 1) for i in range(9, 20)]
    base = _run(model, tokens, plan)
    model.configure_kernels(sink_sdpa=True)
    sink = _run(model, tokens, plan)
    assert model.kernel_receipt()["sink_sdpa_calls"] == 48 * 11
    assert mx.abs(sink - base).max().item() < 1e-5
    model.configure_kernels()
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_small_weights_cast_once_and_follow_updates():
    from mlx2.runtime.models.inkling2.language import _fp32

    from mlx.utils import tree_flatten

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 6))
    # Only the cached weights go BF16 (CPU gather_mm is FP32-only).
    small = ("gate_weight", "forget_weight", "gain_weight", "conv.weight", "static_weight")
    model.load_weights([(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())
                        if k.endswith(small)], strict=False)
    moe = model.model.layers[0].mlp
    assert moe.gate_weight.dtype == mx.bfloat16
    first = _fp32(moe, "gate_weight")
    assert first.dtype == mx.float32 and _fp32(moe, "gate_weight") is first
    assert mx.array_equal(first, moe.gate_weight.astype(mx.float32))
    assert all(v.dtype == mx.bfloat16 for k, v in tree_flatten(model.parameters())
               if k.endswith("gate_weight"))  # the cache is not a parameter
    base = model(tokens)
    assert mx.array_equal(model(tokens), base)  # cached casts are exact
    model.load_weights([("model.layers.0.mlp.gate_weight", moe.gate_weight * 2)], strict=False)
    assert mx.array_equal(_fp32(moe, "gate_weight"), moe.gate_weight.astype(mx.float32))
    assert not mx.array_equal(_fp32(moe, "gate_weight"), first)


def test_inkling2_parser_accepts_template_header_names_and_text_markers():
    from mlx2.adapters.inkling_output import INKLING2_TEXT_MARKERS, InklingOutputParser

    def run(text, **kw):
        parser = InklingOutputParser(chat=True, **kw)
        events = []
        for piece in (text[:7], text[7:23], text[23:]):
            events += parser.push(piece)
        return events + parser.finish("", "stop")

    reply = ("<|message_model|>assistant<|content_thinking|>hm<|end_message|>"
             "<|message_model|><|content_tool_error|>bad call<|end_message|>"
             "<|message_model|>assistant<|content_text|>Hi<|content_model_end_sampling|>")
    for size in range(1, len(reply) + 1):  # every chunk size, markers split
        parser = InklingOutputParser(chat=True, text_markers=INKLING2_TEXT_MARKERS,
                                     header_names=True)
        events = []
        for start in range(0, len(reply), size):
            events += parser.push(reply[start:start + size])
        events += parser.finish("", "stop")
        assert "".join(e.get("reasoning_content", "") for e in events) == "hm"
        assert "".join(e.get("content", "") for e in events) == "bad callHi"
    with pytest.raises(ValueError, match="header"):  # a name needs <|message_model|>
        run("<|message_model|><|content_text|>a<|end_message|>junk<|content_text|>b",
            text_markers=INKLING2_TEXT_MARKERS, header_names=True)
    with pytest.raises(ValueError, match="header"):
        run(reply)  # the v1 parser keeps failing closed
    with pytest.raises(ValueError, match="header"):
        run("<|message_model|>x<|content_xml|>y", text_markers=INKLING2_TEXT_MARKERS,
            header_names=True)


def test_every_loop_pass_uses_its_layers_nvfp4_global_scales():
    """llama.cpp #32704: loop passes 2-3 lost the per-expert NVFP4 global
    scale in copied loop slots (PPL 7.65 -> 130).  Here every visit of a core
    layer must run that layer's own scaled experts, with the installed scale."""
    from mlx2.adapters.inkling2 import NVFP4, install_experts
    model = _tiny_model()
    num, hidden, inter = 8, 128, 64
    layers, installed = {}, {}
    for i in range(24):
        mx.random.seed(100 + i)
        a = _modelopt_nvfp4(mx.random.normal((num, 2 * inter, hidden)) * (1 + i), num)
        b = _modelopt_nvfp4(mx.random.normal((num, hidden, inter)), num)
        layers[i] = {
            "w13_weight": a["weight"], "w13_weight_scale": a["scale"],
            "w13_weight_amax": a["amax"], "w13_weight_outer_scale_denominator": a["denominator"],
            "w2_weight": b["weight"], "w2_weight_scale": b["scale"],
            "w2_weight_amax": b["amax"], "w2_weight_outer_scale_denominator": b["denominator"],
        }
        installed[i] = a["global"]
    install_experts(model, layers, {i: NVFP4 for i in range(24)})
    scaled = type(model.model.layers[0].mlp.switch_mlp.gate_proj)
    owner = {id(model.model.layers[i].mlp.switch_mlp.gate_proj): i for i in range(24)}
    seen = []
    original = scaled.__call__

    def spy(self, *args, **kwargs):
        if id(self) in owner:
            seen.append((owner[id(self)], self.weight_scale_2))
        return original(self, *args, **kwargs)

    scaled.__call__ = spy
    try:
        mx.eval(model(mx.random.randint(0, 500, (1, 4))))
    finally:
        scaled.__call__ = original
    assert [i for i, _ in seen] == list(model.model.layer_indices)
    for i, scale in seen:
        assert mx.allclose(scale, installed[i])


def test_pass_probe_records_every_visit_and_restores_hooks():
    import importlib.util
    from pathlib import Path

    from mlx2.runtime.models.inkling2 import language

    path = Path(__file__).resolve().parents[1] / "scripts/research/inkling2_pass_probe.py"
    spec = importlib.util.spec_from_file_location("inkling2_pass_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [(ord(c) * 7 + i) % 500 for i, c in enumerate(text[:40])]

    model = _tiny_model()
    originals = (language.DecoderLayer.__call__, language._route,
                 language.MessageController.__call__)
    result = probe.run(model, Tok(), lambda q: Tok().encode(q), tokens_per_doc=40, gsm8k=2,
                       chunk=16)
    assert (language.DecoderLayer.__call__, language._route,
            language.MessageController.__call__) == originals
    layers = result["layers"]
    assert [l["layer"] for l in layers] == list(range(6, 18))
    assert layers[0]["visits"] == [6, 18, 30]
    pair = layers[0]["pairs"]["1-2"]
    assert 0 <= pair["jaccard_same_token"] <= 1 and set(pair["by_kind"]) == {
        "code", "gsm8k_chat", "prose"}
    # Controllers run on passes 2-3 only, four per core layer visit.
    assert set(result["controllers"]["6"]) == {
        f"pass{p}.{n}" for p in (2, 3) for n in ("attn", "k", "v", "mlp")}


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_varloop_changes_visits_and_stays_self_consistent(passes):
    from mlx2.adapters.inkling2 import validate_policy

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 21))
    base = _run(model, tokens, [(0, 21)])
    assert model.apc_v2_layout == "inkling2-kv-four-shortconv-v1"
    assert model.loop_receipt()["out_of_distribution"] is False
    model.configure_kernels(loop_passes=passes)
    assert len(model.model.layer_indices) == 12 + 12 * passes
    assert len(model.make_cache()) == 2 * (12 + 12 * passes)
    assert model.apc_v2_layout.endswith(f"-passes{passes}")
    assert model.loop_receipt() == {"passes": passes, "trained_passes": 3,
                                    "visits": 12 + 12 * passes, "out_of_distribution": True}
    varied = _run(model, tokens, [(0, 21)])
    assert not mx.allclose(varied, base)  # the loop count really changed
    decode = _run(model, tokens, [(0, 16)] + [(i, i + 1) for i in range(16, 21)])
    chunks = _run(model, tokens, [(0, 7), (7, 14), (14, 21)])
    assert mx.abs(decode - varied).max().item() < 1e-4
    assert mx.abs(chunks - varied).max().item() < 1e-4
    model.configure_kernels()
    assert mx.array_equal(_run(model, tokens, [(0, 21)]), base)
    assert model.apc_v2_layout == "inkling2-kv-four-shortconv-v1"
    assert validate_policy({"inkling2_loop_passes": passes})[0]["loop_passes"] == passes
    for bad in (0, 9, True, 2.0):
        with pytest.raises(TypeError):
            validate_policy({"inkling2_loop_passes": bad})


def test_hard_no_thinking_prefills_text_channel_only_when_asked():
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            text = f"<prompt effort={kw.get('reasoning_effort')}>"
            return [7, 8, 9] if tokenize else text

        def convert_tokens_to_ids(self, token):
            return {"<|message_model|>": 200001, "<|content_text|>": 200004}[token]

    adapter = object.__new__(Inkling2Adapter)
    adapter.tokenizer = Tok()
    off = {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False}
    on = {"messages": [{"role": "user", "content": "hi"}]}
    adapter.hard_no_thinking = False
    assert adapter.prompt_tokens(off) == [7, 8, 9]  # soft hint only (default)
    assert adapter.thinking_control(off)["mode"] == "template_effort"
    adapter.hard_no_thinking = True
    assert adapter.prompt_tokens(off) == [7, 8, 9, 200001, 200004]
    assert adapter.render_prompt(off).endswith("<|message_model|><|content_text|>")
    assert adapter.prompt_tokens({**on, "reasoning_effort": "none"})[-2:] == [200001, 200004]
    assert adapter.prompt_tokens({**on, "thinking_budget": 0})[-2:] == [200001, 200004]
    assert adapter.prompt_tokens(on) == [7, 8, 9]  # thinking requested: untouched
    with pytest.raises(ValueError, match="conflicts"):
        adapter.prompt_tokens({**off, "reasoning_effort": "high"})
    assert adapter.thinking_control({"prompt": "x"})["mode"] == "raw_prompt"
    assert adapter.thinking_control(off) == {"mode": "hard_text_prefill",
                                             "reasoning_effort": "none",
                                             "policy_hard_no_thinking": True}
    parser = adapter.output_parser(off)
    events = parser.push("Paris<|content_model_end_sampling|>")
    assert events == [{"content": "Paris"}]
    assert validate_policy({"inkling2_hard_no_thinking": True})[0]["hard_no_thinking"] is True
    with pytest.raises(TypeError):
        validate_policy({"inkling2_hard_no_thinking": 1})


def test_prefill_mask_budget_changes_row_chunks_not_results():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 30))
    base = _run(model, tokens, [(0, 30)])
    import mlx2.runtime.models.inkling2.language as lang
    old = lang._MASK_BUDGET_BYTES
    try:
        lang._MASK_BUDGET_BYTES = 4096  # force many row chunks on the reference path
        chunked = _run(model, tokens, [(0, 30)])
    finally:
        lang._MASK_BUDGET_BYTES = old
    model.configure_kernels(prefill_mask_mb=16)
    out = _run(model, tokens, [(0, 30)])
    rows_16 = model.model.layers[5].self_attn.mask_rows_last
    model.configure_kernels()
    _run(model, tokens, [(0, 30)])
    assert model.model.layers[5].self_attn.mask_rows_last == 4 * rows_16  # 64 vs 16 MiB
    assert mx.abs(chunked - base).max().item() < 1e-5
    assert mx.abs(out - base).max().item() < 1e-5
    from mlx2.adapters.inkling2 import validate_policy
    assert validate_policy({"inkling2_prefill_mask_mb": 512})[0]["prefill_mask_mb"] == 512
    with pytest.raises(ValueError):
        validate_policy({"inkling2_prefill_mask_mb": 4096})
    with pytest.raises(TypeError):
        validate_policy({"inkling2_prefill_mask_mb": "512"})
    # The option reaches the row count: 16 MiB vs 64 MiB at a long key span.
    attn = model.model.layers[5].self_attn
    model.configure_kernels(prefill_mask_mb=16)
    assert attn._mask_budget_bytes == 16 << 20
    model.configure_kernels()
    assert attn._mask_budget_bytes is None


def test_flash_prefill_is_gpu_only_and_leaves_cpu_path_unchanged():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 12))
    base = _run(model, tokens, [(0, 12)])
    model.configure_kernels(flash_prefill=True)
    assert mx.array_equal(_run(model, tokens, [(0, 12)]), base)
    assert model.kernel_receipt()["flash_prefill_calls"] == 0


def test_dspark_prefill_reclaim_policy(monkeypatch):
    """Opt-in, boolean, DSpark-only; it reaches the external generator."""
    from mlx2.adapters import external_draft_policy
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    assert validate_policy({"inkling2_dspark_prefill_reclaim": True})[0]["dspark_prefill_reclaim"] is True
    # Unset resolves at load: on for the DSpark route, off otherwise.
    assert validate_policy(None)[0]["dspark_prefill_reclaim"] is None
    assert validate_policy({"inkling2_dspark_prefill_reclaim": False})[0]["dspark_prefill_reclaim"] is False
    for bad in (1, None):
        with pytest.raises(TypeError):
            validate_policy({"inkling2_dspark_prefill_reclaim": bad})
    with pytest.raises(ValueError, match="requires the DSpark route"):
        Inkling2Adapter("/nonexistent", execution_policy={"inkling2_dspark_prefill_reclaim": True})
    seen = []
    monkeypatch.setattr(external_draft_policy.ExternalDraftAdapterMixin, "create_external_batch",
                        lambda self, **kw: seen.append(kw) or "batch")
    adapter = object.__new__(Inkling2Adapter)
    adapter.dspark_prefill_reclaim = False
    assert adapter.create_external_batch() == "batch" and seen[-1] == {}
    adapter.dspark_prefill_reclaim = True
    adapter.create_external_batch()
    assert seen[-1] == {"prefill_allocator_reclaim": True}


@pytest.mark.parametrize("rows", [1, 2])
def test_sink_slot_is_bitwise_reference(rows):
    """``inkling2_sink_slot``: the zero sink row is written into the KV
    buffer's spare capacity and read as a view instead of concatenated. It
    must be bit-identical to the concat path. The run crosses the 256-token
    growth boundary (where the buffer is full and the slot falls back)."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (rows, 262))
    plan = [(0, 200), (200, 256)] + [(i, i + 1) for i in range(256, 262)]
    base = _run(model, tokens, plan)
    model.configure_kernels(sink_slot=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    assert mx.array_equal(got, base)
    assert receipt["sink_slot_calls"] > 0
    # The 256-key fill leaves no spare slot: those global calls fall back.
    assert receipt["sink_slot_fallbacks"] > 0
    model.configure_kernels()
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_sink_slot_after_trim_is_bitwise_reference():
    """A trimmed KVCache (speculative rollback) leaves stale keys past offset;
    the slot overwrites the one it reads, so results stay bit-identical."""
    from mlx2.runtime.models.cache import KVCache

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 30))

    def run():
        cache = model.make_cache()
        outs = [model(tokens[:, :20], cache=cache)]
        outs.append(model(tokens[:, 20:25], cache=cache))
        for plane in cache:
            if type(plane) is KVCache:
                plane.trim(3)
        # One replayed token: its slot (offset 23) still holds a stale key
        # from before the trim, which the slot must overwrite with zeros.
        outs.append(model(tokens[:, 22:23], cache=cache))
        outs.append(model(tokens[:, 23:30], cache=cache))
        return mx.concatenate(outs, axis=1)

    base = run()
    model.configure_kernels(sink_slot=True)
    got = run()
    calls = model.kernel_receipt()["sink_slot_calls"]
    model.configure_kernels()
    assert calls > 0
    assert mx.array_equal(got, base)


# ----- inkling2_verify_splitk (GQA split-K short-block attention) -----


def test_verify_splitk_policy_default_off_and_boolean():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, validate_policy

    assert KERNEL_OPTIONS["inkling2_verify_splitk"] == "verify_splitk"
    assert validate_policy(None)[0]["verify_splitk"] is False
    assert validate_policy({"inkling2_verify_splitk": True})[0]["verify_splitk"] is True
    for bad in (1, None, "true"):
        with pytest.raises(TypeError):
            validate_policy({"inkling2_verify_splitk": bad})
    model = _tiny_model()
    assert model.model.layers[0].self_attn.__dict__.get("_verify_splitk") is None
    model.configure_kernels()
    assert model.kernel_policy["verify_splitk"] is False
    assert model.kernel_receipt()["verify_splitk_calls"] == 0


def test_verify_splitk_declines_on_cpu_and_is_bitwise_reference():
    """On the CPU every short block declines (counted as ``cpu``); prefill
    chunks of 32+ rows decline as ``rows``; output is the reference, bitwise."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 50))
    plan = [(0, 33), (33, 41)] + [(i, i + 1) for i in range(41, 50)]
    base = _run(model, tokens, plan)
    model.configure_kernels(verify_splitk=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    model.configure_kernels()
    assert mx.array_equal(got, base)
    assert receipt["verify_splitk_calls"] == 0
    assert receipt["verify_splitk_declines"] == {"declined:rows": 48, "declined:cpu": 48 * 10}


def test_verify_splitk_decline_reasons():
    from mlx2.runtime.models.inkling2 import splitk

    def arrays(length, dtype=mx.bfloat16, kdtype=None):
        q = mx.zeros((1, 24, length, 128), dtype)
        k = mx.zeros((1, 8, 64, 128), kdtype or dtype)
        return q, k, k

    geo = dict(n_heads=24, n_kv=8, head_dim=128)
    assert splitk.decline_reason(*arrays(8), **geo, device=mx.cpu) == "cpu"
    assert splitk.decline_reason(*arrays(8), **geo, device=mx.gpu) is None
    assert splitk.decline_reason(*arrays(1), **geo, device=mx.gpu) is None
    assert splitk.decline_reason(*arrays(31, mx.float32), **geo, device=mx.gpu) is None
    assert splitk.decline_reason(*arrays(32), **geo, device=mx.gpu) == "rows"
    assert splitk.decline_reason(*arrays(8, mx.float16), **geo, device=mx.gpu) == "dtype"
    assert splitk.decline_reason(*arrays(8, kdtype=mx.float32), **geo, device=mx.gpu) == "dtype"
    q, k, _ = arrays(8)
    assert splitk.decline_reason(q, (k, k), k, **geo, device=mx.gpu) == "cache_format"
    for bad in (dict(head_dim=48), dict(head_dim=256), dict(n_heads=20)):
        assert splitk.decline_reason(*arrays(8), **{**geo, **bad}, device=mx.gpu) == "geometry"
    # A cache type the kernel was not checked against declines before the kernel.
    from mlx2.runtime.models.cache import QuantizedKVCache

    attn = _tiny_model().model.layers[5].self_attn
    assert attn._splitk_decline(QuantizedKVCache(), q, k, k) == "cache_type"


def _engage_splitk_reference(monkeypatch):
    """Make the split-K route engage on the CPU with the composed law in place
    of the Metal kernel: checks the wiring (offset, padding, window, tau-scaled
    q and bias rows, sinks) that the kernel receives."""
    from mlx2.runtime.models.inkling2 import splitk

    seen = []

    def fake(q, k, v, rel, sinks, **kw):
        seen.append((q.shape[2], kw["window"], kw["left_padding"] is not None))
        return splitk.reference_attention(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(splitk, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(splitk, "gqa_splitk_attention", fake)
    return seen


@pytest.mark.parametrize("rows", [1, 2])
def test_verify_splitk_wiring_matches_composed_path(monkeypatch, rows):
    """Engaged for every block (prefill 9, verify 8, decode 1) with a wrapped
    ring (window 8) and log scaling: the arguments passed reproduce the
    composed path bitwise."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (rows, 30))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 30)]
    base = _run(model, tokens, plan)
    seen = _engage_splitk_reference(monkeypatch)
    model.configure_kernels(verify_splitk=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    model.configure_kernels()
    assert receipt["verify_splitk_calls"] == 48 * len(plan) and receipt["verify_splitk_declines"] == {}
    assert {w for _, w, _ in seen} == {0, 8}
    assert mx.array_equal(got, base)


@pytest.mark.parametrize("sliding", ["ring", "unbounded"])
def test_verify_splitk_wiring_with_left_padding(monkeypatch, sliding):
    """Batched caches with left padding (global BatchKVCache; sliding
    BatchRotatingKVCache, or an unbounded BatchKVCache whose keys the model
    crops to the window, shifting the padding): padding reaches the kernel."""
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (2, 24))
    plan = [(0, 12), (12, 20)] + [(i, i + 1) for i in range(20, 24)]

    def run():
        cache = model.make_cache()
        for i in range(0, len(cache), 2):
            cache[i] = (BatchKVCache([0, 3])
                        if type(cache[i]) is KVCache or sliding == "unbounded"
                        else BatchRotatingKVCache(8, [0, 3]))
        return mx.concatenate([model(tokens[:, a:b], cache=cache) for a, b in plan], axis=1)

    base = run()
    seen = _engage_splitk_reference(monkeypatch)
    model.configure_kernels(verify_splitk=True)
    got = run()
    calls = model.kernel_receipt()["verify_splitk_calls"]
    model.configure_kernels()
    assert calls == 48 * len(plan) and all(padded for _, _, padded in seen)
    assert mx.array_equal(got, base)
